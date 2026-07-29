"""
babel_secuencia.py
==================
Bloque C — Estructura secuencial: ¿el pasado predice el futuro?

Cubre C.5 (perplejidad) y C.6 (cadenas de Markov). Las dos tareas comparten
maquinaria a propósito: la entropía condicional y la perplejidad son la misma
medida vista de dos formas, y calcularlas sobre los mismos conteos garantiza
que no se contradigan entre sí.

La pregunta que resuelve el módulo
----------------------------------
El Bloque A midió la entropía de orden 0: cuánta incertidumbre hay en cada
carácter *aislado*. Eso deja abierta la pregunta de verdad interesante: **¿saber
los k caracteres anteriores reduce la incertidumbre del siguiente?**

    H(X)          incertidumbre del carácter suelto
    H(X | ctx_k)  incertidumbre sabiendo los k anteriores
    ganancia      H(X) - H(X | ctx_k)

En texto i.i.d. la ganancia es CERO por construcción: el pasado no informa. En
lenguaje natural es grande y crece con k — la `q` predice la `u`, el artículo
predice el sustantivo. Es el test más directo de estructura que queda.

Por qué el null model es imprescindible aquí
--------------------------------------------
La entropía condicional estimada por conteo tiene **sesgo de muestra finita, y
va en la dirección peligrosa**: sobreestima la estructura. Con orden 3 sobre 29
símbolos hay 24.389 contextos; con 1,1M de caracteres tocan ~45 observaciones
por contexto, y esa escasez hace que las frecuencias observadas parezcan más
deterministas de lo que son.

Es decir: **Babel va a mostrar ganancia de información aunque no tenga ninguna
estructura.** Sin el null model se leería como hallazgo. Con él, el sesgo se
cancela porque afecta a ambos por igual — que es exactamente el papel para el
que se construyó (§5.8).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

ORDENES_DEFECTO = (1, 2, 3)


# ============================================================================
# 1. CONTEOS DE TRANSICIÓN
# ============================================================================
def _preparar(texto: str, excluir: tuple = ("\n",)) -> str:
    """Retira los símbolos excluidos. El '\\n' de Babel es formato, no contenido."""
    if not excluir:
        return texto
    fuera = set(excluir)
    return "".join(ch for ch in texto if ch not in fuera)


def contar_transiciones(
    texto: str,
    orden: int = 1,
    excluir: tuple = ("\n",),
) -> dict[str, Counter]:
    """
    Cuenta, para cada contexto de `orden` caracteres, qué símbolo lo sigue.

    Devuelve {contexto: Counter({siguiente: n})}. Con `orden=0` devuelve un
    único contexto vacío, es decir la distribución marginal.
    """
    t = _preparar(texto, excluir)
    tabla: dict[str, Counter] = defaultdict(Counter)
    if len(t) <= orden:
        return dict(tabla)

    for i in range(len(t) - orden):
        tabla[t[i : i + orden]][t[i + orden]] += 1
    return dict(tabla)


def matriz_transicion(
    texto: str,
    orden: int = 1,
    alfabeto: list[str] | None = None,
    excluir: tuple = ("\n",),
) -> dict:
    """
    Matriz de probabilidades condicionales P(siguiente | contexto).

    Devuelve la matriz densa (contextos × símbolos), sus etiquetas y los
    conteos por contexto. Útil para C.6: comparar visualmente la matriz de
    Babel contra la del español revela de un vistazo si hay dependencias.
    """
    tabla = contar_transiciones(texto, orden, excluir)
    simbolos = sorted(alfabeto) if alfabeto else sorted(
        {ch for c in tabla.values() for ch in c} | set(tabla.keys()))
    if not alfabeto:
        simbolos = sorted({ch for c in tabla.values() for ch in c})

    contextos = sorted(tabla)
    idx = {s: i for i, s in enumerate(simbolos)}
    M = np.zeros((len(contextos), len(simbolos)), dtype=float)
    n_por_contexto = np.zeros(len(contextos), dtype=float)

    for f, ctx in enumerate(contextos):
        c = tabla[ctx]
        total = sum(c.values())
        n_por_contexto[f] = total
        for ch, n in c.items():
            if ch in idx:
                M[f, idx[ch]] = n / total

    return {
        "matriz": M,
        "contextos": contextos,
        "simbolos": simbolos,
        "n_por_contexto": n_por_contexto,
        "orden": orden,
    }


# ============================================================================
# 2. ENTROPÍA CONDICIONAL Y GANANCIA DE INFORMACIÓN
# ============================================================================
def entropia_condicional(
    texto: str,
    orden: int = 1,
    excluir: tuple = ("\n",),
) -> float:
    """
    H(X | contexto de `orden` caracteres), en bits.

    Media de las entropías por contexto, ponderada por la frecuencia del
    contexto. Con `orden=0` equivale a la entropía de Shannon marginal.

    ⚠️ Estimador plug-in: sesgado a la BAJA (aparenta más estructura de la que
    hay) cuando los contextos son escasos. Ver la nota del encabezado: es
    justamente por eso que el resultado solo se interpreta contra el null model.
    """
    tabla = contar_transiciones(texto, orden, excluir)
    if not tabla:
        return 0.0

    total_global = sum(sum(c.values()) for c in tabla.values())
    if total_global == 0:
        return 0.0

    H = 0.0
    for c in tabla.values():
        n_ctx = sum(c.values())
        h_ctx = -sum((n / n_ctx) * math.log2(n / n_ctx) for n in c.values())
        H += (n_ctx / total_global) * h_ctx
    return H


def ganancia_informacion(
    texto: str,
    ordenes: tuple[int, ...] = ORDENES_DEFECTO,
    excluir: tuple = ("\n",),
) -> dict:
    """
    Cuánta incertidumbre elimina conocer los k caracteres anteriores.

    La métrica central de C.6. Devuelve, por orden:
      - `H_k`        : entropía condicional
      - `ganancia_k` : H(X) − H(X | ctx_k)
      - `n_contextos`: contextos observados (para juzgar la escasez)
      - `obs_por_contexto`: observaciones medias por contexto

    En texto i.i.d. la ganancia real es 0. Lo que se mida por encima es sesgo
    de muestra finita, y el null model lo cuantifica.
    """
    H0 = entropia_condicional(texto, 0, excluir)
    salida = {"H0": H0}

    for k in ordenes:
        tabla = contar_transiciones(texto, k, excluir)
        Hk = entropia_condicional(texto, k, excluir)
        n_ctx = len(tabla)
        n_obs = sum(sum(c.values()) for c in tabla.values())
        salida[f"H{k}"] = Hk
        salida[f"ganancia_{k}"] = H0 - Hk
        salida[f"n_contextos_{k}"] = n_ctx
        salida[f"obs_por_contexto_{k}"] = n_obs / n_ctx if n_ctx else 0.0
    return salida


# ============================================================================
# 3. MODELO DE LENGUAJE DE CARACTERES Y PERPLEJIDAD
# ============================================================================
class ModeloCaracteres:
    """
    Modelo de lenguaje de n-gramas de caracteres con backoff interpolado
    (Jelinek-Mercer).

        P(x | ctx_k) = λ·P_ML(x | ctx_k) + (1−λ)·P(x | ctx_{k−1})

    y en la base, la uniforme sobre el alfabeto. El backoff no es un adorno:
    con orden 5 sobre 29 símbolos hay 20,5 millones de contextos posibles y
    ningún corpus de 1M de caracteres los cubre, así que casi todo contexto de
    test es inédito. Sin backoff la perplejidad sería infinita.

    Se implementa aquí en vez de usar `nltk.lm` para que comparta los conteos
    con la entropía condicional: las dos métricas salen del mismo objeto y no
    pueden contradecirse. `nltk.lm` queda disponible como contraste externo.
    """

    def __init__(self, orden: int = 5, lam: float = 0.7, excluir: tuple = ("\n",)):
        self.orden = orden
        self.lam = lam
        self.excluir = excluir
        self.tablas: list[dict[str, Counter]] = []
        self.alfabeto: list[str] = []

    def entrenar(self, texto: str) -> "ModeloCaracteres":
        t = _preparar(texto, self.excluir)
        self.alfabeto = sorted(set(t))
        self.tablas = [contar_transiciones(texto, k, self.excluir)
                       for k in range(self.orden + 1)]
        return self

    def _prob(self, contexto: str, siguiente: str) -> float:
        p = 1.0 / len(self.alfabeto) if self.alfabeto else 0.0
        # De orden 0 hacia arriba, interpolando en cada nivel
        for k in range(0, self.orden + 1):
            if k >= len(self.tablas):
                break
            ctx = contexto[-k:] if k else ""
            c = self.tablas[k].get(ctx)
            if not c:
                continue
            total = sum(c.values())
            p_ml = c.get(siguiente, 0) / total
            p = self.lam * p_ml + (1 - self.lam) * p
        return p

    def log_verosimilitud(self, texto: str) -> tuple[float, int]:
        """Devuelve (suma de log2 P, número de caracteres evaluados)."""
        t = _preparar(texto, self.excluir)
        if len(t) <= self.orden:
            return 0.0, 0
        total = 0.0
        n = 0
        for i in range(self.orden, len(t)):
            p = self._prob(t[i - self.orden : i], t[i])
            if p <= 0:
                p = 1e-12          # el backoff no debería llegar aquí
            total += math.log2(p)
            n += 1
        return total, n

    def perplejidad(self, texto: str) -> float:
        """
        Perplejidad por carácter: 2^(entropía cruzada).

        Interpretación directa: *entre cuántos símbolos equiprobables está
        dudando el modelo* en cada posición. La referencia natural es el tamaño
        del alfabeto (29 aquí): un modelo que no sabe nada duda entre los 29.

        ⚠️ **La perplejidad puede superar 29, y eso significa algo concreto.**
        Un modelo entrenado en español sobre texto de Babel alcanza ~230, es
        decir ocho veces peor que no saber nada. No es un error de cálculo: el
        modelo no está *ignorante*, está **confiadamente equivocado**. Ha
        aprendido que tras `qu` viene `e`, y Babel le pone una `z`; cada
        predicción firme y fallida cuesta muchos bits.

        Dos cosas a tener presentes al interpretarlo:

        1. La magnitud depende del suavizado. Con λ fijo, cada nivel de backoff
           que falla multiplica la probabilidad por (1−λ), así que un contexto
           conocido con continuación inédita se penaliza mucho. Un suavizado
           adaptativo (Kneser-Ney, Witten-Bell) daría números más bajos.
        2. Por eso **la lectura defendible es la comparativa**, no el valor
           absoluto: Babel frente a español retenido, y Babel frente al null
           model, todos bajo el mismo modelo y el mismo suavizado.

        Para interpretación en bits, usar `entropia_cruzada()`: ahí el techo
        uniforme es log2(29) = 4,858 y la comparación es más legible.
        """
        logp, n = self.log_verosimilitud(texto)
        if n == 0:
            return float("nan")
        return 2 ** (-logp / n)

    def entropia_cruzada(self, texto: str) -> float:
        logp, n = self.log_verosimilitud(texto)
        return float("nan") if n == 0 else -logp / n


# ============================================================================
# 4. COMPARACIÓN DE MATRICES DE TRANSICIÓN
# ============================================================================
def divergencia_transiciones(
    texto_a: str,
    texto_b: str,
    orden: int = 1,
    excluir: tuple = ("\n",),
) -> dict:
    """
    Divergencia de Jensen-Shannon media entre las matrices de transición de
    dos textos, ponderada por la frecuencia de cada contexto.

    Se usa JS y no KL porque es simétrica, está acotada en [0, 1] bits y no
    explota cuando un símbolo tiene probabilidad cero en una de las dos — que
    con contextos escasos pasa constantemente.
    """
    ta = contar_transiciones(texto_a, orden, excluir)
    tb = contar_transiciones(texto_b, orden, excluir)
    comunes = set(ta) & set(tb)
    if not comunes:
        return {"js_media": float("nan"), "n_contextos_comunes": 0}

    simbolos = sorted({ch for c in list(ta.values()) + list(tb.values()) for ch in c})
    idx = {s: i for i, s in enumerate(simbolos)}

    def _vec(c: Counter) -> np.ndarray:
        v = np.zeros(len(simbolos))
        for ch, n in c.items():
            v[idx[ch]] = n
        s = v.sum()
        return v / s if s else v

    js_total = peso_total = 0.0
    for ctx in comunes:
        p, q = _vec(ta[ctx]), _vec(tb[ctx])
        m = 0.5 * (p + q)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_pm = np.nansum(np.where(p > 0, p * np.log2(p / m), 0.0))
            kl_qm = np.nansum(np.where(q > 0, q * np.log2(q / m), 0.0))
        js = 0.5 * (kl_pm + kl_qm)
        peso = sum(ta[ctx].values())
        js_total += js * peso
        peso_total += peso

    return {
        "js_media": js_total / peso_total if peso_total else float("nan"),
        "n_contextos_comunes": len(comunes),
        "n_contextos_a": len(ta),
        "n_contextos_b": len(tb),
    }


if __name__ == "__main__":
    print("babel_secuencia.py — smoke test")

    aleatorio = "".join(np.random.default_rng(1).choice(list("abcde"), 20_000))
    estructurado = "abcde" * 4_000

    for etiqueta, txt in (("aleatorio", aleatorio), ("periódico", estructurado)):
        g = ganancia_informacion(txt, ordenes=(1, 2))
        print(f"  {etiqueta:<12} H0={g['H0']:.3f}  H1={g['H1']:.3f}  "
              f"ganancia_1={g['ganancia_1']:.3f}")

    lm = ModeloCaracteres(orden=3).entrenar(aleatorio)
    print(f"  perplejidad(aleatorio | modelo aleatorio) = "
          f"{lm.perplejidad(aleatorio):.2f}  (techo teórico 5)")
    print(f"  perplejidad(periódico | modelo aleatorio) = "
          f"{lm.perplejidad(estructurado):.2f}")
