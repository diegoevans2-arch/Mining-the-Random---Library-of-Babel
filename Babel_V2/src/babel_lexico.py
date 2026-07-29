"""
babel_lexico.py
===============
Bloque D — ¿Cuánto significado emerge por azar?

La versión cuantitativa del experimento mental de Borges. Cubre D.1 (palabras
reales emergentes), D.2 (su distribución entre libros) y D.3 (búsqueda
semántica: cuánto se acerca el azar al sentido).

Más allá de "Babel = null model"
--------------------------------
El resto del estudio demuestra la aleatoriedad por comparación: Babel se
comporta como su gemelo sintético. Aquí se puede ir un paso más allá y
**predecir el resultado en cerrado**, sin medir nada.

Un token de Babel es una secuencia máxima de letras. Dentro de él las 26 letras
son equiprobables, así que la probabilidad de que un token de longitud *k* sea
una palabra real del diccionario es exactamente

    P(k) = N_k / 26^k

con N_k = número de palabras distintas de *k* letras en el léxico. Y como los
límites de palabra (espacio, coma, punto) son 3 de los 29 símbolos, la longitud
de token sigue una geométrica de parámetro 3/29 — media 29/3 − 1 ≈ 8,67, que es
lo que se mide.

Combinando ambas cosas se obtiene el número esperado de palabras emergentes
antes de contar ninguna. Si la observación coincide con la predicción, la
emergencia no solo es indistinguible del azar: **es azar, y se sabe por qué**.

Ese es el argumento más fuerte que puede dar este bloque, porque no depende de
ningún modelo nulo ni de ninguna herramienta: es aritmética.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

# 26 letras a-z: dentro de un token no hay separadores por definición.
ALFABETO_INTRA_TOKEN = 26
# Espacio, coma y punto sobre 29 símbolos de contenido.
P_FRONTERA = 3 / 29


# ============================================================================
# 1. CARGA DE LÉXICOS
# ============================================================================
def cargar_lexico_wordfreq(idioma: str = "es", n_palabras: int = 100_000) -> set[str]:
    """
    Las `n_palabras` más frecuentes del idioma, según `wordfreq`.

    Devuelve un set vacío si la librería no está instalada, para que el bloque
    degrade en vez de reventar.
    """
    try:
        from wordfreq import top_n_list
    except ImportError:
        return set()
    return {p.lower() for p in top_n_list(idioma, n_palabras) if p.isalpha()}


def cargar_lexico_spellchecker(idioma: str = "es") -> set[str]:
    """Diccionario completo de `pyspellchecker` para un idioma."""
    try:
        from spellchecker import SpellChecker
    except ImportError:
        return set()
    try:
        sc = SpellChecker(language=idioma)
    except Exception:
        return set()
    return {p.lower() for p in sc.word_frequency.dictionary if p.isalpha()}


def cargar_lexico(idiomas: tuple[str, ...] = ("es", "en"),
                  n_palabras: int = 100_000) -> dict:
    """
    Une los léxicos de todas las fuentes e idiomas disponibles.

    Se combinan `wordfreq` y `pyspellchecker` a propósito: el primero pondera
    por uso real, el segundo cubre morfología (conjugaciones, plurales) que las
    listas de frecuencia recortan. Para contar *qué puede reconocerse como
    palabra*, la unión es la medida honesta.
    """
    por_fuente: dict[str, set[str]] = {}
    for idioma in idiomas:
        wf = cargar_lexico_wordfreq(idioma, n_palabras)
        sp = cargar_lexico_spellchecker(idioma)
        if wf:
            por_fuente[f"wordfreq_{idioma}"] = wf
        if sp:
            por_fuente[f"spell_{idioma}"] = sp

    union: set[str] = set()
    for s in por_fuente.values():
        union |= s

    return {
        "lexico": union,
        "por_fuente": {k: len(v) for k, v in por_fuente.items()},
        "n_total": len(union),
        "disponible": bool(union),
    }


def perfil_longitudes_lexico(lexico: set[str],
                             longitudes: range | tuple = range(1, 12)) -> dict[int, int]:
    """N_k: cuántas palabras distintas de cada longitud tiene el léxico."""
    c = Counter(len(p) for p in lexico)
    return {k: c.get(k, 0) for k in longitudes}


# ============================================================================
# 2. PREDICCIÓN COMBINATORIA
# ============================================================================
def prob_palabra_por_longitud(lexico: set[str],
                              alfabeto: int = ALFABETO_INTRA_TOKEN,
                              longitudes: range | tuple = range(1, 12)) -> dict:
    """
    P(k) = N_k / alfabeto^k — probabilidad de que una cadena aleatoria de k
    letras sea una palabra real.

    Cae en picado: cada letra extra divide por 26 mientras el diccionario
    crece mucho más despacio. Es la razón matemática de que las "islas de
    significado" de Borges tengan un techo de longitud muy bajo.
    """
    N = perfil_longitudes_lexico(lexico, longitudes)
    return {k: (N[k] / (alfabeto ** k) if N[k] else 0.0) for k in longitudes}


def prediccion_emergentes(tokens: list[str], lexico: set[str],
                          alfabeto: int = ALFABETO_INTRA_TOKEN) -> dict:
    """
    Cuántas palabras reales DEBERÍAN emerger, dada la distribución de
    longitudes observada y el tamaño del léxico. Sin mirar los tokens.

    Devuelve la predicción por longitud y el total, para poder contrastarla
    contra la observación.
    """
    longitudes = Counter(len(t) for t in tokens)
    ks = sorted(longitudes)
    P = prob_palabra_por_longitud(lexico, alfabeto, ks)
    esperado = {k: longitudes[k] * P.get(k, 0.0) for k in ks}
    return {
        "esperado_por_longitud": esperado,
        "esperado_total": sum(esperado.values()),
        "prob_por_longitud": P,
        "tokens_por_longitud": dict(longitudes),
    }


def longitud_media_teorica(p_frontera: float = P_FRONTERA) -> float:
    """
    Media de la longitud de token si los límites de palabra caen al azar.

    Es una geométrica: media = (1 − p)/p. Con 3 separadores sobre 29 símbolos
    da 8,67. Sirve de comprobación de cordura del marco entero — si la longitud
    medida no coincide, el modelo de "letras equiprobables entre separadores"
    no describe el texto.
    """
    return (1 - p_frontera) / p_frontera


# ============================================================================
# 3. OBSERVACIÓN (D.1)
# ============================================================================
def palabras_emergentes(tokens: list[str], lexico: set[str],
                        min_long: int = 3) -> dict:
    """
    Qué tokens son palabras reales del léxico.

    `min_long` aplica el umbral de §5.1: por debajo de 3 letras casi todo
    coincide con algo en algún idioma y la métrica pierde sentido.
    """
    if not lexico:
        return {"disponible": False}

    candidatos = [t for t in tokens if len(t) >= min_long]
    aciertos = [t for t in candidatos if t in lexico]
    por_longitud = Counter(len(t) for t in aciertos)
    distintas = set(aciertos)

    return {
        "disponible": True,
        "n_tokens": len(tokens),
        "n_candidatos": len(candidatos),
        "n_emergentes": len(aciertos),
        "n_distintas": len(distintas),
        "tasa_pct": 100 * len(aciertos) / len(candidatos) if candidatos else 0.0,
        "por_longitud": dict(por_longitud),
        "long_max": max((len(t) for t in aciertos), default=0),
        "mas_frecuentes": Counter(aciertos).most_common(15),
        "mas_largas": sorted(distintas, key=len, reverse=True)[:15],
    }


def zipf_de_emergentes(palabras: list[str], idioma: str = "es") -> dict:
    """
    Cuán comunes son las palabras que emergieron, en escala Zipf de `wordfreq`.

    La escala es logarítmica: 7 ≈ una palabra por cada 10.000 (muy común), 3 ≈
    una por cada 10 millones (rara). Si el azar produce sobre todo palabras
    comunes, es porque las comunes son cortas — no porque el azar "prefiera" el
    sentido.
    """
    try:
        from wordfreq import zipf_frequency
    except ImportError:
        return {}
    if not palabras:
        return {}
    z = [zipf_frequency(p, idioma) for p in palabras]
    z = [v for v in z if v > 0]
    if not z:
        return {}
    return {
        "zipf_media": float(np.mean(z)),
        "zipf_mediana": float(np.median(z)),
        "zipf_min": float(np.min(z)),
        "zipf_max": float(np.max(z)),
        "n_evaluadas": len(z),
    }


# ============================================================================
# 4. DISTRIBUCIÓN ENTRE LIBROS (D.2)
# ============================================================================
def distribucion_por_documento(tokens_por_doc: dict[str, list[str]],
                               lexico: set[str], min_long: int = 3) -> dict:
    """
    Tasa de emergencia en cada documento por separado.

    Si algún libro concentrase palabras reales, sería la primera señal real de
    que no todos se generaron igual. Lo esperable es homogeneidad: misma tasa
    en los diez, dentro del ruido de Poisson.
    """
    filas = {}
    for nombre, tokens in tokens_por_doc.items():
        r = palabras_emergentes(tokens, lexico, min_long)
        if r.get("disponible"):
            filas[nombre] = {"n_emergentes": r["n_emergentes"],
                             "n_candidatos": r["n_candidatos"],
                             "tasa_pct": r["tasa_pct"]}
    if not filas:
        return {}

    tasas = [f["tasa_pct"] for f in filas.values()]
    conteos = [f["n_emergentes"] for f in filas.values()]
    # Con sucesos raros e independientes, var ≈ media (Poisson).
    media = float(np.mean(conteos))
    return {
        "por_documento": filas,
        "tasa_media": float(np.mean(tasas)),
        "tasa_sd": float(np.std(tasas)),
        "razon_varianza_media": float(np.var(conteos) / media) if media else float("nan"),
    }


# ============================================================================
# 5. BÚSQUEDA SEMÁNTICA (D.3)
# ============================================================================
FRASES_SONDA = [
    "el amor es eterno mientras dura",
    "la biblioteca contiene todos los libros posibles",
    "el tiempo es un río que me arrastra",
    "la muerte llega para todos por igual",
    "un hombre camina solo bajo la lluvia",
    "la verdad se esconde en los detalles pequeños",
]


def busqueda_semantica(frases: list[str], chunks_por_condicion: dict[str, list[str]],
                       vectores_por_condicion: dict[str, np.ndarray],
                       embedder) -> dict:
    """
    Para cada frase con significado, el fragmento más parecido de cada condición.

    La pregunta de D.3: *¿qué tan cerca llega el azar al sentido?* Se mide como
    la similitud coseno con el vecino más cercano.

    ⚠️ Interpretación: un valor alto NO significa que Babel contenga esa idea.
    El espacio de embeddings es continuo y siempre existe un punto más cercano
    que los demás; lo informativo es **la diferencia** entre ese máximo en Babel
    y el mismo máximo en texto real, y que Babel coincida con el null model.
    """
    if embedder is None or not frases:
        return {}

    v_frases = np.asarray(embedder.encode(
        list(frases), normalize_embeddings=True, show_progress_bar=False))

    salida: dict[str, dict] = {}
    for cond, V in vectores_por_condicion.items():
        if V is None or len(V) == 0:
            continue
        sims = v_frases @ V.T                      # (n_frases × n_chunks)
        mejores = sims.max(axis=1)
        idx = sims.argmax(axis=1)
        chunks = chunks_por_condicion.get(cond, [])
        salida[cond] = {
            "sim_max_media": float(mejores.mean()),
            "sim_max_max": float(mejores.max()),
            "por_frase": {
                f: {"sim": float(s),
                    "fragmento": chunks[i][:90] if i < len(chunks) else ""}
                for f, s, i in zip(frases, mejores, idx)
            },
        }
    return salida


if __name__ == "__main__":
    print("babel_lexico.py — smoke test")
    print(f"  longitud media teórica de token: {longitud_media_teorica():.2f}")

    lex = {"casa", "perro", "sol", "mar", "libro", "the", "cat", "dog"}
    P = prob_palabra_por_longitud(lex, longitudes=range(3, 6))
    for k, p in P.items():
        print(f"  P(cadena de {k} letras sea palabra) = {p:.3e}")

    tokens = ["sol", "xkqvz", "casa", "mar", "wqxzjf", "the"]
    r = palabras_emergentes(tokens, lex, min_long=3)
    print(f"  emergentes: {r['n_emergentes']}/{r['n_candidatos']} "
          f"({r['tasa_pct']:.1f}%), máx {r['long_max']} letras")
    pred = prediccion_emergentes(tokens, lex)
    print(f"  predicción combinatoria: {pred['esperado_total']:.2e}")
