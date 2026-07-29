"""
babel_transformers.py
=====================
Bloque B — NLP moderno sobre la Biblioteca de Babel.

Provee:
  - Muestreo de ventanas (la unidad de observación del bloque).
  - Detección de idioma con `papluca/xlm-roberta-base-language-detection`.
  - Sentimiento moderno con `cardiffnlp/twitter-xlm-roberta-base-sentiment`.
  - Detección de idioma con fastText (`lid.176.bin`), opcional.
  - Fragmentación subléxica: cuántos sub-tokens necesita el tokenizador por
    carácter, que resulta ser una métrica de aleatoriedad por derecho propio.

Por qué la ventana y no el corpus
---------------------------------
XLM-RoBERTa tiene un contexto de 512 tokens. Sobre 1,1M de caracteres —y con un
tokenizador que ante ruido cae casi a nivel de carácter— salen del orden de mil
ventanas por condición; pasar las 4 condiciones enteras por dos modelos son
horas de CPU, y contrastar contra 100 réplicas nulas es inviable.

La salida no es recortar el rigor sino cambiar la unidad de observación. Cada
ventana es una observación, y el contraste del principio rector (§5.8) se hace
igual: R réplicas nulas × W ventanas dan R valores del estadístico, y ahí se
ubica Babel. Con R=30 y W=30 son ~900 pases por modelo en vez de decenas de
miles, y encima se gana algo que el Bloque A no tenía: variabilidad
intra-corpus, porque cada libro aporta su propia distribución de ventanas.

Diseñado para Jupyter Notebook 7.x. Sin rutas hardcoded; usa pathlib.
"""
from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import numpy as np

# ============================================================================
# MODELOS POR DEFECTO
# ============================================================================
MODELO_IDIOMA = "papluca/xlm-roberta-base-language-detection"
MODELO_SENTIMIENTO = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
URL_FASTTEXT_LID = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"

# Ventana de muestreo, en CARACTERES.
#
# Se fija en caracteres y no en tokens para que las 4 condiciones reciban la
# misma cantidad de texto: comparar "512 tokens de Babel" contra "512 tokens de
# Moby Dick" sería comparar fragmentos de longitud física muy distinta, porque
# el tokenizador fragmenta el ruido mucho más.
#
# El valor sale de medirlo, no de estimarlo. Con XLM-R sobre el corpus real:
#
#     ventana    Babel (máx tokens)    Controles (máx)
#      600 chars        380                  187
#      800 chars        505                  244      <- 7 tokens de margen
#     1000 chars        627 TRUNCA           301
#
# 600 deja margen holgado en la condición más fragmentada, que es la que manda.
CHARS_POR_VENTANA = 600
N_VENTANAS = 30
N_REPLICAS_B = 30


# ============================================================================
# 1. MUESTREO DE VENTANAS
# ============================================================================
def muestrear_ventanas(
    texto: str,
    n_ventanas: int = N_VENTANAS,
    chars: int = CHARS_POR_VENTANA,
    seed: int | None = None,
    quitar_saltos: bool = True,
) -> list[str]:
    """
    Extrae `n_ventanas` fragmentos de `chars` caracteres en posiciones al azar.

    Parameters
    ----------
    quitar_saltos : sustituye '\\n' por espacio.
        Se aplica a TODAS las condiciones por igual, y es deliberado: el salto
        cada 80 caracteres de Babel es un artefacto tipográfico del sitio, no
        una propiedad del texto. Dejarlo delataría a Babel ante un detector de
        idioma por razones que no tienen nada que ver con su contenido —
        exactamente el tipo de confusión que el Bloque A se dedicó a eliminar.
    """
    if not texto:
        return []
    if len(texto) <= chars:
        v = texto.replace("\n", " ") if quitar_saltos else texto
        return [v]

    rng = np.random.default_rng(seed)
    posiciones = rng.integers(0, len(texto) - chars + 1, size=n_ventanas)
    ventanas = [texto[p : p + chars] for p in sorted(posiciones)]
    if quitar_saltos:
        ventanas = [v.replace("\n", " ") for v in ventanas]
    return ventanas


# ============================================================================
# 2. FRAGMENTACIÓN SUBLÉXICA
# ============================================================================
def ratio_tokens_por_char(ventanas: list[str], tokenizador) -> dict:
    """
    Cuántos sub-tokens produce el tokenizador por carácter de texto.

    Es una métrica de aleatoriedad en sí misma, y muy barata: el vocabulario
    de XLM-R está hecho de piezas frecuentes en lenguaje real, así que sobre
    texto natural cubre palabras enteras (~0.25 tokens/char) mientras que sobre
    ruido se ve obligado a bajar casi a nivel de carácter. Es el equivalente
    subléxico de lo que la compresión mide a nivel de bytes.

    También sirve de control de seguridad: si el ratio × `chars` supera 512, las
    ventanas se están truncando y la comparación deja de ser justa.
    """
    if not ventanas:
        return {}
    n_tokens = [len(tokenizador.encode(v, add_special_tokens=False)) for v in ventanas]
    n_chars = [len(v) for v in ventanas]
    ratios = [t / c for t, c in zip(n_tokens, n_chars) if c]
    return {
        "tokens_por_char": float(np.mean(ratios)),
        "tokens_media": float(np.mean(n_tokens)),
        "tokens_max": int(np.max(n_tokens)),
        "supera_512": bool(np.max(n_tokens) > 512),
    }


# ============================================================================
# 3. CARGA DE MODELOS (Hugging Face)
# ============================================================================
def _dispositivo(device: int | None) -> int:
    if device is not None:
        return device
    try:
        import torch

        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def cargar_clasificador(
    nombre_modelo: str,
    device: int | None = None,
    batch_size: int = 8,          # medido: 346 ms/ventana en CPU, mejor que 16 o 32
):
    """
    Devuelve (pipeline, tokenizador). `device=-1` es CPU, `0` la primera GPU.

    `top_k=None` hace que devuelva la distribución completa sobre etiquetas, no
    solo la ganadora: sobre ruido lo interesante no es qué idioma "gana" sino
    cuánta probabilidad concentra, que es la métrica del bloque.
    """
    from transformers import AutoTokenizer, pipeline

    dev = _dispositivo(device)
    tok = AutoTokenizer.from_pretrained(nombre_modelo)
    pipe = pipeline(
        "text-classification",
        model=nombre_modelo,
        tokenizer=tok,
        device=dev,
        top_k=None,
        truncation=True,
        max_length=512,
        batch_size=batch_size,
    )
    return pipe, tok


_ALIAS_SENTIMIENTO = {
    "label_0": "negative",
    "label_1": "neutral",
    "label_2": "positive",
}


def _normalizar_etiqueta(etiqueta: str) -> str:
    e = etiqueta.strip().lower()
    return _ALIAS_SENTIMIENTO.get(e, e)


def clasificar_ventanas(ventanas: list[str], pipe) -> list[dict[str, float]]:
    """Devuelve, por ventana, la distribución {etiqueta: probabilidad}."""
    if not ventanas:
        return []
    salidas = pipe(list(ventanas))
    if salidas and isinstance(salidas[0], dict):   # pipeline sin top_k
        salidas = [[s] for s in salidas]
    return [
        {_normalizar_etiqueta(d["label"]): float(d["score"]) for d in salida}
        for salida in salidas
    ]


# ============================================================================
# 4. MÉTRICAS
# ============================================================================
def metricas_idioma(distribuciones: list[dict[str, float]]) -> dict:
    """
    Resume la salida del detector de idioma sobre un conjunto de ventanas.

    - `prob_top_media` — cuánta probabilidad concentra el modelo en su idioma
      favorito. Es la pregunta del roadmap: *¿con qué seguridad afirma un
      detector que esto es un idioma real?* **Es la métrica que discrimina.**
    - `entropia_idiomas` — dispersión de la etiqueta ganadora entre ventanas.

    Ojo con la interpretación: un softmax siempre suma 1, así que el modelo
    SIEMPRE devuelve un idioma. Que devuelva uno no significa nada; lo que
    significa algo es cuánta probabilidad le pone.

    Sobre la entropía, una advertencia ganada a base de medirla. La hipótesis
    natural es que ante ruido el modelo se pierda y reparta la etiqueta ganadora
    entre idiomas, subiendo la entropía. **No es lo que pasa.** XLM-R etiqueta
    el ruido en alfabeto latino como *hindi* en el 100% de las ventanas, con
    ~0,57 de confianza, y el null model hace exactamente lo mismo. El modelo no
    está perdido: está sistemáticamente equivocado, de forma consistente.

    Así que una entropía baja NO indica estructura lingüística — indica un sesgo
    estable del clasificador. Sin el null model al lado, ese 100% de acuerdo
    entre ventanas sería facilísimo de leer como "señal".
    """
    if not distribuciones:
        return {}

    top_probs = [max(d.values()) for d in distribuciones]
    ganadores = [max(d, key=d.get) for d in distribuciones]
    conteo = Counter(ganadores)
    n = len(distribuciones)
    H = -sum((c / n) * math.log2(c / n) for c in conteo.values())
    H = max(0.0, H)          # evita el -0.0 cuando hay un único idioma
    modal, n_modal = conteo.most_common(1)[0]

    return {
        "prob_top_media": float(np.mean(top_probs)),
        "prob_top_mediana": float(np.median(top_probs)),
        "prob_top_sd": float(np.std(top_probs)),
        "n_idiomas_distintos": len(conteo),
        "entropia_idiomas": H,
        "idioma_modal": modal,
        "pct_modal": 100 * n_modal / n,
        "n_ventanas": n,
    }


def metricas_sentimiento(distribuciones: list[dict[str, float]]) -> dict:
    """
    Resume el sentimiento moderno. `polaridad` = P(positivo) − P(negativo),
    definida así a propósito para poder ponerla al lado de la de TextBlob del
    Bloque A, que vive en el mismo rango [-1, 1].
    """
    if not distribuciones:
        return {}

    pos = [d.get("positive", 0.0) for d in distribuciones]
    neg = [d.get("negative", 0.0) for d in distribuciones]
    neu = [d.get("neutral", 0.0) for d in distribuciones]
    pol = [p - n for p, n in zip(pos, neg)]

    return {
        "positivo_medio": float(np.mean(pos)),
        "negativo_medio": float(np.mean(neg)),
        "neutral_medio": float(np.mean(neu)),
        "polaridad": float(np.mean(pol)),
        "polaridad_sd": float(np.std(pol)),
        "n_ventanas": len(distribuciones),
    }


def analizar_condicion(
    texto: str,
    pipe_idioma=None,
    pipe_sentimiento=None,
    tokenizador=None,
    n_ventanas: int = N_VENTANAS,
    chars: int = CHARS_POR_VENTANA,
    seed: int | None = None,
) -> dict:
    """
    Orquestador: muestrea ventanas de un texto y le pasa los modelos activos.
    Devuelve un dict plano, listo para agregarse en un DataFrame comparativo.
    """
    ventanas = muestrear_ventanas(texto, n_ventanas, chars, seed=seed)
    salida: dict = {"n_ventanas": len(ventanas), "chars_por_ventana": chars}

    if tokenizador is not None:
        salida.update(ratio_tokens_por_char(ventanas, tokenizador))
    if pipe_idioma is not None:
        salida.update(metricas_idioma(clasificar_ventanas(ventanas, pipe_idioma)))
    if pipe_sentimiento is not None:
        s = metricas_sentimiento(clasificar_ventanas(ventanas, pipe_sentimiento))
        salida.update({f"sent_{k}": v for k, v in s.items()})
    return salida


# ============================================================================
# 5. FASTTEXT (opcional)
# ============================================================================
RUTA_FASTTEXT_DEFECTO = Path.home() / ".cache" / "fasttext" / "lid.176.bin"


def descargar_fasttext(ruta: str | Path | None = None) -> Path:
    """
    Descarga `lid.176.bin` (126 MB) si no está ya en caché. Devuelve la ruta.

    Se guarda FUERA del repositorio, en `~/.cache/fasttext/`, por la misma razón
    que Hugging Face cachea sus modelos ahí: son binarios grandes que no deben
    acabar en git.

    En Windows hay que instalar `fasttext-wheel`, no `fasttext`: el segundo no
    publica wheels y requiere compilador C++.
    """
    import urllib.request

    destino = Path(ruta) if ruta else RUTA_FASTTEXT_DEFECTO
    if destino.exists():
        return destino

    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_suffix(destino.suffix + ".parcial")
    urllib.request.urlretrieve(URL_FASTTEXT_LID, parcial)
    parcial.replace(destino)      # descarga atómica: o está entero o no está
    return destino


def cargar_fasttext(ruta_modelo: str | Path | None = None):
    """
    Carga `lid.176.bin`. Devuelve None si falta la librería o el modelo, para
    que el bloque siga corriendo sin él en vez de reventar.
    """
    try:
        import fasttext
    except ImportError:
        return None
    ruta = Path(ruta_modelo) if ruta_modelo else RUTA_FASTTEXT_DEFECTO
    if not ruta.exists():
        return None
    return fasttext.load_model(str(ruta))


def detectar_idioma_fasttext(ventanas: list[str], modelo, k: int = 5) -> list[dict]:
    """Distribución {idioma: prob} por ventana, con las k etiquetas más altas."""
    if modelo is None or not ventanas:
        return []
    salida = []
    for v in ventanas:
        etiquetas, probs = modelo.predict(v.replace("\n", " "), k=k)
        salida.append(
            {e.replace("__label__", ""): float(p) for e, p in zip(etiquetas, probs)}
        )
    return salida


if __name__ == "__main__":
    print("babel_transformers.py — smoke test")
    texto = "abcdefghij klmnopqrst uvwxyz " * 400
    ventanas = muestrear_ventanas(texto, n_ventanas=5, chars=200, seed=1)
    print(f"  ventanas: {len(ventanas)} × {len(ventanas[0])} chars")
    print(f"  muestra : {ventanas[0][:60]!r}")

    falsas = [{"es": 0.6, "en": 0.3, "fr": 0.1}, {"en": 0.5, "es": 0.5}]
    print(f"  metricas_idioma: {metricas_idioma(falsas)}")
    print(f"  metricas_sentimiento: "
          f"{metricas_sentimiento([{'positive': 0.7, 'negative': 0.2, 'neutral': 0.1}])}")
