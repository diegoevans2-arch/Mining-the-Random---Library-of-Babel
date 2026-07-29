"""
babel_pipeline.py
=================
Pipeline básico de minería de texto, refactorizado del notebook original.

Conserva TODOS los pasos clásicos que descubrió el estudio original:
  - Limpieza
  - Tokenización
  - Filtrado de stopwords (multi-idioma)
  - Stemming (multi-idioma)
  - Lematización (spaCy con fallback a NLTK)
  - Análisis de frecuencias (incl. palabras menos frecuentes)
  - Vectorización (BoW, TF-IDF)
  - Análisis de sentimientos (TextBlob)

Diferencia clave con el original: cada función es PURA y reutilizable, para
poder aplicarla idénticamente a las 4 condiciones (babel / null / control_es /
control_en) y demostrar que los "hallazgos" del original son artefactos de la
herramienta, no propiedades del texto.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# --- NLTK ---
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Idiomas para stopwords multi-idioma (replicando el original)
IDIOMAS_STOPWORDS = [
    "spanish", "english", "french", "german", "italian",
    "portuguese", "dutch", "russian", "turkish",
]


# ----------------------------------------------------------------------------
# 1. LIMPIEZA (CORREGIDA)
# ----------------------------------------------------------------------------
def limpiar_texto(
    texto: str,
    preservar_puntuacion: bool = False,
    normalizar_acentos: bool = True,
) -> str:
    """
    Limpieza de texto. Un "token" es una secuencia máxima de letras a-z.

    DECISIÓN METODOLÓGICA (§5): los caracteres no alfabéticos se sustituyen por
    ESPACIO, es decir, actúan como FRONTERA DE PALABRA. No se borran.

    Borrarlos —el idiom clásico `re.sub(r'[^a-z\\s]', '', texto)` que usaba el
    v1— pega los tokens vecinos. En lenguaje natural apenas se nota, porque la
    puntuación casi siempre va pegada a un espacio; pero en Babel la coma y el
    punto son símbolos de contenido que caen en posiciones aleatorias, así que
    el pegado es constante. Medido sobre 1,1M caracteres:

        borrando   : Babel 48.961 tokens, largo medio 20,0 | controles ~4,5
        separando  : Babel 111.797 tokens, largo medio  8,7 | controles ~4,4

    El sesgo era ASIMÉTRICO: inflaba la longitud de token de Babel sin tocar la
    de los controles, justo en la comparación que sostiene el estudio. Además
    dejaba D.1 sin sentido — buscar palabras reales dentro de tokens de 20
    caracteres es vacío por construcción. Con la frontera correcta emergen 602
    palabras del léxico en Babel y 641 en el null model: la misma tasa.

    Esto arregla de paso otros dos pegados que nadie había pedido: los guiones
    bajos de CESS-ESP (`Electricité_de_France`, `51_por_ciento`, que contaban
    como un solo tipo) y los guiones dobles de Moby Dick (`Usher--threadbare`).

    Parameters
    ----------
    preservar_puntuacion : si True, conserva ',' y '.' como símbolos de
        contenido en lugar de tratarlos como frontera.
    normalizar_acentos : si True, á→a, é→e, ñ→n antes de filtrar. Necesario:
        sin esto, sustituir por espacio partiría "corazón" en `coraz` + `n` y
        fragmentaría el vocabulario del control español.
    """
    texto = texto.lower()

    if normalizar_acentos:
        texto = unicodedata.normalize("NFKD", texto)
        texto = "".join(ch for ch in texto if not unicodedata.combining(ch))

    patron = r"[^a-z\s,\.]" if preservar_puntuacion else r"[^a-z\s]"
    texto = re.sub(patron, " ", texto)          # espacio, NO borrado
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


# ----------------------------------------------------------------------------
# 2. TOKENIZACIÓN
# ----------------------------------------------------------------------------
def tokenizar(texto: str) -> list[str]:
    """
    Tokenización por palabras: secuencias máximas de letras a-z.
    Subléxica — no asume ningún idioma.

    Idéntica por definición a babel_core._tokenizar_palabras, para que el
    pipeline clásico y las leyes de Zipf/Heaps cuenten los mismos tokens.
    Se espera recibir la salida de limpiar_texto().
    """
    return re.findall(r"[a-z]+", texto.lower())


# ----------------------------------------------------------------------------
# 3. STOPWORDS MULTI-IDIOMA
# ----------------------------------------------------------------------------
def construir_set_stopwords(idiomas: list[str] = IDIOMAS_STOPWORDS) -> set[str]:
    """Une las stopwords de múltiples idiomas en un solo set."""
    sw = set()
    for idioma in idiomas:
        try:
            sw |= set(stopwords.words(idioma))
        except Exception:
            pass
    return sw


def filtrar_stopwords(
    tokens: list[str],
    set_stopwords: set[str],
    min_longitud: int = 3,
) -> tuple[list[str], dict]:
    """
    Filtra stopwords y tokens muy cortos.

    Devuelve (tokens_conservados, estadisticas) donde estadisticas incluye
    el conteo y porcentaje eliminado — la métrica que el original reportaba
    como "sesgo del ~9%".
    """
    conservados = []
    eliminados_sw = 0
    eliminados_cortos = 0

    for t in tokens:
        if len(t) < min_longitud:
            eliminados_cortos += 1
        elif t in set_stopwords:
            eliminados_sw += 1
        else:
            conservados.append(t)

    n = len(tokens)
    stats = {
        "n_original": n,
        "n_conservados": len(conservados),
        "eliminados_stopwords": eliminados_sw,
        "eliminados_cortos": eliminados_cortos,
        "pct_stopwords": 100 * eliminados_sw / n if n else 0,
        "pct_eliminado_total": 100 * (n - len(conservados)) / n if n else 0,
    }
    return conservados, stats


# ----------------------------------------------------------------------------
# 4. STEMMING
# ----------------------------------------------------------------------------
def aplicar_stemming(tokens: list[str], idioma: str = "spanish") -> tuple[list[str], dict]:
    """
    Aplica stemming Snowball. Devuelve (stems, estadisticas).
    El % de tokens modificados es la métrica de "el stemmer encuentra raíces".
    """
    stemmer = SnowballStemmer(idioma)
    stems = [stemmer.stem(t) for t in tokens]
    modificados = sum(1 for o, s in zip(tokens, stems) if o != s)
    n = len(tokens)
    stats = {
        "n_tokens": n,
        "modificados": modificados,
        "pct_modificado": 100 * modificados / n if n else 0,
        "vocab_antes": len(set(tokens)),
        "vocab_despues": len(set(stems)),
    }
    return stems, stats


# ----------------------------------------------------------------------------
# 5. LEMATIZACIÓN (spaCy con fallback a NLTK)
# ----------------------------------------------------------------------------
def cargar_lematizador(idioma_spacy: str = "es_core_news_sm"):
    """
    Intenta cargar spaCy; si no hay modelo, devuelve un fallback NLTK.
    Devuelve (objeto, tipo) donde tipo in {'spacy', 'nltk'}.
    """
    try:
        import spacy
        nlp = spacy.load(idioma_spacy, disable=["ner", "parser"])
        return nlp, "spacy"
    except Exception:
        return WordNetLemmatizer(), "nltk"


def aplicar_lematizacion(
    tokens: list[str],
    lematizador,
    tipo: str,
    chunk_tokens: int = 20_000,
) -> tuple[list[str], dict]:
    """
    Lematiza con spaCy o NLTK según disponibilidad.

    Parameters
    ----------
    chunk_tokens : tamaño de bloque para spaCy.
        spaCy rechaza documentos de más de 1.000.000 de caracteres (error E088)
        y su uso de memoria crece con la longitud del doc. Con los libros
        completos (~1.3M chars → ~200k tokens) pasar todo junto revienta, así
        que se procesa por bloques y se concatena. La alineación de longitud se
        hace POR BLOQUE: si se hiciera al final, un desajuste en un bloque
        temprano desplazaría todos los lemas posteriores.
    """
    if tipo == "spacy":
        lemmas = []
        for i in range(0, len(tokens), chunk_tokens):
            trozo = tokens[i : i + chunk_tokens]
            doc = lematizador(" ".join(trozo))
            l_trozo = [tok.lemma_ for tok in doc]
            # spaCy puede re-tokenizar; alinear longitud dentro del bloque
            if len(l_trozo) != len(trozo):
                l_trozo = l_trozo[: len(trozo)] + trozo[len(l_trozo):]
            lemmas.extend(l_trozo)
    else:
        lemmas = [lematizador.lemmatize(t) for t in tokens]

    modificados = sum(1 for o, l in zip(tokens, lemmas) if o != l)
    n = len(tokens)
    stats = {
        "metodo": tipo,
        "n_tokens": n,
        "lematizados": modificados,
        "pct_lematizado": 100 * modificados / n if n else 0,
        "vocab_antes": len(set(tokens)),
        "vocab_despues": len(set(lemmas)),
    }
    return lemmas, stats


# ----------------------------------------------------------------------------
# 6. ANÁLISIS DE FRECUENCIAS
# ----------------------------------------------------------------------------
def analizar_frecuencias(tokens: list[str], top_n: int = 20) -> dict:
    """
    Frecuencias de tokens. Incluye el hallazgo del original sobre las
    palabras MENOS frecuentes (¿siempre de 3 caracteres?).
    """
    frec = Counter(tokens)

    # Palabras menos frecuentes (las que aparecen exactamente 1 vez = hapax)
    hapax = [w for w, c in frec.items() if c == 1]
    long_hapax = [len(w) for w in hapax]

    return {
        "n_tipos": len(frec),
        "n_tokens": len(tokens),
        "top": frec.most_common(top_n),
        "n_hapax": len(hapax),
        "long_promedio_hapax": np.mean(long_hapax) if long_hapax else 0,
        "long_min_hapax": min(long_hapax) if long_hapax else 0,
        "distribucion_long_hapax": Counter(long_hapax),
    }


# ----------------------------------------------------------------------------
# 7. VECTORIZACIÓN (BoW + TF-IDF)
# ----------------------------------------------------------------------------
def vectorizar(textos: list[str], max_features: int | None = None) -> dict:
    """
    Calcula BoW y TF-IDF sobre una lista de documentos.

    Nota metodológica: el original usaba max_features=100 sobre ~47k tokens
    únicos (<0.3% del vocabulario). Aquí max_features es explícito; usar None
    para vocabulario completo en análisis de correlación.
    """
    bow_vec = CountVectorizer(max_features=max_features)
    bow = bow_vec.fit_transform(textos)

    tfidf_vec = TfidfVectorizer(max_features=max_features)
    tfidf = tfidf_vec.fit_transform(textos)

    return {
        "bow_matrix": bow,
        "bow_features": bow_vec.get_feature_names_out(),
        "tfidf_matrix": tfidf,
        "tfidf_features": tfidf_vec.get_feature_names_out(),
        "n_features": tfidf.shape[1],
    }


# ----------------------------------------------------------------------------
# 8. ANÁLISIS DE SENTIMIENTOS (TextBlob)
# ----------------------------------------------------------------------------
def analizar_sentimiento(texto: str) -> dict:
    """
    Sentimiento vía TextBlob.

    ADVERTENCIA METODOLÓGICA: sobre texto aleatorio esto es 100% artefacto.
    Se mantiene para DEMOSTRAR que la herramienta alucina sentimiento donde
    no hay ninguno. El valor solo tiene sentido comparado contra el null model.
    """
    from textblob import TextBlob

    blob = TextBlob(texto)
    pol = blob.sentiment.polarity
    subj = blob.sentiment.subjectivity
    clasif = "Positivo" if pol > 0.1 else "Negativo" if pol < -0.1 else "Neutral"
    return {"polaridad": pol, "subjetividad": subj, "clasificacion": clasif}


# ============================================================================
# PIPELINE COMPLETO (orquestador)
# ============================================================================
def pipeline_completo(
    texto: str,
    nombre: str,
    set_sw: set[str],
    lematizador=None,
    tipo_lem: str = "nltk",
    idioma_stem: str = "spanish",
) -> dict:
    """
    Corre el pipeline básico completo sobre UN texto y devuelve todas las
    métricas en un dict plano, listo para agregarse en un DataFrame
    comparativo entre condiciones.
    """
    limpio = limpiar_texto(texto)
    tokens = tokenizar(limpio)
    sin_sw, stats_sw = filtrar_stopwords(tokens, set_sw)
    stems, stats_stem = aplicar_stemming(sin_sw, idioma_stem)

    if lematizador is None:
        lematizador, tipo_lem = cargar_lematizador()
    lemmas, stats_lem = aplicar_lematizacion(sin_sw, lematizador, tipo_lem)

    frec = analizar_frecuencias(lemmas)
    sent = analizar_sentimiento(limpio)

    return {
        "nombre": nombre,
        "n_tokens": stats_sw["n_original"],
        "pct_stopwords": stats_sw["pct_stopwords"],
        "pct_eliminado_total": stats_sw["pct_eliminado_total"],
        "pct_stemming_modificado": stats_stem["pct_modificado"],
        "pct_lematizado": stats_lem["pct_lematizado"],
        "metodo_lematizacion": stats_lem["metodo"],
        "n_tipos": frec["n_tipos"],
        "n_hapax": frec["n_hapax"],
        "long_promedio_hapax": frec["long_promedio_hapax"],
        "long_min_hapax": frec["long_min_hapax"],
        "sentimiento_polaridad": sent["polaridad"],
        "sentimiento_subjetividad": sent["subjetividad"],
        "sentimiento_clasif": sent["clasificacion"],
        # objetos completos para análisis profundo
        "_tokens": tokens,
        "_lemmas": lemmas,
        "_frecuencias": frec,
    }


if __name__ == "__main__":
    import babel_core as bc

    print("babel_pipeline.py — smoke test")
    sw = construir_set_stopwords()
    print(f"  Set de stopwords multi-idioma: {len(sw)} palabras")

    sint = bc.generar_babel_sintetico(50000, seed=1)
    lem, tipo = cargar_lematizador()
    res = pipeline_completo(sint, "Null_test", sw, lem, tipo)
    print(f"  Pipeline sobre null (50k chars):")
    print(f"    % stopwords detectadas: {res['pct_stopwords']:.2f}%")
    print(f"    % stemming modificado:  {res['pct_stemming_modificado']:.2f}%")
    print(f"    % lematizado:           {res['pct_lematizado']:.2f}%")
    print(f"    sentimiento polaridad:  {res['sentimiento_polaridad']:.3f}")
    print(f"    long. mín. hapax:       {res['long_min_hapax']}")
