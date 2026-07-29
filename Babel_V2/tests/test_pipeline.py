"""Pipeline clásico: limpieza, tokenización, stopwords, stemming, lematización."""
import pytest

import babel_pipeline as bp


# --- Limpieza: la decisión §5.10 --------------------------------------------

def test_los_no_alfabeticos_son_frontera_no_se_borran():
    """El corazón de §5.10. Borrar pegaba 'hola' y 'mundo' en 'holamundo'."""
    assert bp.limpiar_texto("hola,mundo") == "hola mundo"
    assert bp.limpiar_texto("uno.dos,tres") == "uno dos tres"


def test_no_pega_tokens_a_traves_de_guiones():
    """Moby Dick: 'Usher--threadbare'."""
    assert bp.limpiar_texto("usher--threadbare") == "usher threadbare"


def test_separa_los_multiword_de_cess_esp():
    """'Electricité_de_France' contaba como un solo tipo."""
    assert bp.limpiar_texto("Electricité_de_France") == "electricite de france"
    assert bp.limpiar_texto("51_por_ciento") == "por ciento"


def test_normaliza_acentos():
    assert bp.limpiar_texto("corazón") == "corazon"
    assert bp.limpiar_texto("MAÑANA") == "manana"
    assert bp.limpiar_texto("café años ñu") == "cafe anos nu"


def test_sin_normalizar_acentos_las_tildes_parten_la_palabra():
    """Justifica por qué normalizar_acentos existe y va en True."""
    assert bp.limpiar_texto("corazón", normalizar_acentos=False) == "coraz n"


def test_preservar_puntuacion_conserva_coma_y_punto():
    assert bp.limpiar_texto("hola,mundo.", preservar_puntuacion=True) == "hola,mundo."


def test_colapsa_espacios_y_recorta():
    assert bp.limpiar_texto("  hola   \n\n  mundo  ") == "hola mundo"


def test_texto_vacio():
    assert bp.limpiar_texto("") == ""


# --- Tokenización -----------------------------------------------------------

def test_tokenizar_solo_devuelve_letras():
    assert bp.tokenizar("hola mundo 123 x_y") == ["hola", "mundo", "x", "y"]


def test_tokenizar_es_case_insensitive():
    assert bp.tokenizar("Hola MUNDO") == ["hola", "mundo"]


def test_tokenizar_texto_vacio():
    assert bp.tokenizar("") == []


# --- Stopwords --------------------------------------------------------------

def test_set_de_stopwords_multiidioma():
    sw = bp.construir_set_stopwords()
    assert len(sw) > 1_000
    assert "de" in sw and "the" in sw


def test_filtrar_separa_cortos_de_stopwords_reales():
    """§5.1: nunca fusionar ambas métricas bajo una etiqueta."""
    sw = {"de", "la", "the"}
    tokens = ["ab", "de", "casa", "la", "xyz", "the", "a"]
    conservados, stats = bp.filtrar_stopwords(tokens, sw, min_longitud=3)

    assert conservados == ["casa", "xyz"]
    assert stats["eliminados_cortos"] == 4        # ab, de, la, a
    assert stats["eliminados_stopwords"] == 1     # the
    assert stats["n_original"] == 7


def test_umbral_cero_no_elimina_por_longitud():
    sw = {"de"}
    _, stats = bp.filtrar_stopwords(["de", "ab"], sw, min_longitud=1)
    assert stats["eliminados_cortos"] == 0
    assert stats["eliminados_stopwords"] == 1


def test_filtrar_lista_vacia():
    _, stats = bp.filtrar_stopwords([], set(), min_longitud=3)
    assert stats["pct_stopwords"] == 0


# --- Stemming ---------------------------------------------------------------

def test_stemming_devuelve_un_stem_por_token():
    tokens = ["casas", "corriendo", "gatos"]
    stems, stats = bp.aplicar_stemming(tokens, "spanish")
    assert len(stems) == len(tokens)
    assert stats["n_tokens"] == 3
    assert 0 <= stats["pct_modificado"] <= 100


def test_stemming_reduce_el_vocabulario():
    stems, stats = bp.aplicar_stemming(["casa", "casas"], "spanish")
    assert stats["vocab_despues"] <= stats["vocab_antes"]


# --- Lematización -----------------------------------------------------------

def test_lematizacion_preserva_la_longitud():
    lem, tipo = bp.cargar_lematizador()
    tokens = [f"palabra{i%50}" for i in range(1_000)]
    lemmas, stats = bp.aplicar_lematizacion(tokens, lem, tipo)
    assert len(lemmas) == len(tokens)
    assert stats["n_tokens"] == 1_000


def test_lematizacion_por_bloques_no_desalinea():
    """El fix del error E088: spaCy rechaza documentos de más de 1M de chars."""
    lem, tipo = bp.cargar_lematizador()
    if tipo != "spacy":
        pytest.skip("spaCy no disponible; el chunking solo aplica a spaCy")
    tokens = [f"tok{i:05d}" for i in range(50_000)]   # ~400k chars unidos
    lemmas, _ = bp.aplicar_lematizacion(tokens, lem, tipo, chunk_tokens=5_000)
    assert len(lemmas) == len(tokens)


# --- Frecuencias ------------------------------------------------------------

def test_analizar_frecuencias_identifica_hapax():
    frec = bp.analizar_frecuencias(["uno", "uno", "dos", "tres"])
    assert frec["n_tokens"] == 4
    assert frec["n_tipos"] == 3
    assert frec["n_hapax"] == 2                  # dos, tres
    assert frec["long_min_hapax"] == 3


def test_frecuencias_lista_vacia():
    frec = bp.analizar_frecuencias([])
    assert frec["n_hapax"] == 0
    assert frec["long_promedio_hapax"] == 0


# --- Sentimiento ------------------------------------------------------------

def test_sentimiento_devuelve_rango_valido():
    try:
        s = bp.analizar_sentimiento("this is a wonderful and happy day")
    except ImportError:
        pytest.skip("TextBlob no instalado")
    assert -1.0 <= s["polaridad"] <= 1.0
    assert 0.0 <= s["subjetividad"] <= 1.0
    assert s["clasificacion"] in {"Positivo", "Negativo", "Neutral"}
