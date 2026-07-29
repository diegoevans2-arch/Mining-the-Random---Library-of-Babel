"""
Bloque D — palabras reales emergentes y búsqueda semántica.

La parte que más valor tiene es la predicción combinatoria: se puede validar
con aritmética, sin corpus ni modelos.
"""
import numpy as np
import pytest

import babel_lexico as bl


LEXICO = {"casa", "perro", "sol", "mar", "luz", "libro", "tiempo",
          "the", "cat", "dog", "sun", "sea"}


# ============================================================================
# PREDICCIÓN COMBINATORIA — el núcleo del bloque
# ============================================================================

def test_longitud_media_teorica_coincide_con_lo_medido():
    """
    3 separadores sobre 29 símbolos -> geométrica de media 8,67.
    El valor medido sobre el corpus real es 8,7: el modelo describe el texto.
    """
    assert bl.longitud_media_teorica() == pytest.approx(8.667, abs=0.01)


def test_la_probabilidad_cae_con_la_longitud():
    """Cada letra extra divide por 26; el diccionario no crece tan rápido."""
    P = bl.prob_palabra_por_longitud(LEXICO, longitudes=range(3, 7))
    assert P[3] > P[4] > P[5]


def test_la_probabilidad_es_n_sobre_alfabeto_elevado_k():
    """La fórmula exacta, sin aproximaciones."""
    lex = {"abc", "xyz"}                       # 2 palabras de 3 letras
    P = bl.prob_palabra_por_longitud(lex, alfabeto=26, longitudes=(3,))
    assert P[3] == pytest.approx(2 / 26 ** 3)


def test_longitud_sin_palabras_da_probabilidad_cero():
    P = bl.prob_palabra_por_longitud({"abc"}, longitudes=(3, 9))
    assert P[9] == 0.0


def test_la_prediccion_pondera_por_longitud_de_token():
    """Tokens largos aportan casi nada al total esperado."""
    tokens = ["abc"] * 1_000 + ["abcdefgh"] * 1_000
    pred = bl.prediccion_emergentes(tokens, {"abc"})
    assert pred["esperado_por_longitud"][3] > 0
    assert pred["esperado_por_longitud"][8] == 0.0
    assert pred["esperado_total"] == pytest.approx(1_000 / 26 ** 3)


def test_la_prediccion_registra_las_longitudes_observadas():
    pred = bl.prediccion_emergentes(["ab", "abc", "abc"], LEXICO)
    assert pred["tokens_por_longitud"] == {2: 1, 3: 2}


def test_perfil_de_longitudes_del_lexico():
    N = bl.perfil_longitudes_lexico(LEXICO, longitudes=range(3, 5))
    assert N[3] == sum(1 for p in LEXICO if len(p) == 3)


# ============================================================================
# OBSERVACIÓN — D.1
# ============================================================================

def test_cuenta_las_palabras_reales():
    tokens = ["sol", "xkqvz", "casa", "mar", "wqxzjf"]
    r = bl.palabras_emergentes(tokens, LEXICO, min_long=3)
    assert r["n_emergentes"] == 3               # sol, casa, mar
    assert r["n_candidatos"] == 5
    assert r["tasa_pct"] == pytest.approx(60.0)


def test_el_umbral_de_longitud_filtra_antes_de_contar():
    """§5.1: por debajo de 3 letras casi todo coincide con algo."""
    tokens = ["ab", "sol", "casa", "tiempo"]    # 2, 3, 4 y 6 letras
    assert bl.palabras_emergentes(tokens, LEXICO, min_long=3)["n_candidatos"] == 3
    assert bl.palabras_emergentes(tokens, LEXICO, min_long=4)["n_candidatos"] == 2
    assert bl.palabras_emergentes(tokens, LEXICO, min_long=5)["n_candidatos"] == 1


def test_reporta_la_palabra_mas_larga():
    r = bl.palabras_emergentes(["sol", "tiempo"], LEXICO, min_long=3)
    assert r["long_max"] == 6


def test_distingue_apariciones_de_palabras_distintas():
    r = bl.palabras_emergentes(["sol", "sol", "mar"], LEXICO, min_long=3)
    assert r["n_emergentes"] == 3               # apariciones
    assert r["n_distintas"] == 2                # tipos


def test_sin_lexico_degrada_limpiamente():
    r = bl.palabras_emergentes(["sol"], set())
    assert r["disponible"] is False


def test_sin_tokens_no_revienta():
    r = bl.palabras_emergentes([], LEXICO)
    assert r["n_emergentes"] == 0
    assert r["tasa_pct"] == 0.0


# ============================================================================
# DISTRIBUCIÓN ENTRE DOCUMENTOS — D.2
# ============================================================================

def test_tasa_homogenea_entre_documentos():
    docs = {f"libro_{i}": ["sol", "casa", "xkqv", "wzpt"] for i in range(5)}
    r = bl.distribucion_por_documento(docs, LEXICO)
    assert len(r["por_documento"]) == 5
    assert r["tasa_sd"] == pytest.approx(0.0)


def test_detecta_un_documento_atipico():
    docs = {f"libro_{i}": ["xkqv", "wzpt"] for i in range(4)}
    docs["raro"] = ["sol", "casa", "mar", "luz"]
    r = bl.distribucion_por_documento(docs, LEXICO)
    assert r["tasa_sd"] > 0


def test_razon_varianza_media_para_sucesos_raros():
    """Con sucesos raros e independientes, var/media ~ 1 (Poisson)."""
    rng = np.random.default_rng(0)
    docs = {}
    for i in range(30):
        n = int(rng.poisson(4))
        docs[f"d{i}"] = ["sol"] * n + ["xkqv"] * (50 - n)
    r = bl.distribucion_por_documento(docs, LEXICO)
    assert 0.3 < r["razon_varianza_media"] < 3.0


def test_distribucion_sin_documentos():
    assert bl.distribucion_por_documento({}, LEXICO) == {}


# ============================================================================
# BÚSQUEDA SEMÁNTICA — D.3
# ============================================================================

class EmbedderFalso:
    """Devuelve vectores deterministas y normalizados según el texto."""

    def encode(self, textos, normalize_embeddings=True, show_progress_bar=False):
        V = np.array([[len(t) % 7, len(t) % 5, 1.0] for t in textos], dtype=float)
        return V / np.linalg.norm(V, axis=1, keepdims=True)


def test_busqueda_devuelve_el_mejor_por_condicion():
    emb = EmbedderFalso()
    chunks = {"Babel": ["aaa", "bbbbbb"], "Español": ["cc", "dddd"]}
    vecs = {k: emb.encode(v) for k, v in chunks.items()}
    r = bl.busqueda_semantica(["hola mundo"], chunks, vecs, emb)
    assert set(r) == {"Babel", "Español"}
    for cond in r:
        assert -1.0 <= r[cond]["sim_max_media"] <= 1.0
        assert len(r[cond]["por_frase"]) == 1


def test_busqueda_adjunta_el_fragmento_encontrado():
    emb = EmbedderFalso()
    chunks = {"Babel": ["abcdefghij" * 20]}
    vecs = {"Babel": emb.encode(chunks["Babel"])}
    r = bl.busqueda_semantica(["frase"], chunks, vecs, emb)
    frag = list(r["Babel"]["por_frase"].values())[0]["fragmento"]
    assert 0 < len(frag) <= 90


def test_busqueda_sin_embedder():
    assert bl.busqueda_semantica(["x"], {}, {}, None) == {}


def test_busqueda_ignora_condiciones_vacias():
    emb = EmbedderFalso()
    r = bl.busqueda_semantica(["x"], {"A": []}, {"A": np.zeros((0, 3))}, emb)
    assert r == {}


def test_hay_frases_sonda_definidas():
    assert len(bl.FRASES_SONDA) >= 5
    assert all(isinstance(f, str) and f for f in bl.FRASES_SONDA)


# ============================================================================
# LÉXICOS EXTERNOS — se saltan si no están instalados
# ============================================================================

def test_carga_de_lexicos_externos():
    r = bl.cargar_lexico(idiomas=("es",), n_palabras=5_000)
    if not r["disponible"]:
        pytest.skip("wordfreq / pyspellchecker no instalados")
    assert r["n_total"] > 1_000
    assert all(p.isalpha() for p in list(r["lexico"])[:100])


def test_el_lexico_externo_tiene_palabras_de_varias_longitudes():
    r = bl.cargar_lexico(idiomas=("es",), n_palabras=20_000)
    if not r["disponible"]:
        pytest.skip("wordfreq / pyspellchecker no instalados")
    N = bl.perfil_longitudes_lexico(r["lexico"], longitudes=range(3, 9))
    assert sum(1 for k in N if N[k] > 0) >= 5
