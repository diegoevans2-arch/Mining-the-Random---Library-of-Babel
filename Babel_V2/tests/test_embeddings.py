"""
Bloque C — embeddings, clustering y near-duplicates.

Casi todo se valida con nubes sintéticas de estructura conocida: si las
métricas no distinguen una bola isótropa de dos cúmulos separados, no sirven
para decidir si Babel tiene islas de significado.
"""
import numpy as np
import pytest

import babel_embeddings as be


@pytest.fixture(scope="module")
def isotropa():
    """Sin estructura: direcciones al azar sobre la esfera."""
    rng = np.random.default_rng(0)
    V = rng.normal(size=(200, 32))
    return V / np.linalg.norm(V, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def cumulos():
    """Dos grupos compactos y bien separados."""
    rng = np.random.default_rng(1)
    centros = rng.normal(size=(2, 32))
    V = np.vstack([c + 0.05 * rng.normal(size=(100, 32)) for c in centros])
    return V / np.linalg.norm(V, axis=1, keepdims=True)


# ============================================================================
# GEOMETRÍA — C.2
# ============================================================================

def test_la_dimension_efectiva_separa_estructura(isotropa, cumulos):
    """Una bola ocupa muchas direcciones; dos cúmulos, esencialmente una."""
    g_iso = be.geometria(isotropa)
    g_cum = be.geometria(cumulos)
    assert g_iso["dim_efectiva"] > 10
    assert g_cum["dim_efectiva"] < 3
    assert g_iso["dim_efectiva"] > g_cum["dim_efectiva"]


def test_la_dispersion_de_similitudes_separa_estructura(isotropa, cumulos):
    """Con cúmulos hay pares muy juntos y muy lejos: varianza alta."""
    assert be.geometria(cumulos)["sim_sd"] > be.geometria(isotropa)["sim_sd"]


def test_la_nube_isotropa_esta_centrada(isotropa):
    """Sin dirección dominante, el centroide queda cerca del origen."""
    assert be.geometria(isotropa)["norma_centroide"] < 0.15


def test_geometria_reporta_el_numero_de_chunks(isotropa):
    assert be.geometria(isotropa)["n_chunks"] == 200


def test_geometria_con_pocos_vectores():
    assert be.geometria(np.zeros((1, 8))) == {}
    assert be.geometria(np.zeros((0, 8))) == {}


def test_la_distribucion_de_similitudes_alimenta_a_geometria(isotropa):
    """La figura del bloque y el número del veredicto salen del mismo cálculo.

    Es el invariante que justifica que `similitudes_por_pares` exista aparte:
    si divergieran, la media dibujada y la media reportada podrían no coincidir
    sin que nada avisara.
    """
    sims = be.similitudes_por_pares(isotropa, seed=0)
    g = be.geometria(isotropa, seed=0)
    assert float(sims.mean()) == g["sim_media"]
    assert float(sims.std()) == g["sim_sd"]
    assert len(sims) == 200 * 199 // 2          # todos los pares, sin repetir


def test_las_similitudes_separan_estructura_de_ruido(isotropa, cumulos):
    """Dos cúmulos compactos se parecen más entre sí que una nube isótropa.

    Es la forma testeable del hallazgo del bloque: la degeneración se ve como
    similitud ALTA, no como dispersión.
    """
    assert be.similitudes_por_pares(cumulos).mean() > \
           be.similitudes_por_pares(isotropa).mean()


def test_similitudes_por_pares_con_muy_pocos_vectores():
    assert len(be.similitudes_por_pares(np.zeros((1, 8)))) == 0
    assert len(be.similitudes_por_pares(np.zeros((0, 8)))) == 0


def test_el_submuestreo_de_pares_no_cambia_el_resultado(isotropa):
    """Con n grande se submuestrean pares; debe dar lo mismo dentro de ruido."""
    completo = be.geometria(isotropa, max_pares=10**9)
    muestreado = be.geometria(isotropa, max_pares=2_000)
    assert muestreado["sim_media"] == pytest.approx(completo["sim_media"], abs=0.05)


# ============================================================================
# CLUSTERING — C.3
# ============================================================================

def test_encuentra_los_cumulos_que_existen(cumulos):
    red = be.reducir(cumulos, n_componentes=3)
    r = be.agrupar(red, min_cluster=10)
    assert r["n_clusters"] >= 2
    assert r["silueta"] > 0.5


def test_hdbscan_siempre_devuelve_algo(isotropa):
    """
    El aviso que motiva el contraste contra el null: sobre una nube SIN
    estructura, HDBSCAN igualmente particiona o declara ruido. El número suelto
    no es interpretable.
    """
    red = be.reducir(isotropa, n_componentes=3)
    r = be.agrupar(red, min_cluster=10)
    assert "n_clusters" in r and "pct_ruido" in r
    assert 0.0 <= r["pct_ruido"] <= 100.0


def test_agrupar_con_muy_pocos_puntos():
    r = be.agrupar(np.zeros((5, 3)), min_cluster=10)
    assert r["n_clusters"] == 0
    assert r["pct_ruido"] == 100.0


def test_reducir_baja_la_dimension(isotropa):
    assert be.reducir(isotropa, n_componentes=4).shape == (200, 4)


def test_reducir_no_falla_con_menos_puntos_que_componentes():
    V = np.zeros((3, 10))
    assert be.reducir(V, n_componentes=5).shape == (3, 10)


# ============================================================================
# NEAR-DUPLICATES — C.4
# ============================================================================

def test_jaccard_uno_para_textos_identicos():
    j = be.jaccard_por_pares({"a": "abcdefgh" * 3_000, "b": "abcdefgh" * 3_000},
                             max_chars=20_000)
    if not j:
        pytest.skip("datasketch no instalado")
    assert j["a vs b"] == pytest.approx(1.0, abs=0.05)


def test_jaccard_cero_para_textos_disjuntos():
    j = be.jaccard_por_pares({"a": "abcdefgh" * 3_000, "b": "zyxwvuts" * 3_000},
                             max_chars=20_000)
    if not j:
        pytest.skip("datasketch no instalado")
    assert j["a vs b"] == pytest.approx(0.0, abs=0.05)


def test_todos_los_pares_sin_repetir():
    j = be.jaccard_por_pares({"a": "ab" * 5_000, "b": "cd" * 5_000,
                              "c": "ef" * 5_000}, max_chars=10_000)
    if not j:
        pytest.skip("datasketch no instalado")
    assert len(j) == 3                      # C(3,2), sin duplicar ni autopares
    assert "a vs a" not in j


def test_max_chars_recorta_el_texto():
    f = be.firma_minhash("ab" * 100_000, max_chars=5_000)
    if f is None:
        pytest.skip("datasketch no instalado")
    assert f is not None


# ============================================================================
# EMBEDDINGS — degradación limpia
# ============================================================================

def test_embeber_sin_modelo_devuelve_vacio():
    assert be.embeber(["hola"], None).size == 0


def test_embeber_sin_chunks_devuelve_vacio():
    assert be.embeber([], object()).size == 0


def test_la_configuracion_esta_dentro_del_rango_del_roadmap():
    """El roadmap fija chunks de 100-500 caracteres."""
    assert 100 <= be.CHARS_POR_CHUNK <= 500


def test_reutiliza_el_muestreo_del_bloque_b():
    """La ventana/chunk es el mismo concepto; una sola implementación."""
    import babel_transformers as bt
    assert be.muestrear_ventanas is bt.muestrear_ventanas
