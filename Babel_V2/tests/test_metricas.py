"""Métricas de aleatoriedad: entropía, compresión, Zipf, Heaps, χ²."""
import math

import pytest

import babel_core as bc


# --- Entropía ---------------------------------------------------------------

def test_entropia_de_texto_constante_es_cero():
    assert bc.entropia_shannon("aaaaaaaa", 1) == 0.0


def test_entropia_maxima_con_simbolos_equiprobables():
    assert bc.entropia_shannon("abcd" * 250, 1) == pytest.approx(2.0)


def test_entropia_de_texto_vacio_no_revienta():
    assert bc.entropia_shannon("", 1) == 0.0


def test_entropia_maxima_teorica():
    assert bc.entropia_maxima(30, 1) == pytest.approx(math.log2(30))
    assert bc.entropia_maxima(30, 2) == pytest.approx(2 * math.log2(30))


def test_aleatorio_tiene_mas_entropia_que_repetitivo():
    aleatorio = bc.generar_babel_sintetico(50_000, seed=1)
    repetitivo = "abc " * 12_500
    assert bc.entropia_shannon(aleatorio, 1) > bc.entropia_shannon(repetitivo, 1)


# --- Compresión -------------------------------------------------------------

def test_solo_devuelve_los_algoritmos_pedidos():
    t = bc.generar_babel_sintetico(10_000, seed=1)
    assert set(bc.ratios_compresion(t, algoritmos=("gzip",))) == {"gzip"}
    assert set(bc.ratios_compresion(t, algoritmos=("gzip", "bz2"))) == {"gzip", "bz2"}


def test_por_defecto_calcula_los_de_stdlib():
    r = bc.ratios_compresion(bc.generar_babel_sintetico(10_000, seed=1))
    assert {"gzip", "bz2", "lzma"} <= set(r)


def test_lo_repetitivo_comprime_y_lo_aleatorio_no():
    repetitivo = bc.ratios_compresion("abcdefgh" * 5_000, algoritmos=("gzip",))["gzip"]
    aleatorio = bc.ratios_compresion(
        bc.generar_babel_sintetico(40_000, seed=1), algoritmos=("gzip",))["gzip"]
    assert repetitivo < 0.05
    assert aleatorio > 0.5


def test_texto_vacio_devuelve_dict_vacio():
    assert bc.ratios_compresion("") == {}


# --- Zipf -------------------------------------------------------------------

def test_excluir_hapax_recorta_el_tramo_de_ajuste():
    t = bc.generar_babel_sintetico(200_000, seed=1)
    con_cola = bc.ley_zipf(t)
    sin_cola = bc.ley_zipf(t, excluir_hapax=True)
    assert sin_cola["n_rangos_ajuste"] < con_cola["n_rangos_ajuste"]
    # La cola horizontal arruina el ajuste; sin ella la recta aparece
    assert sin_cola["r2"] > con_cola["r2"]
    assert sin_cola["pendiente"] < con_cola["pendiente"]


def test_zipf_sobre_texto_aleatorio_da_pendiente_zipfiana():
    """Miller/Mandelbrot: el ruido TAMBIÉN cumple Zipf."""
    t = bc.generar_babel_sintetico(300_000, seed=2)
    z = bc.ley_zipf(t, excluir_hapax=True)
    assert -1.2 < z["pendiente"] < -0.7
    assert z["r2"] > 0.8


def test_zipf_sin_tokens_no_revienta():
    z = bc.ley_zipf("")
    assert math.isnan(z["pendiente"])


# --- Heaps ------------------------------------------------------------------

def test_heaps_en_aleatorio_es_casi_lineal():
    t = bc.generar_babel_sintetico(200_000, seed=1)
    h = bc.ley_heaps(t)
    assert h["beta"] > 0.9
    assert h["r2"] > 0.99


def test_heaps_devuelve_tasa_de_tipos():
    h = bc.ley_heaps(bc.generar_babel_sintetico(100_000, seed=1))
    assert 0 < h["tasa_tipos"] <= 1
    assert h["tasa_tipos"] == pytest.approx(h["Vs"][-1] / h["Ns"][-1])


def test_n_min_restringe_los_puntos_del_ajuste():
    t = bc.generar_babel_sintetico(200_000, seed=1)
    todos = bc.ley_heaps(t)
    recortado = bc.ley_heaps(t, n_min=10_000)
    assert recortado["n_puntos_ajuste"] < todos["n_puntos_ajuste"]


def test_heaps_con_texto_minusculo_no_revienta():
    h = bc.ley_heaps("una dos")
    assert math.isnan(h["beta"])


def test_vocabulario_repetitivo_satura():
    """Contraste: texto con vocabulario cerrado tiene beta bajo."""
    h = bc.ley_heaps("alfa beta gamma delta " * 5_000)
    assert h["beta"] < 0.3


# --- Chi cuadrado -----------------------------------------------------------

def test_chi2_usa_el_espacio_completo_de_categorias():
    """Regresión §6/E.5 #3: descartar las de conteo 0 sesga hacia 'uniforme'."""
    t = bc.generar_babel_sintetico(200_000, seed=1)
    uni = bc.chi2_uniformidad(t, n=1)
    bi = bc.chi2_uniformidad(t, n=2)
    assert uni["n_categorias"] == 29
    assert bi["n_categorias"] == 29 ** 2 == 841


def test_chi2_cuenta_las_categorias_vacias():
    """Con poco texto quedan bigramas sin observar: deben entrar como cero."""
    t = bc.generar_babel_sintetico(2_000, seed=1)
    bi = bc.chi2_uniformidad(t, n=2)
    assert bi["n_categorias"] == 841
    assert bi["n_categorias_vacias"] > 0


def test_chi2_no_rechaza_sobre_aleatorio_uniforme():
    t = bc.generar_babel_sintetico(200_000, probs_contenido=None, seed=4)
    assert bc.chi2_uniformidad(t, n=1)["p_valor"] > 0.01


def test_chi2_rechaza_sobre_texto_sesgado():
    t = "aaaaaaaaab" * 20_000
    r = bc.chi2_uniformidad(t, n=1)
    assert r["p_valor"] < 1e-10


def test_chi2_excluye_el_salto_de_linea_por_defecto():
    t = bc.generar_babel_sintetico(50_000, seed=1)
    assert bc.chi2_uniformidad(t, n=1)["n_categorias"] == 29    # no 30


def test_chi2_reporta_la_frecuencia_esperada():
    """Sirve para saber si el test es fiable (E>=5)."""
    r = bc.chi2_uniformidad(bc.generar_babel_sintetico(100_000, seed=1), n=1)
    assert r["frec_esperada"] > 5


def test_chi2_con_texto_vacio_no_revienta():
    r = bc.chi2_uniformidad("")
    assert r["n_categorias"] == 0
