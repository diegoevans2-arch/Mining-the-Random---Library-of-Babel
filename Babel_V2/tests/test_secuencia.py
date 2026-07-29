"""
Bloque C — estructura secuencial (C.5 perplejidad, C.6 Markov).

Los tests usan texto sintético con estructura conocida: si la ganancia de
información no es cero sobre i.i.d. y máxima sobre texto periódico, la métrica
no mide lo que dice medir.
"""
import math

import numpy as np
import pytest

import babel_secuencia as bs


@pytest.fixture(scope="module")
def aleatorio():
    """i.i.d. uniforme sobre 5 símbolos: H = log2(5) = 2,322 y ganancia nula."""
    return "".join(np.random.default_rng(1).choice(list("abcde"), 40_000))


@pytest.fixture(scope="module")
def periodico():
    """Determinista: conocer 1 carácter basta para predecir el siguiente."""
    return "abcde" * 8_000


# ============================================================================
# CONTEOS Y MATRICES
# ============================================================================

def test_cuenta_las_transiciones_correctas():
    tabla = bs.contar_transiciones("abcabc", orden=1, excluir=())
    assert tabla["a"]["b"] == 2
    assert tabla["b"]["c"] == 2
    assert tabla["c"]["a"] == 1


def test_orden_cero_es_la_distribucion_marginal():
    tabla = bs.contar_transiciones("aaab", orden=0, excluir=())
    assert set(tabla) == {""}
    assert tabla[""]["a"] == 3 and tabla[""]["b"] == 1


def test_excluye_el_salto_de_linea_por_defecto():
    """El '\\n' de Babel es formato, no contenido."""
    tabla = bs.contar_transiciones("ab\ncd", orden=1)
    assert "\n" not in tabla
    assert tabla["b"]["c"] == 1        # el salto no rompe la secuencia


def test_texto_mas_corto_que_el_orden():
    assert bs.contar_transiciones("ab", orden=5, excluir=()) == {}


def test_la_matriz_tiene_filas_de_probabilidad():
    m = bs.matriz_transicion("abcabcabc", orden=1, excluir=())
    filas = m["matriz"].sum(axis=1)
    assert np.allclose(filas, 1.0)
    assert m["matriz"].shape == (len(m["contextos"]), len(m["simbolos"]))


def test_la_matriz_registra_las_observaciones_por_contexto():
    m = bs.matriz_transicion("aaab", orden=1, excluir=())
    assert m["n_por_contexto"].sum() == 3      # 3 transiciones en 4 caracteres


# ============================================================================
# ENTROPÍA CONDICIONAL Y GANANCIA — el núcleo de C.6
# ============================================================================

def test_entropia_de_orden_cero_es_la_de_shannon(aleatorio):
    H = bs.entropia_condicional(aleatorio, orden=0)
    assert H == pytest.approx(math.log2(5), abs=0.01)


def test_sin_ganancia_sobre_texto_iid(aleatorio):
    """Lo esencial: en i.i.d. el pasado NO informa."""
    g = bs.ganancia_informacion(aleatorio, ordenes=(1, 2))
    assert abs(g["ganancia_1"]) < 0.02
    assert abs(g["ganancia_2"]) < 0.10


def test_ganancia_maxima_sobre_texto_determinista(periodico):
    """En 'abcde' repetido, un carácter basta para predecir el siguiente."""
    g = bs.ganancia_informacion(periodico, ordenes=(1,))
    assert g["H1"] == pytest.approx(0.0, abs=1e-9)
    assert g["ganancia_1"] == pytest.approx(g["H0"], abs=1e-9)


def test_la_ganancia_no_decrece_con_el_orden(periodico):
    """Más contexto nunca puede informar menos (en el estimador plug-in)."""
    g = bs.ganancia_informacion(periodico, ordenes=(1, 2, 3))
    assert g["ganancia_1"] <= g["ganancia_2"] + 1e-9
    assert g["ganancia_2"] <= g["ganancia_3"] + 1e-9


def test_reporta_la_escasez_de_contextos(aleatorio):
    """
    Sin esto no se puede juzgar el sesgo: la ganancia aparente crece cuando
    las observaciones por contexto caen, aunque no haya estructura alguna.
    """
    g = bs.ganancia_informacion(aleatorio, ordenes=(1, 3))
    assert g["n_contextos_1"] == 5                 # 5 símbolos
    assert g["n_contextos_3"] <= 125               # 5^3
    assert g["obs_por_contexto_1"] > g["obs_por_contexto_3"]


def test_el_sesgo_de_muestra_finita_es_real(aleatorio):
    """
    Documenta el artefacto que motiva el módulo: sobre i.i.d. PURO, la ganancia
    medida crece con el orden aunque la verdadera sea cero, porque los
    contextos se quedan sin observaciones.
    """
    g = bs.ganancia_informacion(aleatorio, ordenes=(1, 4))
    assert g["obs_por_contexto_4"] < 100
    assert g["ganancia_4"] > g["ganancia_1"]       # sesgo, no estructura


# ============================================================================
# MODELO DE LENGUAJE Y PERPLEJIDAD — C.5
# ============================================================================

def test_perplejidad_de_iid_se_acerca_al_alfabeto(aleatorio):
    """Un modelo honesto sobre i.i.d. de 5 símbolos duda entre ~5."""
    lm = bs.ModeloCaracteres(orden=3).entrenar(aleatorio)
    assert lm.perplejidad(aleatorio) == pytest.approx(5.0, abs=0.6)


def test_perplejidad_baja_sobre_texto_predecible(periodico):
    lm = bs.ModeloCaracteres(orden=3).entrenar(periodico)
    assert lm.perplejidad(periodico) < 2.0


def test_el_modelo_penaliza_el_texto_que_no_se_parece(periodico, aleatorio):
    """Entrenado en periódico, el ruido le resulta mucho más caro."""
    lm = bs.ModeloCaracteres(orden=3).entrenar(periodico)
    assert lm.perplejidad(aleatorio) > lm.perplejidad(periodico)


def test_puede_superar_el_tamano_del_alfabeto(periodico, aleatorio):
    """
    Peor que uniforme = el modelo está confiadamente equivocado, no ignorante.
    Es el resultado real sobre Babel con un modelo entrenado en español.
    """
    lm = bs.ModeloCaracteres(orden=3).entrenar(periodico)
    assert lm.perplejidad(aleatorio) > 5.0


def test_entropia_cruzada_y_perplejidad_son_coherentes(aleatorio):
    lm = bs.ModeloCaracteres(orden=2).entrenar(aleatorio)
    h = lm.entropia_cruzada(aleatorio)
    assert lm.perplejidad(aleatorio) == pytest.approx(2 ** h, rel=1e-9)


def test_el_backoff_evita_probabilidad_cero():
    """Sin backoff, un contexto inédito daría perplejidad infinita."""
    lm = bs.ModeloCaracteres(orden=4).entrenar("abcde" * 500)
    logp, n = lm.log_verosimilitud("edcba" * 100)
    assert n > 0
    assert math.isfinite(logp)


def test_texto_mas_corto_que_el_orden_no_revienta():
    lm = bs.ModeloCaracteres(orden=5).entrenar("abcde" * 200)
    assert math.isnan(lm.perplejidad("ab"))


# ============================================================================
# DIVERGENCIA ENTRE MATRICES DE TRANSICIÓN
# ============================================================================

def test_divergencia_nula_consigo_mismo(aleatorio):
    d = bs.divergencia_transiciones(aleatorio, aleatorio, orden=1)
    assert d["js_media"] == pytest.approx(0.0, abs=1e-12)


def test_divergencia_alta_entre_textos_distintos(aleatorio, periodico):
    d = bs.divergencia_transiciones(aleatorio, periodico, orden=1)
    assert d["js_media"] > 0.5


def test_la_divergencia_esta_acotada(aleatorio, periodico):
    """Jensen-Shannon en log2 vive en [0, 1]; KL no, y por eso no se usa."""
    d = bs.divergencia_transiciones(aleatorio, periodico, orden=1)
    assert 0.0 <= d["js_media"] <= 1.0


def test_divergencia_sin_contextos_comunes():
    d = bs.divergencia_transiciones("aaaa", "bbbb", orden=1, excluir=())
    assert d["n_contextos_comunes"] == 0
    assert math.isnan(d["js_media"])
