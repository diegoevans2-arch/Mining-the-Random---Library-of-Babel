"""
Bloque B. La mayoría de los tests usan un pipeline falso: la lógica de muestreo
y de métricas no necesita 2 GB de modelos para validarse, y así la suite sigue
corriendo en cualquier máquina.
"""
import math

import pytest

import babel_transformers as bt


# ============================================================================
# MUESTREO DE VENTANAS
# ============================================================================

def test_devuelve_el_numero_y_tamano_pedidos():
    texto = "abcdefghij" * 5_000
    vs = bt.muestrear_ventanas(texto, n_ventanas=12, chars=300, seed=1)
    assert len(vs) == 12
    assert all(len(v) == 300 for v in vs)


def test_reproducible_por_semilla():
    texto = "abcdefghij" * 5_000
    a = bt.muestrear_ventanas(texto, n_ventanas=8, chars=200, seed=42)
    b = bt.muestrear_ventanas(texto, n_ventanas=8, chars=200, seed=42)
    c = bt.muestrear_ventanas(texto, n_ventanas=8, chars=200, seed=43)
    assert a == b
    assert a != c


def test_quita_los_saltos_de_linea():
    """El salto cada 80 chars es tipografía del sitio, no contenido."""
    texto = ("x" * 79 + "\n") * 200
    vs = bt.muestrear_ventanas(texto, n_ventanas=5, chars=300, seed=1)
    assert all("\n" not in v for v in vs)
    assert all(len(v) == 300 for v in vs)


def test_puede_conservar_los_saltos():
    texto = ("x" * 79 + "\n") * 200
    vs = bt.muestrear_ventanas(texto, n_ventanas=3, chars=300, seed=1,
                               quitar_saltos=False)
    assert any("\n" in v for v in vs)


def test_texto_mas_corto_que_la_ventana():
    vs = bt.muestrear_ventanas("hola mundo", n_ventanas=10, chars=800)
    assert vs == ["hola mundo"]


def test_texto_vacio():
    assert bt.muestrear_ventanas("", n_ventanas=5) == []


def test_las_ventanas_salen_del_texto():
    texto = "abcdefghij" * 5_000
    for v in bt.muestrear_ventanas(texto, n_ventanas=10, chars=250, seed=7):
        assert v in texto


# ============================================================================
# MÉTRICAS DE IDIOMA
# ============================================================================

def test_modelo_seguro_y_consistente():
    """Todas las ventanas al mismo idioma con alta probabilidad."""
    dists = [{"es": 0.95, "en": 0.05} for _ in range(10)]
    m = bt.metricas_idioma(dists)
    assert m["prob_top_media"] == pytest.approx(0.95)
    assert m["n_idiomas_distintos"] == 1
    assert m["entropia_idiomas"] == pytest.approx(0.0)
    assert m["idioma_modal"] == "es"
    assert m["pct_modal"] == 100.0


def test_modelo_perdido_reparte_entre_idiomas():
    """Lo que se espera sobre ruido: cada ventana cae en un idioma distinto."""
    idiomas = ["es", "en", "fr", "de"]
    dists = [{i: 0.3 if i == idiomas[k % 4] else 0.23 for i in idiomas}
             for k in range(8)]
    m = bt.metricas_idioma(dists)
    assert m["n_idiomas_distintos"] == 4
    assert m["entropia_idiomas"] == pytest.approx(2.0)   # log2(4)
    assert m["prob_top_media"] == pytest.approx(0.3)


def test_la_entropia_de_idiomas_discrimina():
    seguro = bt.metricas_idioma([{"es": 0.9, "en": 0.1}] * 8)
    perdido = bt.metricas_idioma(
        [{"es": 0.9, "en": 0.1}, {"en": 0.9, "es": 0.1}] * 4)
    assert perdido["entropia_idiomas"] > seguro["entropia_idiomas"]


def test_la_entropia_nunca_es_negativa():
    """Con un solo idioma el cálculo daba -0.0, que ensucia tablas y gráficos."""
    m = bt.metricas_idioma([{"hi": 0.57, "en": 0.43}] * 5)
    assert m["entropia_idiomas"] == 0.0
    assert not math.copysign(1, m["entropia_idiomas"]) < 0


def test_entropia_baja_no_implica_estructura():
    """
    Documenta el hallazgo que refutó la hipótesis inicial: XLM-R etiqueta el
    ruido como hindi en el 100% de las ventanas, así que la entropía baja
    igual que en lenguaje real. Quien discrimina es prob_top_media.
    """
    ruido = bt.metricas_idioma([{"hi": 0.57, "en": 0.2, "ur": 0.23}] * 10)
    natural = bt.metricas_idioma([{"es": 0.99, "pt": 0.01}] * 10)
    assert ruido["entropia_idiomas"] == natural["entropia_idiomas"] == 0.0
    assert ruido["prob_top_media"] < natural["prob_top_media"]


def test_metricas_idioma_sin_datos():
    assert bt.metricas_idioma([]) == {}


# ============================================================================
# MÉTRICAS DE SENTIMIENTO
# ============================================================================

def test_polaridad_es_positivo_menos_negativo():
    """Definida así para poder ponerla al lado de la de TextBlob."""
    m = bt.metricas_sentimiento(
        [{"positive": 0.7, "negative": 0.2, "neutral": 0.1}])
    assert m["polaridad"] == pytest.approx(0.5)
    assert -1.0 <= m["polaridad"] <= 1.0


def test_sentimiento_neutro_da_polaridad_cero():
    m = bt.metricas_sentimiento(
        [{"positive": 0.25, "negative": 0.25, "neutral": 0.5}])
    assert m["polaridad"] == pytest.approx(0.0)


def test_etiquetas_ausentes_cuentan_como_cero():
    m = bt.metricas_sentimiento([{"positive": 1.0}])
    assert m["negativo_medio"] == 0.0
    assert m["polaridad"] == pytest.approx(1.0)


def test_metricas_sentimiento_sin_datos():
    assert bt.metricas_sentimiento([]) == {}


# ============================================================================
# NORMALIZACIÓN DE ETIQUETAS Y PIPELINE
# ============================================================================

class PipeFalso:
    """Imita un pipeline de transformers con top_k=None."""

    def __init__(self, respuesta):
        self.respuesta = respuesta
        self.recibido = None

    def __call__(self, ventanas):
        self.recibido = ventanas
        return [self.respuesta for _ in ventanas]


def test_normaliza_las_etiquetas_label_n():
    """cardiffnlp puede devolver LABEL_0/1/2 en vez de negative/neutral/positive."""
    pipe = PipeFalso([{"label": "LABEL_2", "score": 0.8},
                      {"label": "LABEL_0", "score": 0.2}])
    dists = bt.clasificar_ventanas(["a", "b"], pipe)
    assert dists == [{"positive": 0.8, "negative": 0.2}] * 2


def test_respeta_las_etiquetas_de_idioma():
    pipe = PipeFalso([{"label": "es", "score": 0.6}, {"label": "en", "score": 0.4}])
    assert bt.clasificar_ventanas(["x"], pipe) == [{"es": 0.6, "en": 0.4}]


def test_acepta_pipeline_sin_top_k():
    """Si alguien lo construye sin top_k=None, devuelve un dict por ventana."""
    class PipeUnico:
        def __call__(self, ventanas):
            return [{"label": "es", "score": 0.9} for _ in ventanas]

    assert bt.clasificar_ventanas(["x", "y"], PipeUnico()) == [{"es": 0.9}] * 2


def test_clasificar_sin_ventanas():
    assert bt.clasificar_ventanas([], PipeFalso([])) == []


def test_analizar_condicion_encadena_todo():
    pipe_id = PipeFalso([{"label": "es", "score": 0.4}, {"label": "en", "score": 0.6}])
    pipe_st = PipeFalso([{"label": "positive", "score": 0.5},
                         {"label": "negative", "score": 0.5}])
    r = bt.analizar_condicion("abcde " * 2_000, pipe_idioma=pipe_id,
                              pipe_sentimiento=pipe_st, n_ventanas=6,
                              chars=200, seed=1)
    assert r["n_ventanas"] == 6
    assert r["chars_por_ventana"] == 200
    assert r["idioma_modal"] == "en"
    assert r["sent_polaridad"] == pytest.approx(0.0)


# ============================================================================
# FRAGMENTACIÓN SUBLÉXICA
# ============================================================================

class TokenizadorFalso:
    """Un token por carácter: el peor caso, que es el que da el ruido."""

    def encode(self, texto, add_special_tokens=False):
        return list(range(len(texto)))


def test_ratio_de_tokens_por_caracter():
    vs = ["abcd", "efgh"]
    r = bt.ratio_tokens_por_char(vs, TokenizadorFalso())
    assert r["tokens_por_char"] == pytest.approx(1.0)
    assert r["tokens_media"] == pytest.approx(4.0)
    assert r["supera_512"] is False


def test_avisa_si_las_ventanas_se_truncan():
    """Si se supera el contexto de 512, la comparación deja de ser justa."""
    r = bt.ratio_tokens_por_char(["x" * 600], TokenizadorFalso())
    assert r["supera_512"] is True


def test_ratio_sin_ventanas():
    assert bt.ratio_tokens_por_char([], TokenizadorFalso()) == {}


# ============================================================================
# FASTTEXT — degradación limpia
# ============================================================================

def test_fasttext_devuelve_none_si_falta_el_modelo(tmp_path):
    assert bt.cargar_fasttext(tmp_path / "no_existe.bin") is None


def test_deteccion_fasttext_sin_modelo_no_revienta():
    assert bt.detectar_idioma_fasttext(["hola"], None) == []


def test_descargar_fasttext_no_redescarga(tmp_path):
    """131 MB: si ya está en caché, ni se toca la red."""
    ya_existe = tmp_path / "lid.176.bin"
    ya_existe.write_bytes(b"contenido falso")
    assert bt.descargar_fasttext(ya_existe) == ya_existe
    assert ya_existe.read_bytes() == b"contenido falso"


def test_la_cache_de_fasttext_vive_fuera_del_repo():
    """Un binario de 131 MB no debe acabar en git."""
    assert ".cache" in bt.RUTA_FASTTEXT_DEFECTO.parts
    assert bt.RUTA_FASTTEXT_DEFECTO.name == "lid.176.bin"


# ============================================================================
# CONFIGURACIÓN DEL BLOQUE
# ============================================================================

def test_la_ventana_deja_margen_bajo_el_contexto_de_512():
    """
    Medido con XLM-R sobre el corpus real: la condición más fragmentada (ruido)
    va a ~0,60 tokens/char. La ventana por defecto tiene que quedar por debajo
    de 512 tokens con margen, o las comparaciones se truncan de forma desigual.
    """
    peor_ratio = 0.62
    assert bt.CHARS_POR_VENTANA * peor_ratio < 512
