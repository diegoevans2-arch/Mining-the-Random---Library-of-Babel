"""
Invariantes que cruzan módulos, y los defectos concretos que la validación del
Bloque A destapó. Cada test aquí corresponde a un artefacto que llegó a pasar
por hallazgo antes de detectarse.
"""
import pytest

import babel_core as bc
import babel_pipeline as bp


def test_los_dos_modulos_tokenizan_igual(texto_babel):
    """
    Regresión §6/E.5 #4.

    `babel_core._tokenizar_palabras` alimenta Zipf y Heaps; `babel_pipeline`
    alimenta el pipeline clásico. Llegaron a contar 111.797 y 48.961 tokens del
    MISMO texto, así que el estudio reportaba dos vocabularios distintos.
    """
    por_core = bc._tokenizar_palabras(texto_babel)
    por_pipeline = bp.tokenizar(bp.limpiar_texto(texto_babel))
    assert por_core == por_pipeline


def test_los_dos_modulos_tokenizan_igual_con_texto_sucio():
    """El caso que los separaba: puntuación, dígitos, guiones bajos, acentos."""
    sucio = "Electricité_de_France -Fpa- 51_por_ciento, año.dos--tres"
    assert bc._tokenizar_palabras(sucio) == bp.tokenizar(bp.limpiar_texto(sucio))


def test_condiciones_normalizadas_al_mismo_tamano(corpus, control_en, control_es):
    """
    Regresión §6/E.5 #5.

    `N_REF = max(...)` sobre los libros dejaba a Babel más corto que el resto, y
    los controles NLTK son MÁS cortos que un libro de Babel: el tope real lo
    ponen ellos. Entropía, compresión, Zipf y Heaps dependen todas de N.
    """
    n_ref = min(
        min(c.n_chars for c in corpus),
        control_en.n_chars,
        control_es.n_chars,
    )
    condiciones = {
        "Babel": corpus[0].texto[:n_ref],
        "Control_EN": control_en.texto[:n_ref],
        "Control_ES": control_es.texto[:n_ref],
    }
    assert len({len(t) for t in condiciones.values()}) == 1
    # El invariante que rompía max(): ningún control alcanza el tamaño de Babel
    assert control_en.n_chars < corpus[0].n_chars
    assert control_es.n_chars < corpus[0].n_chars


def test_el_null_no_es_distinguible_por_entropia(texto_babel, probs, formato):
    """
    El artefacto que puso la entropía en percentil 100: un null sin paginar
    queda sistemáticamente por debajo. Con el formato replicado, coincide.
    """
    kw = dict(n_chars=len(texto_babel), probs_contenido=probs,
              ancho_linea=formato["ancho_linea"], seed=99)
    h_real = bc.entropia_shannon(texto_babel, 1)
    h_pag = bc.entropia_shannon(
        bc.generar_babel_sintetico(lineas_por_pagina=formato["lineas_por_pagina"], **kw), 1)
    h_sin = bc.entropia_shannon(
        bc.generar_babel_sintetico(lineas_por_pagina=None, **kw), 1)

    assert abs(h_pag - h_real) < abs(h_sin - h_real)
    assert h_pag == pytest.approx(h_real, abs=1e-3)


def test_babel_y_null_no_tienen_stopwords_reales(texto_babel, texto_null):
    """
    El '9%' con el umbral estándar: se desploma en AMBOS por igual. Si Babel se
    separase del null aquí, habría algo que explicar.
    """
    sw = bp.construir_set_stopwords()
    tasas = []
    for t in (texto_babel, texto_null):
        tokens = bp.tokenizar(bp.limpiar_texto(t))
        reales = sum(1 for x in tokens if len(x) >= 3 and x in sw)
        tasas.append(100 * reales / len(tokens))
    assert all(tasa < 1.0 for tasa in tasas)
    assert abs(tasas[0] - tasas[1]) < 0.5


def test_zipf_no_separa_babel_del_lenguaje_natural(texto_babel, control_es):
    """
    Miller/Mandelbrot (§2.1). Si este test empieza a fallar porque Babel se
    aleja del español, sospechar del ajuste antes que del corpus.
    """
    z_babel = bc.ley_zipf(texto_babel, excluir_hapax=True)
    z_es = bc.ley_zipf(control_es.texto[:len(texto_babel)], excluir_hapax=True)
    assert abs(z_babel["pendiente"] - z_es["pendiente"]) < 0.25


def test_heaps_si_separa(texto_babel, control_es):
    """El discriminador que Zipf no es."""
    h_babel = bc.ley_heaps(texto_babel)
    h_es = bc.ley_heaps(control_es.texto[:len(texto_babel)])
    assert h_babel["beta"] > 0.9
    assert h_es["beta"] < 0.85
    # El separador más crudo, sin ajuste de curva de por medio
    assert h_babel["tasa_tipos"] > 0.5
    assert h_es["tasa_tipos"] < 0.3


def test_babel_y_null_comprimen_igual(texto_babel, texto_null):
    r_babel = bc.ratios_compresion(texto_babel, algoritmos=("gzip",))["gzip"]
    r_null = bc.ratios_compresion(texto_null, algoritmos=("gzip",))["gzip"]
    assert r_babel == pytest.approx(r_null, abs=0.01)


def test_lenguaje_natural_comprime_mucho_mejor(texto_babel, control_en):
    r_babel = bc.ratios_compresion(texto_babel, algoritmos=("gzip",))["gzip"]
    r_en = bc.ratios_compresion(
        control_en.texto[:len(texto_babel)], algoritmos=("gzip",))["gzip"]
    assert r_en < r_babel - 0.15
