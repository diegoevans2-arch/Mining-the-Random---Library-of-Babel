"""El null model tiene que ser un gemelo estadístico: alfabeto Y formato."""
from collections import Counter

import pytest

import babel_core as bc

from conftest import N_TEST


def test_longitud_exacta():
    for n in (1_000, 50_000, 123_457):
        assert len(bc.generar_babel_sintetico(n, seed=1)) == n


def test_usa_solo_el_alfabeto_canonico():
    t = bc.generar_babel_sintetico(50_000, seed=1)
    assert set(t) <= set(bc.ALFABETO_CONTENIDO) | {"\n"}


def test_reproducible_por_semilla():
    assert bc.generar_babel_sintetico(10_000, seed=7) == \
           bc.generar_babel_sintetico(10_000, seed=7)
    assert bc.generar_babel_sintetico(10_000, seed=7) != \
           bc.generar_babel_sintetico(10_000, seed=8)


def test_sin_paginar_no_hay_lineas_en_blanco():
    t = bc.generar_babel_sintetico(50_000, lineas_por_pagina=None, seed=1)
    assert "\n\n" not in t


def test_paginado_inserta_lineas_en_blanco():
    t = bc.generar_babel_sintetico(50_000, ancho_linea=80,
                                   lineas_por_pagina=40, seed=1)
    vacias = sum(1 for ln in t.split("\n") if not ln)
    esperadas = len([ln for ln in t.split("\n") if ln]) // 40
    assert vacias == pytest.approx(esperadas, abs=1)


def test_la_paginacion_cambia_la_frecuencia_de_salto():
    """La razón de ser de §5.9: 1/79,06 contra 1/81."""
    kw = dict(n_chars=200_000, ancho_linea=80, seed=1)
    con = bc.generar_babel_sintetico(lineas_por_pagina=40, **kw)
    sin = bc.generar_babel_sintetico(lineas_por_pagina=None, **kw)
    f_con = con.count("\n") / len(con)
    f_sin = sin.count("\n") / len(sin)
    assert f_sin == pytest.approx(1 / 81, rel=0.01)
    assert f_con > f_sin
    assert 1 / f_con == pytest.approx(79.06, abs=0.3)


def test_null_uniforme_es_uniforme():
    t = bc.generar_babel_sintetico(300_000, probs_contenido=None,
                                   lineas_por_pagina=None, seed=3)
    c = Counter(ch for ch in t if ch != "\n")
    frecs = [n / sum(c.values()) for n in c.values()]
    assert len(c) == 29
    assert max(frecs) - min(frecs) < 0.004      # ~1/29 = 0.0345


def test_calibrado_respeta_las_probabilidades():
    probs = {ch: 0.0 for ch in bc.ALFABETO_CONTENIDO}
    probs["a"], probs["b"] = 0.75, 0.25
    t = bc.generar_babel_sintetico(20_000, probs_contenido=probs,
                                   lineas_por_pagina=None, seed=5)
    c = Counter(ch for ch in t if ch != "\n")
    assert set(c) == {"a", "b"}
    assert c["a"] / (c["a"] + c["b"]) == pytest.approx(0.75, abs=0.02)


def test_replicas_son_distintas_entre_si():
    reps = bc.generar_replicas_null(5, 5_000, seed_base=100)
    assert len({r.texto for r in reps}) == 5
    assert all(r.condicion == "null" for r in reps)


def test_replicas_propagan_el_formato():
    reps = bc.generar_replicas_null(3, 50_000, ancho_linea=80,
                                    lineas_por_pagina=40, seed_base=1)
    for r in reps:
        assert "\n\n" in r.texto


# --- Contra el corpus real --------------------------------------------------

def test_null_replica_el_formato_de_babel(texto_babel, texto_null):
    """Regresión §6/E.5 #2. Es lo que sacó la entropía del percentil 100."""
    assert len(texto_null) == len(texto_babel) == N_TEST
    assert texto_null.count("\n") == texto_babel.count("\n")

    vacias = lambda t: sum(1 for ln in t.split("\n") if not ln)
    assert vacias(texto_null) == vacias(texto_babel)

    f_real = bc.perfil_formato(texto_babel)
    f_null = bc.perfil_formato(texto_null)
    assert f_null["ancho_linea"] == f_real["ancho_linea"]
    assert f_null["n_lineas_contenido"] == f_real["n_lineas_contenido"]
    assert f_null["freq_salto"] == pytest.approx(f_real["freq_salto"])


def test_entropia_del_null_iguala_a_babel(texto_babel, texto_null):
    """Con el formato replicado deben coincidir; sin él diferían en 4e-4 bits."""
    h_real = bc.entropia_shannon(texto_babel, 1)
    h_null = bc.entropia_shannon(texto_null, 1)
    assert h_null == pytest.approx(h_real, abs=1e-3)


def test_perfil_formato_detecta_410_paginas(formato):
    assert formato["ancho_linea"] == 80
    assert formato["lineas_por_pagina"] == 40
    assert formato["n_paginas"] == 410
    assert formato["n_lineas_contenido"] == 16_400
    assert formato["n_lineas_vacias"] == 409
