"""
Fixtures compartidas de la suite.

Localiza `src/` y los .txt de Babel con la misma lógica que el notebook, para
que los tests corran igual desde la raíz del repo, desde Babel_V2/ o desde
tests/. Si el corpus no está disponible, los tests que lo necesitan se saltan en
vez de fallar: la suite tiene que poder correr en un clone sin datos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]      # Babel_V2/
RUTA_SRC = RAIZ_PROYECTO / "src"
sys.path.insert(0, str(RUTA_SRC))

import babel_core as bc          # noqa: E402
import babel_pipeline as bp      # noqa: E402

# Tamaño de trabajo para los tests que necesitan texto realista pero no el
# corpus entero. Suficiente para que el formato de página se manifieste
# (410 páginas × 40 líneas × 80 chars ≈ 3.281 chars por página).
N_TEST = 200_000


def _localizar_datos() -> Path | None:
    """Misma prioridad que el notebook: data/ del proyecto -> data/ de la raíz
    del repo -> carpeta padre -> proyecto. Los 10 libros viven en <repo>/data/."""
    candidatos = (
        RAIZ_PROYECTO / "data",
        RAIZ_PROYECTO.parent / "data",
        RAIZ_PROYECTO.parent,
        RAIZ_PROYECTO,
    )
    for d in candidatos:
        if d.exists() and any(d.glob("*.txt")):
            return d
    return None


@pytest.fixture(scope="session")
def ruta_datos() -> Path:
    d = _localizar_datos()
    if d is None:
        pytest.skip("No se encontraron los .txt de Babel")
    return d


@pytest.fixture(scope="session")
def corpus(ruta_datos) -> list:
    """Los libros de Babel, parseados (sin título ni footer)."""
    return bc.cargar_corpus_babel(ruta_datos)


@pytest.fixture(scope="session")
def corpus_crudo(ruta_datos) -> list:
    """Los mismos libros SIN parsear, para contrastar el efecto del parser."""
    return bc.cargar_corpus_babel(ruta_datos, parsear=False)


@pytest.fixture(scope="session")
def libro(corpus):
    return corpus[0]


@pytest.fixture(scope="session")
def probs(corpus) -> dict:
    return bc.estimar_prob_contenido(corpus)


@pytest.fixture(scope="session")
def formato(libro) -> dict:
    return bc.perfil_formato(libro.texto)


@pytest.fixture(scope="session")
def texto_babel(libro) -> str:
    return libro.texto[:N_TEST]


@pytest.fixture(scope="session")
def texto_null(probs, formato) -> str:
    """Null model calibrado con el alfabeto Y el formato reales."""
    return bc.generar_babel_sintetico(
        N_TEST,
        probs_contenido=probs,
        ancho_linea=formato["ancho_linea"],
        lineas_por_pagina=formato["lineas_por_pagina"],
        seed=1234,
    )


def _control(fn, nombre):
    try:
        return fn()
    except LookupError:
        pytest.skip(f"Corpus NLTK de {nombre} no descargado")


@pytest.fixture(scope="session")
def control_en():
    return _control(bc.cargar_control_ingles, "Moby Dick")


@pytest.fixture(scope="session")
def control_es():
    return _control(bc.cargar_control_espanol, "CESS-ESP")


# --- Material sintético, sin dependencia del corpus real --------------------

LIBRO_SINTETICO = (
    "titulo de prueba\n"
    "\n"
    + "\n".join("abcde fghij, klmno. pqrst uvwxy" for _ in range(12))
    + "\n\n\n"
    "Book Location:4-w1-s1-v09\n"
)


@pytest.fixture
def libro_sintetico() -> str:
    """Reproduce la forma de un .txt descargado: título, cuerpo, footer."""
    return LIBRO_SINTETICO
