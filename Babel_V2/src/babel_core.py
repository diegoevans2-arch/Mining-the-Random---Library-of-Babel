"""
babel_core.py
=============
Núcleo del estudio "Minando lo Aleatorio v2" — Bloque A (Fundamentos estadísticos).

Provee:
  - Carga del corpus Babel (auto-detección de archivos).
  - Generación del null model (Babel sintético) calibrado al alfabeto real.
  - Carga de corpus de control (Moby Dick, CESS-ESP) desde NLTK, offline.
  - Métricas de aleatoriedad: entropía de Shannon, ratios de compresión,
    Zipf, Heaps, chi-cuadrado de uniformidad.

Diseñado para Jupyter Notebook 7.x. Sin rutas hardcoded; usa pathlib.
"""
from __future__ import annotations

import bz2
import gzip
import lzma
import math
import random
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# zstandard: compresor de vanguardia (Meta). Opcional pero recomendado.
try:
    import zstandard as zstd
    _HAS_ZSTD = True
except ImportError:
    _HAS_ZSTD = False


# ============================================================================
# ALFABETO REAL DE LA BIBLIOTECA DE BABEL
# ----------------------------------------------------------------------------
# Calibrado empíricamente sobre las muestras reales: exactamente 30 símbolos.
# 26 letras minúsculas a-z + espacio + coma + punto + salto de línea.
# NO existen: mayúsculas, tildes, ñ, dígitos ni otros signos.
# La distribución observada es ~uniforme sobre los 29 símbolos "de contenido";
# el '\n' aparece a intervalo fijo por el formato de 80 columnas.
# ============================================================================
ALFABETO_CONTENIDO = list(string.ascii_lowercase) + [" ", ",", "."]  # 29 símbolos

# ----------------------------------------------------------------------------
# FORMATO FÍSICO DEL LIBRO (medido sobre los 10 archivos reales)
# ----------------------------------------------------------------------------
# Cada libro son 410 páginas × 40 líneas × 80 caracteres, con una LÍNEA EN
# BLANCO separando páginas:
#     16.400 líneas de contenido + 409 líneas en blanco = 16.809 líneas
#     80×16.400 + 16.808 saltos = 1.328.808 chars   <- coincide exacto
# Las 410 páginas son el número que Borges fija en el cuento; el sitio lo
# implementa literalmente.
#
# Esto IMPORTA para el null model: ignorar las líneas en blanco deja la
# frecuencia de '\n' en 1/81 cuando la real es 1/79,06, y eso solo basta para
# que la entropía de Shannon separe Babel del null (H1 4,8944 vs 4,8940) por
# un motivo que no tiene nada que ver con aleatoriedad.
ANCHO_LINEA_BABEL = 80        # caracteres por línea antes del '\n'
LINEAS_POR_PAGINA_BABEL = 40  # líneas por página; las páginas se separan con ''

# Algoritmos de compresión disponibles (ver ratios_compresion)
ALGORITMOS_COMPRESION = ("gzip", "bz2", "lzma", "zstd")

# Una línea pertenece al CUERPO de Babel si solo usa el alfabeto de contenido.
_RE_LINEA_CONTENIDO = re.compile(r"^[a-z ,.]*$")


@dataclass
class Corpus:
    """Contenedor uniforme para cualquier texto bajo análisis."""
    nombre: str
    condicion: str  # 'babel' | 'null' | 'control_es' | 'control_en'
    texto: str
    meta: dict = field(default_factory=dict)

    @property
    def n_chars(self) -> int:
        return len(self.texto)


# ----------------------------------------------------------------------------
# 1. CARGA DEL CORPUS BABEL
# ----------------------------------------------------------------------------
def parsear_libro_babel(texto: str, nombre_archivo: str | None = None) -> tuple[str, dict]:
    """
    Separa el CUERPO aleatorio de la metadata que añade libraryofbabel.info.

    Formato observado en los .txt descargados:

        línea 0          : título del libro (mismo string que el nombre de archivo)
        línea 1          : vacía
        líneas 2..n-4    : cuerpo, 80 caracteres por línea, alfabeto [a-z ,.]
        líneas n-3, n-2  : vacías
        última línea     : 'Book Location:4-w1-s1-v09'

    El título y el footer introducen mayúsculas, dígitos, ':' y '-' — símbolos
    que NO pertenecen al alfabeto de 30 de Babel. Si entran al análisis
    contaminan el perfil del alfabeto, la calibración del null model y
    cualquier detección de near-duplicates (el footer es idéntico en los 10
    libros, así que MinHash lo reportaría como coincidencia real).

    Lo descartado queda registrado en `meta` para poder auditarlo.

    Returns
    -------
    (cuerpo, meta) con meta = {'titulo', 'location', 'lineas_descartadas'}
    """
    lineas = texto.replace("\r\n", "\n").split("\n")
    meta: dict = {"titulo": None, "location": None, "lineas_descartadas": []}

    # --- 1. Título: primera línea no vacía ---------------------------------
    # No se distingue por charset (el título también es [a-z ,.]), así que se
    # identifica estructuralmente: coincide con el nombre del archivo, o bien
    # es más corta que el ancho de línea y va seguida de una línea vacía.
    i = 0
    while i < len(lineas) and not lineas[i].strip():
        i += 1
    if i < len(lineas):
        candidato = lineas[i]
        stem = Path(nombre_archivo).stem if nombre_archivo else None
        es_titulo = (stem is not None and candidato.strip() == stem.strip()) or (
            len(candidato) < ANCHO_LINEA_BABEL
            and i + 1 < len(lineas)
            and not lineas[i + 1].strip()
        )
        if es_titulo:
            meta["titulo"] = candidato
            lineas = lineas[i + 1 :]

    # --- 2. Cualquier línea fuera del alfabeto de Babel es metadata --------
    cuerpo = []
    for ln in lineas:
        if _RE_LINEA_CONTENIDO.match(ln):
            cuerpo.append(ln)
        else:
            meta["lineas_descartadas"].append(ln)
            if ln.strip().lower().startswith("book location"):
                meta["location"] = ln.strip()

    # --- 3. Recortar líneas vacías de los extremos -------------------------
    while cuerpo and not cuerpo[0]:
        cuerpo.pop(0)
    while cuerpo and not cuerpo[-1]:
        cuerpo.pop()

    return "\n".join(cuerpo), meta


def cargar_corpus_babel(
    ruta_datos: str | Path,
    patron: str = "*.txt",
    parsear: bool = True,
) -> list[Corpus]:
    """
    Carga todos los .txt de una carpeta como corpus Babel.
    Auto-detecta cuántos archivos hay (funciona con 2 muestras o 10 libros).

    Parameters
    ----------
    ruta_datos : ruta a la carpeta con los .txt de Babel.
    patron : glob para filtrar archivos.
    parsear : si True (por defecto), retira título y footer 'Book Location'
        vía parsear_libro_babel. Pasar False para trabajar con el crudo.
    """
    ruta = Path(ruta_datos)
    archivos = sorted(ruta.glob(patron))
    if not archivos:
        raise FileNotFoundError(
            f"No se encontraron archivos '{patron}' en {ruta.resolve()}"
        )

    corpus = []
    for i, fpath in enumerate(archivos, 1):
        crudo = fpath.read_text(encoding="utf-8")
        if parsear:
            texto, meta_parse = parsear_libro_babel(crudo, nombre_archivo=fpath.name)
        else:
            texto, meta_parse = crudo, {}
        corpus.append(
            Corpus(
                nombre=f"Babel_{i:02d}",
                condicion="babel",
                texto=texto,
                meta={
                    "archivo": fpath.name,
                    "ruta": str(fpath),
                    "n_chars_crudo": len(crudo),
                    **meta_parse,
                },
            )
        )
    return corpus


# ----------------------------------------------------------------------------
# 2. PERFIL DE ALFABETO (calibración del null model)
# ----------------------------------------------------------------------------
def perfil_alfabeto(texto: str) -> dict:
    """Devuelve inventario, conteos y probabilidades empíricas de caracteres."""
    c = Counter(texto)
    total = sum(c.values())
    return {
        "inventario": sorted(c.keys()),
        "n_simbolos": len(c),
        "conteos": dict(c),
        "probabilidades": {ch: n / total for ch, n in c.items()},
    }


def perfil_formato(texto: str) -> dict:
    """
    Mide el formato FÍSICO observado: ancho de línea, líneas por página y
    frecuencia de salto de línea.

    Sirve para calibrar el null model con la disposición real en lugar de
    asumirla. El '\\n' no es un símbolo aleatorio más: aparece en posiciones
    deterministas y su frecuencia entra directo en la entropía, así que un
    gemelo estadístico legítimo tiene que reproducirla.
    """
    lineas = texto.split("\n")
    largos = Counter(len(l) for l in lineas if l)
    if not largos:
        return {
            "ancho_linea": ANCHO_LINEA_BABEL,
            "lineas_por_pagina": None,
            "n_lineas_contenido": 0,
            "n_lineas_vacias": 0,
            "n_paginas": 0,
            "freq_salto": 0.0,
        }

    n_contenido = sum(largos.values())
    n_vacias = sum(1 for l in lineas if not l)
    n_paginas = n_vacias + 1
    return {
        "ancho_linea": largos.most_common(1)[0][0],
        "lineas_por_pagina": round(n_contenido / n_paginas) if n_vacias else None,
        "n_lineas_contenido": n_contenido,
        "n_lineas_vacias": n_vacias,
        "n_paginas": n_paginas,
        "freq_salto": texto.count("\n") / len(texto) if texto else 0.0,
        "distribucion_largos": dict(largos.most_common(5)),
    }


def estimar_prob_contenido(corpus_babel: list[Corpus]) -> dict[str, float]:
    """
    Estima la probabilidad de cada símbolo de contenido (excluyendo '\\n')
    agregando todos los libros de Babel. Sirve para calibrar el null model.
    Si un símbolo del alfabeto canónico no aparece, recibe prob ~0 suavizada.
    """
    c = Counter()
    for corp in corpus_babel:
        c.update(ch for ch in corp.texto if ch != "\n")

    total = sum(c.values())
    probs = {}
    for ch in ALFABETO_CONTENIDO:
        probs[ch] = c.get(ch, 0) / total if total else 0.0

    # Renormalizar por si hubiera símbolos fuera del alfabeto canónico
    s = sum(probs.values())
    if s > 0:
        probs = {ch: p / s for ch, p in probs.items()}
    return probs


# ----------------------------------------------------------------------------
# 3. NULL MODEL — generación de Babel sintético
# ----------------------------------------------------------------------------
def generar_babel_sintetico(
    n_chars: int,
    probs_contenido: dict[str, float] | None = None,
    ancho_linea: int = ANCHO_LINEA_BABEL,
    lineas_por_pagina: int | None = LINEAS_POR_PAGINA_BABEL,
    seed: int | None = None,
) -> str:
    """
    Genera un texto que sigue EXACTAMENTE la misma ley generadora que Babel:
    muestreo i.i.d. sobre el alfabeto de contenido, dispuesto en líneas de
    `ancho_linea` caracteres agrupadas en páginas de `lineas_por_pagina`
    líneas separadas por una línea en blanco.

    Si probs_contenido es None, usa distribución uniforme sobre los 29 símbolos
    (la hipótesis nula más pura: aleatoriedad perfecta).

    Parameters
    ----------
    n_chars : longitud objetivo del texto generado.
    probs_contenido : probabilidades por símbolo; None => uniforme.
    ancho_linea : cada cuántos chars de contenido se inserta '\\n'.
    lineas_por_pagina : líneas por página; se inserta una línea en blanco entre
        páginas. None => sin paginación (comportamiento de la primera versión).

        Reproducir esto NO es cosmético. Los libros reales traen 409 líneas en
        blanco además de las 16.400 de contenido, así que la frecuencia real de
        '\\n' es 1/79,06 y no 1/81. Sin paginar, el null model quedaba con una
        entropía sistemáticamente menor (4,8940 vs 4,8944) y Babel salía
        "distinguible" en el veredicto Monte Carlo por un motivo puramente de
        formato.
    seed : semilla para reproducibilidad.
    """
    rng = np.random.default_rng(seed)

    if probs_contenido is None:
        simbolos = ALFABETO_CONTENIDO
        pesos = None  # uniforme
    else:
        simbolos = list(probs_contenido.keys())
        pesos = np.array(list(probs_contenido.values()), dtype=float)
        pesos = pesos / pesos.sum()

    # Chars totales por línea de contenido, contando el overhead de saltos:
    #   total(L) = ancho*L + (L - n_pag) + 2*(n_pag - 1) = L*(ancho + 1 + 1/P) - 2
    # Se genera con margen de una página y se recorta al final a n_chars.
    extra_pagina = (1 / lineas_por_pagina) if lineas_por_pagina else 0
    por_linea = ancho_linea + 1 + extra_pagina
    n_lineas = int(n_chars / por_linea) + (lineas_por_pagina or 1) + 2
    n_contenido = n_lineas * ancho_linea

    idx = rng.choice(len(simbolos), size=n_contenido, p=pesos)
    contenido = np.array(simbolos)[idx]

    # Cortar en líneas de `ancho_linea` caracteres
    lineas = [
        "".join(contenido[i : i + ancho_linea])
        for i in range(0, n_contenido, ancho_linea)
    ]

    if lineas_por_pagina:
        # Agrupar en páginas y separarlas con una línea en blanco
        paginas = [
            "\n".join(lineas[i : i + lineas_por_pagina])
            for i in range(0, len(lineas), lineas_por_pagina)
        ]
        texto = "\n\n".join(paginas)
    else:
        texto = "\n".join(lineas)

    return texto[:n_chars]


def generar_replicas_null(
    n_replicas: int,
    n_chars: int,
    probs_contenido: dict[str, float] | None = None,
    ancho_linea: int = ANCHO_LINEA_BABEL,
    lineas_por_pagina: int | None = LINEAS_POR_PAGINA_BABEL,
    seed_base: int = 42,
) -> list[Corpus]:
    """Genera N réplicas del null model, cada una con semilla distinta."""
    replicas = []
    for r in range(n_replicas):
        texto = generar_babel_sintetico(
            n_chars=n_chars,
            probs_contenido=probs_contenido,
            ancho_linea=ancho_linea,
            lineas_por_pagina=lineas_por_pagina,
            seed=seed_base + r,
        )
        replicas.append(
            Corpus(
                nombre=f"Null_{r:03d}",
                condicion="null",
                texto=texto,
                meta={"seed": seed_base + r},
            )
        )
    return replicas


# ----------------------------------------------------------------------------
# 4. CORPUS DE CONTROL (offline, vía NLTK)
# ----------------------------------------------------------------------------
def cargar_control_ingles(n_chars: int | None = None) -> Corpus:
    """Moby Dick (Melville, 1851). Dominio público. ~1.24M chars."""
    from nltk.corpus import gutenberg

    texto = gutenberg.raw("melville-moby_dick.txt")
    if n_chars:
        texto = texto[:n_chars]
    return Corpus(
        nombre="Control_EN_MobyDick",
        condicion="control_en",
        texto=texto,
        meta={"fuente": "NLTK gutenberg", "obra": "Moby Dick", "autor": "Melville"},
    )


def cargar_control_espanol(n_chars: int | None = None) -> Corpus:
    """Corpus CESS-ESP (español anotado). Estándar académico. ~1.1M chars."""
    from nltk.corpus import cess_esp

    # cess_esp viene tokenizado; reconstruimos texto plano
    texto = " ".join(cess_esp.words())
    if n_chars:
        texto = texto[:n_chars]
    return Corpus(
        nombre="Control_ES_CESS",
        condicion="control_es",
        texto=texto,
        meta={"fuente": "NLTK cess_esp", "tipo": "corpus anotado español"},
    )


# ============================================================================
# MÉTRICAS DE ALEATORIEDAD
# ============================================================================

# ----------------------------------------------------------------------------
# 5a. ENTROPÍA DE SHANNON
# ----------------------------------------------------------------------------
def entropia_shannon(texto: str, n: int = 1) -> float:
    """
    Entropía de Shannon (bits/símbolo) sobre n-gramas de caracteres.

    n=1 : entropía de caracteres individuales.
    n=2 : entropía de bigramas.
    n=3 : entropía de trigramas.

    Máximo teórico para alfabeto de tamaño A: log2(A^n) = n*log2(A).
    Texto aleatorio uniforme se acerca al máximo; texto natural queda por debajo.
    """
    if n == 1:
        unidades = texto
    else:
        unidades = [texto[i : i + n] for i in range(len(texto) - n + 1)]

    c = Counter(unidades)
    total = sum(c.values())
    if total == 0:
        return 0.0

    H = -sum((cnt / total) * math.log2(cnt / total) for cnt in c.values())
    return H


def entropia_maxima(n_simbolos: int, n: int = 1) -> float:
    """Entropía máxima teórica para un alfabeto dado y orden de n-grama."""
    return n * math.log2(n_simbolos)


# ----------------------------------------------------------------------------
# 5b. RATIOS DE COMPRESIÓN
# ----------------------------------------------------------------------------
def ratios_compresion(
    texto: str,
    algoritmos: tuple[str, ...] | list[str] | None = None,
) -> dict[str, float]:
    """
    Ratio = tamaño_comprimido / tamaño_original (en bytes).

    Interpretación:
      - Cercano a 1.0  => incompresible => alta aleatoriedad.
      - Bajo (0.3-0.5) => muy compresible => estructura/redundancia (texto natural).

    Usa 4 algoritmos: gzip, bz2, lzma (stdlib) + zstandard (Meta, vanguardia).

    Parameters
    ----------
    algoritmos : subconjunto de ALGORITMOS_COMPRESION; None = todos.
        IMPORTANTE en bucles Monte Carlo: pedir solo ('gzip',) si es lo único
        que se va a usar. lzma preset=9 sobre 1.3M chars incompresibles cuesta
        varios segundos por llamada, y calcular los 4 para descartar 3
        multiplica por ~5 el tiempo total de las N réplicas.
    """
    pedidos = tuple(algoritmos) if algoritmos else ALGORITMOS_COMPRESION

    raw = texto.encode("utf-8")
    n0 = len(raw)
    if n0 == 0:
        return {}

    out = {}
    if "gzip" in pedidos:
        out["gzip"] = len(gzip.compress(raw, compresslevel=9)) / n0
    if "bz2" in pedidos:
        out["bz2"] = len(bz2.compress(raw, compresslevel=9)) / n0
    if "lzma" in pedidos:
        out["lzma"] = len(lzma.compress(raw, preset=9)) / n0
    if "zstd" in pedidos and _HAS_ZSTD:
        cctx = zstd.ZstdCompressor(level=19)
        out["zstd"] = len(cctx.compress(raw)) / n0
    return out


# ----------------------------------------------------------------------------
# 5c. LEY DE ZIPF
# ----------------------------------------------------------------------------
def _tokenizar_palabras(texto: str) -> list[str]:
    """
    Tokenización por palabras: secuencias máximas de letras a-z.

    Debe coincidir con babel_pipeline.limpiar_texto + tokenizar, o el estudio
    reporta dos recuentos distintos del mismo texto. Antes usaba `\\b\\w+\\b`,
    donde `\\w` incluye dígitos, guion bajo y letras acentuadas: sobre Babel
    daba lo mismo (no tiene ninguno de los tres), pero en CESS-ESP mantenía
    unidos los multiword del corpus (`Electricité_de_France`, `51_por_ciento`)
    como un único tipo, inflando el vocabulario del control español y por tanto
    su Heaps β y su pendiente de Zipf.
    """
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    return re.findall(r"[a-z]+", texto)


def ley_zipf(
    texto: str,
    min_rango: int = 1,
    max_rango: int | None = None,
    excluir_hapax: bool = False,
) -> dict:
    """
    Ajusta la ley de Zipf: log(freq) = c - s*log(rango).

    El exponente 's' (pendiente) caracteriza la distribución:
      - Texto natural: s ≈ 1.0 (ley de Zipf clásica).
      - Texto aleatorio: s significativamente distinto / peor ajuste.

    Devuelve pendiente, intercepto, R² y datos para graficar.

    Parameters
    ----------
    excluir_hapax : si True, ajusta solo sobre los rangos con frecuencia > 1.
        En texto aleatorio casi todo el vocabulario son hapax (frecuencia 1):
        esa cola es una recta HORIZONTAL en log-log que domina el ajuste por
        peso numérico y produce pendiente y R² sin significado (R²≈0.4 tanto
        en Babel como en el null model). Excluirla es lo que hace que la
        pendiente sea comparable entre condiciones. Por defecto False para no
        alterar en silencio los resultados ya publicados del Bloque A.
    """
    tokens = _tokenizar_palabras(texto)
    if not tokens:
        return {"pendiente": np.nan, "r2": np.nan, "rangos": [], "frecuencias": []}

    frec = Counter(tokens)
    frec_ordenadas = np.array(sorted(frec.values(), reverse=True))
    rangos = np.arange(1, len(frec_ordenadas) + 1)

    # Recorte de rango para el ajuste (evitar cola ruidosa de hapax)
    hi = max_rango or len(rangos)
    if excluir_hapax:
        n_no_hapax = int((frec_ordenadas > 1).sum())
        hi = min(hi, n_no_hapax)
    mask = (rangos >= min_rango) & (rangos <= hi)
    log_r = np.log10(rangos[mask])
    log_f = np.log10(frec_ordenadas[mask])

    # Ajuste lineal en espacio log-log
    if len(log_r) < 2:
        return {"pendiente": np.nan, "r2": np.nan, "rangos": rangos, "frecuencias": frec_ordenadas}

    coef = np.polyfit(log_r, log_f, 1)
    pendiente, intercepto = coef[0], coef[1]

    # R²
    pred = np.polyval(coef, log_r)
    ss_res = np.sum((log_f - pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "pendiente": pendiente,         # s (idealmente ~ -1)
        "intercepto": intercepto,
        "r2": r2,
        "rangos": rangos,
        "frecuencias": frec_ordenadas,
        "n_tipos": len(frec),
        "n_tokens": len(tokens),
        "n_rangos_ajuste": int(mask.sum()),  # sobre cuántos rangos se ajustó
    }


# ----------------------------------------------------------------------------
# 5d. LEY DE HEAPS
# ----------------------------------------------------------------------------
def ley_heaps(texto: str, n_puntos: int = 50, n_min: int | None = None) -> dict:
    """
    Ley de Heaps: V(N) = K * N^beta, donde V = vocabulario, N = tokens.

      - Texto natural: beta ≈ 0.4-0.6 (vocabulario crece sublinealmente).
      - Texto aleatorio: beta ≈ 1.0 (casi cada token nuevo es palabra nueva).

    El exponente beta es uno de los discriminadores MÁS limpios entre
    aleatoriedad y lenguaje real para este proyecto.

    Parameters
    ----------
    n_min : ajustar solo con los puntos de N >= n_min. None = todos (desde N=10).

        El beta ≈ 0.4-0.6 del lenguaje natural es un ASINTÓTICO. En el arranque
        de la curva (N pequeño) todo texto crece casi linealmente, también el
        natural, así que incluir ese tramo empuja beta hacia arriba: con los
        controles a 1.1M chars da 0.76 (EN) y 0.80 (ES) desde N=10, pero 0.55 y
        0.76 desde N=50.000.

        Lo informativo no es el número suelto sino cómo se comporta al mover el
        corte: en Babel beta es PLANO (~0.97 en todo rango, el vocabulario crece
        linealmente de verdad), mientras que en lenguaje natural DECRECE al
        excluir el arranque, que es la firma de la saturación real del léxico.
    """
    tokens = _tokenizar_palabras(texto)
    N = len(tokens)
    if N < 10:
        return {"beta": np.nan, "K": np.nan, "Ns": [], "Vs": []}

    # Muestrear el crecimiento de vocabulario en n_puntos a lo largo del texto
    posiciones = np.unique(np.logspace(1, np.log10(N), n_puntos).astype(int))
    vistos = set()
    Vs, Ns = [], []
    ptr = 0
    for pos in posiciones:
        while ptr < pos:
            vistos.add(tokens[ptr])
            ptr += 1
        Ns.append(pos)
        Vs.append(len(vistos))

    Ns, Vs = np.array(Ns), np.array(Vs)

    # Selección del tramo de ajuste
    mask = Ns >= n_min if n_min else np.ones(len(Ns), dtype=bool)
    if mask.sum() < 2:
        return {"beta": np.nan, "K": np.nan, "r2": np.nan, "Ns": Ns, "Vs": Vs,
                "n_puntos_ajuste": int(mask.sum())}

    log_N = np.log10(Ns[mask])
    log_V = np.log10(Vs[mask])
    coef = np.polyfit(log_N, log_V, 1)
    beta, log_K = coef[0], coef[1]

    pred = np.polyval(coef, log_N)
    ss_res = np.sum((log_V - pred) ** 2)
    ss_tot = np.sum((log_V - log_V.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "beta": beta,              # exponente de Heaps (~0.5 natural, ~1.0 aleatorio)
        "K": 10 ** log_K,
        "r2": r2,
        "Ns": Ns,
        "Vs": Vs,
        "n_puntos_ajuste": int(mask.sum()),
        "tasa_tipos": float(Vs[-1] / Ns[-1]),  # V/N final: 0.77 Babel vs 0.09 inglés
    }


# ----------------------------------------------------------------------------
# 5e. CHI-CUADRADO DE UNIFORMIDAD
# ----------------------------------------------------------------------------
def chi2_uniformidad(
    texto: str,
    n: int = 1,
    excluir: tuple = ("\n",),
    alfabeto: str | list[str] | None = None,
) -> dict:
    """
    Test chi-cuadrado de bondad de ajuste contra distribución UNIFORME.

    H0: los n-gramas de caracteres se distribuyen uniformemente.
      - Texto aleatorio: NO rechaza H0 (p alto) a nivel de caracteres.
      - Texto natural: rechaza H0 fuertemente (p≈0).

    CORRECCIÓN respecto a la primera versión: las categorías con conteo CERO
    deben entrar en el test. Antes el vector de observados se construía con
    `Counter(...).values()`, es decir solo los n-gramas presentes, y el
    esperado se calculaba sobre ese k reducido. Eso descarta en silencio las
    categorías ausentes y sesga el test HACIA "uniforme" — justo la conclusión
    que el estudio quiere poder defender. Ahora el espacio de categorías se
    construye completo como A^n.

    Parameters
    ----------
    alfabeto : símbolos que definen el espacio de categorías. None => se
        infiere de los símbolos observados en el texto.

    Devuelve estadístico, p-valor y grados de libertad.
    """
    from itertools import product

    from scipy.stats import chisquare

    t = "".join(ch for ch in texto if ch not in excluir)
    if not t:
        return {"chi2": np.nan, "p_valor": np.nan, "gl": 0, "n_categorias": 0}

    simbolos = sorted(set(alfabeto)) if alfabeto is not None else sorted(set(t))
    if n == 1:
        unidades = list(t)
    else:
        unidades = [t[i : i + n] for i in range(len(t) - n + 1)]

    c = Counter(unidades)
    # Espacio COMPLETO de categorías (A^n), incluidas las de conteo 0
    categorias = ["".join(p) for p in product(simbolos, repeat=n)]
    observado = np.array([c.get(cat, 0) for cat in categorias], dtype=float)
    k = len(observado)
    if k < 2:
        return {"chi2": np.nan, "p_valor": np.nan, "gl": 0, "n_categorias": k}

    esperado = np.full(k, observado.sum() / k)
    chi2, p = chisquare(f_obs=observado, f_exp=esperado)

    return {
        "chi2": float(chi2),
        "p_valor": float(p),
        "gl": k - 1,
        "n_categorias": k,
        "n_categorias_vacias": int((observado == 0).sum()),
        "frec_esperada": float(esperado[0]),  # <5 => test poco fiable
    }


if __name__ == "__main__":
    # Smoke test rápido
    print("babel_core.py — smoke test")
    print(f"  Alfabeto de contenido: {len(ALFABETO_CONTENIDO)} símbolos")
    print(f"  zstandard disponible: {_HAS_ZSTD}")
    sint = generar_babel_sintetico(500, seed=1)
    print(f"  Muestra sintética (200 chars):\n{sint[:200]}")
    print(f"  Entropía H1: {entropia_shannon(sint, 1):.3f} bits/símbolo")
    print(f"  Ratios compresión: {ratios_compresion(sint)}")

    # Parser de metadata: reproduce el formato real de libraryofbabel.info
    crudo = "abc def\n\n" + sint + "\n\n\nBook Location:4-w1-s1-v09\n"
    cuerpo, meta = parsear_libro_babel(crudo, nombre_archivo="abc def.txt")
    print("\n  parsear_libro_babel:")
    print(f"    título detectado   : {meta['titulo']!r}")
    print(f"    location detectada : {meta['location']!r}")
    print(f"    chars crudo/cuerpo : {len(crudo)} -> {len(cuerpo)}")
    fuera = sorted(set(cuerpo) - set(ALFABETO_CONTENIDO) - {"\n"})
    print(f"    símbolos fuera del alfabeto en el cuerpo: {fuera or 'ninguno'}")
