"""
babel_viz.py
============
Capa de estilo y figuras del estudio.

Dos cometidos:

1. **Un sistema visual único.** Paleta, tipografía y cromo de ejes centralizados,
   para que las diez figuras del notebook se lean como una sola familia en vez de
   como diez decisiones sueltas de matplotlib.

2. **Recuperar las figuras del v1.** El estudio original produjo dos gráficos que
   el v2 había dejado atrás: la nube de palabras menos frecuentes y el mapa de
   correlación entre documentos. Se reconstruyen aquí — pero **con el null model
   dentro**, que es lo que convierte una ilustración en un argumento.

Sobre la paleta
---------------
Se usa la paleta de referencia del sistema de visualización, sin modificar y en
su orden de slots. Ese orden es el mecanismo de seguridad para daltonismo, no una
elección estética: sus resultados están declarados en la referencia (peor par
adyacente CVD ΔE 9,1 en modo claro; los tres primeros slots validan además
todos-los-pares, que es lo que exige un scatter).

Reglas que se respetan aquí y conviene no romper al añadir figuras:

- **El color sigue a la entidad.** Babel es azul en las diez figuras. Filtrar o
  reordenar no repinta a los supervivientes.
- **Nunca doble eje.** Dos magnitudes de escala distinta van en dos paneles.
- **Secuencial = un solo tono, claro→oscuro.** Nada de arcoíris para magnitud.
- **Rejilla y ejes en línea sólida de un pelo**, un tono por encima del fondo.
  Nunca discontinua: el trazo discontinuo se lee como "umbral" o "proyección".
- **Leyenda siempre con dos o más series**, y etiquetas directas solo en lo que
  importa — nunca un número sobre cada punto.
"""
from __future__ import annotations

import math

import numpy as np

# ============================================================================
# PALETA
# ============================================================================
# Slots categóricos, en orden. No cambiar el orden ni añadir un noveno color:
# más allá del octavo hay que plegar a "Otros" o usar múltiplos pequeños.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
         "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Asignación fija entidad -> slot. Estable en todo el estudio.
COLOR_CONDICION = {
    "Babel": SLOTS[0],        # azul
    "Null": SLOTS[1],         # naranja
    "Control_EN": SLOTS[2],   # aqua
    "Control_ES": SLOTS[3],   # amarillo
}

# Cromo. La distribución nula es fondo, no serie: va en gris.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SEC = "#52514e"
MUTED = "#898781"
REJILLA = "#e1e0d9"
EJE = "#c3c2b7"
NULO_RELLENO = "#dcdbd4"      # histograma de la distribución nula

# Rampa secuencial de un solo tono (azul), claro -> oscuro.
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
              "#256abf", "#184f95", "#0d366b"]


def cmap_secuencial():
    """Colormap de un solo tono para magnitud. Nunca un arcoíris."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("babel_azul", RAMPA_AZUL)


def cmap_divergente():
    """
    Azul ↔ gris ↔ rojo, para magnitudes con signo (correlaciones).

    Una correlación es **divergente**, no secuencial: el cero significa "nada" y
    los dos extremos significan cosas opuestas. Codificarla con un arcoíris —lo
    que hacía el v1 con `viridis`— destruye esa lectura: el punto medio queda
    teñido de verde, como si +0,0 fuese un valor con contenido, y los dos
    extremos no se leen como opuestos.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "babel_div", ["#0d366b", "#2a78d6", "#9ec5f4",
                      "#f0efec",
                      "#f0a6a5", "#e34948", "#8f2423"])


def aplicar_estilo():
    """
    rcParams del estudio. Llamar una vez al principio del notebook.

    Rejilla y ejes sólidos de un pelo: el trazo discontinuo añade ruido visual
    y se lee como umbral cuando solo es una rejilla.
    """
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SEC,
        "axes.titlecolor": TINTA,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "axes.edgecolor": EJE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",          # sólida, nunca discontinua
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 8,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def colores_de(condiciones) -> list[str]:
    """Colores en el orden de las condiciones dadas, con reserva gris."""
    return [COLOR_CONDICION.get(c, MUTED) for c in condiciones]


# ============================================================================
# PANEL DE VEREDICTO (el patrón que se repite en cada bloque)
# ============================================================================
def panel_veredicto(ax, valores_null, valor_babel, valor_en=None, valor_es=None,
                    etiqueta="", leyenda=False):
    """
    Distribución nula como histograma gris + líneas de referencia.

    La nula es el FONDO contra el que se lee todo, no una serie más: por eso va
    en gris y las tres entidades comparables se llevan los tres primeros slots
    categóricos, que son los que validan todos-los-pares.
    """
    vn = np.asarray(valores_null)
    ax.hist(vn, bins=22, color=NULO_RELLENO, edgecolor=SUPERFICIE,
            linewidth=1.2, label=f"Null model (N={len(vn)})" if leyenda else None)

    p2, p97 = np.percentile(vn, [2.5, 97.5])
    ax.axvspan(p2, p97, color=NULO_RELLENO, alpha=0.55, zorder=0,
               label="IC 95% nulo" if leyenda else None)

    ax.axvline(valor_babel, color=COLOR_CONDICION["Babel"], linewidth=2.5,
               label="Babel" if leyenda else None)
    if valor_en is not None:
        ax.axvline(valor_en, color=COLOR_CONDICION["Control_EN"], linewidth=2.0,
                   label="Moby Dick" if leyenda else None)
    if valor_es is not None:
        ax.axvline(valor_es, color=COLOR_CONDICION["Control_ES"], linewidth=2.0,
                   label="CESS-ESP" if leyenda else None)

    dentro = p2 <= valor_babel <= p97
    # Nada de ✓/✗: la fuente del estudio es Segoe UI y no trae U+2713 ni U+2717,
    # así que matplotlib los dibuja como un cuadro vacío. Se usan palabras, que
    # además se leen sin descifrar el código.
    ax.set_title(f"{etiqueta}  ·  {'dentro' if dentro else 'FUERA'}")
    ax.set_ylabel("réplicas")
    if leyenda:
        ax.legend(loc="upper right")
    return dentro


# ============================================================================
# FIGURA RECUPERADA 1 — MAPA DE CORRELACIÓN ENTRE DOCUMENTOS
# ============================================================================
def matriz_similitud_documentos(textos: dict[str, str], max_features: int | None = 5_000):
    """
    Similitud coseno TF-IDF entre documentos.

    Reconstruye el "mapa de correlación" que el estudio original presentó como
    prueba de que los libros comparten ADN estadístico — **pero incluyendo
    réplicas del null model entre los documentos**.

    Ahí está todo el argumento: si el bloque Babel-Babel se ve igual que el
    bloque Babel-Null, la correlación que el v1 leyó como parentesco es
    simplemente lo que produce cualquier par de textos del mismo alfabeto.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    nombres = list(textos)
    vec = TfidfVectorizer(max_features=max_features)
    M = vec.fit_transform([textos[n] for n in nombres])
    Mn = M.toarray()
    norma = np.linalg.norm(Mn, axis=1, keepdims=True)
    norma[norma == 0] = 1.0
    Mn = Mn / norma
    return {"similitud": Mn @ Mn.T, "nombres": nombres,
            "n_features": Mn.shape[1]}


def resumen_por_bloques(S: np.ndarray, nombres: list[str],
                        grupos: dict[str, list[int]]) -> dict:
    """
    Similitud media dentro y entre grupos de documentos.

    Convierte la impresión visual del mapa en números comparables: si
    dentro-Babel ≈ Babel-vs-Null, la lectura del v1 queda descartada sin
    depender de que el ojo interprete bien un mapa de calor.
    """
    salida = {}
    claves = list(grupos)
    for i, a in enumerate(claves):
        for b in claves[i:]:
            ia, ib = grupos[a], grupos[b]
            if a == b:
                if len(ia) < 2:
                    continue
                vals = [S[x, y] for k, x in enumerate(ia) for y in ia[k + 1:]]
            else:
                vals = [S[x, y] for x in ia for y in ib]
            if vals:
                salida[f"{a} vs {b}" if a != b else f"dentro de {a}"] = float(np.mean(vals))
    return salida


# ============================================================================
# FIGURA RECUPERADA 1bis — CORRELACIÓN ENTRE PALABRAS (la del v1)
# ============================================================================
def correlacion_palabras(textos: list[str], top_n: int = 100) -> dict:
    """
    Correlación de Pearson **entre palabras**, a lo largo de los documentos.

    Reproduce exactamente lo que hacía el v1: TF-IDF sin `max_features`, se
    escogen las `top_n` palabras por TF-IDF acumulado, y se correlacionan sus
    columnas. El resultado es una matriz palabra×palabra.

    ⚠️ **El detalle que lo decide todo: n = número de DOCUMENTOS, no de
    palabras.** Con 10 libros, cada correlación se estima con 10 puntos. Eso
    hace que su desviación típica bajo no-asociación sea enorme (~1/3), y que
    aparezcan valores de ±0,9 continuamente. El v1 leyó esos extremos como
    estructura; son el comportamiento normal de un estimador con 10 muestras.
    """
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer

    vec = TfidfVectorizer()          # sin max_features, como el v1
    M = vec.fit_transform(textos)
    df = pd.DataFrame(M.toarray(), columns=vec.get_feature_names_out())
    top = df.sum().nlargest(top_n).index
    C = df[top].corr()

    vals = C.values[np.triu_indices_from(C.values, k=1)]
    vals = vals[~np.isnan(vals)]
    return {
        "matriz": C,
        "palabras": list(top),
        "n_documentos": len(textos),
        "n_vocabulario": df.shape[1],
        "correlaciones": vals,
        "media": float(vals.mean()) if len(vals) else float("nan"),
        "maxima": float(vals.max()) if len(vals) else float("nan"),
        "minima": float(vals.min()) if len(vals) else float("nan"),
        "n_pares": len(vals),
        "n_picos_05": int((np.abs(vals) > 0.5).sum()),
    }


def densidad_nula_correlacion(n_docs: int, rejilla: np.ndarray | None = None):
    """
    Densidad teórica del coeficiente de Pearson bajo **no asociación**.

        f(r) ∝ (1 − r²)^((n−4)/2)

    Es la referencia que faltaba en el v1. Superponerla sobre el histograma de
    correlaciones observadas convierte una impresión ("hay picos altísimos") en
    una comprobación ("los hay exactamente los que debe haber").
    """
    r = np.linspace(-0.999, 0.999, 601) if rejilla is None else rejilla
    if n_docs <= 3:
        return r, np.zeros_like(r)
    dens = (1 - r ** 2) ** ((n_docs - 4) / 2)
    area = np.trapezoid(dens, r) if hasattr(np, "trapezoid") else np.trapz(dens, r)
    return r, dens / area


def prob_superar_umbral(n_docs: int, umbral: float = 0.5) -> float:
    """
    P(|r| > umbral) bajo no asociación, vía la transformación t de Student.

    Con n=10 documentos y umbral 0,5 da ≈ 0,14: sobre 4.950 pares de palabras,
    unos 700 superarán 0,5 sin que exista ninguna relación. El v1 contó 676 y
    los presentó como "picos detectados".
    """
    from scipy import stats

    if n_docs <= 2 or umbral >= 1:
        return float("nan")
    t = umbral * math.sqrt(n_docs - 2) / math.sqrt(1 - umbral ** 2)
    return float(2 * stats.t.sf(t, df=n_docs - 2))


# ============================================================================
# FIGURA RECUPERADA 2 — NUBE DE PALABRAS MENOS FRECUENTES
# ============================================================================
def nube_de_palabras(tokens: list[str], solo_hapax: bool = True,
                     max_palabras: int = 150, seed: int = 42):
    """
    Nube de las palabras MENOS frecuentes (hapax: las que aparecen una vez).

    El v1 la presentó como hallazgo. Aquí solo tiene sentido enfrentada a la
    misma nube del null model: dos nubes de galimatías indistinguibles dicen
    más que cualquier tabla.

    Se colorea con la rampa de un solo tono, no con colores al azar: el tamaño
    ya codifica la frecuencia y el color no debe competir con él.

    Devuelve None si `wordcloud` no está instalado.
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        return None

    from collections import Counter

    frec = Counter(tokens)
    if solo_hapax:
        palabras = [p for p, n in frec.items() if n == 1]
        pesos = {p: 1 for p in palabras[:max_palabras * 3]}
    else:
        pesos = dict(frec.most_common(max_palabras * 3))
    if not pesos:
        return None

    def _color(word, font_size, position, orientation, random_state=None, **kw):
        idx = min(len(RAMPA_AZUL) - 1, max(0, int(font_size / 12)))
        return RAMPA_AZUL[idx]

    wc = WordCloud(width=900, height=500, background_color=SUPERFICIE,
                   max_words=max_palabras, color_func=_color,
                   random_state=seed, prefer_horizontal=0.95)
    return wc.generate_from_frequencies(pesos)


# ============================================================================
# E.2 — GRAFO DE CO-OCURRENCIA DE BIGRAMAS
# ============================================================================
def grafo_bigramas(texto: str, top_n: int = 40, excluir: tuple = ("\n",)):
    """
    Grafo dirigido de los `top_n` bigramas de caracteres más frecuentes.

    Es otra vista de la matriz de transición de C.6, pero mucho más elocuente
    para un lector no técnico: en español unos pocos arcos dominan (`qu`, `es`,
    `de`); en Babel los arcos son todos del mismo grosor, una malla uniforme.

    Devuelve (grafo, pesos_normalizados).
    """
    import networkx as nx
    from collections import Counter

    fuera = set(excluir)
    t = "".join(ch for ch in texto if ch not in fuera)
    bigramas = Counter(t[i:i + 2] for i in range(len(t) - 1))
    top = bigramas.most_common(top_n)
    if not top:
        return nx.DiGraph(), {}

    maximo = top[0][1]
    G = nx.DiGraph()
    pesos = {}
    for bg, n in top:
        a, b = bg[0], bg[1]
        G.add_edge(a, b, weight=n / maximo)
        pesos[bg] = n / maximo
    return G, pesos


if __name__ == "__main__":
    print("babel_viz.py — smoke test")
    print(f"  slots categóricos: {len(SLOTS)}")
    print(f"  condiciones mapeadas: {COLOR_CONDICION}")

    textos = {"a": "hola mundo " * 200, "b": "hola mundo " * 200,
              "c": "xkqv zplm " * 200}
    r = matriz_similitud_documentos(textos)
    S = r["similitud"]
    print(f"  similitud a-b (idénticos): {S[0, 1]:.3f}")
    print(f"  similitud a-c (distintos): {S[0, 2]:.3f}")

    G, pesos = grafo_bigramas("abcabcabc", top_n=5)
    print(f"  grafo de bigramas: {G.number_of_nodes()} nodos, "
          f"{G.number_of_edges()} arcos")
