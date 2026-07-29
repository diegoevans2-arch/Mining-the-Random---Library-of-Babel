"""
babel_embeddings.py
===================
Bloque C — Modelado no supervisado: ¿emergen "islas de significado"?

Cubre C.1 (BERTopic), C.2 (embeddings), C.3 (UMAP + HDBSCAN) y C.4 (MinHash).
Es el bloque más alineado con el concepto borgeano: si aparecen tópicos o
clusters coherentes, sería señal; si no, es la prueba más fuerte de aleatoriedad.

Una pasada de embeddings alimenta C.1, C.2 y C.3
------------------------------------------------
Embeber es el único paso caro: BGE-M3 son 568M de parámetros y en CPU cuesta
~292 ms por chunk de 200 caracteres. Calcular los vectores una vez y reutilizarlos
para geometría, clustering y tópicos evita pagar tres veces lo mismo.

Qué se mide y por qué
---------------------
La intuición ingenua dice que el ruido "no se agrupará". Es al revés, y conviene
tenerlo claro antes de mirar los números: para un modelo de embeddings, **todos
los fragmentos de ruido se parecen entre sí** — son intercambiables, no hay tema
que los distinga. Lo esperable es una nube compacta y sin estructura interna.

El lenguaje natural hace lo contrario: cada fragmento habla de algo, así que la
nube se estira y aparecen vecindarios.

Por eso las métricas no son "¿hay clusters?" sino:

  - `sim_media` / `sim_sd` — cómo de parecidos son los chunks entre sí y cuánto
    varía ese parecido. Ruido: media alta, varianza baja. Natural: al revés.
  - `dim_efectiva` — cuántas dimensiones ocupa realmente la nube (razón de
    participación de los autovalores). Una bola isótropa ocupa muchas; una nube
    con temas se concentra en pocas direcciones dominantes.
  - `n_clusters`, `pct_ruido`, `silueta` — lo que HDBSCAN encuentra.

Y todas, como siempre, contra el null model (§5.8): HDBSCAN **siempre** devuelve
algo, así que el número suelto no dice nada.
"""
from __future__ import annotations

import numpy as np

# La ventana/chunk es el mismo concepto que en el Bloque B; se reutiliza la
# implementación en vez de duplicarla.
from babel_transformers import muestrear_ventanas  # noqa: F401

MODELO_EMBEDDINGS = "BAAI/bge-m3"

# Medido sobre esta CPU: 200 chars = 292 ms/chunk, 300 = 432, 500 = 932.
# El coste crece más que linealmente (atención cuadrática), así que el punto
# eficiente está en el extremo bajo del rango 100-500 que fija el roadmap.
CHARS_POR_CHUNK = 200
N_CHUNKS = 200
BATCH_SIZE = 8          # medido: 8 mejor que 16 en CPU (cuello de cómputo)


# ============================================================================
# 1. EMBEDDINGS
# ============================================================================
def cargar_embedder(nombre: str = MODELO_EMBEDDINGS, device: str = "cpu"):
    """Carga el modelo de embeddings. Devuelve None si falta la librería."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    return SentenceTransformer(nombre, device=device)


def embeber(chunks: list[str], modelo, batch_size: int = BATCH_SIZE) -> np.ndarray:
    """
    Vectores normalizados (n_chunks × dim).

    Se normalizan para que el producto escalar sea directamente la similitud
    coseno: simplifica todas las métricas de abajo y es lo que espera HDBSCAN
    con métrica euclídea sobre la esfera.
    """
    if modelo is None or not chunks:
        return np.zeros((0, 0))
    return np.asarray(modelo.encode(
        list(chunks), batch_size=batch_size, show_progress_bar=False,
        normalize_embeddings=True))


# ============================================================================
# 2. GEOMETRÍA DE LA NUBE (C.2)
# ============================================================================
def similitudes_por_pares(vectores: np.ndarray, max_pares: int = 20_000,
                          seed: int | None = 0) -> np.ndarray:
    """
    Similitud coseno entre pares de chunks. Submuestrea si n² se dispara.

    Se expone aparte de `geometria()` —que la consume— para que la figura del
    bloque y el número del veredicto salgan del MISMO cálculo. Si la
    distribución se recalculase por su cuenta en el notebook, la media dibujada
    y la media reportada podrían divergir sin que nada avisara.

    Los vectores llegan normalizados desde `embeber()`, así que el producto
    escalar ya es la similitud coseno.
    """
    n = len(vectores)
    if n < 2:
        return np.empty(0)

    rng = np.random.default_rng(seed)
    if n * (n - 1) // 2 > max_pares:
        i = rng.integers(0, n, size=max_pares)
        j = rng.integers(0, n, size=max_pares)
        distintos = i != j
        return np.einsum("ij,ij->i", vectores[i[distintos]], vectores[j[distintos]])

    G = vectores @ vectores.T
    return G[np.triu_indices(n, k=1)]


def geometria(vectores: np.ndarray, max_pares: int = 20_000,
              seed: int | None = 0) -> dict:
    """
    Describe la nube de embeddings sin agrupar nada.

    `dim_efectiva` es la razón de participación de los autovalores de la
    covarianza: (Σλ)² / Σλ². Vale 1 si toda la varianza está en una dirección y
    d si está repartida por igual. Mide "en cuántas direcciones vive la nube"
    sin tener que fijar un umbral arbitrario de varianza explicada.
    """
    n = len(vectores)
    if n < 2:
        return {}

    sims = similitudes_por_pares(vectores, max_pares=max_pares, seed=seed)

    # Vecino más cercano de cada punto
    G = vectores @ vectores.T
    np.fill_diagonal(G, -np.inf)
    sim_vecino = G.max(axis=1)

    centrado = vectores - vectores.mean(axis=0)
    lam = np.linalg.svd(centrado, compute_uv=False) ** 2
    dim_efectiva = (lam.sum() ** 2) / (lam ** 2).sum() if lam.sum() else 0.0

    return {
        "n_chunks": n,
        "sim_media": float(sims.mean()),
        "sim_sd": float(sims.std()),
        "sim_p5": float(np.percentile(sims, 5)),
        "sim_p95": float(np.percentile(sims, 95)),
        "sim_vecino_media": float(sim_vecino.mean()),
        "dim_efectiva": float(dim_efectiva),
        "norma_centroide": float(np.linalg.norm(vectores.mean(axis=0))),
    }


# ============================================================================
# 3. REDUCCIÓN Y CLUSTERING (C.3)
# ============================================================================
def reducir(vectores: np.ndarray, n_componentes: int = 5,
            n_vecinos: int = 15, seed: int = 42) -> np.ndarray:
    """
    UMAP a `n_componentes` dimensiones. Si UMAP no está, cae a PCA.

    Se reduce antes de agrupar porque HDBSCAN sufre la maldición de la
    dimensionalidad: en 1024 dimensiones las distancias se concentran y casi
    todo acaba etiquetado como ruido, independientemente de si hay estructura.
    """
    if len(vectores) <= n_componentes:
        return vectores
    try:
        import umap
        reductor = umap.UMAP(n_components=n_componentes, n_neighbors=n_vecinos,
                             metric="cosine", random_state=seed)
        return reductor.fit_transform(vectores)
    except ImportError:
        from sklearn.decomposition import PCA
        return PCA(n_components=n_componentes, random_state=seed).fit_transform(vectores)


def agrupar(vectores_reducidos: np.ndarray, min_cluster: int = 10) -> dict:
    """
    HDBSCAN sobre los vectores reducidos.

    ⚠️ HDBSCAN SIEMPRE devuelve un resultado. Encontrar 4 clusters no significa
    que haya 4 grupos: significa que el algoritmo particionó lo que le diste.
    El número solo es interpretable frente a la distribución nula.

    `pct_ruido` es tan informativo como `n_clusters`: HDBSCAN etiqueta como
    ruido lo que no encaja en ninguna densidad, así que un porcentaje alto es
    la forma que tiene el algoritmo de decir "aquí no hay grupos".
    """
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score

    n = len(vectores_reducidos)
    if n < min_cluster * 2:
        return {"n_clusters": 0, "pct_ruido": 100.0, "silueta": float("nan")}

    # copy=True explícito: el array reducido se reutiliza después para graficar
    # y HDBSCAN puede modificarlo in situ. Además fija el comportamiento ante
    # el cambio de defecto anunciado en sklearn 1.10.
    etiquetas = HDBSCAN(min_cluster_size=min_cluster,
                        copy=True).fit_predict(vectores_reducidos)
    reales = etiquetas[etiquetas >= 0]
    n_clusters = len(set(reales.tolist()))
    pct_ruido = 100.0 * (etiquetas < 0).sum() / n

    silueta = float("nan")
    if n_clusters >= 2:
        mascara = etiquetas >= 0
        if mascara.sum() > n_clusters:
            silueta = float(silhouette_score(
                vectores_reducidos[mascara], etiquetas[mascara]))

    return {
        "n_clusters": n_clusters,
        "pct_ruido": pct_ruido,
        "silueta": silueta,
        "etiquetas": etiquetas,
    }


# ============================================================================
# 4. TÓPICOS (C.1)
# ============================================================================
def modelar_topicos(chunks: list[str], vectores: np.ndarray,
                    min_topic_size: int = 10, seed: int = 42) -> dict:
    """
    BERTopic sobre embeddings ya calculados.

    Se le pasan los vectores para que no vuelva a embeber: es el paso caro y ya
    está hecho.

    El valor de esta tarea es pedagógico, y el roadmap lo dice explícitamente:
    *"BERTopic sobre ruido puede producir tópicos espurios → ESE es el hallazgo"*.
    Ver la herramienta bautizando tópicos con aplomo sobre ruido puro es el
    mismo argumento que justifica conservar TextBlob (§5.2).

    Devuelve None si BERTopic no está instalado.
    """
    try:
        from bertopic import BERTopic
    except ImportError:
        return None

    try:
        from umap import UMAP
        reductor = UMAP(n_components=5, n_neighbors=15, metric="cosine",
                        random_state=seed)
        modelo = BERTopic(umap_model=reductor, min_topic_size=min_topic_size,
                          verbose=False, calculate_probabilities=False)
    except ImportError:
        modelo = BERTopic(min_topic_size=min_topic_size, verbose=False,
                          calculate_probabilities=False)

    topicos, _ = modelo.fit_transform(list(chunks), embeddings=vectores)
    info = modelo.get_topic_info()
    reales = [t for t in set(topicos) if t >= 0]

    palabras = {}
    for t in sorted(reales)[:5]:
        palabras[t] = [p for p, _ in (modelo.get_topic(t) or [])[:8]]

    return {
        "n_topicos": len(reales),
        "pct_outliers": 100.0 * sum(1 for t in topicos if t < 0) / len(topicos),
        "tamano_mayor": int(info["Count"].max()) if len(info) else 0,
        "palabras_por_topico": palabras,
        "modelo": modelo,
    }


# ============================================================================
# 5. NEAR-DUPLICATES (C.4)
# ============================================================================
def firma_minhash(texto: str, k: int = 8, num_perm: int = 128,
                  max_chars: int | None = 200_000):
    """
    Firma MinHash sobre shingles de `k` caracteres.

    `max_chars` recorta el texto porque el número de shingles es ~len(texto) y
    el coste es lineal en ellos: con 200k caracteres hay 200.000 shingles por
    libro, de sobra para estimar Jaccard con 128 permutaciones.

    Devuelve None si `datasketch` no está instalado.
    """
    try:
        from datasketch import MinHash
    except ImportError:
        return None

    t = texto[:max_chars] if max_chars else texto
    m = MinHash(num_perm=num_perm)
    m.update_batch([t[i : i + k].encode("utf8") for i in range(len(t) - k + 1)])
    return m


def jaccard_por_pares(textos: dict[str, str], k: int = 8,
                      num_perm: int = 128, max_chars: int | None = 200_000) -> dict:
    """
    Jaccard estimado entre cada par de textos, sobre shingles de k caracteres.

    Ataca directamente la afirmación del estudio original de que "los libros
    comparten ADN estadístico". Con k=8 sobre un alfabeto de 29 hay 29^8 ≈ 5·10¹¹
    shingles posibles y solo 200.000 observados por libro: la probabilidad de
    coincidencia por azar es despreciable, así que cualquier solapamiento real
    saltaría a la vista.
    """
    firmas = {n: firma_minhash(t, k, num_perm, max_chars) for n, t in textos.items()}
    if any(f is None for f in firmas.values()):
        return {}

    nombres = list(firmas)
    pares = {}
    for i, a in enumerate(nombres):
        for b in nombres[i + 1 :]:
            pares[f"{a} vs {b}"] = float(firmas[a].jaccard(firmas[b]))
    return pares


if __name__ == "__main__":
    print("babel_embeddings.py — smoke test")

    rng = np.random.default_rng(0)
    # Nube isótropa (sin estructura) frente a dos cúmulos separados
    isotropa = rng.normal(size=(200, 32))
    isotropa /= np.linalg.norm(isotropa, axis=1, keepdims=True)

    centros = rng.normal(size=(2, 32))
    cumulos = np.vstack([c + 0.05 * rng.normal(size=(100, 32)) for c in centros])
    cumulos /= np.linalg.norm(cumulos, axis=1, keepdims=True)

    for etiqueta, V in (("isótropa", isotropa), ("dos cúmulos", cumulos)):
        g = geometria(V)
        print(f"  {etiqueta:<13} sim_media={g['sim_media']:+.3f}  "
              f"sim_sd={g['sim_sd']:.3f}  dim_efectiva={g['dim_efectiva']:.1f}")

    textos = {"a": "abcdefgh" * 5_000, "b": "abcdefgh" * 5_000, "c": "zyxwvuts" * 5_000}
    j = jaccard_por_pares(textos, max_chars=20_000)
    print(f"  jaccard: {j if j else 'datasketch no instalado'}")
