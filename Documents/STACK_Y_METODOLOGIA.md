# 🔬 Stack técnico y metodología

> **Propósito de este documento:** inventario razonado de las herramientas
> empleadas en *Minando lo Aleatorio*, qué aporta cada una, en qué estado del
> arte se encuentra, y —sobre todo— **por qué se eligió**. Sirve como
> referencia técnica del proyecto y como documentación del enfoque analítico
> aplicado.

**Proyecto:** Minando lo Aleatorio — Text Mining sobre la Biblioteca de Babel
**Autor:** Diego Evans Pérez Paz
**Curso:** Complejidad y Caos — Enero 2026
**Estado documentado:** Bloques A, B, C y D completados · 29 de julio de 2026
**Veredicto acumulado:** **24 métricas indistinguibles del null model**

---

## Índice

1. [El problema y por qué define el stack](#1-el-problema-y-por-qué-define-el-stack)
2. [Stack por capas](#2-stack-por-capas)
3. [Metodología](#3-metodología)
4. [Decisiones técnicas documentadas](#4-decisiones-técnicas-documentadas)
5. [Los artefactos del instrumento](#5-los-artefactos-del-instrumento)
6. [Ingeniería y reproducibilidad](#6-ingeniería-y-reproducibilidad)
7. [Resultados](#7-resultados)
8. [Competencias aplicadas](#8-competencias-aplicadas)
9. [Limitaciones conocidas](#9-limitaciones-conocidas)

---

## 1. El problema y por qué define el stack

El proyecto analiza 10 libros extraídos de [libraryofbabel.info](https://libraryofbabel.info)
—13.288.080 caracteres de texto generado al azar sobre un alfabeto de 30
símbolos— con la pregunta: *¿deja rastros la verdadera aleatoriedad?*

La primera versión del estudio aplicó un pipeline clásico de NLP y encontró
"patrones sorprendentes": ~9% de stopwords, lematización exitosa, sentimiento
positivo, correlaciones entre libros.

**El problema no eran las mediciones. Era que no había forma de saber si venían
del texto o del instrumento.** Sin modelo nulo ni grupo de control, ninguna
afirmación era defendible.

Ese diagnóstico determina todo el stack: cada herramienta se elige por lo que
aporta a **distinguir señal de artefacto**, no por su capacidad bruta.

El estudio se organiza en cuatro bloques, cada uno con una pregunta propia y un
veredicto Monte Carlo independiente:

| Bloque | Pregunta | Módulo | Métricas |
|---|---|---|---|
| **A** — Fundamentos | ¿Distinguen las métricas clásicas Babel del ruido? | `babel_core`, `babel_pipeline` | 7 |
| **B** — NLP moderno | ¿Y los transformers? | `babel_transformers` | 5 |
| **C** — Estructura y geometría | ¿Hay memoria secuencial? ¿Hay clusters? | `babel_secuencia`, `babel_embeddings` | 8 |
| **D** — Significado emergente | ¿Cuánto significado emerge por azar? | `babel_lexico` | 4 |

**24 de 24 indistinguibles.** Ninguna herramienta —de 2001 a 2024, de un
stemmer de reglas a un encoder de 568M de parámetros— separa Babel de su gemelo
sintético.

---

## 2. Stack por capas

### 2.1. Núcleo científico

| Herramienta | Versión | Para qué se usa aquí | Estado del arte |
|---|---|---|---|
| **Python** | 3.12.7 | Lenguaje base | Estándar de facto en ciencia de datos |
| **NumPy** | 1.26.4 | Arrays, muestreo aleatorio (`default_rng`, PCG64), ajustes `polyfit` | Fundacional y estable. El API `Generator` (2019) sustituye al legacy `RandomState`; usarlo es la práctica correcta actual |
| **pandas** | 2.2.2 | DataFrames comparativos, exportación CSV | Estándar dominante. Alternativas modernas (Polars, DuckDB) ganan en volumen; a esta escala pandas es la elección sensata |
| **SciPy** | 1.13.1 | `chisquare`, `percentileofscore`, distribución *t* de Student para la densidad nula de correlaciones | Fundacional y estable |
| **Matplotlib** | 3.10.8 | Figuras (14) | Estándar de facto |
| **Seaborn** | 0.13.2 | Estética y grillas estadísticas | Maduro. Su API de objetos (0.12+) es la dirección actual |

### 2.2. Minería de texto clásica

| Herramienta | Versión | Para qué se usa aquí | Estado del arte |
|---|---|---|---|
| **NLTK** | 3.9.1 | Stopwords (9 idiomas), `SnowballStemmer`, corpus Gutenberg y CESS-ESP | **Legacy consciente.** Data de 2001; hoy es más biblioteca de recursos y docencia que herramienta de producción. Se usa por sus *corpus* y para reproducir el pipeline original |
| **spaCy** | 3.8.14 | Lematización (`es_core_news_sm` 3.8.0) con `disable=["ner","parser"]` | **Vigente en producción.** El estándar industrial para pipelines lingüísticos: rápido, tipado, con modelos versionados |
| **TextBlob** | — | Análisis de sentimiento | **Obsoleto, y conservado a propósito.** Léxico de reglas sobre el corpus de Pattern (2012). Se mantiene como *demostración de artefacto*, no como medición válida |
| **scikit-learn** | 1.9.0 | `CountVectorizer`, `TfidfVectorizer`, `HDBSCAN`, `PCA`, `silhouette_score` | Estándar de facto para ML clásico |

### 2.3. Teoría de la información

| Herramienta | Versión | Para qué se usa aquí | Estado del arte |
|---|---|---|---|
| **gzip / bz2 / lzma** | stdlib | Ratios de compresión como proxy de complejidad de Kolmogorov | Algoritmos maduros (DEFLATE 1996, BWT 1996, LZMA 1998). Su valor aquí es la **robustez y ubicuidad**, no la novedad |
| **zstandard** | 0.23.0 | Cuarto compresor, de generación reciente | **Vigente.** Zstd (Meta, 2016) es hoy el compresor de propósito general de referencia: mejor frontera velocidad/ratio que gzip, adoptado en Linux, Btrfs, Hadoop |
| **Implementación propia** | `babel_secuencia` | Entropía condicional, ganancia de información, matrices de transición de orden 1-4, divergencia Jensen-Shannon, modelo de n-gramas de caracteres con backoff interpolado | Ver §2.8 |

> **Por qué cuatro algoritmos.** La compresión estima complejidad de Kolmogorov,
> que es incomputable. Usar cuatro familias distintas (DEFLATE, BWT, cadena LZ,
> entropía moderna) convierte una medida indirecta en una **convergencia de
> evidencias**: si los cuatro coinciden, el resultado no depende del algoritmo.

### 2.4. NLP moderno

| Herramienta | Versión | Para qué se usa aquí | Estado del arte |
|---|---|---|---|
| **PyTorch** | 2.13.0+cpu | Motor de inferencia | **Vigente y dominante** en investigación. Compilación CPU deliberada: el estudio debe correr sin GPU |
| **transformers** | 5.14.1 | API de modelos, `pipeline`, tokenizadores | **Vigente.** La librería de referencia del ecosistema Hugging Face |
| **sentence-transformers** | 5.6.1 | Carga y *encoding* de BGE-M3, normalización L2 de los vectores | **Vigente.** El envoltorio estándar para modelos de embeddings; la v5 (2025) unifica embeddings densos, dispersos y *rerankers* |
| **tokenizers** | 0.22.2 | Tokenización subword (implementación en Rust) | **Vigente.** Estándar de rendimiento |
| **sentencepiece** | 0.2.2 | Algoritmo de subword requerido por XLM-R y BGE-M3 | Maduro (2018). Junto con BPE, la base de la tokenización moderna |
| **safetensors** | 0.8.0 | Serialización de pesos sin `pickle` | **Vigente.** Sustituye al formato `.bin` por motivos de seguridad |
| **fastText** | 0.9.2 | Identificación de idioma subléxica | **Antiguo pero insustituido en su nicho.** Ver §2.9 |

### 2.5. Modelado no supervisado — Bloque C

| Herramienta | Versión | Para qué se usa aquí | Estado del arte |
|---|---|---|---|
| **umap-learn** | 0.5.12 | Reducción de 1024 → 5 dimensiones antes de agrupar. Fallback automático a `PCA` si falta | **Vigente y dominante.** UMAP (McInnes et al., 2018) desplazó a t-SNE como reducción no lineal por defecto: preserva estructura global mejor y escala. Es lo que BERTopic usa internamente |
| **HDBSCAN** | `sklearn` 1.9.0 | Clustering por densidad sobre los vectores reducidos | **Vigente.** Campello et al. (2013). Desde sklearn 1.3 está en la librería estándar: el paquete `hdbscan` suelto ya no hace falta. La ventaja sobre k-means es que **no exige fijar el número de clusters** y admite puntos sin asignar — decisivo aquí, porque la hipótesis es que no hay clusters |
| **BERTopic** | 0.17.4 | Modelado de tópicos sobre los embeddings ya calculados | **Vigente.** Grootendorst (2022) es hoy el enfoque de referencia para topic modeling neuronal, y ha desplazado a LDA en la práctica. Modular: acepta embeddings precomputados, lo que evita recalcular BGE-M3 |
| **datasketch** | 2.0.0 | MinHash sobre shingles de caracteres para similitud Jaccard entre libros | **Maduro y vigente en su nicho.** MinHash (Broder, 1997) es la técnica canónica de deduplicación aproximada; se usa hoy para depurar corpus de entrenamiento de LLMs a escala de billones de tokens |
| **numba / pynndescent** | 0.60.0 / 0.6.0 | Dependencias de UMAP: compilación JIT y búsqueda aproximada de vecinos | Infraestructura, no elección de diseño |

> **Nota sobre `hdbscan` 0.8.44.** El paquete suelto está instalado porque
> BERTopic lo arrastra como dependencia, pero `babel_embeddings` importa
> `sklearn.cluster.HDBSCAN`. Una sola implementación en el código del estudio.

### 2.6. Recursos léxicos — Bloque D

| Herramienta | Versión | Para qué se usa aquí | Estado del arte |
|---|---|---|---|
| **wordfreq** | 3.1.1 | Las 100.000 palabras más frecuentes de español e inglés, más la escala Zipf de frecuencia de cada palabra emergente | **Vigente y sin equivalente directo.** Speer et al.: frecuencias agregadas sobre múltiples fuentes (Wikipedia, subtítulos, prensa, redes) en ~40 idiomas, con una escala logarítmica interpretable |
| **pyspellchecker** | 0.9.0 | Diccionario completo por idioma, como segunda fuente léxica | Maduro. Implementación en Python puro del algoritmo de Norvig. Aporta **cobertura morfológica** que la lista de frecuencias no tiene |

> **Por qué dos fuentes y no las listas Aspell del roadmap.** La unión de
> `wordfreq` y `pyspellchecker` da más de 100.000 palabras por idioma y cubre
> flexión verbal y plurales, que es donde una lista de frecuencias truncada
> falla. Aspell habría añadido una dependencia externa al sistema —contraria a
> la reproducibilidad del §6.3— sin aportar cobertura nueva medible.
>
> **Lo que estas dos librerías arreglan es un problema de circularidad.** La
> medición preliminar del Bloque A cruzaba los tokens de Babel contra un léxico
> construido *con los propios corpus de control*: el vocabulario de referencia
> dependía de los textos con los que se comparaba. Un diccionario externo, fijo
> y publicado rompe esa dependencia, y es lo que hace defendible la predicción
> combinatoria de §3.9.

### 2.7. Modelos empleados

| Modelo | Tamaño en caché | Qué es | Estado del arte |
|---|---|---|---|
| `BAAI/bge-m3` | 4.353 MB | Encoder de embeddings multilingüe: 568M parámetros, 1024 dimensiones, contexto de 8.192 tokens, 100+ idiomas | **Frontera abierta.** Chen et al. (2024). Uno de los mejores embedders multilingües disponibles con pesos públicos. Ver §2.9 |
| `papluca/xlm-roberta-base-language-detection` | 1.074 MB | Fine-tune de XLM-R base para detección de idioma | **Caballo de batalla, no frontera.** Ver abajo |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 2.126 MB | XLM-R re-preentrenado sobre ~198M tweets, ajustado a sentimiento en 8 idiomas | Ídem. El tamaño duplicado se debe a que el repositorio incluye pesos en formato PyTorch **y** TensorFlow |
| `lid.176.bin` (fastText) | 131 MB | Clasificador lineal sobre embeddings de n-gramas de caracteres, 176 idiomas | Ver §2.9 |

**Sobre XLM-RoBERTa** (Conneau et al., 2019, Facebook AI): encoder-only, 278M
parámetros, 100 idiomas, entrenado sobre 2,5 TB de CommonCrawl filtrado. Fue
estado del arte en comprensión multilingüe hacia 2020. **En 2026 es un caballo
de batalla maduro, no vanguardia.** Sucesores más capaces al mismo tamaño:
mDeBERTa-v3, y la generación de encoders modernizados (contexto extendido,
atención eficiente) que siguió a ModernBERT.

**Sobre BGE-M3** (Chen et al., 2024, BAAI): el nombre viene de *Multi-Linguality,
Multi-Granularity, Multi-Functionality* — produce simultáneamente embeddings
densos, dispersos y multi-vector. Aquí solo se usa el denso. Sustituyó a
`paraphrase-multilingual-MiniLM-L12-v2` (2021, 384 dim, límite de 128 tokens)
por la razón que se explica en §2.9: **no por necesidad técnica, sino para
cerrar una objeción.**

### 2.8. Lo que no se tomó de una librería

Tres piezas se implementaron a mano, y en los tres casos por la misma razón:
**para que dos métricas que deben ser coherentes salgan del mismo objeto.**

| Pieza | Alternativa descartada | Por qué |
|---|---|---|
| `similitudes_por_pares()` — expuesta aparte de `geometria()`, que la consume | Recalcular la distribución en el notebook para dibujarla | La figura del bloque C y el número del veredicto salen del **mismo** cálculo. Si la figura recalculase por su cuenta, la media dibujada y la media reportada podrían divergir sin que nada avisara. Fijado con un test |
| `ModeloCaracteres` — n-gramas de caracteres con backoff interpolado | `nltk.lm` | Comparte los conteos con la entropía condicional. Las dos métricas salen del mismo `Counter`, así que **no pueden contradecirse**. Con `nltk.lm` habría dos estimaciones independientes de lo mismo y ninguna forma de saber cuál creer |
| Entropía condicional y ganancia de información | `scipy.stats.entropy` sobre tablas construidas aparte | Ídem: el estimador plug-in y su sesgo (§5.3) tienen que ser visibles y auditables, no estar enterrados en una llamada de librería |
| Densidad nula del coeficiente de Pearson | Simulación Monte Carlo | Tiene forma cerrada vía la transformación *t* de Student. Calcularla es exacto, inmediato y no introduce ruido de muestreo — y ese cálculo **es** el hallazgo de §3.10 |

> Es la excepción al principio de "usar la librería estándar", y tiene un
> criterio: se implementa a mano cuando **la relación entre dos números importa
> más que cada número por separado.**

### 2.9. Nota honesta sobre la vanguardia

El estudio usa modelos de dos generaciones distintas a propósito, porque **el
criterio para elegir modelo depende de la dirección del argumento.**

**Donde el argumento es "hasta la herramienta estándar falla" → herramienta
madura.** Los tres modelos de los bloques A y B no son estado del arte actual:
dos son arquitectura de 2019 con ajustes comunitarios de 2021-2022, y fastText
es de 2016 y ni siquiera es un transformer. Razones:

1. **Fuerza argumental.** Si un modelo de frontera alucina sobre ruido, la
   respuesta es "usaste la herramienta equivocada". Si alucina **la herramienta
   estándar de producción**, no hay respuesta posible. `lid.176` es lo que se
   usa realmente para filtrar CommonCrawl y curar corpus de entrenamiento de
   LLMs: que asigne 0,110 de probabilidad a Babel es un dato sobre la práctica
   real, no sobre un juguete.

2. **Reproducibilidad.** Modelos públicos, versionados, gratuitos y ejecutables
   en CPU. Un modelo de frontera vía API haría el estudio irreproducible.

3. **Calibración documentada.** Son modelos suficientemente estudiados como para
   que su comportamiento sea interpretable.

**Donde el argumento es "ni siquiera el mejor encuentra nada" → herramienta de
frontera.** En el Bloque C la conclusión es *negativa* —no hay estructura
semántica— y un negativo obtenido con un modelo débil no vale nada: la objeción
inmediata sería "quizá con un modelo mejor sí aparecería". Por eso se sustituyó
el `MiniLM` de 2021 previsto en el roadmap por **BGE-M3**, que es lo mejor
disponible con pesos abiertos.

> **Versiones anteriores de este documento anunciaban esa sustitución como
> trabajo futuro. Se ejecutó.** El coste fue 4.353 MB en caché —el repositorio
> trae los pesos duplicados en `.bin` y `.safetensors`— y ~292 ms por chunk en
> CPU. A cambio, la conclusión del bloque no admite la objeción del modelo
> insuficiente.

**Lo que queda fuera y hay que decir:** no se ha probado un modelo de frontera
*cerrado* (vía API) sobre el corpus, ni un LLM generativo al que se le pida
interpretar los fragmentos. Ver §9.1.

### 2.10. Visualización

| Herramienta | Versión | Para qué se usa aquí |
|---|---|---|
| **wordcloud** | 1.9.6 | Nube de las palabras *menos* frecuentes — la figura del v1, reproducida |
| **networkx** | 3.3 | Grafo dirigido de bigramas de caracteres |
| **`babel_viz`** | — | Paleta, tipografía y cromo centralizados. `aplicar_estilo()` se llama una vez en la celda de imports, así que un Run All unifica las 14 figuras |

**Sistema visual, reglas fijas:**

- **Color ligado a la entidad, no a la serie.** Babel es azul en las catorce
  figuras; el null model, gris de fondo; los controles, aqua y amarillo. Un
  lector que aprende el código en la figura 1 lo tiene aprendido para el resto.

  > **La regla no se cumplía y hubo que arreglarlo (28 de julio).** La celda del
  > veredicto Monte Carlo —figura `06`, la que encabeza el README del
  > repositorio— no llamaba a `panel_veredicto()`: pintaba Babel en el verde por
  > defecto de matplotlib (`#2ca02c`) y los controles en rojo y naranja. Es
  > decir, la figura más visible del estudio usaba un código de color que
  > contradecía al de las otras trece. Reejecutar el bloque no lo corregía,
  > porque el defecto estaba en el código de la celda, no en los datos.
  > Reescrita para usar `panel_veredicto()`.

- **La figura muestra la métrica, no una proyección de la métrica.** La figura
  `09` coloreaba los puntos por cluster de HDBSCAN — el canal más visible para
  la cantidad que §3.8 documenta como casi inservible — y los mostraba sobre
  UMAP, que **normaliza la densidad local por construcción**. Resultado: las
  cuatro nubes salían iguales y el hallazgo del bloque (Babel colapsa: 0,888
  contra 0,371) no aparecía en ninguna parte. Rehecha el 28 de julio con la
  distribución de similitudes arriba y el UMAP debajo, coloreado por entidad.

  > **Por qué NO se usó hexbin ni curvas de nivel**, que eran las dos
  > alternativas consideradas. Hexbin necesita miles de puntos para superar a un
  > scatter y aquí hay 200 por panel. Y contornear la densidad del UMAP habría
  > sido un error **metodológico**, no estético: pondría una superficie de
  > densidad de aspecto científico sobre una proyección cuya densidad es un
  > artefacto de `n_neighbors`/`min_dist`, no una propiedad de los datos. Es
  > exactamente el error que este estudio existe para documentar. Las curvas de
  > nivel sí son legítimas en la figura `12`, donde la superficie es una matriz
  > de correlación real.

- **Ningún carácter fuera del repertorio de la fuente.** La tipografía del
  estudio es Segoe UI, que **no contiene `✓` (U+2713) ni `✗` (U+2717)**:
  matplotlib los dibujaba como un cuadro vacío en los siete títulos de la figura
  `06`. `panel_veredicto()` usa ahora las palabras `dentro` / `FUERA`, que
  además se leen sin descifrar un código. Verificado contra el `cmap` de
  `segoeui.ttf`, no a ojo.
- **Magnitud en un solo tono** claro→oscuro. La matriz de transición de C.6 pasó
  de `viridis` a rampa azul por esto.
- **Signo en paleta divergente** (azul ↔ gris ↔ rojo). La correlación tiene
  signo: el cero significa "nada" y los extremos, cosas opuestas. Un arcoíris
  tiñe de verde el punto medio, como si el 0,0 tuviese contenido.
- Nunca doble eje. Rejilla y ejes en línea **sólida** de un pelo — la discontinua
  se lee como umbral. Leyenda solo con dos o más series.

### 2.11. Ingeniería

| Herramienta | Versión | Para qué se usa aquí |
|---|---|---|
| **pytest** | 7.4.4 | **180 tests**, incluidos tests de regresión de invariantes científicos |
| **Jupyter Notebook** | 7.2.2 | Capa de presentación (76 celdas) |
| **pathlib, importlib, unicodedata, re, itertools, collections, math** | stdlib | Rutas portables, recarga de módulos, normalización Unicode, tokenización, espacios de categorías, conteos, logaritmos |
| **git** | — | Control de versiones |

---

## 3. Metodología

El aparato metodológico es la aportación central del proyecto. Diez piezas.

### 3.1. Null model como hipótesis nula formal

Se genera texto sintético que sigue **la misma ley generadora** que Babel:
muestreo i.i.d. sobre el alfabeto real calibrado, dispuesto en el mismo formato
físico (410 páginas × 40 líneas × 80 caracteres, con línea en blanco entre
páginas).

Es un *gemelo estadístico*: si Babel es genuinamente aleatorio, debe ser
indistinguible de él en toda métrica.

**Fundamento:** Ferrer-i-Cancho & Solé (2001) consolidaron la lectura de que las
construcciones de texto aleatorio deben tratarse como modelos nulos, no como
explicaciones del lenguaje.

### 3.2. Controles positivos

*Moby Dick* (Melville, 1851) y el corpus CESS-ESP. Ambos de dominio público —
requisito de reproducibilidad. Aportan la referencia de "cómo se ve la
estructura lingüística genuina".

### 3.3. Normalización de tamaño

Las 4 condiciones se recortan al mismo número de caracteres (1.104.297, fijado
por el control más corto), con `assert` de verificación. Entropía, compresión,
Zipf y Heaps dependen todas de *N*: sin igualar tamaños la comparación no es
válida.

### 3.4. Veredicto Monte Carlo

Se generan réplicas del null model, se calcula la distribución de cada métrica y
se ubica el valor de Babel dentro de ella mediante percentil e intervalo de
confianza del 95%.

Convierte la pregunta impresionista en un contraste de hipótesis formal. El
número de réplicas se ajusta al coste de la métrica, nunca al revés:

| Bloque | Réplicas | Unidad de observación | Coste |
|---|---|---|---|
| A — métricas clásicas | R=100 | corpus completo (1,1M chars) | ~110 s |
| B — transformers | R=25 × W=25 ventanas | ventana de 600 chars | 11,5 min |
| B — fastText | R=100 × W=100 ventanas | ventana de 600 chars | 22 s |
| C — secuencial | R=100 | corpus completo | ~40 min |
| C — embeddings | R=15 × W=200 chunks | chunk de 200 chars | ~25 min |
| D — léxico | R=100 | corpus completo | ~4 min |

### 3.5. Tablas de sensibilidad

Toda elección de parámetro con efecto sobre los resultados se acompaña de la
tabla de su alternativa:

- **Umbral de longitud de token** (sin filtro / ≥2 / ≥3 / ≥5)
- **Tramo de ajuste de Heaps** (N≥10 / 100 / 1.000 / 10.000 / 50.000)
- **Ajuste de Zipf** con y sin la cola de hapax
- **Orden del contexto** en la entropía condicional (k = 1, 2, 3 y **4**)

No es cosmética: en tres de los cuatro casos la tabla **cambió la
interpretación**. El cuarto —el orden k=4— produjo el artefacto más grave del
proyecto (§5.3), y solo se ve porque se calculó el orden que *no* se iba a usar.

### 3.6. Principio rector

> Cualquier "patrón" o "hallazgo" nuevo DEBE contrastarse contra el null model
> antes de considerarse real.

Se aplicó sin excepción, incluso cuando obligó a rediseñar. El análisis de
sentimiento se contrastaba inicialmente contra una sola réplica; se reincorporó
al Monte Carlo completo por coherencia con este principio, y el resultado cambió.

### 3.7. La unidad de observación se elige, no se hereda

**Problema:** los modelos tienen ventana de contexto finita y coste alto por
pasada. XLM-RoBERTa admite 512 tokens; sobre 1,1M de caracteres salen ~1.000
ventanas por condición, y contrastarlo contra 100 réplicas es inviable.

**Solución:** cambiar la unidad de observación, no recortar el rigor. Cada
ventana (o chunk) es una observación; *R* réplicas × *W* unidades producen *R*
valores del estadístico, y ahí se ubica Babel.

```
Transformers (Bloque B):  R=25  × W=25  ventanas de 600 chars
fastText     (Bloque B):  R=100 × W=100 ventanas de 600 chars
BGE-M3       (Bloque C):  R=15  × W=200 chunks   de 200 chars
```

El principio rector queda intacto y se gana **variabilidad intra-corpus**: cada
libro aporta su propia distribución de fragmentos.

**El tamaño se fija midiendo, no estimando.** La ventana de 600 caracteres del
Bloque B salió de comprobar que a 1.000 caracteres se truncaba *solo el ruido*
—627 tokens frente a 301 en los controles— lo que habría sesgado exactamente la
comparación que sostiene el estudio. El chunk de 200 caracteres del Bloque C se
fija por la razón inversa: es lo bastante corto para que un texto de 1,1M
caracteres dé 200 chunks independientes sin solaparse.

### 3.8. La geometría antes que el clustering

En el Bloque C, la tentación es correr HDBSCAN, contar clusters y concluir. El
diseño lo prohíbe: **primero se describe la nube de embeddings, y solo después
se intenta agruparla.**

`geometria()` calcula similitud media entre chunks, similitud con el vecino más
cercano, dispersión y dimensión efectiva **sin agrupar nada**. Son magnitudes
que no dependen de ningún hiperparámetro de clustering.

La razón es empírica, y es uno de los resultados del bloque: el recuento de
clusters de HDBSCAN resultó **casi inservible** —su intervalo de confianza nulo
sobre el porcentaje de puntos sin asignar va del 51,3% al 96,0% sobre 15
réplicas del *mismo* proceso—. Si el diseño hubiera dependido de él, el bloque
no habría concluido nada. Las métricas de geometría, en cambio, tienen
intervalos estrechos y separan las condiciones con claridad.

> **Generalizable:** cuando una herramienta siempre devuelve *algo*, hay que
> medir cuánto varía ese algo sin que cambie la entrada, antes de interpretarlo.

### 3.9. Predicción en cerrado — el resultado que no necesita null model

La pieza metodológicamente más fuerte del estudio, y la única que no depende de
ninguna herramienta ni de ningún modelo nulo.

**La idea:** si Babel es ruido i.i.d. sobre 26 letras, cuántas palabras reales
deben emerger no es algo que haya que medir. Es algo que se puede **calcular
antes de mirar el texto**:

$$P(k) = \frac{N_k}{26^k}$$

donde $N_k$ es el número de palabras de longitud $k$ en el diccionario. Cada
token de longitud $k$ es una tirada con esa probabilidad; el total esperado es la
suma sobre todos los tokens.

| Condición | Predichas | Observadas | **Obs/Pred** |
|---|---|---|---|
| **Babel** | 3.860,9 | 3.766 | **0,98** |
| **Null** | 3.864,0 | 3.798 | **0,98** |
| Control_EN | 16.725,3 | 150.914 | 9,02 |
| Control_ES | 10.449,1 | 117.267 | 11,22 |

Sin ajustar un solo parámetro. La longitud media de token, predicha como
geométrica de parámetro $3/29$ en **8,67**, se midió en **8,75**.

**Esto cambia la categoría de la conclusión.** En A, B y C se demuestra que la
emergencia de estructura es *indistinguible* del azar. Aquí se demuestra que
**es azar y se conoce la fórmula generadora**. Y la razón Obs/Pred se convierte
en una medida directa de intención: el lenguaje natural produce entre 9 y 11
veces más palabras válidas de las que el azar explicaría.

### 3.10. El tamaño de muestra como diagnóstico

Mismo patrón que §3.9 aplicado al hallazgo más vistoso del estudio original.

El v1 correlacionaba **palabras entre sí** a lo largo de los 10 documentos y
reportaba *"Picos detectados (correlación > 0.5): 676"*. Ese 10 es el tamaño de
muestra de cada coeficiente, y ahí está todo: bajo no-asociación, con n=10,

$$P(|r| > 0{,}5) = 0{,}1411$$

Sobre las $\binom{100}{2} = 4.950$ parejas de palabras salen **699 pares por
puro azar**. El hallazgo estrella del v1 está un **3,2% por debajo** de lo que
predice el ruido.

| | picos \|r\|>0,5 | % de pares | esperado por azar |
|---|---|---|---|
| Babel | 683 | 13,8% | 14,1% |
| Null | 697 | 14,1% | 14,1% |
| **Moby Dick partido en 10** | **1.082** | **21,9%** | 14,1% |

**La métrica no está rota**: el lenguaje natural la supera con holgura. El
problema eran los 10 documentos. Con n=50 la tasa de falsos picos cae al 0,02%.

> Igual que §3.9, no dice solo "Babel es indistinguible del azar": dice
> **cuánto azar había que esperar y por qué**. Las dos son la misma jugada
> —calcular la predicción del ruido en cerrado— aplicada a preguntas distintas.

---

## 4. Decisiones técnicas documentadas

Cada una nació de un problema concreto, y todas están registradas con su
justificación en `CONTEXT.md §5`.

| # | Decisión | Motivo |
|---|---|---|
| 1 | Umbral de token `>=3` con tabla de sensibilidad | El estudio original fusionaba "tokens cortos" y "stopwords" bajo una etiqueta |
| 2 | Conservar TextBlob pese a estar obsoleto | Es la demostración de que la herramienta alucina; eliminarlo perdería el argumento |
| 3 | Null model calibrado con la distribución real, no uniforme pura | Para ser gemelo estadístico debe replicar la ley observada |
| 4 | Corpus de control solo de dominio público | Validez académica y reproducibilidad |
| 5 | N=100 réplicas Monte Carlo a tamaño completo | Balance tiempo/precisión |
| 6 | Cuatro algoritmos de compresión | Convergencia de evidencias sobre una medida indirecta |
| 7 | Conservar todos los pasos del pipeline original | El v2 debe reinterpretar lo del v1, no borrarlo |
| 8 | **Principio rector** (§3.6) | Lección central del proyecto |
| 9 | El null model replica el **formato físico**, no solo el alfabeto | Sin la paginación, la frecuencia de `\n` difiere y la entropía separa por tipografía |
| 10 | Un token es una secuencia máxima de letras; los no alfabéticos son **frontera**, no se borran | Borrarlos pega tokens; el sesgo era asimétrico entre Babel y controles |
| 11 | Ajuste de Zipf sin la cola de hapax | En texto aleatorio la cola es ~96% de los rangos y es horizontal en log-log |
| 12 | χ² con el espacio **completo** de categorías `A^n` | Descartar las de conteo cero sesga el test hacia "uniforme" |
| 13 | Heaps con tabla de sensibilidad al tramo | β≈0,5 es un asintótico, no un valor a cualquier escala |
| 14 | Sentimiento incorporado al Monte Carlo | Coherencia con el principio rector |

**Añadidas en los bloques C y D:**

| # | Decisión | Motivo |
|---|---|---|
| 15 | Perplejidad con n-gramas de **caracteres**, no de palabras | A nivel de palabra el modelo asigna probabilidad ≈0 a cada token de Babel: la perplejidad sale **infinita** y no informa. A nivel de carácter da un número real y enlaza con la entropía del Bloque A |
| 16 | `ModeloCaracteres` propio en vez de `nltk.lm` | Comparte los conteos con la entropía condicional: las dos métricas salen del mismo objeto y no pueden contradecirse (§2.8) |
| 17 | Se calcula el orden k=4 **sabiendo que colapsa**, y se publica | Es la exhibición del sesgo del estimador plug-in. Ocultarlo sería quedarse con el rango cómodo sin decir dónde acaba (§5.3) |
| 18 | BGE-M3 en lugar del `MiniLM` previsto en el roadmap | La conclusión del bloque es negativa; un negativo con modelo débil no cierra la objeción "quizá con un modelo mejor" (§2.9) |
| 19 | Geometría de la nube **antes** que clustering | El recuento de clusters de HDBSCAN varía enormemente sin que cambie el texto; las métricas de geometría no (§3.8) |
| 20 | BERTopic sobre embeddings **precomputados** | Evita recalcular BGE-M3 y garantiza que topic modeling y clustering vean exactamente los mismos vectores |
| 21 | Léxico externo (`wordfreq` + `pyspellchecker`), no el vocabulario de los controles | Rompe la circularidad: el diccionario de referencia no puede depender de los textos con los que se compara (§2.6) |
| 22 | Se reporta D.3 pese a ser un resultado **negativo con confundente** | El negativo también es resultado. Se documenta el confundente en vez de retirar la métrica (§7.5) |
| 23 | Correlación en paleta **divergente**, no `viridis` | Una correlación tiene signo: el cero significa "nada". Un arcoíris tiñe de verde el punto medio como si el 0,0 tuviese contenido |

---

## 5. Los artefactos del instrumento

La sección con más valor metodológico del proyecto. Siete "hallazgos" de
corridas intermedias resultaron ser **defectos del aparato de medición**, no
propiedades de Babel.

### 5.1. Los cuatro del Bloque A

| Artefacto | Síntoma | Causa real | Corrección |
|---|---|---|---|
| **Zipf** | Percentil 99 | El ajuste incluía la cola de hapax: recta horizontal en log-log que domina por peso numérico (R²≈0,35 en Babel *y* en el null) | Excluir la cola. Babel pasa a percentil 34 |
| **χ²** | Percentil 100 | El null se calibraba con las frecuencias de una muestra de 9.475 caracteres: heredaba su ruido y le sumaba ruido fresco. El null *rechazaba* uniformidad (p=0,040) mientras Babel no (p=0,491) | Calibrar con los 13,3M. Percentil 65 |
| **Entropía** | Percentil 100 | El null no replicaba la paginación: frecuencia de `\n` de 1/81 frente a 1/79,06 real. Diferencia suficiente para separar por tipografía | Replicar las 410 páginas. Percentil 53 |
| **Sentimiento** | Babel al doble que el null (0,486 vs 0,195) | La limpieza borraba comas y puntos en vez de tratarlos como frontera: pegaba tokens y dejaba a TextBlob sin léxico reconocible | Frontera de palabra. Ambos ≈0,21 |

### 5.2. El quinto: incoherencia entre módulos

`babel_core` y `babel_pipeline` contaban **tokens distintos del mismo texto**
(111.797 frente a 48.961) porque tokenizaban de forma diferente. El estudio
reportaba dos vocabularios del mismo corpus. Unificado en la decisión #10.

### 5.3. El sexto, y el más grave: el estimador plug-in a orden 4

**El artefacto que habría invertido la conclusión del estudio.**

La ganancia de información mide cuánta incertidumbre elimina conocer los *k*
caracteres anteriores. Hasta k=3 se comporta como debe: Babel ≈ null ≈ 0, los
controles muy por encima. A **k=4** ocurre esto:

| Condición | k=1 | k=2 | k=3 | **k=4** | obs/contexto k=4 |
|---|---|---|---|---|---|
| Babel | 0,0005 | 0,0155 | 0,5258 | **3,7446** | 2,0 |
| Null | 0,0005 | 0,0157 | 0,5254 | **3,7442** | 2,0 |
| Control_EN | 0,9087 | 1,6198 | 2,2263 | 2,6823 | 17,9 |
| Control_ES | 1,0852 | 1,6835 | 2,2663 | 2,7475 | 21,2 |

**A orden 4 el estimador atribuye a Babel un 36% MÁS de estructura secuencial
que al español.**

**Causa.** El estimador plug-in de la entropía —usar las frecuencias empíricas
como si fuesen las probabilidades— es **sesgado a la baja**, y el sesgo crece
cuando hay pocas observaciones por celda. A orden 4 el texto aleatorio genera
más de medio millón de contextos distintos con **2,0 observaciones cada uno**:
cada distribución condicional empírica parece casi determinista, y la entropía
condicional se desploma. El lenguaje natural resiste porque concentra su masa en
muchos menos contextos distintos —18 a 21 observaciones por contexto—, así que
sus frecuencias empíricas todavía estiman algo.

Es decir: **el número no mide estructura del texto, mide escasez de datos.** Y
la escasez es máxima justo donde el texto es más aleatorio, lo que produce
exactamente la conclusión invertida.

**Detección.** El null model. Da 3,7442 frente a los 3,7446 de Babel — cuatro
diezmilésimas. Un valor absurdo que el gemelo estadístico reproduce con esa
precisión no es un hallazgo: es una propiedad del estimador.

> **Sin el null model, este bloque habría publicado que el ruido tiene más
> memoria que la literatura.** No es un error sutil que habría bajado la
> calidad del trabajo; es la conclusión contraria a la correcta, con números que
> la respaldan.

**Corrección.** El veredicto se limita a k ∈ {1, 2, 3}, donde el estimador es
fiable. El k=4 se calcula y se publica igualmente, como exhibición del límite
(decisión #17).

### 5.4. El séptimo: n=10

El caso de §3.10, listado aquí porque tiene la misma estructura que los
anteriores: **un número real, producido por un procedimiento correcto,
interpretado como hallazgo cuando era la predicción del azar.**

676 picos de correlación en el v1, contra 699 esperados por puro tamaño de
muestra. La diferencia con los seis artefactos previos es que este no se
corrige cambiando código, sino **cambiando n**: con 50 documentos en vez de 10,
la tasa de falsos picos pasa del 14,1% al 0,02%.

> **La moraleja, y el criterio que ordena todo el proyecto:** antes de creerse un
> hallazgo hay que descartar que lo esté produciendo el instrumento. El
> instrumento incluye el estimador, la herramienta, la limpieza previa **y el
> diseño muestral**.

---

## 6. Ingeniería y reproducibilidad

### 6.1. Arquitectura

**El repositorio separa las dos fases del estudio** (reorganizado el 27 de julio
de 2026; antes el v1 vivía disperso en la raíz):

```
Babel_V1/                   # FASE 1 — el estudio original, conservado íntegro
├── MINERIA_BIBLIOTECA_BABEL_v1.ipynb
└── figures/                # Las 6 figuras del v1

Babel_V2/                   # FASE 2 — la reconstrucción
Documents/                  # CONTEXT.md + este documento
data/                       # Los 10 libros (13 MB, irreproducibles)
```

Conservar el v1 no es sentimentalismo: **el v2 es un argumento sobre él.** Sus
conclusiones se enuncian como reinterpretaciones de hallazgos concretos, así que
borrar el original dejaría al v2 sin el objeto al que se refiere.

Dentro del v2, separación estricta entre lógica y presentación:

```
src/                        # Motor: funciones puras, testeables, sin visualización
├── babel_core.py           # Null model, controles, métricas de aleatoriedad
├── babel_pipeline.py       # Pipeline clásico de minería de texto
├── babel_transformers.py   # Bloque B: ventanas, transformers, fastText
├── babel_secuencia.py      # Bloque C: Markov, entropía condicional, perplejidad
├── babel_embeddings.py     # Bloque C: BGE-M3, geometría, UMAP, HDBSCAN, BERTopic, MinHash
├── babel_lexico.py         # Bloque D: léxico externo, predicción combinatoria
└── babel_viz.py            # Paleta, estilo, figuras del v1 revisitadas

MINERIA_BIBLIOTECA_BABEL_v2.ipynb   # Presentación: orquesta, muestra, narra (76 celdas)
tests/                              # 180 tests
outputs/                            # 27 tablas CSV
figures/                            # 14 figuras PNG
```

El notebook original mezclaba lógica y presentación en celdas de cientos de
líneas, lo que lo hacía imposible de mantener y testear.

**Los módulos nuevos respetan la misma frontera:** `babel_embeddings` devuelve
vectores y diccionarios de métricas, nunca un `Axes`; `babel_viz` es la única
excepción y lo es por diseño —es el módulo *de* presentación, y aun así separa
el cálculo (`correlacion_palabras`, `prob_superar_umbral`) del dibujo.

### 6.2. Degradación limpia ante dependencias ausentes

Los bloques C y D dependen de siete librerías que el Bloque A no necesitaba. El
código no asume ninguna:

| Función | Si falta la dependencia |
|---|---|
| `cargar_embedder()` | Devuelve `None` en vez de lanzar; el notebook salta el bloque |
| `reducir()` | Cae a `PCA` de sklearn si no hay UMAP |
| `modelar_topicos()` | Devuelve `None` si no hay BERTopic |
| `cargar_lexico()` | Une las fuentes **disponibles**; funciona con una sola |
| `cargar_fasttext()` | Devuelve `None` si falta librería o modelo |
| `conftest.py` | **Omite** los tests en vez de fallarlos |

Es lo que permite que 180 tests corran en 15 segundos en una máquina sin los
3,5 GB de modelos descargados.

### 6.3. Suite de tests

**180 tests, ~15 segundos.** Estructura:

| Archivo | Tests | Cubre |
|---|---|---|
| `conftest.py` | — | Fixtures; localiza `src/` y los datos; **omite en vez de fallar** si faltan corpus o dependencias opcionales |
| `test_parseo.py` | 13 | Separación de metadata, contaminación del alfabeto |
| `test_null_model.py` | 13 | Generación, reproducibilidad por semilla, calibración, paginación |
| `test_metricas.py` | 24 | Entropía, compresión, Zipf, Heaps, χ² y casos límite |
| `test_pipeline.py` | 22 | Limpieza, tokenización, stopwords, stemming, lematización por bloques |
| `test_transformers.py` | 30 | Muestreo, métricas, normalización de etiquetas (con *pipeline* simulado) |
| `test_secuencia.py` | 23 | Transiciones, entropía condicional, ganancia, perplejidad, divergencia JS |
| `test_embeddings.py` | 19 | Chunking, geometría, reducción, clustering, MinHash (con embedder simulado) |
| `test_lexico.py` | 24 | Carga de léxicos, `P(k)`, predicción combinatoria, emergentes, por documento |
| `test_regresiones.py` | 9 | Invariantes entre módulos y los artefactos de §5 |

Lo distintivo no es la cobertura sino **qué se testea**: además de las funciones,
se fijan los **invariantes científicos** del estudio. Que Zipf *no* separe Babel
del español y que Heaps *sí* lo haga son tests. Si empiezan a fallar, la alarma
suena antes de llegar a las conclusiones.

Los tests de `test_transformers.py` y `test_embeddings.py` usan modelos
simulados: la lógica de muestreo, geometría y métricas no requiere 7,5 GB de
pesos para validarse.

### 6.4. Reproducibilidad

- **Semilla fijada** (`SEED = 2026`) y propagada a todo muestreo, generación,
  UMAP y HDBSCAN
- **Sin rutas hardcoded:** `pathlib` y auto-detección del corpus en varias ubicaciones
- **Dependencias fijadas** en `requirements.txt`, con marcadores de plataforma
  donde el paquete difiere (`fasttext-wheel` en Windows)
- **Recarga explícita de módulos** (`importlib.reload`) más verificación de API
  en la celda de imports: sin ello, un kernel que ya importó una versión antigua
  falla con `AttributeError` varias celdas más abajo, con un mensaje que no
  apunta a la causa
- **Modelos cacheados fuera del repositorio** (`~/.cache/huggingface`,
  `~/.cache/fasttext`)
- **Modo rápido** (`MODO_RAPIDO_CHARS`) y banderas por bloque
  (`EJECUTAR_BLOQUE_B`, `EJECUTAR_BLOQUE_C2`) para smoke test antes de la
  corrida larga

### 6.5. Rendimiento

Optimizaciones aplicadas tras medir, no tras suponer:

| Problema | Efecto |
|---|---|
| El bucle Monte Carlo calculaba 4 compresores para usar 1, y la entropía dos veces | Cómputo descartado eliminado |
| spaCy rechaza documentos de más de 1M de caracteres (error E088) | Procesamiento por bloques de 20k tokens; sin ello la lematización fallaba a escala real |
| Ventana de transformers dimensionada a ojo | Medida: a 800 caracteres el ruido llegaba a 505 tokens (7 de margen sobre 512) y a 1.000 se truncaba **solo el ruido**. Fijada en 600 |
| `batch_size` por defecto | Medido: 8 resultó más rápido que 16 y 32 en CPU, tanto para XLM-R como para BGE-M3 |
| BERTopic recalculando embeddings | Se le pasan los vectores ya calculados: el Bloque C entero hace **una sola** pasada de BGE-M3 |
| MinHash sobre el corpus completo | Acotado por `max_chars`: la firma converge mucho antes de 1,1M caracteres |
| Correlación de palabras sobre 1,1M chars por documento | Medido: se estabiliza mucho antes. Fijado en 200.000 |

**Corrida completa ≈ 1 hora** en CPU (12 núcleos, sin GPU):

| Bloque | Tiempo |
|---|---|
| A — fundamentos | ~3 min |
| B — NLP moderno | ~12 min |
| C — secuencial + embeddings | ~42 min |
| D — léxico | ~4 min |
| Figuras del v1 revisitadas | ~4 min |

---

## 7. Resultados

### 7.1. Bloque A — 7 de 7 métricas indistinguibles

| Métrica | Babel | Null μ | IC 95% nulo | Percentil |
|---|---|---|---|---|
| Entropía relativa (%) | 99,7454 | 99,7454 | [99,745, 99,746] | 53 |
| Compresión gzip | 0,6512 | 0,6513 | [0,651, 0,651] | 49 |
| Pendiente Zipf | −0,9386 | −0,9372 | [−0,949, −0,927] | 34 |
| Heaps β | 0,9697 | 0,9695 | [0,968, 0,974] | 70 |
| χ² p-valor | 0,4687 | 0,3784 | [0,015, 0,952] | 65 |
| Sentimiento (polaridad) | 0,2254 | 0,2051 | [0,085, 0,320] | 60 |
| Sentimiento (subjetividad) | 0,5167 | 0,5214 | [0,438, 0,605] | 46 |

El lenguaje natural sí se separa: entropía relativa 71,4% y 67,9% frente a
99,7%; compresión 0,405 y 0,368 frente a 0,651; **tipos únicos / tokens 0,084 y
0,124 frente a 0,769**.

### 7.2. Bloque B — 5 de 5 métricas indistinguibles

| Métrica | Babel | Null μ | Percentil | Ctrl_EN | Ctrl_ES |
|---|---|---|---|---|---|
| Fragmentación (tokens/char) | 0,6029 | 0,6019 | 60 | 0,270 | 0,276 |
| XLM-R prob. de idioma | 0,5417 | 0,5323 | 68 | 0,953 | 0,991 |
| fastText prob. de idioma | 0,1099 | 0,1102 | 43 | 0,967 | 0,963 |
| Sentimiento moderno (polaridad) | −0,1113 | −0,1109 | 40 | −0,184 | −0,190 |
| Sentimiento moderno (neutral) | 0,4472 | 0,4474 | 48 | 0,316 | 0,391 |

### 7.3. Bloque C — estructura secuencial: 3 de 3 indistinguibles

| Métrica | Babel | Null μ | IC 95% nulo | Pctil | Ctrl_EN | Ctrl_ES |
|---|---|---|---|---|---|---|
| Ganancia de información (k=1) | 0,0005 | 0,0005 | [0,0005, 0,0006] | 71 | 0,909 | 1,085 |
| Ganancia de información (k=2) | 0,0155 | 0,0156 | [0,0154, 0,0159] | 29 | 1,620 | 1,683 |
| Ganancia de información (k=3) | 0,5255 | 0,5255 | [0,5243, 0,5269] | 53 | 2,226 | 2,266 |

Conocer los tres caracteres anteriores elimina **0,53 bits** de incertidumbre en
Babel y **2,23** en inglés. El texto aleatorio no tiene memoria, y la métrica lo
detecta con un margen de más de cuatro veces.

*(El orden k=4 y su colapso: §5.3.)*

**Perplejidad — un gradiente, no un interruptor.** Modelo de caracteres de orden
5 entrenado sobre el 80% de CESS-ESP, evaluado sobre 200.000 caracteres de cada
condición:

| Texto evaluado | bits/char | Perplejidad | vs uniforme (29) |
|---|---|---|---|
| Español retenido | 2,261 | 4,8 | 0,17× |
| Inglés (Moby Dick) | 5,990 | 63,6 | 2,19× |
| **Babel** | **7,912** | **240,9** | **8,31×** |
| Null model | 7,906 | 239,8 | 8,27× |

El inglés da la escala: un modelo de español lo predice mal (63,6, por encima
del techo uniforme de 29) pero **cuatro veces mejor que a Babel**, porque
comparte alfabeto y estadística de bigramas. Babel llega a 8,31× peor que no
saber nada: el modelo está *confiadamente equivocado*, no ignorante.

**Matrices de transición.** Divergencia de Jensen-Shannon media:

| Comparación | JS |
|---|---|
| Babel vs Null | **0,0003** |
| Babel vs Control_ES | 0,5424 |
| Null vs Control_ES | 0,5428 |

> **Matiz que hay que declarar:** la divergencia contra el español mezcla dos
> efectos, porque CESS-ESP en crudo usa 96 símbolos (mayúsculas, dígitos,
> puntuación) frente a los 29 de Babel. Parte de ese 0,54 es diferencia de
> alfabeto, no de estructura. La comparación limpia —mismo alfabeto exacto— es
> Babel contra Null: **0,0003**.

### 7.4. Bloque C — geometría de embeddings: 5 de 5 indistinguibles

BGE-M3 sobre chunks de 200 caracteres, R=15 réplicas × W=200 chunks:

| Métrica | Babel | Null μ | IC 95% nulo | Pctil | Ctrl_EN | Ctrl_ES |
|---|---|---|---|---|---|---|
| Similitud media entre chunks | 0,8878 | 0,8874 | [0,884, 0,891] | 53,3 | 0,464 | 0,371 |
| Dispersión de similitudes | 0,0239 | 0,0241 | [0,022, 0,028] | 60,0 | 0,060 | 0,062 |
| Dimensión efectiva | 86,90 | 86,78 | [83,86, 89,15] | 46,7 | 82,66 | 88,20 |
| Clusters HDBSCAN | 2,5 | 2,27 | [0,70, 3,65] | 66,7 | 4,0 | 4,0 |
| % puntos sin cluster | 69,25 | 71,63 | [51,25, 95,98] | 33,3 | 56,5 | 28,0 |

**La ausencia de estructura se manifiesta como uniformidad, no como dispersión.**
Los fragmentos de Babel se parecen entre sí **más del doble** que los de la
literatura española (0,888 frente a 0,371). Para el modelo, dos trozos de ruido
son casi el mismo objeto: son intercambiables porque no hay nada que los
distinga. Babel no "deja de agruparse" — **colapsa en una bola**.

> Invierte la intuición de partida, que era que el ruido se dispersaría. Conviene
> tenerlo escrito antes de mirar cualquier resultado de clustering.

**BERTopic halla más tópicos en el ruido que en el español:**

| Condición | Tópicos | % outliers |
|---|---|---|
| **Babel** | **3** | 64,5 |
| Null | 2 | 85,0 |
| Control_EN | 2 | 45,0 |
| **Control_ES** | **2** | 20,0 |

Tres tópicos en ruido puro contra dos en español real. Mismo argumento que
TextBlob: la herramienta no falla en silencio, falla **con aplomo**. Lo que sí
ordena bien las condiciones es el % de outliers (20 → 45 → 64,5 → 85).

**El "ADN estadístico compartido", liquidado.** MinHash sobre shingles de 8
caracteres, 128 permutaciones: **Jaccard = 0,000000 en los 21 pares**. Entre
libros de Babel, entre Babel y el null, entre Babel y el español. Ni una sola
secuencia de ocho caracteres compartida.

### 7.5. Bloque D — significado emergente: 4 de 4 indistinguibles

| Métrica | Babel | Null μ | IC 95% nulo | Pctil |
|---|---|---|---|---|
| Tasa de palabras reales (%) | 4,2714 | 4,2787 | [4,159, 4,412] | 49 |
| Palabras emergentes | 3.758,7 | 3.768,2 | [3.655,8, 3.888,1] | 43 |
| Palabras distintas | 2.907,9 | 2.918,1 | [2.838,3, 2.990,6] | 41 |
| Longitud máxima | 5,5 | 5,56 | [5,00, 6,00] | 46 |

*(La predicción en cerrado que explica estos números: §3.9.)*

**Por qué las islas de significado tienen techo de longitud:**

| k | N_k | 26^k | P(k) |
|---|---|---|---|
| 3 | 6.395 | 17.576 | **0,364** |
| 5 | 20.753 | 1,19·10⁷ | 0,00175 |
| 9 | 42.816 | 5,43·10¹² | **7,89·10⁻⁹** |

Con tres letras, el **36% de todas las cadenas posibles** son palabras reales.
Con nueve, 8 entre mil millones. El diccionario crece, pero el espacio de
cadenas crece por 26 en cada paso: esa carrera perdida es toda la explicación.
La palabra más larga de Babel tiene **5 letras**; el español llega a 19.

**Homogeneidad entre libros:** la tasa de emergencia va de 4,196% a 4,372% en
los 10 libros, con razón varianza/media compatible con Poisson. No hay un libro
"más significativo" que otro.

**D.3 — el negativo documentado.** Búsqueda semántica con sondas en español
(*"el amor es eterno mientras dura"* y similares) contra los chunks de cada
condición:

| Condición | Similitud máxima media |
|---|---|
| Babel | 0,4252 |
| Null | 0,4314 |
| Control_EN | 0,5108 |
| Control_ES | 0,4621 |

Babel alcanza 0,425; el español, 0,462. Un 8% de diferencia. **No significa que
Babel casi contenga esas ideas: significa que la métrica no sirve para lo que
parecía.** BGE-M3 concentra todo el texto en un cono estrecho —el mismo
fenómeno que la similitud media de 0,888—, el suelo para ruido puro ya está en
0,42 y el rango útil es minúsculo.

> **Confundente detectado y declarado:** el inglés puntúa más alto (0,5108) que
> el español (0,4621) frente a sondas *en español*. Causa probable: CESS-ESP es
> prensa económica, temáticamente lejos de sondas sobre amor, tiempo y muerte,
> mientras Moby Dick es prosa narrativa. La comparación entre controles no es
> limpia. Se reporta completa porque el negativo también es resultado, pero no
> sostiene ninguna conclusión por sí sola.

### 7.6. Cinco resultados con contenido propio

**Zipf no discrimina aleatoriedad.** Babel da pendiente −0,940 (R²=0,858) y el
español real −0,968 (R²=0,988). El texto aleatorio cumple la ley de Zipf: es el
resultado del "mono tipeando" de Miller (1957) y Mandelbrot (1953), reproducido
sobre un corpus concreto. Heaps sí separa. Contrastar ambas es lo que enseña
*cuáles* métricas sirven.

**Dos detectores de idioma fallan de forma opuesta.** fastText reparte la
probabilidad y se queda en 0,110: reconoce que no sabe. XLM-R concentra 0,542 en
**hindi**, en el 100% de las ventanas, con la misma consistencia con la que
etiqueta el español real. Un transformer de 278M de parámetros está seguro y
equivocado; un clasificador lineal sobre n-gramas es honesto. **La capacidad del
modelo no predice su calibración.**

**El sentimiento moderno invierte a TextBlob.** TextBlob lee el ruido *más
positivo* que la literatura (+0,225 frente a +0,083); el modelo moderno le da
más masa neutral (0,447 frente a 0,316 y 0,391), o sea se compromete **menos**
cuando no hay nada que leer. Los dos alucinan; solo uno lo hace de forma
detectable como error.

**El ruido colapsa, no se dispersa.** La intuición de que la ausencia de
estructura semántica se vería como una nube difusa es falsa: los fragmentos de
Babel son *más* parecidos entre sí que los de cualquier texto real, porque no
tienen rasgos que los diferencien. Es el hallazgo que hay que tener presente
antes de interpretar cualquier clustering sobre texto degenerado.

**El significado emergente tiene fórmula cerrada.** `P(k) = N_k / 26^k` predijo
3.861 palabras reales antes de contar ninguna; aparecieron 3.766, un 2% de
error. Y la razón entre observadas y predichas —0,98 en Babel, 9,02 y 11,22 en
los controles— es una medida directa de cuánto de un texto no lo explica el
azar. **Es el único resultado del estudio que no depende de ningún modelo nulo
ni de ninguna herramienta de NLP.**

---

## 8. Competencias aplicadas

Cada afirmación remite a evidencia verificable en el repositorio.

| Competencia | Evidencia concreta |
|---|---|
| **Diseño experimental** | Modelo nulo como hipótesis nula formal, controles positivos, normalización de tamaño con verificación automática, unidad de observación elegida por bloque y dimensionada por medición |
| **Inferencia estadística** | Monte Carlo en cuatro bloques, percentiles e intervalos de confianza, χ² con espacio de categorías completo, regresión en log-log, densidad nula del coeficiente de Pearson en forma cerrada |
| **Detección de artefactos** | Siete "hallazgos" identificados como defectos del instrumento (§5), cada uno con mecanismo diagnosticado y corrección verificada — incluido uno que habría invertido la conclusión del bloque |
| **Conocimiento de los estimadores, no solo de las librerías** | Sesgo del estimador plug-in de entropía diagnosticado, cuantificado y publicado en vez de evitado (§5.3) |
| **Lectura crítica de resultados propios** | Tres hipótesis de diseño refutadas por los datos y documentadas como tales: la dispersión de idiomas sobre ruido, la dimensión efectiva como discriminador, y la dispersión geométrica del ruido |
| **NLP clásico** | Pipeline completo: limpieza, tokenización, stopwords multi-idioma, stemming, lematización, frecuencias, vectorización |
| **NLP moderno** | Transformers en CPU con diseño de ventanas medido; embeddings con BGE-M3; contraste entre modelo neuronal y clasificador lineal |
| **Modelado no supervisado** | UMAP + HDBSCAN + BERTopic sobre embeddings precomputados, con la variabilidad del propio clustering medida antes de interpretarlo (§3.8) |
| **Teoría de la información** | Entropía de Shannon sobre n-gramas, entropía condicional y ganancia de información hasta orden 4, modelo de lenguaje de caracteres con backoff, divergencia Jensen-Shannon, compresión como proxy de complejidad de Kolmogorov con cuatro algoritmos |
| **Modelado probabilístico en cerrado** | Predicción combinatoria de palabras emergentes con 2% de error y sin parámetros ajustados; longitud media de token derivada como geométrica; tasa de falsos picos derivada de la *t* de Student (§3.9, §3.10) |
| **Ingeniería de software** | Separación lógica/presentación en 7 módulos, 180 tests incluidos invariantes científicos, degradación limpia ante siete dependencias opcionales |
| **Optimización basada en medición** | Siete cuellos de botella corregidos tras cronometrarlos, no tras suponerlos |
| **Comunicación visual** | Sistema de color ligado a la entidad y consistente en 14 figuras; codificación elegida por tipo de magnitud (divergente para signo, un solo tono para magnitud) |
| **Documentación técnica** | 23 decisiones metodológicas registradas con su justificación; correcciones marcadas explícitamente en vez de sustituidas en silencio |
| **Criterio sobre el estado del arte** | Elección deliberada de herramientas maduras donde el argumento es "hasta la estándar falla", y de frontera abierta donde es "ni siquiera la mejor encuentra nada" (§2.9) |

---

## 9. Limitaciones conocidas

Se listan porque su ausencia sería la principal debilidad del documento.

### 9.1. Vigentes

1. **No se ha probado un modelo de frontera cerrado.** El Bloque C usa BGE-M3,
   que es lo mejor disponible con pesos abiertos, pero no se ha comprobado qué
   haría un embedder comercial de última generación ni un LLM generativo al que
   se le pida interpretar los fragmentos. La restricción es deliberada
   (reproducibilidad, §2.9) pero sigue siendo una limitación real.

2. **Los modelos de los bloques A y B no son estado del arte actual** (§2.9).
   Fue deliberado y argumentado, y para esos bloques el argumento *requiere*
   herramientas estándar. Aun así, no se ha comprobado el comportamiento de un
   detector de idioma moderno sobre el mismo corpus.

3. **El diseño muestreado limita la granularidad del percentil.** R=25 en
   transformers da percentiles de 4% en 4%; R=15 en embeddings, de 6,7% en 6,7%.
   Suficiente dado el tamaño de los efectos observados —los valores de Babel
   caen en mitad de la distribución nula, no en el borde— pero no comparable al
   Monte Carlo de 100 réplicas de los bloques A, C-secuencial y D.

4. **La compresión estadística mide entropía, no complejidad algorítmica pura**
   (Vitányi). Se usa como estimador práctico validado contra controles, que es la
   práctica estándar defendible, pero conviene explicitarlo.

5. **El estimador de entropía condicional solo es fiable hasta k=3** en este
   corpus (§5.3). El límite no es del método sino del tamaño: con más texto, el
   orden fiable subiría. No se ha calculado dónde estaría el corte para un
   corpus de 100M de caracteres.

6. **La divergencia contra el español mezcla alfabeto y estructura** (§7.3). Los
   0,54 de Jensen-Shannon no son atribuibles solo a diferencia de estructura
   secuencial, porque CESS-ESP usa 96 símbolos frente a 29. La comparación
   limpia es Babel vs Null.

7. **La búsqueda semántica (D.3) tiene rango útil minúsculo y un confundente
   temático** entre los dos controles (§7.5). Se reporta como negativo
   documentado; no sostiene ninguna conclusión por sí sola.

8. **Un solo dominio.** Todas las conclusiones proceden de un corpus generado por
   un único procedimiento. La generalización a otras fuentes de aleatoriedad no
   está comprobada.

### 9.2. Resueltas desde la versión anterior de este documento

> Se conservan explícitamente en vez de borrarse, por coherencia con el criterio
> de marcar las correcciones en lugar de sustituirlas en silencio.

| Limitación anterior | Cómo se resolvió |
|---|---|
| **"El léxico de referencia para palabras emergentes es circular."** La medición preliminar (602 palabras en Babel frente a 641 en el null) usaba un vocabulario construido con los propios corpus de control | Bloque D. `wordfreq` + `pyspellchecker` aportan un diccionario externo, fijo y publicado. La medición formal da 3.766 palabras emergentes en Babel frente a 3.798 en el null, y además permite la predicción en cerrado de §3.9, que no habría sido posible con un léxico circular |
| **"El modelo de embeddings previsto está desfasado."** `paraphrase-multilingual-MiniLM-L12-v2` es de 2021, 384 dimensiones y límite de 128 tokens | Sustituido por BGE-M3 (2024, 568M parámetros, 1024 dim). Ver §2.9 |

---

## Apéndice — Entorno verificado

Versiones capturadas del entorno de ejecución real, no de `requirements.txt`.

```
Python              3.12.7 (Anaconda)      Windows 11 · 12 núcleos lógicos · sin GPU

NÚCLEO                                     NLP MODERNO
numpy               1.26.4                 torch               2.13.0+cpu
pandas              2.2.2                  transformers        5.14.1
scipy               1.13.1                 sentence-transformers 5.6.1
matplotlib          3.10.8                 tokenizers          0.22.2
seaborn             0.13.2                 sentencepiece       0.2.2
scikit-learn        1.9.0                  safetensors         0.8.0
                                           huggingface_hub     1.24.0
NLP CLÁSICO                                fasttext-wheel      0.9.2
nltk                3.9.1
spacy               3.8.14                 MODELADO NO SUPERVISADO
  es_core_news_sm   3.8.0                  umap-learn          0.5.12
  en_core_web_sm    3.8.0                  bertopic            0.17.4
textblob            —                      datasketch          2.0.0
zstandard           0.23.0                 numba               0.60.0
                                           pynndescent         0.6.0
LÉXICO                                     hdbscan             0.8.44 ¹
wordfreq            3.1.1
pyspellchecker      0.9.0                  VISUALIZACIÓN / INFRA
                                           wordcloud           1.9.6
                                           networkx            3.3
                                           pytest              7.4.4
                                           notebook            7.2.2
                                           IPython             8.27.0
```

¹ Instalado como dependencia de BERTopic. El estudio usa `sklearn.cluster.HDBSCAN`.

**Modelos en caché** (`~/.cache/huggingface`, `~/.cache/fasttext`):

| Modelo | Tamaño |
|---|---|
| `BAAI/bge-m3` | 4.353 MB |
| `cardiffnlp/twitter-xlm-roberta-base-sentiment` | 2.126 MB |
| `papluca/xlm-roberta-base-language-detection` | 1.074 MB |
| `lid.176.bin` (fastText) | 131 MB |
| **Total** | **7.684 MB** |

**Huella en disco:** 1.136 MB de paquetes + 7.684 MB de modelos = **8,8 GB**.
Los modelos se cachean fuera del repositorio y se comparten entre proyectos.

---

## Referencias metodológicas

**Texto aleatorio y leyes de frecuencia**
- Miller, G. A. (1957). Some effects of intermittent silence. *American Journal of Psychology*, 70(2).
- Mandelbrot, B. (1953). An informational theory of the statistical structure of language.
- Ferrer-i-Cancho, R., & Solé, R. (2001). Two regimes in the frequency of words and the origins of complex lexicons. *Journal of Quantitative Linguistics*.
- Piantadosi, S. T. (2014). Zipf's word frequency law in natural language: A critical review. *Psychonomic Bulletin & Review*, 21(5).

**Teoría de la información y estimación de entropía**
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System Technical Journal*.
- Li, M., & Vitányi, P. (2008). *An Introduction to Kolmogorov Complexity and Its Applications*. Springer.
- Miller, G. A. (1955). Note on the bias of information estimates. *Information Theory in Psychology*. [Sesgo del estimador plug-in — §5.3]
- Paninski, L. (2003). Estimation of entropy and mutual information. *Neural Computation*, 15(6). [Ídem]

**Modelos y algoritmos empleados**
- Conneau, A. et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale. [XLM-RoBERTa]
- Joulin, A. et al. (2016). Bag of Tricks for Efficient Text Classification. [fastText]
- Chen, J. et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. [BGE-M3]
- McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection. arXiv:1802.03426.
- Campello, R., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. *PAKDD*. [HDBSCAN]
- Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
- Broder, A. (1997). On the resemblance and containment of documents. *Compression and Complexity of Sequences*. [MinHash]
- Speer, R. et al. `wordfreq`: frecuencias de palabras en 40+ idiomas. [Recurso léxico]

**Referencia conceptual**
- Borges, J. L. (1941). *La Biblioteca de Babel*.

> Las referencias deben verificarse antes de cualquier publicación formal.
> Se listan como punto de partida.

---

**Documento actualizado el 29 de julio de 2026** ·
Versiones, tamaños de caché y conteos verificados sobre el entorno de ejecución
real (Python 3.12.7, Anaconda, Windows 11, CPU 12 núcleos, sin GPU) y sobre las
27 tablas de `outputs/`.
