# 📚 Minando lo Aleatorio — Documento de Contexto y Roadmap

> **Propósito de este documento:** handoff completo del proyecto para continuar el
> desarrollo en Claude Code. Contiene el contexto científico, el estado del arte
> académico, la arquitectura actual, las decisiones metodológicas ya tomadas
> (que NO deben revertirse) y el roadmap técnico con specs de implementación.

**Autor del estudio:** Diego Evans Pérez Paz
**Curso:** Complejidad y Caos — Enero 2026
**Estado actual:** Bloques A, B, C y D completados y **validados sobre el corpus
completo** (13.288.080 caracteres). Veredicto: **24 de 24 métricas
indistinguibles del null model**.
**Última actualización:** 29 de julio de 2026

---

---

## 🔴 ESTADO Y SIGUIENTE PASO — leer esto primero

**Última sesión:** 29 de julio de 2026. Bloques A, B, C, D y E.5 completados y
validados. **24 métricas indistinguibles del null model.**

### Lo que está PENDIENTE DE EJECUTAR ahora mismo

**Nada.** Run All completo el 29 de julio (08:22–09:12), más la celda de la
figura 09 a las 09:50. Las 14 figuras y las 27 tablas están al día.

> **Prueba de reproducibilidad, de regalo:** las cinco tablas de veredicto
> salieron **idénticas** a las de la corrida del 26 de julio, hasta el último
> decimal publicado. La única diferencia en los 27 CSV es ruido de coma flotante
> en el dígito 16 de la divergencia JS. `SEED = 2026` se propaga de verdad.

> **Por qué la 36.** No usaba `bviz.panel_veredicto()`: pintaba Babel en el verde
> por defecto de matplotlib y los controles en rojo/naranja, contradiciendo el
> código de color de las otras trece figuras. Y los títulos llevaban `✓`/`✗`, que
> Segoe UI no tiene, así que salían como cuadros vacíos. **Reejecutar el bloque no
> lo arreglaba** —el defecto estaba en el código, no en los datos—; el Run All del
> 28 de julio produjo una figura idéntica. Corregido en la celda 36 y en
> `babel_viz.panel_veredicto()`.

> **Por qué la 56.** La figura `09` coloreaba los puntos **por cluster de
> HDBSCAN** (`cmap="tab10"`), es decir, entregaba su canal más visible a la
> cantidad que el propio estudio documenta como casi inservible (IC 95% nulo del
> 51% al 96%). Además el azul significaba "cluster 0" en los cuatro paneles, así
> que el mismo color quería decir cosas distintas. Y sobre todo: **UMAP normaliza
> la densidad local por construcción**, de modo que las cuatro nubes salían
> iguales y el hallazgo del bloque —Babel colapsa, 0,888 contra 0,371— no se veía
> en ninguna parte. La figura contradecía su propio pie.
>
> Rehecha: arriba la distribución de similitudes por pares, que **es** la métrica;
> abajo la proyección UMAP coloreada **por entidad**, conservada para exhibir que
> la herramienta popular no muestra el efecto.
>
> **Descartado a propósito:** hexbin (200 puntos por panel, hacen falta miles) y
> curvas de nivel sobre el UMAP —esto último habría sido un error metodológico,
> no solo estético: pondría una superficie de densidad de aspecto científico
> encima de una proyección cuya densidad es un artefacto del algoritmo. Es
> exactamente el error que el estudio existe para documentar.

Ambas verificadas en aislado antes de tocar nada: Babel sale `#2a78d6`, cero
caracteres sin glifo y cero colisiones de etiqueta.

El resto está corrido: `outputs/` tiene las 27 tablas y `figures/` las 14
figuras. Los CSV no cambiaron entre corridas —mismo `SEED = 2026`, mismos
valores— así que ninguna cifra de la documentación se mueve.

### Cómo trabajar con el autor (importante)

> **Diego ejecuta el notebook él mismo.** Cuando se modifiquen celdas: avisarle
> qué cambió y cuánto tarda, y esperar. **No lanzar scripts en paralelo que
> repliquen la lógica del notebook ni escribir en `outputs/`** — se hizo al
> principio y obligaba a correr todo dos veces. Sí es válido usar scripts
> sueltos para *diagnosticar*: cronometrar, calibrar un parámetro, verificar una
> función aislada.

### Cola de trabajo, por orden acordado

1. ~~**`STACK_Y_METODOLOGIA.md`**~~ — **HECHO** el 27 de julio. Cubre los cuatro
   bloques: los siete módulos, BGE-M3, UMAP/HDBSCAN/BERTopic/datasketch/wordfreq,
   la predicción combinatoria (§3.9 de ese documento), el hallazgo del n=10
   (§3.10), el colapso del estimador a k=4 como sexto artefacto (§5.3), los
   negativos documentados y los conteos actuales.
2. ~~**`requirements.txt`**~~ — **HECHO** el 27 de julio. Añadidas las ocho
   dependencias de C y D (`sentence-transformers`, `umap-learn`, `bertopic`,
   `datasketch`, `wordfreq`, `pyspellchecker`, `wordcloud`, `networkx`) y los
   comandos de descarga de los corpus de NLTK, TextBlob y fastText, que antes
   solo vivían en §4.4 de este documento.
3. ~~**Los dos README**~~ — **HECHOS** el 27 de julio, junto con la
   reorganización del repositorio en dos fases (§4). Ahora son **tres**:
   - `README.md` (raíz) — portada del proyecto, en inglés, con las figuras
     embebidas y el contraste v1 → v2. Es la cara pública del repo.
   - `Babel_V1/README.md` — **nuevo**. Qué encontró la fase 1 y qué pasó con
     cada hallazgo.
   - `Babel_V2/README.md` — reescrito: los cuatro bloques, los siete artefactos,
     conteos correctos.
4. **Conclusión global del notebook** ← siguiente, y es el hueco más serio que
   queda. La sección 11 se llama "Conclusiones del Bloque A" y su última
   subsección es "Próximos pasos (Bloque B y siguientes)", obsoleta desde hace
   dos bloques. El estudio tiene 24 métricas repartidas en cinco tablas de
   veredicto (`08`, `12`, `15`, `19`, `24`) y **ninguna que las reúna**. Iría
   bien acompañado de un `28_veredicto_global.csv`.
5. **Commit del v2.** El `.gitignore` ya existe (27 de julio) y está verificado:
   deja fuera `__pycache__/`, `.pytest_cache/` y `.ipynb_checkpoints/`, y
   versiona a propósito los 10 `.txt`, `outputs/` y `figures/` — el porqué de
   cada excepción está comentado en el propio archivo.

   > **Corrección:** versiones previas de este documento decían "nada está
   > trackeado". Es falso. El v1 (notebook y PNG) y los 10 libros **ya estaban
   > commiteados** desde `fee12cd`. Lo que nunca se ha añadido es todo el v2:
   > `Babel_V2/` y `Documents/` enteros, **66 archivos**.

   Pendiente además: `.ipynb_checkpoints/…-checkpoint.ipynb` (7,2 MB, copia
   duplicada del notebook del v1) está trackeado desde el commit inicial. El
   `.gitignore` no lo saca —solo afecta a lo no trackeado—; hay que hacerlo a
   mano con `git rm --cached -r .ipynb_checkpoints/`, que lo deja en disco.
6. **Roadmap E restante:** E.1 (plotly), E.3 (datashader), E.4 (config YAML),
   E.6 (Obsidian). E.2 quedó cubierto por el grafo de bigramas de §10.3.
7. **Verificar las referencias** de §8 antes de cualquier publicación.

### Datos operativos

- **Python:** `%LOCALAPPDATA%\anaconda3\python.exe` (3.12.7). `python` no está en
  el PATH del shell de la herramienta; usar la ruta completa.
- **No hay Node.js**, así que el validador de paletas del sistema de
  visualización no se puede ejecutar aquí.
- **Sin GPU.** 12 núcleos. Caché de Hugging Face: 7,9 GB.
- **Run All completo ≈ 1 hora** (A ~3 min, B ~12, C ~42, D ~4, sección 10 ~4).
- **Tests:** `python -m pytest` desde `Babel_V2/` → 180 en ~15 s.

---

> 📄 **Documento hermano:** [`STACK_Y_METODOLOGIA.md`](STACK_Y_METODOLOGIA.md) —
> inventario razonado del stack, estado del arte de cada herramienta y
> metodología aplicada. `CONTEXT.md` es el handoff operativo (qué hacer y qué no
> revertir); `STACK_Y_METODOLOGIA.md` es la referencia técnica (qué se usó y por
> qué).

## 📑 Tabla de contenidos

1. [Contexto del proyecto](#1-contexto-del-proyecto)
2. [Estado del arte académico](#2-estado-del-arte-académico)
3. [Estado actual del proyecto](#3-estado-actual-del-proyecto)
4. [Arquitectura y estructura de archivos](#4-arquitectura-y-estructura-de-archivos)
5. [Decisiones metodológicas (NO revertir)](#5-decisiones-metodológicas-no-revertir)
6. [Roadmap técnico](#6-roadmap-técnico)
7. [Principios de trabajo](#7-principios-de-trabajo)
8. [Referencias](#8-referencias)

---

## 1. Contexto del proyecto

### 1.1. La pregunta

El proyecto aplica técnicas de minería de texto y NLP a documentos extraídos
aleatoriamente de [libraryofbabel.info](https://libraryofbabel.info), una
implementación digital de la "Biblioteca de Babel" de Borges donde cada libro es
una combinación aleatoria de un alfabeto reducido.

**Pregunta central:** *¿Deja rastros la verdadera aleatoriedad? ¿Puede el caos
generar patrones? ¿O cualquier "patrón" detectado es un artefacto de las
herramientas de análisis?*

### 1.2. El dataset

| Parámetro | Valor |
|-----------|-------|
| Libros analizados | 10 libros aleatorios |
| Archivo crudo por libro | ~1.328.860 caracteres |
| **Cuerpo por libro** (sin metadata) | **1.328.808 caracteres** — idéntico en los 10 |
| Total del corpus | 13.288.080 caracteres |
| Alfabeto | **30 símbolos**: `a-z` (26) + espacio + coma + punto + `\n` |
| Distribución | uniforme sobre los 29 símbolos de contenido: rango observado 3,4351%–3,4578% contra un teórico de 3,4483% |
| **Formato físico** | **410 páginas × 40 líneas × 80 caracteres** |
| Frecuencia real de `\n` | **1/79,06** (no 1/81) |
| Tamaño de comparación | 1.104.297 caracteres por condición, lo fija CESS-ESP |

> ⚠️ **El alfabeto de 30 símbolos solo aplica al CUERPO.** Los archivos
> descargados traen metadata alrededor: una línea de título y un footer
> `Book Location:4-w1-s1-v09`. Sin retirarlos el corpus tiene **44 símbolos**,
> incluidos `B`, `L`, `:`, `-` y los dígitos `0-9`. Lo hace
> `parsear_libro_babel()`, que además registra lo descartado para poder
> auditarlo. Las `Book Location` son **distintas en los 10 libros**, así que no
> son un duplicado trivial pero sí contaminan la calibración del null model.

> ⚠️ **Las 410 páginas no son un detalle decorativo.** Es el número que Borges
> fija en el cuento y el sitio lo implementa literalmente. Las páginas se
> separan con una línea en blanco: 16.400 líneas de contenido + 409 en blanco =
> 16.809 líneas, y `80×16.400 + 16.808 saltos = 1.328.808` exacto. Esas 409
> líneas en blanco suben la frecuencia de `\n` de 1/81 a 1/79,06 — diferencia
> suficiente para que la entropía de Shannon separe Babel de un null model que
> no las replique. Ver §5.9.

### 1.3. Origen: el estudio v1 y su problema

El notebook original (`MINERIA_BIBLIOTECA_BABEL.ipynb`) ejecutaba un pipeline
clásico de NLP y reportaba "hallazgos sorprendentes":

- "~9% de stopwords" detectadas en texto aleatorio.
- Lematización y stemming "exitosos".
- Sentimiento "positivo" (vía TextBlob).
- "Mapa de correlación" mostrando que los libros comparten "ADN estadístico".

**El problema metodológico fundamental:** el estudio no tenía forma de distinguir
si estos patrones eran *propiedades del texto* o *artefactos de las herramientas*.
Sin un grupo de control ni un modelo nulo, las afirmaciones no eran defendibles.

**El Bloque A (ya completado) resuelve esto.** Ver sección 3.

---

## 2. Estado del arte académico

Esta sección fundamenta el enfoque en la literatura. Es directamente relevante
porque **el problema que el v1 enfrentó sin saberlo es un problema clásico y
resuelto en la literatura de lingüística estadística.**

### 2.1. Texto aleatorio y Ley de Zipf: el debate del "mono tipeando"

El hallazgo más importante para este proyecto: **la Ley de Zipf — que las
frecuencias de palabras siguen una distribución de potencias — NO es evidencia de
estructura lingüística, porque el texto puramente aleatorio también la cumple.**

Este resultado se remonta a Miller (1957) y Mandelbrot (1953). Un "mono tipeando
al azar" en una máquina de escribir, que ocasionalmente golpea la barra
espaciadora creando límites de palabra, genera una distribución de frecuencias
Zipfiana **aunque las palabras se creen enteramente al azar, letra por letra**
(Miller, 1957; Li, 1992; Conrad & Mitzenmacher, 2004).

Más aún, Miller observó que el exponente de la ley de potencias en su modelo
estaba cerca de **−1** para 26 letras — el mismo valor icónico que Zipf documentó
para lenguajes naturales. Perline (2015) demostró formalmente que el exponente
tiende a −1 conforme el tamaño del alfabeto crece.

**Implicación directa para este proyecto:** Ferrer-i-Cancho & Solé (2001) y el
trabajo posterior consolidaron la visión de que **las construcciones de texto
aleatorio deben tratarse como *null models*, no como explicaciones realistas de
la estructura del lenguaje.** Como lo expresa Ferrer-i-Cancho:

> *"Debido a su extrema simplicidad, el modelo de Mandelbrot funciona como una
> especie de hipótesis nula para la Ley de Zipf. Si no pudiera refutarse, la
> conclusión sería que es imposible discernir entre una muestra de lenguaje real
> y una secuencia aleatoria de letras y espacios."*

**Esto es exactamente lo que el Bloque A implementa:** el null model como
hipótesis nula formal, y la demostración de que las métricas ingenuas (incluida
Zipf) NO distinguen Babel de ruido — pero las métricas correctas (Heaps β,
entropía relativa, compresión, χ²) SÍ lo hacen frente al lenguaje natural.

### 2.2. La crítica de Howes y por qué importan los controles

Howes (1968) señaló una falacia en el argumento de Miller que es directamente
aplicable al error del v1: el hecho de que el texto aleatorio *reproduzca* una
propiedad del lenguaje no implica que esa propiedad *carezca de significado* en el
lenguaje real. La única forma de saberlo es **comparar contra controles genuinos**
— justamente lo que el Bloque A añade con Moby Dick y CESS-ESP.

### 2.3. Entropía de Shannon, compresión y complejidad de Kolmogorov

El marco teórico para "medir aleatoriedad" proviene de la teoría de la información:

- **Entropía de Shannon** (Shannon, 1948): mide la incertidumbre promedio por
  símbolo de una fuente probabilística. Texto aleatorio uniforme maximiza la
  entropía; el lenguaje natural queda por debajo por su redundancia.

- **Complejidad de Kolmogorov**: la longitud del programa más corto que genera una
  secuencia. Una secuencia es aleatoria si no puede describirse con nada más corto
  que ella misma — es decir, **si no puede comprimirse** (Li & Vitányi).

- **Compresión como proxy práctico:** dado que la complejidad de Kolmogorov es
  incomputable, la compresión sin pérdida (gzip, bz2, lzma, zstd) se usa como
  estimador práctico. Si una secuencia comprime bien, tiene patrones; si no
  comprime, es efectivamente aleatoria. La relación formal es
  *"entropía = complejidad de Kolmogorov esperada"*.

> ⚠️ **Matiz académico importante** (Vitányi; alignmentforum 2022): la entropía de
> Shannon *subestima* la estructura determinista y la compresión estadística mide
> entropía, no complejidad algorítmica pura. Para este proyecto esto NO es un
> problema — usamos ambas como métricas convergentes y las validamos contra
> controles, que es la práctica estándar defendible. Pero conviene mencionarlo en
> el paper para mostrar conciencia de las limitaciones.

### 2.4. Leyes de Heaps

La Ley de Heaps (crecimiento del vocabulario $V = K \cdot N^\beta$) es un
discriminador más robusto que Zipf: en lenguaje natural $\beta \approx 0.4-0.6$
(el vocabulario se satura), mientras que en texto aleatorio $\beta \approx 1.0$
(casi cada token nuevo es una "palabra" nueva, porque el espacio de combinaciones
es enorme). **Esta métrica sí separa limpiamente Babel del lenguaje natural**, a
diferencia de Zipf.

### 2.5. Síntesis: qué aporta este proyecto

El proyecto no reinventa la teoría, pero la aplica de forma pedagógica y rigurosa
a un caso concreto y borgeano. Su valor está en:

1. Demostrar empíricamente, con un dataset concreto, que las herramientas de NLP
   "encuentran" lenguaje donde no lo hay.
2. Contrastar múltiples métricas (Zipf que falla, Heaps que funciona) para enseñar
   *cuáles* métricas discriminan aleatoriedad y cuáles no.
3. Usar el aparato completo de null model + controles + Monte Carlo, que es
   metodología de primer nivel raramente aplicada en trabajos introductorios.

---

## 3. Estado actual del proyecto

### 3.1. Bloque A — COMPLETADO Y VALIDADO ✅

El notebook `MINERIA_BIBLIOTECA_BABEL_v2.ipynb` (41 células) ejecutado sin errores
sobre el **corpus completo**: 10 libros, 13.288.080 caracteres de cuerpo,
normalizados a 1.104.297 por condición.

**Fundamentos estadísticos:**
- ✅ Null model calibrado al alfabeto real Y al formato físico (410×40×80).
- ✅ Controles de lenguaje natural: Moby Dick (EN) + CESS-ESP (ES), dominio público.
- ✅ **7 métricas formales**: entropía de Shannon, ratios de compresión
  (4 algoritmos), Zipf, Heaps, χ² de uniformidad, y sentimiento (polaridad y
  subjetividad, incorporados al Monte Carlo por el principio rector §5.8).
- ✅ Veredicto Monte Carlo con N=100 réplicas a tamaño completo.

**Pipeline clásico conservado** (requisito del estudio — todo lo del v1 se mantiene):
- ✅ Limpieza, tokenización, stopwords multi-idioma, stemming, lematización,
  frecuencias, BoW/TF-IDF, sentimiento.
- ✅ Ejecutado sobre las 4 condiciones en paralelo para comparación.

### 3.2. El veredicto

**Babel es indistinguible del null model en las 7 métricas.**

| Métrica | Babel | Null μ | IC 95% nulo | Percentil |
|---------|-------|--------|-------------|-----------|
| Entropía relativa (%) | 99,7454 | 99,7454 | [99,745, 99,746] | 53 |
| Compresión gzip | 0,6512 | 0,6513 | [0,651, 0,651] | 49 |
| Pendiente Zipf | −0,9386 | −0,9372 | [−0,949, −0,927] | 34 |
| Heaps β | 0,9697 | 0,9695 | [0,968, 0,974] | 70 |
| χ² p-valor | 0,4687 | 0,3784 | [0,015, 0,952] | 65 |
| Sentimiento (polaridad) | 0,2254 | 0,2051 | [0,085, 0,320] | 60 |
| Sentimiento (subjetividad) | 0,5167 | 0,5214 | [0,438, 0,605] | 46 |

Y el lenguaje natural sí se separa:

| | Babel | Null | Moby Dick | CESS-ESP |
|---|---|---|---|---|
| Entropía relativa | 99,7% | 99,7% | 71,4% | 67,9% |
| Compresión gzip / bz2 | 0,651 / 0,622 | 0,651 / 0,622 | 0,405 / 0,308 | 0,368 / 0,279 |
| Heaps β | 0,971 | 0,969 | 0,762 | 0,790 |
| **Tipos únicos / tokens** | **0,769** | **0,771** | **0,084** | **0,124** |
| % stemming modificado | 8,48 | 8,56 | 32,15 | 80,33 |
| % lematizado | 1,55 | 1,61 | 7,94 | 35,57 |
| χ² p-valor | 0,85 | 0,62 | ≈ 0 | ≈ 0 |

### 3.3. Los hallazgos del v1, reinterpretados

| Afirmación original | Babel | Null | Veredicto |
|---------------------|-------|------|-----------|
| "~9% de stopwords" | 7,606% sin filtro · 0,115% con ≥3 | 7,494% · 0,130% | Número **real**; el null lo reproduce → combinatoria |
| "Stemming exitoso" | 8,48% | 8,56% | Transformación mecánica |
| "Lematización exitosa" | 1,55% | 1,61% | Ídem |
| "Menos frecuentes: siempre 3 chars" | mínimo 1 | mínimo 1 | No se sostiene |
| "Sentimiento positivo" | +0,225 | +0,205 | Idéntico al null (ver §3.5) |
| "Los libros comparten ADN estadístico" | — | — | Lo comparten con el null igual |

> **Corrección respecto a versiones previas de este documento:** se afirmaba que
> el 9% eran "tokens cortos, no stopwords reales (~0,07%)". Con la tokenización
> corregida (§5.10) eso ya no se sostiene: el 7,6% **sí** son stopwords según
> NLTK, secuencias de 1-2 letras que son palabras funcionales en alguno de los 9
> idiomas cargados (`a`, `y`, `de`, `el`). Lo que las convierte en artefacto es
> que el null model las produce a la misma tasa.

### 3.4. Zipf no discrimina — el resultado que valida §2.1

Babel: pendiente **−0,940** (R²=0,858). CESS-ESP: **−0,968** (R²=0,988).

El texto aleatorio cumple la ley de Zipf. Es Miller (1957) y Mandelbrot (1953)
reproducido sobre un corpus concreto, y confirma empíricamente lo que §2.1 plantea
como núcleo del estado del arte.

Heaps sí separa, y su **comportamiento ante el tramo de ajuste** separa aún mejor:

| Condición | β (N≥10) | β (N≥1.000) | β (N≥50.000) |
|-----------|----------|-------------|--------------|
| Babel | 0,971 | 0,963 | 0,972 |
| Null | 0,969 | 0,964 | 0,972 |
| Control_EN | 0,762 | 0,658 | 0,549 |
| Control_ES | 0,790 | 0,735 | 0,744 |

β es **plano** en Babel (el vocabulario crece linealmente de verdad) y
**decreciente** en lenguaje natural (saturación real del léxico). El β≈0,4-0,6 de
manual es un asintótico, no un valor que deba salir a cualquier escala.

### 3.5. TextBlob: alucina más cuanto menos texto real tiene

Babel +0,225 y null +0,205 — la misma polaridad. Pero los controles dan **0,083
(Moby Dick)** y **0,039 (CESS-ESP)**, prácticamente neutros.

**TextBlob lee más sentimiento positivo en texto aleatorio que en literatura.**
Esto es lo que justifica la decisión §5.2 de conservarlo.

### 3.6. Cuánto significado emerge por azar

Cruzando los tokens de Babel contra un léxico construido con los propios corpus de
control: **602 palabras reales en Babel, 641 en el null model.** La misma tasa, y
ninguna supera las 5 letras. En un millón de caracteres, las "islas de significado"
topan en cinco.

(Medición preliminar; D.1 la formalizará con diccionarios externos.)

### 3.7. Cuatro artefactos del instrumento, documentados

Las corridas intermedias produjeron cuatro "hallazgos" que resultaron ser defectos
del aparato de medición. Se dejan documentados porque **son** la lección del
proyecto:

1. **Zipf en percentil 99** — el ajuste incluía la cola de hapax, casi todo el
   vocabulario en texto aleatorio y una recta horizontal en log-log. → §5.11
2. **χ² en percentil 100** — el null model se calibraba con las frecuencias de una
   muestra de 9.475 caracteres, heredaba su ruido y le sumaba ruido fresco encima.
   El null *rechazaba* uniformidad (p=0,040) mientras Babel no (p=0,491). Se
   resuelve al calibrar con los 13,3M: el rango de probabilidades pasa de
   3,02%–3,82% a 3,4351%–3,4578%.
3. **Entropía en percentil 100** — el null no replicaba la paginación. → §5.9
4. **Sentimiento de Babel al doble que el del null** (0,486 vs 0,195) — la limpieza
   borraba comas y puntos en vez de tratarlos como frontera. → §5.10

Además, `babel_core` y `babel_pipeline` contaban tokens **distintos del mismo
texto** (111.797 vs 48.961 en Babel) porque tokenizaban de forma diferente.
Unificado en §5.10.

### 3.8. Bloque B — NLP moderno (COMPLETADO ✅)

Módulo `babel_transformers.py`. **Veredicto: 5 de 5 métricas indistinguibles.**

| Métrica | Babel | Null μ | IC 95% nulo | Pctil | Ctrl_EN | Ctrl_ES |
|---------|-------|--------|-------------|-------|---------|---------|
| Fragmentación (tokens/char) | 0,6029 | 0,6019 | [0,597, 0,606] | 60 | 0,270 | 0,276 |
| XLM-R prob. de idioma | 0,5417 | 0,5323 | [0,479, 0,604] | 68 | 0,953 | 0,991 |
| fastText prob. de idioma | 0,1099 | 0,1102 | [0,105, 0,116] | 43 | 0,967 | 0,963 |
| Sentimiento moderno (polaridad) | −0,1113 | −0,1109 | [−0,119, −0,103] | 40 | −0,184 | −0,190 |
| Sentimiento moderno (neutral) | 0,4472 | 0,4474 | [0,442, 0,452] | 48 | 0,316 | 0,391 |

**La pregunta del roadmap, contestada:** fastText asigna **0,110** de probabilidad
a que un libro de Babel sea algún idioma real, contra 0,967 y 0,963 en los
controles. Factor de casi nueve.

**El hallazgo del bloque: los dos detectores fallan de forma opuesta.** fastText
reparte y se queda en 0,11 — reconoce que no sabe. XLM-R concentra 0,54 en
**hindi**, en el 100% de las ventanas, con la misma consistencia con la que
etiqueta el español real. Un transformer de 278M de parámetros está seguro y
equivocado; un clasificador lineal sobre n-gramas de caracteres es honesto. La
capacidad del modelo no predice su calibración.

> Trampa a evitar: ese 100% de acuerdo entre ventanas, sin el null model al
> lado, sería trivial de leer como "el modelo detecta estructura consistente".
> La entropía de idiomas es **cero**, igual que en el español real. Solo el null
> model revela que es sesgo del clasificador. La hipótesis inicial —que ante
> ruido el modelo repartiría entre idiomas— quedó refutada.

**El sentimiento moderno invierte a TextBlob:**

| | Babel | Null | Moby Dick | CESS-ESP |
|---|---|---|---|---|
| TextBlob (Bloque A) | +0,225 | +0,205 | +0,083 | +0,039 |
| XLM-R sentiment | −0,111 | −0,111 | −0,184 | −0,190 |

Los dos alucinan, en sentidos opuestos. Lo que importa: el modelo moderno da al
ruido más masa **neutral** (0,447) que a los controles (0,316 y 0,391), es
decir, se compromete **menos** cuando no hay nada que leer. TextBlob hacía justo
lo contrario.

**Fragmentación subléxica**, el discriminador barato: 0,603 tokens/char en Babel
contra 0,270 y 0,276. Análogo subléxico de la compresión del Bloque A, sale
gratis con el tokenizador y no requiere una sola pasada del modelo.

---

### 3.9. Bloque C — Estructura secuencial (C.5 y C.6 COMPLETADOS ✅)

Módulo `babel_secuencia.py`. **Veredicto: 3 de 3 métricas indistinguibles.**

| Métrica | Babel | Null μ | IC 95% nulo | Pctil | Ctrl_EN | Ctrl_ES |
|---|---|---|---|---|---|---|
| Ganancia de información (k=1) | 0,0005 | 0,0005 | [0,0005, 0,0006] | 71 | 0,909 | 1,085 |
| Ganancia de información (k=2) | 0,0155 | 0,0156 | [0,0154, 0,0159] | 29 | 1,620 | 1,683 |
| Ganancia de información (k=3) | 0,5255 | 0,5255 | [0,5243, 0,5269] | 53 | 2,226 | 2,266 |

#### El artefacto más grave del proyecto, y el mejor documentado

| Condición | k=1 | k=2 | k=3 | **k=4** | obs/contexto k=4 |
|---|---|---|---|---|---|
| Babel | 0,0005 | 0,0155 | 0,5258 | **3,7446** | 2,0 |
| Null | 0,0005 | 0,0157 | 0,5254 | **3,7442** | 2,0 |
| Control_EN | 0,9087 | 1,6198 | 2,2263 | 2,6823 | 17,9 |
| Control_ES | 1,0852 | 1,6835 | 2,2663 | 2,7475 | 21,2 |

**A orden 4 el estimador plug-in atribuye a Babel un 36% MÁS de estructura
secuencial que al español.** Con 555.976 contextos posibles y 2,0 observaciones
por contexto, las distribuciones empíricas parecen deterministas y la entropía
condicional se desploma. El lenguaje natural resiste mejor porque concentra su
masa en muchos menos contextos distintos.

El null model da 3,7442 frente a 3,7446. **Sin él, este bloque habría publicado
que el ruido tiene más memoria que la literatura.**

#### Perplejidad (C.5) — un gradiente, no un interruptor

Modelo de caracteres de orden 5 entrenado sobre el 80% de CESS-ESP:

| Texto evaluado | bits/char | Perplejidad | vs uniforme (29) |
|---|---|---|---|
| Español retenido | 2,261 | 4,8 | 0,17× |
| Inglés (Moby Dick) | 5,990 | 63,6 | 2,19× |
| Babel | 7,912 | 240,9 | 8,31× |
| Null model | 7,906 | 239,8 | 8,27× |

El inglés da valor a la escala: un modelo de español lo predice mal (63,6, por
encima del techo uniforme de 29) pero **cuatro veces mejor que a Babel**, porque
comparte alfabeto y estadística de bigramas. Babel llega a 8,31× peor que no
saber nada: el modelo está *confiadamente equivocado*, no ignorante.

#### Matrices de transición (C.6)

| Comparación | Divergencia Jensen-Shannon |
|---|---|
| Babel vs Null | **0,0003** |
| Babel vs Control_ES | 0,5424 |
| Null vs Control_ES | 0,5428 |

> **Matiz:** la divergencia contra el español mezcla dos efectos, porque
> CESS-ESP en crudo usa 96 símbolos (mayúsculas, dígitos, puntuación) frente a
> los 29 de Babel. Parte de ese 0,54 es diferencia de alfabeto. La comparación
> limpia —mismo alfabeto exacto— es Babel contra Null: 0,0003.

### 3.10. Bloque C — Modelado no supervisado (C.1-C.4 COMPLETADOS ✅)

Módulo `babel_embeddings.py`, sobre **BGE-M3** (BAAI, 2024; 568M parámetros,
1024 dimensiones). **Veredicto: 5 de 5 métricas indistinguibles.**

| Métrica | Babel | Null μ | IC 95% nulo | Pctil |
|---|---|---|---|---|
| Similitud media entre chunks | 0,8878 | 0,8874 | [0,884, 0,891] | 53,3 |
| Dispersión de similitudes | 0,0239 | 0,0241 | [0,022, 0,028] | 60,0 |
| Dimensión efectiva | 86,90 | 86,78 | [83,86, 89,15] | 46,7 |
| Clusters HDBSCAN | 2,5 | 2,27 | [0,70, 3,65] | 66,7 |
| % puntos sin cluster | 69,25 | 71,63 | [51,25, 95,98] | 33,3 |

#### La ausencia de estructura se manifiesta como uniformidad, no como dispersión

| | Babel | Null | Moby Dick | CESS-ESP |
|---|---|---|---|---|
| Similitud media entre chunks | **0,885** | 0,889 | 0,464 | **0,371** |
| Similitud con el vecino más cercano | 0,922 | 0,923 | 0,617 | 0,566 |
| Dispersión de similitudes | 0,025 | 0,022 | 0,060 | 0,062 |

Los fragmentos de Babel se parecen entre sí **más del doble** que los de la
literatura española. Para el modelo, dos trozos de ruido son casi el mismo
objeto: son intercambiables porque no hay nada que los distinga. Babel no "deja
de agruparse": **colapsa en una bola**.

> Esto invierte la intuición de partida y conviene tenerlo escrito antes de
> mirar cualquier resultado de clustering en bloques futuros.

#### Dos negativos documentados

1. **La dimensión efectiva no discrimina** (86,90 / 86,78 / 82,66 / 88,20). Se
   diseñó esperando que la nube de ruido ocupase más direcciones; en 1024
   dimensiones la cola de autovalores domina la razón de participación por igual
   en las cuatro condiciones.

2. **El recuento de clusters de HDBSCAN es casi inservible.** El IC 95% nulo del
   porcentaje de puntos sin cluster va de **51,3% a 96,0%** sobre 15 réplicas del
   *mismo* proceso. El de número de clusters, [0,70, 3,65], se solapa con los 4
   de ambos controles. Confirmación empírica de que HDBSCAN siempre devuelve
   algo y ese algo varía enormemente sin que cambie el texto.

#### BERTopic halla más tópicos en el ruido que en el español (C.1)

| Condición | Tópicos | % outliers |
|---|---|---|
| **Babel** | **3** | 64,5 |
| Null | 2 | 85,0 |
| Control_EN | 2 | 45,0 |
| **Control_ES** | **2** | 20,0 |

Tres tópicos en ruido puro contra dos en Cervantes. Mismo argumento que §5.2:
la herramienta no falla en silencio, falla **con aplomo**. Lo que sí ordena bien
las condiciones es el % de outliers (20 → 45 → 64,5 → 85).

#### El "ADN estadístico compartido", liquidado (C.4)

MinHash sobre shingles de 8 caracteres, 128 permutaciones: **Jaccard = 0,000000
en los 21 pares**. Entre libros de Babel, entre Babel y el null, entre Babel y el
español. Ni una sola secuencia de ocho caracteres compartida.

---

### 3.11. Bloque D — Significado emergente (COMPLETADO ✅)

Módulo `babel_lexico.py`. **Veredicto: 4 de 4 métricas indistinguibles.**

| Métrica | Babel | Null μ | IC 95% nulo | Pctil |
|---|---|---|---|---|
| Tasa de palabras reales (%) | 4,2714 | 4,2787 | [4,159, 4,412] | 49 |
| Palabras emergentes | 3.758,7 | 3.768,2 | [3.655,8, 3.888,1] | 43 |
| Palabras distintas | 2.907,9 | 2.918,1 | [2.838,3, 2.990,6] | 41 |
| Longitud máxima | 5,5 | 5,56 | [5,00, 6,00] | 46 |

#### El resultado más fuerte del estudio: la predicción en cerrado

| Condición | Tokens (≥3) | Long. media | Predichas | Observadas | **Obs/Pred** | Máx. |
|---|---|---|---|---|---|---|
| **Babel** | 87.686 | 8,75 | 3.860,9 | 3.766 | **0,98** | 5 |
| Null | 87.997 | 8,74 | 3.864,0 | 3.798 | **0,98** | 5 |
| Control_EN | 153.493 | 4,38 | 16.725,3 | 150.914 | 9,02 | 18 |
| Control_ES | 123.441 | 4,89 | 10.449,1 | 117.267 | 11,22 | 19 |

`P(k) = N_k / 26^k` predijo **3.861** palabras reales antes de contar ninguna;
aparecieron **3.766**. Sin ajustar parámetros, sin usar el null model, sin usar
ninguna herramienta de NLP. Y la longitud media de token, predicha como
geométrica de parámetro 3/29 en 8,67, se midió en 8,75.

> **Esto cambia la categoría de la conclusión.** En A, B y C se demuestra que la
> emergencia de estructura es *indistinguible* del azar. Aquí se demuestra que
> **es azar y se conoce la fórmula generadora.** Es el único resultado del
> estudio que no depende de ningún modelo nulo ni de ninguna herramienta.

Los controles producen **entre 9 y 11 veces más** palabras válidas de las que el
azar explicaría: esa razón es la firma de la intención.

#### Por qué las islas de significado tienen techo de longitud

| k | N_k | 26^k | P(k) |
|---|---|---|---|
| 3 | 6.395 | 17.576 | **0,364** |
| 5 | 20.753 | 1,19·10⁷ | 0,00175 |
| 9 | 42.816 | 5,43·10¹² | **7,89·10⁻⁹** |

Con tres letras, el **36% de todas las cadenas posibles** son palabras reales.
Con nueve, 8 entre mil millones. El diccionario crece, pero el espacio de
cadenas crece por 26 en cada paso: esa carrera perdida es toda la explicación.
La palabra más larga de Babel tiene 5 letras; el español llega a 19.

#### D.3 — la métrica más débil del estudio

| Condición | Similitud máxima media |
|---|---|
| Babel | 0,4252 |
| Null | 0,4314 |
| Control_EN | 0,5108 |
| Control_ES | 0,4621 |

Babel alcanza 0,425 frente a sondas como *"el amor es eterno mientras dura"*; el
español, 0,462. Un 8% de diferencia. **No significa que Babel casi contenga esas
ideas: significa que la métrica no sirve para lo que parecía.** BGE-M3 concentra
todo el texto en un cono estrecho, el suelo para ruido puro ya está en 0,42 y el
rango útil es minúsculo.

> **Confundente detectado:** el inglés puntúa más alto (0,5108) que el español
> (0,4621) frente a sondas *en español*. Causa probable: CESS-ESP es prensa
> económica, temáticamente lejos de sondas sobre amor, tiempo y muerte, mientras
> Moby Dick es prosa narrativa. La comparación entre controles no es limpia. Se
> reporta completa porque el negativo también es resultado, pero no sostiene
> ninguna conclusión por sí sola.

### 3.12. Las figuras del v1, revisitadas — y el hallazgo del n=10

Módulo `babel_viz.py`. **Sección 10 del notebook: ejecutada** (26 de julio).
Genera `11_correlacion_palabras.png`, `12_paisaje_correlacion_3d.png`,
`13_nube_palabras_raras.png`, `14_grafo_bigramas.png` y la tabla
`27_correlacion_palabras.csv`.

§5.7 exige conservar todos los pasos del pipeline original, y dos figuras del v1
se habían quedado fuera: el mapa de correlación entre palabras (con sus seis
vistas 3D) y la nube de palabras menos frecuentes.

#### El hallazgo: el número estrella del v1 es la predicción del azar

El v1 correlacionaba **palabras entre sí** (`df_tfidf_seleccion.corr()`, o sea
las columnas del TF-IDF) **a lo largo de los 10 documentos**. Ese 10 es el
tamaño de muestra de cada coeficiente, y ahí está todo:

Con n=10, bajo no-asociación, `P(|r| > 0,5) = 0,1411`. Sobre las C(100,2) = 4.950
parejas de palabras salen **699 pares por puro azar**.

> **El v1 reportó "Picos detectados (correlación > 0.5): 676".**
> Su hallazgo más vistoso está un **3,2%** por debajo de lo que predice el ruido.

Verificado además empíricamente sobre el corpus:

| | picos \|r\|>0,5 | % de pares | esperado por azar |
|---|---|---|---|
| Babel | 683 | 13,8% | 14,1% |
| Null | 697 | 14,1% | 14,1% |
| **Moby Dick partido en 10** | **1.082** | **21,9%** | 14,1% |

> Cifras de `outputs/27_correlacion_palabras.csv`, la corrida real del 26 de
> julio. Versiones previas de este documento daban 1.197 y 24,2% para Moby
> Dick: eran una estimación escrita antes de ejecutar la sección 10. El
> argumento no cambia — el lenguaje natural sigue superando con holgura el
> 14,1% que predice el azar.

**La métrica no está rota**: el lenguaje natural la supera con holgura. El
problema eran los 10 documentos. Con n=50 la tasa de falsos picos cae al 0,02%.

Este resultado tiene la misma estructura que el de §3.11: no dice solo "Babel es
indistinguible del azar", dice **cuánto azar había que esperar y por qué**.

#### Sistema visual

`babel_viz.py` centraliza paleta, tipografía y cromo. Color ligado a la entidad:
Babel azul en todas las figuras, null model en el gris de fondo, controles en
aqua y amarillo. `aplicar_estilo()` se llama en la celda de imports, así que un
Run All completo unifica las catorce figuras.

Dos correcciones de codificación respecto a versiones previas:

1. **La correlación va en paleta divergente** (azul ↔ gris ↔ rojo), no en
   `viridis`. Una correlación tiene signo: el cero significa "nada" y los
   extremos, cosas opuestas. Un arcoíris tiñe de verde el punto medio, como si
   el 0,0 tuviese contenido.
2. **La matriz de transición de C.6 pasó de `viridis` a rampa de un solo tono.**
   Es magnitud, no divergente.

> Reglas del sistema visual, a no romper al añadir figuras: nunca doble eje;
> magnitud en un solo tono claro→oscuro; rejilla y ejes en línea **sólida** de un
> pelo (la discontinua se lee como umbral); leyenda siempre con dos o más series
> y etiquetas directas solo en lo que importa. **No hay Node en la máquina**, así
> que el validador de paletas no se puede correr: se usa la paleta de referencia
> sin modificar y en su orden de slots documentado.

---

## 4. Arquitectura y estructura de archivos

**Reorganizado el 27 de julio de 2026** en dos fases explícitas. Antes el v1 vivía
disperso en la raíz (notebook, 6 PNG y los 10 `.txt` mezclados); ahora cada fase
es una carpeta con su propio README.

```
Mining the Random - Library of Babel/          # raíz del repositorio
├── Babel_V1/                                  # FASE 1 — el estudio original
│   ├── MINERIA_BIBLIOTECA_BABEL_v1.ipynb      # Conservado, con sus outputs embebidos
│   ├── figures/                               # Las 6 figuras del v1
│   └── README.md                              # Qué encontró y qué pasó con cada hallazgo
│
├── Babel_V2/                                  # FASE 2 — la reconstrucción
│   ├── MINERIA_BIBLIOTECA_BABEL_v2.ipynb      # Notebook principal (dashboard, 76 celdas)
│   ├── README.md                              # Los cuatro bloques, con figuras
│   ├── requirements.txt                       # A, B, C y D (actualizado el 27/07)
│   ├── src/                                   # LÓGICA (motor reutilizable)
│   │   ├── babel_core.py                      # Null model, controles, métricas
│   │   ├── babel_pipeline.py                  # Pipeline clásico de text mining
│   │   ├── babel_transformers.py              # Bloque B: NLP moderno
│   │   ├── babel_secuencia.py                 # Bloque C: Markov y perplejidad
│   │   ├── babel_embeddings.py                # Bloque C: BGE-M3, UMAP, BERTopic
│   │   ├── babel_lexico.py                    # Bloque D: palabras emergentes
│   │   └── babel_viz.py                       # Paleta, estilo, figuras del v1
│   ├── tests/                                 # 180 tests (pytest, ~15 s)
│   │   ├── conftest.py, test_parseo.py, test_null_model.py,
│   │   ├── test_metricas.py, test_pipeline.py, test_regresiones.py,
│   │   └── test_transformers.py, test_secuencia.py, test_embeddings.py,
│   │       test_lexico.py
│   ├── pytest.ini
│   ├── outputs/                               # 27 tablas CSV
│   └── figures/                               # 14 figuras PNG
│
├── Documents/
│   ├── CONTEXT.md                             # Este documento (handoff operativo)
│   └── STACK_Y_METODOLOGIA.md                 # Referencia técnica (stack + metodología)
│
├── data/                                      # Los 10 libros de Babel (13 MB)
├── README.md                                  # Portada del repositorio, con figuras
└── .gitignore
```

> ⚠️ **Los `.txt` ya NO están en la raíz: viven en `data/`.** El notebook los
> auto-detecta buscando en orden: `data/` dentro del proyecto → **`data/` de la
> raíz del repo** → carpeta padre → el propio proyecto; gana la primera con
> archivos `.txt`. El segundo candidato se añadió el 27 de julio y es **el único
> cambio de código que exigió la reorganización**; la lista es estrictamente
> aditiva, así que cualquier disposición anterior sigue funcionando.
> `tests/conftest.py` replica la misma lista.

> ⚠️ **Trampa conocida en el último candidato.** `RUTA_PROYECTO` es `Babel_V2/`,
> que contiene `requirements.txt` — y `.glob("*.txt")` lo captura. Si `data/`
> desapareciera, la detección "encontraría" 1 archivo y el parser reventaría con
> un `ValueError` que no apunta a la causa. Pasó al mover los libros, antes de
> actualizar `conftest.py`. No se ha cambiado la semántica de la búsqueda porque
> es una decisión de diseño; queda anotado.

> ⚠️ **El notebook del v1 no corre desde su nueva ubicación** — ni corría desde
> la anterior. Su celda 2 tiene la ruta absoluta
> `C:/Users/diego.perezp/Documents/DATA SCIENCE/Mineria de Texto/MUESTRA_BIBLIOTECA_BABEL/`,
> que no apunta al repositorio. Se conserva como registro histórico con los
> outputs embebidos, no como algo reejecutable. La ruta se dejó tal cual en vez
> de parchearla en silencio.

### 4.1. Filosofía arquitectónica

**Separación estricta lógica / presentación:**
- Los `.py` en `src/` contienen toda la lógica: funciones puras, testeables,
  reutilizables. NO contienen visualización ni narrativa.
- El notebook es la capa de presentación: importa los módulos, orquesta el
  análisis, muestra tablas y gráficos, y contiene la narrativa markdown.

Esto se decidió porque el v1 mezclaba lógica y presentación en celdas de cientos
de líneas, lo que lo hacía imposible de mantener y testear.

### 4.2. `babel_core.py` — API principal

| Función | Propósito |
|---------|-----------|
| `parsear_libro_babel(texto, archivo)` | **Separa cuerpo de metadata** (título + `Book Location`); devuelve lo descartado para auditoría |
| `cargar_corpus_babel(ruta, parsear=True)` | Carga y auto-detecta los .txt; parsea por defecto |
| `perfil_alfabeto(texto)` | Inventario, conteos y probabilidades de caracteres |
| `perfil_formato(texto)` | **Mide** ancho de línea, líneas por página y frecuencia de `\n` |
| `estimar_prob_contenido(corpus)` | Calibra probabilidades del alfabeto |
| `generar_babel_sintetico(n, probs, ancho_linea, lineas_por_pagina, seed)` | Genera 1 texto del null model, **con paginación** |
| `generar_replicas_null(N, n, probs, ancho_linea, lineas_por_pagina)` | Genera N réplicas Monte Carlo |
| `cargar_control_ingles(n)` | Moby Dick vía NLTK (1.242.990 chars) |
| `cargar_control_espanol(n)` | CESS-ESP vía NLTK (1.104.297 chars) |
| `entropia_shannon(texto, n)` | Entropía de n-gramas de caracteres |
| `entropia_maxima(n_simbolos, n)` | Techo teórico `n·log2(A)` |
| `ratios_compresion(texto, algoritmos)` | gzip, bz2, lzma, zstd; **selectivo** para bucles Monte Carlo |
| `ley_zipf(texto, min_rango, max_rango, excluir_hapax)` | Pendiente Zipf + R² |
| `ley_heaps(texto, n_puntos, n_min)` | Exponente β + R² + `tasa_tipos` (V/N) |
| `chi2_uniformidad(texto, n, excluir, alfabeto)` | Test χ² con **espacio completo de categorías A^n** |

### 4.3. `babel_pipeline.py` — API principal

| Función | Propósito |
|---------|-----------|
| `limpiar_texto(texto, preservar_puntuacion, normalizar_acentos)` | Limpieza; los no alfabéticos pasan a **espacio** (frontera de palabra) |
| `tokenizar(texto)` | Tokenización subléxica: secuencias máximas de `a-z` |
| `construir_set_stopwords(idiomas)` | Set multi-idioma (9 idiomas) |
| `filtrar_stopwords(tokens, sw, min_long)` | Filtro + estadísticas separadas |
| `aplicar_stemming(tokens, idioma)` | Snowball stemmer |
| `cargar_lematizador()` | spaCy con fallback a NLTK |
| `aplicar_lematizacion(tokens, lem, tipo)` | Lematización |
| `analizar_frecuencias(tokens)` | Frecuencias + análisis de hapax |
| `vectorizar(textos, max_features)` | BoW + TF-IDF |
| `analizar_sentimiento(texto)` | TextBlob (como demo de artefacto) |
| `pipeline_completo(...)` | Orquestador de todo el pipeline |

### 4.4. `babel_transformers.py` — API principal (Bloque B)

| Función | Propósito |
|---------|-----------|
| `muestrear_ventanas(texto, n, chars, seed)` | **La unidad de observación del bloque**; quita los saltos de línea por igual en las 4 condiciones |
| `ratio_tokens_por_char(ventanas, tok)` | Fragmentación subléxica + aviso de truncado |
| `cargar_clasificador(modelo, device, batch_size)` | Devuelve `(pipeline, tokenizador)`; CPU por defecto |
| `clasificar_ventanas(ventanas, pipe)` | Distribución `{etiqueta: prob}` por ventana; normaliza `LABEL_0/1/2` |
| `metricas_idioma(distribuciones)` | `prob_top_media` (la que discrimina), entropía, idioma modal |
| `metricas_sentimiento(distribuciones)` | Polaridad definida como P(pos) − P(neg), comparable con TextBlob |
| `analizar_condicion(texto, ...)` | Orquestador: muestrea y aplica los modelos activos |
| `descargar_fasttext(ruta)` | Descarga `lid.176.bin` a `~/.cache/fasttext/`, fuera del repo |
| `cargar_fasttext(ruta)` | Devuelve `None` si falta librería o modelo, en vez de reventar |
| `detectar_idioma_fasttext(ventanas, modelo, k)` | Distribución de idiomas por ventana |

### 4.4. Cómo correr

```bash
pip install -r Babel_V2/requirements.txt
python -m spacy download es_core_news_sm
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
python -c "import nltk; nltk.download('gutenberg'); nltk.download('cess_esp'); nltk.download('stopwords'); nltk.download('wordnet')"
# Abrir Babel_V2/MINERIA_BIBLIOTECA_BABEL_v2.ipynb en Jupyter 7.x y Run All
```

Corrida completa ≈ 3 minutos (~30 s el pipeline clásico, ~110 s el Monte Carlo).
Para smoke test, `MODO_RAPIDO_CHARS = 200_000` en la celda de configuración.

> ⚠️ **Al editar cualquier `.py` de `src/`:** la celda de imports recarga los
> módulos con `importlib.reload` y verifica que el kernel tenga la API esperada.
> Sin eso Python sirve la versión cacheada y las funciones nuevas fallan con
> `AttributeError` varias celdas más abajo, con un mensaje que no apunta a la
> causa. Esto costó una corrida completa.

---

## 5. Decisiones metodológicas (NO revertir)

> ⚠️ **Para Claude Code:** estas decisiones se tomaron deliberadamente con el autor.
> No las cambies sin consultarle. Cada una tiene una justificación explícita.

1. **Umbral de longitud de token = `>=3`** (estándar en text mining). Se acompaña de
   una tabla de sensibilidad (sin filtro / ≥2 / ≥3 / ≥5) para blindar la elección.
   **Razón:** el v1 fusionaba "tokens cortos" con "stopwords" bajo una etiqueta,
   generando el falso 9%. Siempre reportar ambas métricas por SEPARADO.

2. **TextBlob se conserva a propósito** como demostración de que la herramienta
   alucina sentimiento sobre ruido. NO es una medición válida. **No lo elimines**;
   es un punto pedagógico. En el Bloque B se le añadirá un contraste con modelos
   modernos, pero el TextBlob original se queda como referencia del artefacto.

3. **Null model calibrado con la distribución real de Babel**, no uniforme puro.
   **Razón:** para ser un gemelo estadístico legítimo debe replicar la ley
   generadora observada, incluyendo el `\n` posicional cada 80 caracteres.

4. **Controles de dominio público exclusivamente** (Moby Dick 1851, CESS-ESP).
   **Razón:** validez académica y reproducibilidad. Se descartaron Huxley y Asimov
   por copyright (Brave New World no entra a dominio público hasta 2028; Asimov
   sigue protegido por décadas). NO uses textos con copyright.

5. **N=100 réplicas Monte Carlo, todas a tamaño completo.** Confirmado con el autor
   (se bajó de 200 a 100 para balancear tiempo/precisión).

6. **4 algoritmos de compresión** (gzip, bz2, lzma de stdlib + zstandard de Meta).
   **Razón:** zstandard aporta validez de vanguardia; los de stdlib dan robustez.

7. **Todos los pasos del pipeline clásico del v1 se conservan.** Requisito explícito
   del autor: el v2 debe pasar por los elementos básicos de minería de texto y
   preservar lo descubierto en el v1, aunque sea para reinterpretarlo.

8. **PRINCIPIO RECTOR:** cualquier "patrón" o "hallazgo" nuevo (de cualquier bloque
   futuro) DEBE contrastarse contra el null model antes de considerarse real. Esta
   es la lección central del proyecto.

---

> Las siguientes se tomaron durante la validación sobre el corpus completo
> (julio 2026). Cada una nació de un artefacto que se hizo pasar por hallazgo.

9. **El null model replica el FORMATO FÍSICO, no solo el alfabeto.**
   410 páginas × 40 líneas × 80 caracteres, con línea en blanco entre páginas.
   `perfil_formato()` lo mide sobre el corpus y `generar_babel_sintetico()` lo
   reproduce.
   **Razón:** sin la línea en blanco, la frecuencia de `\n` del null queda en
   1/81 contra 1/79,06 del real. Eso solo bastaba para que la entropía diera
   4,8940 en el null contra 4,8944 en Babel y el veredicto lo marcara como
   "distinguible" en percentil 100. Corregido, Babel cae en percentil 53. Es la
   aplicación literal de §5.3: si el null ha de ser un gemelo, tiene que serlo
   también en la ley de formato.

10. **Un token es una secuencia máxima de letras `a-z`. Los no alfabéticos son
    FRONTERA DE PALABRA, no se borran.** Con normalización de acentos previa
    (á→a, ñ→n). Definición idéntica en `babel_core` y `babel_pipeline`.
    **Razón:** el idiom clásico `re.sub(r'[^a-z\s]', '', texto)` que usaba el v1
    pega los tokens vecinos. En lenguaje natural apenas se nota porque la
    puntuación va pegada a un espacio; en Babel la coma y el punto caen en
    posiciones aleatorias, así que el pegado es constante. El sesgo era
    **asimétrico**: largo medio de token en Babel 20,0 contra 8,7, sin mover a
    los controles — justo en la comparación que sostiene el estudio. Además los
    dos módulos contaban distinto el mismo texto (111.797 vs 48.961 tokens en
    Babel), porque `_tokenizar_palabras` ya usaba `\b\w+\b`.
    **Efectos colaterales, todos deseables:** desaparecen los multiword de
    CESS-ESP unidos por guion bajo (`Electricité_de_France`, `51_por_ciento`,
    ~2.400 tipos espurios) y los `--` de Moby Dick.

11. **El ajuste de Zipf excluye la cola de hapax** (`EXCLUIR_HAPAX_ZIPF=True`).
    Se reportan ambos ajustes para trazabilidad.
    **Razón:** en texto aleatorio los hapax son ~96% de los rangos y forman una
    recta horizontal en log-log que domina el ajuste por peso numérico. Con cola
    incluida, Babel y el null dan R²≈0,35 y pendientes ≈−0,17: comparar esas
    cifras no informa de nada. Sin cola, Babel da −0,940 con R²=0,858, que es el
    fenómeno Zipfiano real y lo que permite el contraste con el español (−0,968).

12. **El χ² usa el espacio COMPLETO de categorías `A^n`,** incluidas las de
    conteo cero.
    **Razón:** construir el vector de observados con `Counter(...).values()`
    descarta en silencio las categorías ausentes y calcula el esperado sobre un
    k reducido, sesgando el test **hacia "uniforme"** — precisamente la
    conclusión que el estudio quiere poder defender. Crítico con `n=2` (841
    categorías).

13. **Heaps se reporta con tabla de sensibilidad al tramo de ajuste**
    (`n_min` ∈ {10, 100, 1.000, 10.000, 50.000}) y con la tasa `V/N`.
    **Razón:** β≈0,4-0,6 en lenguaje natural es un **asintótico**; el arranque
    de la curva crece casi linealmente en cualquier texto y empuja β hacia
    arriba. Lo informativo no es el número suelto sino que β sea plano en Babel
    y decreciente en los controles. Misma práctica de blindaje que §5.1.

14. **El sentimiento entra al veredicto Monte Carlo**, con polaridad y
    subjetividad.
    **Razón:** contrastarlo contra una sola réplica nula viola §5.8. Al meterlo
    en la distribución completa se ve que Babel (+0,225) y el null (+0,205) son
    la misma cosa, y que los controles reales quedan por debajo de ambos.

---

## 6. Roadmap técnico

El Bloque A (fundamentos) está completo y validado. Los siguientes bloques añaden
herramientas modernas de NLP/ML, cada uno respondiendo a una pregunta científica
distinta.

**Prioridad revisada: ~~E.5~~ → ~~B~~ → ~~C~~ → ~~D~~ → resto de E.**

> **Estado: E.5, B, C y D completados.** Queda solo el resto del Bloque E
> (E.1-E.4, E.6): visualización interactiva, grafos, datashader, configuración
> YAML y el vault de Obsidian. Es infraestructura, no ciencia nueva.
>
> El orden dentro de C se alteró a propósito: C.5 y C.6 primero por no requerir
> descargas, y resultaron ser las de mayor valor metodológico (§3.9). Después
> C.1-C.4 con BGE-M3 (§3.10).
>
> **D dependía de una corrección previa:** la tokenización de §5.10. Con la
> anterior, buscar palabras reales en tokens de 20 caracteres de media era
> vacío por construcción. Sin ese arreglo, el mejor resultado del estudio
> (§3.11) no habría existido.

> **E.5 (tests) ya está hecho, y era lo primero por una razón.** La validación
> del Bloque A destapó siete defectos en `src/`: metadata sin parsear, `N_REF`
> calculado con `max` en vez de `min`, χ² descartando categorías vacías, la cola
> de hapax en Zipf, compresores calculados y descartados, el límite de 1M chars
> de spaCy, y la paginación ausente en el null model. Los bloques B–E añaden
> cuatro módulos nuevos encima de esa base; ahora hay **180 tests** que los cubren.
>
> **Correr `python -m pytest` antes y después de tocar `src/`.**

### ⚠️ Tres cosas que la validación cambia para los bloques siguientes

**1. La tokenización (§5.10) reabre D.1.** Con la tokenización anterior el largo
medio de token en Babel era 20 caracteres y buscar palabras reales en ellos era
vacío por construcción. Ahora es 8,7 con mediana 6 — el rango donde una palabra
real puede emerger por azar. Medición preliminar: 602 palabras del léxico en
Babel contra 641 en el null, ninguna de más de 5 letras. D.1 debe formalizarlo
con diccionarios externos (`wordfreq`, Aspell), no con el vocabulario de los
controles.

**2. C.5 necesita n-gramas de CARACTERES, no de palabras.** Un modelo de lenguaje
a nivel de palabra entrenado sobre Cervantes asigna probabilidad ≈0 a cada token
de Babel, así que la perplejidad sale infinita y no informa. A nivel de carácter
da un número real y enlaza directamente con la entropía ya medida en el Bloque A.

**3. C.4 está limpio.** Era un riesgo señalado: el footer `Book Location` podía
aparecer como falso duplicado. Verificado que las 10 coordenadas son distintas y
que, tras el parser, hay **0 8-gramas compartidos** entre libros. MinHash puede
correr sin ese sesgo.

> Recordatorio para cualquier bloque que genere texto sintético: usar
> `perfil_formato()` y pasar `ancho_linea` y `lineas_por_pagina` al generador. Un
> null model sin paginar no es un gemelo válido (§5.9).

### 🔷 Bloque B — NLP moderno (reemplazar herramientas legacy)

**Objetivo:** contrastar las herramientas clásicas (TextBlob, NLTK) con modelos
modernos, y usar detección de idioma probabilística como nueva métrica.

**Tareas de implementación:**

- [x] **B.1** — **COMPLETADO.** Módulo `babel_transformers.py` con los dos modelos.
  Resultados en §3.8. La métrica de idioma discrimina (0,54 vs 0,95-0,99) y el
  sentimiento moderno invierte a TextBlob.
- [x] **B.2** — **COMPLETADO.** fastText `lid.176.bin`, y resultó ser la mejor
  métrica del bloque: 0,110 contra 0,967/0,963. Además es ~1.000× más rápido que
  un transformer (63.000 ventanas/s), así que corre el Monte Carlo **completo**
  con las mismas 100 réplicas del Bloque A, sin muestreo.

  > **Instalación en Windows:** usar `fasttext-wheel`, no `fasttext`. El segundo
  > no publica wheels y requiere compilador C++. `fasttext-wheel` 0.9.2 sí tiene
  > wheel para cp312 y se instala sin incidencias.
  >
  > El modelo (131 MB) se cachea en `~/.cache/fasttext/`, **fuera del repo**,
  > vía `bt.descargar_fasttext()`.
- [x] **B.3** — *(parcial)* spaCy ya carga con `disable=["ner","parser"]` y
  `aplicar_lematizacion()` procesa por bloques de 20k tokens — obligatorio, porque
  spaCy rechaza documentos de más de 1M de caracteres (error E088) y a escala real
  reventaba. **Falta** el `nlp.pipe()` con batch de verdad.
- [x] **B.4** — **COMPLETADO.** Las 4 condiciones contrastadas contra el null
  model. 11,5 min en CPU. Resultados en §3.8, secciones 7.x del notebook, y
  tablas `11`–`13` en `outputs/`.

**Specs técnicas — medidas, no estimadas:**

- **La ventana es la unidad de observación, no el corpus.** Con 512 tokens de
  contexto sobre 1,1M de caracteres salen ~1.000 ventanas por condición: pasar
  todo por dos modelos son horas, y 100 réplicas nulas es inviable. El diseño
  usado —R réplicas × W ventanas → R valores del estadístico— mantiene §5.8
  intacto y añade variabilidad intra-corpus. `R=25, W=25` para transformers;
  `R=100, W=100` para fastText, que se lo puede permitir.

- **Tamaño de ventana: 600 caracteres**, fijado en caracteres y no en tokens
  para que las 4 condiciones reciban la misma cantidad de texto físico. Medido
  con XLM-R sobre el corpus real:

  | ventana | Babel (máx tokens) | Controles (máx) |
  |---------|--------------------|-----------------|
  | 600 chars | 380 | 187 |
  | 800 chars | 505 | 244 — 7 de margen |
  | 1000 chars | 627 **TRUNCA** | 301 |

  Con 1.000 se truncaría **solo el ruido**, sesgando justo la comparación que
  sostiene el estudio.

- **Rendimiento en CPU (12 núcleos, sin GPU):** 346 ms/ventana con
  `batch_size=8`, que resultó mejor que 16 y 32. Los modelos pesan ~1,1 GB cada
  uno y tardan ~60 s en cargar.

- **Contraste obligatorio contra null model** (principio rector §5.8).

### 🔷 Bloque C — Modelado no supervisado (buscar "islas de significado")

**Objetivo:** el bloque más alineado con el concepto borgeano. Si emergen tópicos o
clusters coherentes, es señal; si no, es la prueba más fuerte de aleatoriedad.

**Tareas de implementación:**

- [x] **C.1** — **COMPLETADO.** BERTopic sobre chunks de 200 caracteres,
  reutilizando los embeddings ya calculados. Halló **3 tópicos en Babel contra 2
  en español**. Resultados en §3.10.
- [x] **C.2** — **COMPLETADO**, pero **no con el modelo del roadmap**.

  > **Cambio de modelo, decidido con el autor.**
  > `paraphrase-multilingual-MiniLM-L12-v2` es de 2021, 384 dimensiones y límite
  > de 128 tokens. Se sustituyó por **`BAAI/bge-m3`** (2024; 568M parámetros,
  > 1024 dim). El motivo no es el contexto largo —con chunks de 200 caracteres
  > sobra— sino **blindar la conclusión**: si el modelo multilingüe más fuerte
  > disponible no encuentra estructura, se agota la objeción de que "quizá con
  > un modelo mejor sí aparecería".
  >
  > Coste: 4.564 MB en caché (el repo trae los pesos duplicados en `.bin` y
  > `.safetensors`) y ~292 ms por chunk en CPU.

- [x] **C.3** — **COMPLETADO.** UMAP + HDBSCAN. Figura `09_embeddings_umap.png`.
  El paquete `hdbscan` suelto no hace falta: `sklearn.cluster.HDBSCAN` existe
  desde sklearn 1.3.
- [x] **C.4** — **COMPLETADO.** MinHash con `datasketch`: Jaccard 0,000000 en
  los 21 pares. Ver §3.10.
- [x] **C.5** — **COMPLETADO.** Modelo de n-gramas de **caracteres** (orden 5)
  entrenado sobre el 80% de CESS-ESP. Resultados en §3.9.

  > **Desviación del roadmap, justificada.** El plan decía n-gramas de palabras
  > sobre Cervantes. A nivel de palabra el modelo asigna probabilidad ≈0 a cada
  > token de Babel y la perplejidad sale infinita: no informa. A nivel de
  > carácter da un número real y enlaza con la entropía del Bloque A.
  >
  > Se implementó en `babel_secuencia.ModeloCaracteres` en vez de usar
  > `nltk.lm` para que comparta los conteos con la entropía condicional: las dos
  > métricas salen del mismo objeto y no pueden contradecirse. `nltk.lm` sigue
  > disponible como contraste externo si se quiere.

- [x] **C.6** — **COMPLETADO.** Matrices de transición de orden 1-4, entropía
  condicional, ganancia de información y divergencia Jensen-Shannon.
  Resultados en §3.9 y figura `08_matrices_transicion.png`.

**Specs técnicas:**
- BERTopic sobre ruido puede producir "tópicos" espurios → ESE es el hallazgo.
  Documentar coherencia baja vs. la de un corpus real.
- Comparar SIEMPRE la estructura de clusters de Babel contra la del null model y la
  de los controles.

### 🔷 Bloque D — Búsqueda activa de "palabras reales"

**Objetivo:** versión cuantitativa del experimento mental de Borges: ¿cuánto
significado emerge por azar?

**Tareas de implementación:**

- [x] **D.1** — **COMPLETADO.** `babel_lexico.py` con `wordfreq` +
  `pyspellchecker` (las listas Aspell no hicieron falta: la unión de ambas
  fuentes da 100k+ palabras y cubre morfología). Resultados en §3.11.

  > **Aportación no prevista en el roadmap:** la **predicción combinatoria en
  > cerrado**. `P(k) = N_k / 26^k` permite calcular cuántas palabras deberían
  > emerger *antes* de contarlas. Acertó con un 2% de error. Es el único
  > resultado del estudio que no depende de ningún modelo nulo.

- [x] **D.2** — **COMPLETADO.** Tasa homogénea entre los 10 libros, con razón
  varianza/media compatible con Poisson.
- [x] **D.3** — **COMPLETADO**, con resultado negativo documentado: la métrica
  tiene suelo en 0,42 para ruido puro y rango útil minúsculo, más un confundente
  temático entre los controles. Ver §3.11.

**Specs técnicas:**
- `wordfreq` da frecuencias reales por idioma — útil para ponderar "cuán común" es
  cada palabra emergente.
- El hallazgo esperado: la tasa de palabras reales en Babel = tasa en null model,
  y ambas explicables por combinatoria pura.

### 🔷 Bloque E — Visualización e infraestructura

**Objetivo:** llevar el proyecto a calidad de publicación.

**Tareas de implementación:**

- [ ] **E.1** — Migrar figuras clave a **plotly** (interactivas) para el reporte.
- [ ] **E.2** — Grafos de co-ocurrencia de bigramas con `networkx` / `pyvis`.
- [ ] **E.3** — `datashader` para visualizar distribuciones de millones de puntos.
- [ ] **E.4** — Configuración por YAML (parámetros del estudio centralizados).
- [x] **E.5** — **COMPLETADO.** Suite `pytest` en `Babel_V2/tests/`: **180 tests,
  ~15 s**. Diez archivos (creció con B, C y D):

  | Archivo | Tests | Cubre |
  |---------|-------|-------|
  | `conftest.py` | — | Fixtures; localiza `src/` y los datos como el notebook; **se salta** en vez de fallar si faltan corpus o dependencias opcionales |
  | `test_parseo.py` | 13 | `parsear_libro_babel`, contaminación de metadata, invariantes del corpus |
  | `test_null_model.py` | 13 | Generación, reproducibilidad, calibración, **paginación** |
  | `test_metricas.py` | 24 | Entropía, compresión, Zipf, Heaps, χ² + casos límite |
  | `test_pipeline.py` | 22 | Limpieza, tokenización, stopwords, stemming, lematización por bloques |
  | `test_transformers.py` | 30 | Muestreo, métricas, normalización de etiquetas (pipeline simulado) |
  | `test_secuencia.py` | 23 | Transiciones, entropía condicional, ganancia, perplejidad, divergencia JS |
  | `test_embeddings.py` | 19 | Chunking, geometría, reducción, clustering, MinHash (embedder simulado) |
  | `test_lexico.py` | 24 | Carga de léxicos, `P(k)`, predicción combinatoria, emergentes, por documento |
  | `test_regresiones.py` | 9 | Invariantes entre módulos y los artefactos de §3.7 |

  Correr con `python -m pytest` desde `Babel_V2/`.

  > **Hallazgo de la propia suite:** `dozwqk k jrq.txt` guarda internamente el
  > título `.dozwqk k jrq` — el punto inicial se perdió al guardar el archivo.
  > El parser lo detectó igualmente gracias a su heurística de formato de
  > respaldo, pero conviene saberlo: el nombre de archivo **no** es una fuente
  > fiable del título.
- [ ] **E.6** — (Opcional) documentación del vault en formato Obsidian.

**Specs técnicas:**
- Mantener la separación lógica/presentación: la infra no debe contaminar `src/`.

---

## 7. Principios de trabajo

> Preferencias del autor, a respetar en todo el desarrollo.

1. **Rigor antes que vistosidad.** Fundamentos estadísticos sólidos antes de añadir
   herramientas impresionantes. Todo hallazgo se contrasta contra el null model.

2. **Stack de vanguardia.** Usar librerías, frameworks y patrones actuales y
   recomendados por la comunidad (Hugging Face, sentence-transformers, BERTopic,
   UMAP, plotly, zstandard, etc.).

3. **No traducir términos técnicos.** Mantener `dataframe`, `query`, `pipeline`,
   `feature`, `deployment`, `embedding`, etc. en su forma original.

4. **Separación lógica / presentación.** `.py` modulares como motor, notebook como
   dashboard. Nunca meter lógica pesada en celdas del notebook.

5. **Workflow diagnóstico-primero.** Revisar y entender completamente antes de
   generar código. Enumerar pasos y consultar dudas de lógica antes de codificar.

6. **Entorno objetivo:** Jupyter Notebook 7.4.5. (Nota: si en algún momento hay SQL,
   es para AWS Athena — dialecto Presto/Trino.)

7. **Corpus abiertos / sin copyright** para cualquier control o recurso externo.

8. **Documentar decisiones metodológicas** formalmente (tablas de sensibilidad,
   notas explícitas) para blindar el trabajo ante revisión académica.

---

## 8. Referencias

**Texto aleatorio y Ley de Zipf:**
- Miller, G. A. (1957). Some effects of intermittent silence. *American Journal of
  Psychology*, 70(2), 311–314. [Modelo del "mono tipeando"]
- Mandelbrot, B. (1953). An informational theory of the statistical structure of
  language. *Communication Theory*.
- Li, W. (1992). Random texts exhibit Zipf's-law-like word frequency distribution.
  *IEEE Transactions on Information Theory*, 38(6), 1842–1845.
- Conrad, B., & Mitzenmacher, M. (2004). Power laws for monkeys typing randomly:
  the case of unequal probabilities. *IEEE Trans. Information Theory*, 50(7).
- Ferrer-i-Cancho, R., & Solé, R. (2001). Two regimes in the frequency of words and
  the origins of complex lexicons. *Journal of Quantitative Linguistics*.
- Perline, R. (2015). The random division of the unit interval and the approximate
  −1 exponent in the monkey-at-the-typewriter model. arXiv:1512.02177.
- Howes, D. (1968). Zipf's law and Miller's random-monkey model. *American Journal
  of Psychology*.
- Piantadosi, S. T. (2014). Zipf's word frequency law in natural language: A
  critical review and future directions. *Psychonomic Bulletin & Review*, 21(5).
  [Revisión crítica exhaustiva y moderna]

**Teoría de la información y complejidad:**
- Shannon, C. E. (1948). A mathematical theory of communication. *Bell System
  Technical Journal*.
- Li, M., & Vitányi, P. (2008). *An Introduction to Kolmogorov Complexity and Its
  Applications*. Springer. [Relación entropía ↔ compresión ↔ complejidad]
- Grünwald, P., & Vitányi, P. (2004). Shannon information and Kolmogorov complexity.

**Referencia conceptual:**
- Borges, J. L. (1941). *La Biblioteca de Babel*.

> **Nota:** las referencias con fechas y volúmenes deben verificarse antes de la
> publicación final. Se listan aquí como punto de partida para la sección de
> estado del arte del paper.
