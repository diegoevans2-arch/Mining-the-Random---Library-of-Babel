"""Parseo de los .txt: separar el cuerpo aleatorio de la metadata del sitio."""
import babel_core as bc


# --- Sobre material sintético (no necesita el corpus real) ------------------

def test_extrae_titulo_y_location(libro_sintetico):
    cuerpo, meta = bc.parsear_libro_babel(
        libro_sintetico, nombre_archivo="titulo de prueba.txt")
    assert meta["titulo"] == "titulo de prueba"
    assert meta["location"] == "Book Location:4-w1-s1-v09"
    assert "Book Location" not in cuerpo


def test_el_cuerpo_no_pierde_contenido(libro_sintetico):
    cuerpo, _ = bc.parsear_libro_babel(
        libro_sintetico, nombre_archivo="titulo de prueba.txt")
    assert cuerpo.count("\n") == 11          # 12 líneas de contenido
    assert cuerpo.startswith("abcde")
    assert cuerpo.endswith("uvwxy")          # sin líneas vacías en los extremos


def test_registra_lo_descartado(libro_sintetico):
    _, meta = bc.parsear_libro_babel(libro_sintetico)
    assert any("Book Location" in ln for ln in meta["lineas_descartadas"])


def test_texto_sin_metadata_sobrevive_intacto():
    limpio = "abcde fghij\nklmno pqrst\nuvwxy zabcd"
    cuerpo, meta = bc.parsear_libro_babel(limpio, nombre_archivo="otro.txt")
    assert cuerpo == limpio
    assert meta["titulo"] is None
    assert meta["location"] is None


def test_texto_vacio_no_revienta():
    cuerpo, meta = bc.parsear_libro_babel("")
    assert cuerpo == ""
    assert meta["location"] is None


# --- Sobre el corpus real ---------------------------------------------------

def test_cuerpo_solo_usa_el_alfabeto_de_30(corpus):
    """Regresión §6/E.5 #1: sin esto el corpus tiene 44 símbolos."""
    simbolos = set("".join(c.texto for c in corpus))
    permitidos = set(bc.ALFABETO_CONTENIDO) | {"\n"}
    assert simbolos <= permitidos, f"fuera del alfabeto: {sorted(simbolos - permitidos)}"
    assert len(simbolos) == 30


def test_sin_parsear_el_corpus_esta_contaminado(corpus_crudo):
    """El contraste que justifica el parser."""
    simbolos = set("".join(c.texto for c in corpus_crudo))
    contaminantes = simbolos - set(bc.ALFABETO_CONTENIDO) - {"\n"}
    assert contaminantes, "se esperaba metadata en los archivos crudos"
    assert {"B", "L", ":"} <= contaminantes
    assert any(d in contaminantes for d in "0123456789")


def test_todos_los_libros_tienen_el_mismo_cuerpo(corpus):
    largos = {c.n_chars for c in corpus}
    assert len(largos) == 1, f"longitudes distintas: {sorted(largos)}"


def test_cada_libro_declara_su_location(corpus):
    locs = [c.meta["location"] for c in corpus]
    assert all(l and l.startswith("Book Location") for l in locs)
    # Son coordenadas distintas: el footer no es un duplicado trivial entre libros
    assert len(set(locs)) == len(locs)


def test_el_titulo_esta_contenido_en_el_nombre_de_archivo(corpus):
    """
    El título va dentro del archivo y el nombre del archivo lo repite... casi.

    `dozwqk k jrq.txt` guarda internamente el título `.dozwqk k jrq`: el punto
    inicial se perdió al guardar, porque un nombre de archivo que empieza por
    punto da problemas en varios sistemas. Por eso el parser NO puede depender
    solo de esta coincidencia, y lleva una heurística de formato como respaldo.
    """
    for c in corpus:
        stem = c.meta["archivo"].rsplit(".txt", 1)[0]
        titulo = c.meta["titulo"]
        assert titulo is not None, f"{c.nombre}: no se detectó título"
        assert titulo.endswith(stem), f"{c.nombre}: {titulo!r} vs {stem!r}"


def test_detecta_el_titulo_aunque_no_coincida_con_el_archivo(libro_sintetico):
    """El respaldo por formato: línea corta seguida de línea vacía."""
    cuerpo, meta = bc.parsear_libro_babel(
        libro_sintetico, nombre_archivo="nombre completamente distinto.txt")
    assert meta["titulo"] == "titulo de prueba"
    assert not cuerpo.startswith("titulo")


def test_detecta_el_titulo_sin_nombre_de_archivo(libro_sintetico):
    """Cuando no hay archivo del que fiarse, solo queda la heurística."""
    cuerpo, meta = bc.parsear_libro_babel(libro_sintetico)
    assert meta["titulo"] == "titulo de prueba"
    assert not cuerpo.startswith("titulo")


def test_parsear_solo_quita_metadata(corpus, corpus_crudo):
    """El parser no debe comerse contenido: la diferencia es pequeña y acotada."""
    for limpio, crudo in zip(corpus, corpus_crudo):
        quitado = crudo.n_chars - limpio.n_chars
        assert 0 < quitado < 200, f"{limpio.nombre}: se quitaron {quitado} chars"
