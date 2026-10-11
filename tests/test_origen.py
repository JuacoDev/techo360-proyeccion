"""app/origen.py: la regla real/proyeccion y su equivalencia con el filtro SQL.

pytest tests/test_origen.py   (o: python -m unittest discover -s tests)
"""
import unittest

from sqlalchemy import create_engine, text

from app.origen import SQL_SOLO_REALES, es_proyeccion, es_real

ORIGENES_REALES = [
    "TerriData",
    "IGAC Catastro",
    "DANE-CNPV2018",
    # Nombre de la FUENTE oficial del DANE, no un relleno de este pipeline:
    "DANE-proyeccion-poblacion-2024",
    "DANE-proyeccion-poblacion-2042",
]
ORIGENES_PROYECCION = [
    "proyeccion-ultimo-valor-observado-2024",
    "proyeccion-tendencia-nacional-2018-2024",
    "proyeccion-snapshot-2026",
    "proyeccion-techo360-ultimo-valor",
    "proyeccion-regresion-lineal-4puntos",
    "TerriData-proyeccion",
]


class TestEsProyeccion(unittest.TestCase):
    def test_origenes_reales(self):
        for origen in ORIGENES_REALES:
            self.assertFalse(es_proyeccion(origen), origen)
            self.assertTrue(es_real(origen), origen)

    def test_origenes_de_proyeccion(self):
        for origen in ORIGENES_PROYECCION:
            self.assertTrue(es_proyeccion(origen), origen)
            self.assertFalse(es_real(origen), origen)

    def test_no_es_un_simple_contiene_proyeccion(self):
        self.assertIn("proyeccion", "DANE-proyeccion-poblacion-2030")
        self.assertTrue(es_real("DANE-proyeccion-poblacion-2030"))

    def test_sin_origen_es_real(self):
        for origen in (None, ""):
            self.assertFalse(es_proyeccion(origen))
            self.assertTrue(es_real(origen))

    def test_distingue_mayusculas_como_like_de_postgres(self):
        self.assertFalse(es_proyeccion("Proyeccion-x"))
        self.assertFalse(es_proyeccion("x-Proyeccion"))


class TestHelperCoincideConElFiltroSql(unittest.TestCase):
    """SQL_SOLO_REALES (que usan las consultas del repositorio) y es_real() deben dar
    el mismo resultado. SQLite con case_sensitive_like imita el LIKE de Postgres."""

    def test_equivalencia(self):
        muestras = ORIGENES_REALES + ORIGENES_PROYECCION + ["Proyeccion-x", "x-Proyeccion", "proyeccion", "proyeccion-", "-proyeccion", "mi-proyeccion-final"]
        engine = create_engine("sqlite://")
        with engine.begin() as con:
            con.execute(text("PRAGMA case_sensitive_like = ON"))
            con.execute(text("CREATE TABLE t (originVar TEXT)"))
            for m in muestras:
                con.execute(text("INSERT INTO t VALUES (:o)"), {"o": m})
            solo_reales_sql = {r[0] for r in con.execute(text(f"SELECT originVar FROM t WHERE {SQL_SOLO_REALES}"))}
        self.assertEqual(solo_reales_sql, {m for m in muestras if es_real(m)})


if __name__ == "__main__":
    unittest.main()
