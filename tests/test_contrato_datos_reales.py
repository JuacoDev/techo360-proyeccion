"""Contrato de calcular_zonas() contra los DATOS REALES de produccion, en MODO LECTURA.

NO corre por defecto: exige TECHO360_DATOS_REALES=1 y DATABASE_URL. Pensado para un
contenedor temporal `:verify` (Tailscale -> postgres-principal), igual que la
verificacion del 22-sep-2026. Fuerza una sesion de solo lectura (SHOW lo confirma) y
nunca escribe.

    docker run --rm --env-file <solo DATABASE_URL> -e TECHO360_DATOS_REALES=1 \
        -v <repo>/tests:/app/tests techo360/proyeccion:verify \
        sh -c "pip install -q pytest && python -m pytest -q tests/test_contrato_datos_reales.py"
"""
import os
import unittest

import pandas as pd
from sqlalchemy import create_engine, text

from app.origen import es_proyeccion, es_real
from app.repositories import proyeccion as repo
from app.services import proyeccion as servicio

HABILITADO = os.environ.get("TECHO360_DATOS_REALES") == "1" and bool(os.environ.get("DATABASE_URL"))
COBERTURAS = ("cobertura_acueducto", "cobertura_alcantarillado")


@unittest.skipUnless(HABILITADO, "requiere TECHO360_DATOS_REALES=1 y DATABASE_URL (solo lectura)")
class TestContratoConDatosReales(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_engine(os.environ["DATABASE_URL"], connect_args={"options": "-c default_transaction_read_only=on"})
        repo._engine = cls.engine  # el repositorio usa ESTA sesion de solo lectura
        with cls.engine.connect() as con:
            assert con.execute(text("SHOW default_transaction_read_only")).scalar() == "on"
        cls.cache = {}

    @classmethod
    def tearDownClass(cls):
        cls.engine.dispose()

    def filas_bd(self, indicador, anio):
        with self.engine.connect() as con:
            df = pd.read_sql(text("""SELECT codigoDivipola AS c, valueVar AS v, originVar AS o, intervaloConfianza AS ic
                                     FROM MapCalorDB WHERE nameVar=:i AND yearM=:a AND codigoDivipola IS NOT NULL AND valueVar IS NOT NULL"""),
                             con, params={"i": indicador, "a": anio})
        return {str(r.c).zfill(5): r for r in df.itertuples(index=False) if str(r.c).zfill(5) != "01001"}

    def zonas(self, indicador, anio):
        k = (indicador, anio)
        if k not in self.cache:
            self.cache[k] = {z["codigo_municipio"]: z for z in servicio.calcular_zonas(indicador, anio)}
        return self.cache[k]

    # ---- 1. ninguna fila de la BD tiene origen nulo (el helper las trataria como reales)
    def test_no_hay_origenes_nulos(self):
        with self.engine.connect() as con:
            n = con.execute(text("SELECT count(*) FROM MapCalorDB WHERE originVar IS NULL OR originVar = ''")).scalar()
        self.assertEqual(n, 0)

    # ---- 2. la regla coincide con el filtro SQL, sobre los origenes REALES de la BD
    def test_helper_coincide_con_el_filtro_sql_en_la_bd(self):
        with self.engine.connect() as con:
            todos = {r[0] for r in con.execute(text("SELECT DISTINCT originVar FROM MapCalorDB"))}
            reales_sql = {r[0] for r in con.execute(text(f"SELECT DISTINCT originVar FROM MapCalorDB WHERE {repo.SQL_SOLO_REALES}"))}
        self.assertEqual(reales_sql, {o for o in todos if es_real(o)})

    # ---- 3. coberturas 2025-26: ya existen filas proyeccion-* -> ahora salen como proyeccion
    def test_cobertura_2025_2026_sale_como_proyeccion(self):
        for indicador in COBERTURAS:
            for anio in (2025, 2026):
                bd = self.filas_bd(indicador, anio)
                proy = {c: r for c, r in bd.items() if es_proyeccion(r.o)}
                self.assertGreater(len(proy), 900, (indicador, anio))  # hoy ~980
                z = self.zonas(indicador, anio)
                for c, r in proy.items():
                    self.assertFalse(z[c]["es_real"], (indicador, anio, c))
                    self.assertEqual(z[c]["origen"], r.o)
                    self.assertAlmostEqual(z[c]["valor"], min(max(float(r.v), 0), 100), places=2)
                # no hay cobertura real posterior a 2024: NINGUN municipio puede salir real
                self.assertFalse(any(x["es_real"] for x in z.values()), (indicador, anio))

    # ---- 4. contrato general en varios (indicador, año)
    def test_contrato_general(self):
        casos = [("cobertura_acueducto", 2025), ("cobertura_acueducto", 2030), ("cobertura_alcantarillado", 2026),
                 ("cobertura_alcantarillado", 2042), ("densidad_poblacional", 2025), ("densidad_poblacional", 2030), ("densidad_poblacional", 2042)]
        for indicador, anio in casos:
            bd = self.filas_bd(indicador, anio)
            z = self.zonas(indicador, anio)
            self.assertGreater(len(z), 1000, (indicador, anio))
            self.assertNotIn("01001", z)
            for c, item in z.items():
                self.assertIsNotNone(item["origen"], (indicador, anio, c))
                self.assertGreaterEqual(item["fiabilidad"], 0.0)
                self.assertLessEqual(item["fiabilidad"], 100.0)
                if c in bd:  # fila en BD -> su origen decide
                    self.assertEqual(item["es_real"], es_real(bd[c].o), (indicador, anio, c, bd[c].o))
                    self.assertEqual(item["origen"], bd[c].o)
                else:        # respaldo calculado -> NUNCA real
                    self.assertFalse(item["es_real"], (indicador, anio, c))
                    self.assertTrue(es_proyeccion(item["origen"]))

    # ---- 5. el respaldo de cobertura es el ULTIMO VALOR REAL, verificado con SQL independiente
    def test_respaldo_de_cobertura_es_el_ultimo_valor_real(self):
        for indicador in COBERTURAS:
            z = self.zonas(indicador, 2035)  # no hay filas en BD para 2035 -> todo es respaldo
            with self.engine.connect() as con:
                df = pd.read_sql(text(f"""SELECT codigoDivipola AS c, yearM AS a, valueVar AS v FROM MapCalorDB
                                          WHERE nameVar=:i AND {repo.SQL_SOLO_REALES} AND codigoDivipola IS NOT NULL AND valueVar IS NOT NULL"""),
                                 con, params={"i": indicador})
            df["c"] = df.c.astype(str).str.zfill(5)
            ultimo = df.sort_values(["c", "a"]).groupby("c").tail(1).set_index("c")
            ultimo = ultimo[ultimo.index != "01001"]
            self.assertEqual(set(z), set(ultimo.index))
            for c, fila in ultimo.iterrows():
                self.assertFalse(z[c]["es_real"])
                self.assertEqual(z[c]["origen"], servicio.ORIGEN_ULTIMO_VALOR)
                self.assertAlmostEqual(z[c]["valor"], min(max(float(fila.v), 0), 100), places=2, msg=c)

    # ---- 6. la regresion de densidad solo usa puntos REALES
    def test_la_regresion_de_densidad_solo_usa_filas_reales(self):
        with self.engine.connect() as con:
            df = pd.read_sql(text("SELECT codigoDivipola AS c, yearM AS a, originVar AS o FROM MapCalorDB WHERE nameVar='densidad_poblacional' AND codigoDivipola IS NOT NULL"), con)
        df["c"] = df.c.astype(str).str.zfill(5)
        reales = {(r.c, int(r.a)) for r in df.itertuples(index=False) if es_real(r.o)}
        proyectadas = {(r.c, int(r.a)) for r in df.itertuples(index=False) if es_proyeccion(r.o)}
        self.assertTrue(proyectadas, "se esperaba al menos una fila proyectada de densidad (TerriData-proyeccion)")
        puntos = repo.obtener_puntos_todos_los_municipios("densidad_poblacional", n_puntos=4)
        self.assertGreater(len(puntos), 1000)
        for c, serie in puntos.items():
            for p in serie:
                self.assertIn((c, p["anio"]), reales, (c, p))
                self.assertNotIn((c, p["anio"]), proyectadas, (c, p))

    # ---- 7. densidad sin fila (anio fuera de la carga DANE) -> regresion, es_real False
    def test_densidad_sin_fila_cae_en_la_regresion(self):
        z = self.zonas("densidad_poblacional", 2043)
        self.assertGreater(len(z), 1000)
        for item in z.values():
            self.assertFalse(item["es_real"])
            self.assertEqual(item["origen"], servicio.ORIGEN_REGRESION)
            self.assertGreaterEqual(item["valor"], 0.0)


if __name__ == "__main__":
    unittest.main()
