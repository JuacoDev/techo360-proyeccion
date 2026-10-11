"""Contrato de calcular_zonas() tras el etiquetado real/proyeccion (2026-10-10).
Sin BD: el repositorio se mockea.

Reglas que fija:
  1. Si hay fila en BD para (municipio, indicador, año) -> valor tal cual, es_real = not
     es_proyeccion(origen), con origen y fiabilidad.
  2. Si NO hay fila -> respaldo con es_real=False: ultimo valor real (cobertura) o
     regresion (densidad).
  3. Las regresiones siguen EXCLUYENDO filas proyectadas al ajustar (filtro SQL).
  4. fiabilidad siempre en [0, 100].

pytest tests/test_contrato_zonas.py   (o: python -m unittest discover -s tests)
"""
import unittest
from unittest.mock import patch

import pandas as pd

from app.repositories import proyeccion as repo
from app.services import proyeccion as servicio


def fila(valor, origen, ic=None):
    return {"valor": valor, "origen": origen, "intervalo_confianza": ic}


@patch.object(servicio, "obtener_puntos_todos_los_municipios", return_value={})
@patch.object(servicio, "obtener_ultimo_valor_real_todos_municipios", return_value={})
class TestFilaEnBd(unittest.TestCase):
    def _zonas(self, indicador, anio, filas):
        with patch.object(servicio, "obtener_filas_todos_municipios_para_anio", return_value=filas):
            return {z["codigo_municipio"]: z for z in servicio.calcular_zonas(indicador, anio)}

    def test_fila_proyeccion_sale_como_proyeccion(self, *_):
        # el caso real de hoy: cobertura 2025-26 ya existe como proyeccion-ultimo-valor-observado-2024
        z = self._zonas("cobertura_acueducto", 2025, {"05001": fila(96.75, "proyeccion-ultimo-valor-observado-2024", 60.0)})["05001"]
        self.assertFalse(z["es_real"])
        self.assertEqual(z["valor"], 96.75)
        self.assertEqual(z["origen"], "proyeccion-ultimo-valor-observado-2024")
        self.assertEqual(z["fiabilidad"], 60.0)

    def test_fila_real_sale_como_real(self, *_):
        z = self._zonas("cobertura_acueducto", 2024, {"05001": fila(96.0, "TerriData")})["05001"]
        self.assertTrue(z["es_real"])
        self.assertEqual(z["origen"], "TerriData")
        self.assertEqual(z["fiabilidad"], 100.0)

    def test_dane_proyeccion_poblacion_es_real(self, *_):
        z = self._zonas("densidad_poblacional", 2030, {"05001": fila(2500.5, "DANE-proyeccion-poblacion-2030")})["05001"]
        self.assertTrue(z["es_real"])

    def test_terridata_proyeccion_es_proyeccion(self, *_):
        z = self._zonas("densidad_poblacional", 2030, {"05001": fila(10.0, "TerriData-proyeccion", 40.0)})["05001"]
        self.assertFalse(z["es_real"])
        self.assertEqual(z["fiabilidad"], 40.0)

    def test_proyeccion_sin_intervalo_usa_el_minimo(self, *_):
        z = self._zonas("cobertura_alcantarillado", 2026, {"05001": fila(50.0, "proyeccion-x", None)})["05001"]
        self.assertEqual(z["fiabilidad"], 30.0)

    def test_toda_fila_proyeccion_es_false_y_toda_real_es_true(self, *_):
        origenes = ["TerriData", "DANE-CNPV2018", "IGAC Catastro", "DANE-proyeccion-poblacion-2030",
                    "proyeccion-ultimo-valor-observado-2024", "proyeccion-techo360-ultimo-valor", "TerriData-proyeccion",
                    "proyeccion-snapshot-2026"]
        filas = {f"{i:05d}": fila(10.0 + i, o, 55.0) for i, o in enumerate(origenes, start=1)}
        zonas = self._zonas("densidad_poblacional", 2030, filas)
        for i, o in enumerate(origenes, start=1):
            self.assertEqual(zonas[f"{i:05d}"]["es_real"], not (o.startswith("proyeccion-") or o.endswith("-proyeccion")), o)

    def test_fiabilidad_siempre_entre_0_y_100(self, *_):
        filas = {"00001": fila(1.0, "TerriData"), "00002": fila(1.0, "proyeccion-x", 250.0),
                 "00003": fila(1.0, "proyeccion-x", -5.0), "00004": fila(1.0, "proyeccion-x", None)}
        for z in self._zonas("cobertura_acueducto", 2026, filas).values():
            self.assertGreaterEqual(z["fiabilidad"], 0.0)
            self.assertLessEqual(z["fiabilidad"], 100.0)

    def test_excluye_el_agregado_nacional(self, *_):
        zonas = self._zonas("densidad_poblacional", 2030, {"01001": fila(1.0, "TerriData-proyeccion"), "05001": fila(2.0, "TerriData")})
        self.assertNotIn("01001", zonas)

    def test_cobertura_de_la_fila_se_recorta_a_0_100(self, *_):
        zonas = self._zonas("cobertura_acueducto", 2025, {"00001": fila(130.0, "proyeccion-x", 50.0), "00002": fila(-4.0, "proyeccion-x", 50.0)})
        self.assertEqual(zonas["00001"]["valor"], 100.0)
        self.assertEqual(zonas["00002"]["valor"], 0.0)


class TestRespaldoDeCobertura(unittest.TestCase):
    def _zonas(self, anio, filas, ultimos, indicador="cobertura_acueducto"):
        with patch.object(servicio, "obtener_filas_todos_municipios_para_anio", return_value=filas), \
             patch.object(servicio, "obtener_ultimo_valor_real_todos_municipios", return_value=ultimos), \
             patch.object(servicio, "obtener_puntos_todos_los_municipios", side_effect=AssertionError("la cobertura ya no usa regresion")):
            return {z["codigo_municipio"]: z for z in servicio.calcular_zonas(indicador, anio)}

    def test_devuelve_el_ultimo_valor_real_y_es_real_false(self):
        z = self._zonas(2030, {}, {"05001": {"anio": 2024, "valor": 81.25}})["05001"]
        self.assertEqual(z["valor"], 81.25)
        self.assertFalse(z["es_real"])
        self.assertEqual(z["origen"], "proyeccion-techo360-ultimo-valor")

    def test_no_extrapola_pendiente(self):
        # una serie con tendencia fuerte NO cambia el respaldo: solo cuenta el ultimo valor
        z2030 = self._zonas(2030, {}, {"05001": {"anio": 2024, "valor": 70.0}})["05001"]["valor"]
        z2042 = self._zonas(2042, {}, {"05001": {"anio": 2024, "valor": 70.0}})["05001"]["valor"]
        self.assertEqual(z2030, z2042)

    def test_fiabilidad_por_antiguedad(self):
        esperado = {2025: 60.0, 2026: 50.0, 2027: 40.0, 2028: 30.0, 2042: 30.0}
        for anio, fiab in esperado.items():
            z = self._zonas(anio, {}, {"05001": {"anio": 2024, "valor": 70.0}})["05001"]
            self.assertEqual(z["fiabilidad"], fiab, anio)

    def test_una_fila_de_bd_gana_al_respaldo(self):
        zonas = self._zonas(2025, {"05001": fila(55.0, "proyeccion-ultimo-valor-observado-2024", 60.0)},
                            {"05001": {"anio": 2024, "valor": 99.0}, "05002": {"anio": 2023, "valor": 40.0}})
        self.assertEqual(zonas["05001"]["valor"], 55.0)       # la fila, no el respaldo
        self.assertEqual(zonas["05002"]["valor"], 40.0)       # sin fila -> respaldo
        self.assertEqual(zonas["05002"]["origen"], "proyeccion-techo360-ultimo-valor")

    def test_se_recorta_a_0_100_y_vale_para_alcantarillado(self):
        z = self._zonas(2030, {}, {"05001": {"anio": 2024, "valor": 107.0}}, "cobertura_alcantarillado")["05001"]
        self.assertEqual(z["valor"], 100.0)

    def test_excluye_el_agregado_nacional(self):
        zonas = self._zonas(2030, {}, {"01001": {"anio": 2024, "valor": 70.0}, "05001": {"anio": 2024, "valor": 60.0}})
        self.assertNotIn("01001", zonas)


class TestRespaldoDeDensidad(unittest.TestCase):
    PUNTOS = [{"anio": 2020, "valor": 80.0}, {"anio": 2021, "valor": 90.0}, {"anio": 2022, "valor": 95.0}, {"anio": 2023, "valor": 99.0}]

    def _zonas(self, anio, filas, puntos):
        with patch.object(servicio, "obtener_filas_todos_municipios_para_anio", return_value=filas), \
             patch.object(servicio, "obtener_ultimo_valor_real_todos_municipios", side_effect=AssertionError("densidad no usa ultimo valor")), \
             patch.object(servicio, "obtener_puntos_todos_los_municipios", return_value=puntos):
            return {z["codigo_municipio"]: z for z in servicio.calcular_zonas("densidad_poblacional", anio)}

    def test_sigue_siendo_regresion_lineal_y_es_real_false(self):
        z = self._zonas(2030, {}, {"05001": self.PUNTOS})["05001"]
        self.assertGreater(z["valor"], 99.0)  # la recta sube
        self.assertFalse(z["es_real"])
        self.assertEqual(z["origen"], "proyeccion-regresion-lineal-4puntos")
        self.assertGreaterEqual(z["fiabilidad"], 0.0)
        self.assertLessEqual(z["fiabilidad"], 100.0)

    def test_si_hay_fila_no_se_recalcula(self):
        z = self._zonas(2030, {"05001": fila(2500.0, "DANE-proyeccion-poblacion-2030")}, {"05001": self.PUNTOS})["05001"]
        self.assertEqual(z["valor"], 2500.0)
        self.assertTrue(z["es_real"])

    def test_no_proyecta_hacia_un_anio_ya_cubierto_por_los_puntos(self):
        zonas = self._zonas(2022, {}, {"05001": self.PUNTOS})
        self.assertNotIn("05001", zonas)


class TestLasRegresionesExcluyenFilasProyectadas(unittest.TestCase):
    """Las consultas que alimentan una regresion filtran proyeccion-*; la que alimenta
    calcular_zonas con la fila del año NO filtra (necesita verlas para etiquetarlas)."""

    FILTRO_PREFIJO = "originVar NOT LIKE 'proyeccion-%'"
    FILTRO_SUFIJO = "originVar NOT LIKE '%-proyeccion'"

    def _capturar(self, funcion, *args, respuestas):
        consultas = []
        cola = list(respuestas)

        def falso_read_sql(consulta, *_a, **_k):
            consultas.append(str(consulta))
            return cola.pop(0)

        with patch.object(repo.pd, "read_sql", side_effect=falso_read_sql), patch.object(repo, "_obtener_engine", return_value=object()):
            funcion(*args)
        return consultas

    def _assert_filtra(self, consultas):
        self.assertTrue(consultas)
        for c in consultas:
            self.assertIn(self.FILTRO_PREFIJO, c)
            self.assertIn(self.FILTRO_SUFIJO, c)

    def test_puntos_de_todos_los_municipios(self):
        df = pd.DataFrame({"anio": [2020], "valor": [1.0], "codigo_divipola": ["05001"]})
        self._assert_filtra(self._capturar(repo.obtener_puntos_todos_los_municipios, "densidad_poblacional", respuestas=[df]))

    def test_datos_de_un_municipio_para_anios(self):
        df = pd.DataFrame({"anio": [2030], "valor": [1.0]})
        self._assert_filtra(self._capturar(repo.obtener_datos_municipio_para_anios, "05001", "densidad_poblacional", 2027, 2030, respuestas=[df]))

    def test_contexto_de_un_municipio(self):
        zonas = pd.DataFrame({"codigo_municipio": ["05001"], "codigo_departamento": ["05"], "municipio_nombre": ["Medellin"]})
        reg = pd.DataFrame({"municipio_nombre": ["Medellin"] * 2, "anio": [2022, 2023], "valor": [1.0, 2.0], "codigo_divipola": ["05001"] * 2, "origen": ["TerriData"] * 2})
        consultas = self._capturar(repo.obtener_contexto_municipio, "05001", "densidad_poblacional", respuestas=[zonas, reg])
        self._assert_filtra(consultas[1:])  # la 1.a es de zonas_municipio

    def test_ultimo_valor_real_filtra(self):
        df = pd.DataFrame({"codigo_divipola": ["05001"], "anio": [2024], "valor": [90.0]})
        self._assert_filtra(self._capturar(repo.obtener_ultimo_valor_real_todos_municipios, "cobertura_acueducto", 2030, respuestas=[df]))

    def test_filas_del_anio_NO_filtra(self):
        df = pd.DataFrame({"valor": [1.0], "codigo_divipola": ["05001"], "origen": ["proyeccion-x"], "intervalo_confianza": [60.0]})
        consultas = self._capturar(repo.obtener_filas_todos_municipios_para_anio, "cobertura_acueducto", 2025, respuestas=[df])
        for c in consultas:
            self.assertNotIn("NOT LIKE", c)


if __name__ == "__main__":
    unittest.main()
