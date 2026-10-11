"""Pruebas de app/services/proyeccion.py para la extension a
cobertura_acueducto/cobertura_alcantarillado (score de privacion por
proyeccion). No usan base de datos: las funciones del repositorio se
mockean, igual que hace calcular_zonas()/calcular_serie_proyectada() al
recibirlas ya importadas.

Se corre con: python -m unittest discover -s tests
"""
import unittest
from unittest.mock import patch

from app.schemas import SolicitudProyeccion
from app.services import proyeccion as servicio


class TestIndicadoresSoportados(unittest.TestCase):
    def test_densidad_sigue_soportada(self):
        self.assertIn("densidad_poblacional", servicio.INDICADORES_SOPORTADOS)

    def test_acueducto_y_alcantarillado_soportados(self):
        self.assertIn("cobertura_acueducto", servicio.INDICADORES_SOPORTADOS)
        self.assertIn("cobertura_alcantarillado", servicio.INDICADORES_SOPORTADOS)

    def test_deficit_no_soportado(self):
        # Decision de producto: el deficit no tiene serie temporal (censo
        # fijo 2018), no se proyecta aqui por regresion.
        self.assertNotIn("deficit_habitacional_cuantitativo", servicio.INDICADORES_SOPORTADOS)


class TestCalcularZonasGuard(unittest.TestCase):
    """calcular_zonas con el repositorio mockeado. Desde 2026-10-10: si hay fila en BD
    se devuelve tal cual; si no, el respaldo de cobertura es el ULTIMO VALOR REAL
    (no la regresion de 4 puntos) y el de densidad sigue siendo la regresion. El detalle
    del contrato esta en tests/test_contrato_zonas.py."""

    def test_rechaza_indicador_no_soportado(self):
        with self.assertRaises(ValueError):
            servicio.calcular_zonas("deficit_habitacional_cuantitativo", 2030)

    @patch.object(servicio, "obtener_ultimo_valor_real_todos_municipios")
    @patch.object(servicio, "obtener_filas_todos_municipios_para_anio")
    def test_acepta_cobertura_acueducto(self, mock_filas, mock_ultimo):
        mock_filas.return_value = {}
        mock_ultimo.return_value = {"05001": {"anio": 2023, "valor": 96.0}}

        resultado = servicio.calcular_zonas("cobertura_acueducto", 2030)

        self.assertEqual(len(resultado), 1)
        self.assertEqual(resultado[0]["codigo_municipio"], "05001")
        self.assertEqual(resultado[0]["valor"], 96.0)  # ultimo valor real, sin regresion
        self.assertFalse(resultado[0]["es_real"])

    @patch.object(servicio, "obtener_ultimo_valor_real_todos_municipios")
    @patch.object(servicio, "obtener_filas_todos_municipios_para_anio")
    def test_cobertura_se_recorta_a_100(self, mock_filas, mock_ultimo):
        mock_filas.return_value = {}
        mock_ultimo.return_value = {"05001": {"anio": 2023, "valor": 104.0}}

        resultado = servicio.calcular_zonas("cobertura_acueducto", 2030)

        self.assertEqual(resultado[0]["valor"], 100.0)

    @patch.object(servicio, "obtener_filas_todos_municipios_para_anio")
    @patch.object(servicio, "obtener_puntos_todos_los_municipios")
    def test_densidad_no_tiene_techo(self, mock_puntos, mock_filas):
        # Densidad: respaldo por regresion lineal y SIN techo fisico (no se recorta a 100).
        mock_puntos.return_value = {
            "05001": [
                {"anio": 2020, "valor": 80.0},
                {"anio": 2021, "valor": 90.0},
                {"anio": 2022, "valor": 95.0},
                {"anio": 2023, "valor": 99.0},
            ]
        }
        mock_filas.return_value = {}

        resultado = servicio.calcular_zonas("densidad_poblacional", 2030)

        self.assertGreater(resultado[0]["valor"], 100.0)
        self.assertFalse(resultado[0]["es_real"])


class TestProyectarSerieConTecho(unittest.TestCase):
    def test_serie_acueducto_se_recorta(self):
        contexto = {
            "ultimo_anio": 2023,
            "puntos": [
                {"anio": 2020, "valor": 80.0},
                {"anio": 2021, "valor": 90.0},
                {"anio": 2022, "valor": 95.0},
                {"anio": 2023, "valor": 99.0},
            ],
        }
        serie = servicio._proyectar_serie(contexto, 2026, "cobertura_acueducto")
        self.assertTrue(all(p["valor"] <= 100.0 for p in serie))

    def test_serie_densidad_sin_techo(self):
        contexto = {
            "ultimo_anio": 2023,
            "puntos": [
                {"anio": 2020, "valor": 80.0},
                {"anio": 2021, "valor": 90.0},
                {"anio": 2022, "valor": 95.0},
                {"anio": 2023, "valor": 99.0},
            ],
        }
        serie = servicio._proyectar_serie(contexto, 2026, "densidad_poblacional")
        self.assertTrue(any(p["valor"] > 100.0 for p in serie))


class TestCalcularSerieProyectadaGuard(unittest.TestCase):
    def test_rechaza_deficit(self):
        solicitud = SolicitudProyeccion(
            codigo_municipio="05001", indicador="deficit_habitacional_cuantitativo", anio_fin=2030
        )
        with self.assertRaises(ValueError):
            servicio.calcular_serie_proyectada(solicitud)

    @patch.object(servicio, "obtener_datos_municipio_para_anios")
    @patch.object(servicio, "obtener_contexto_municipio")
    def test_acepta_alcantarillado(self, mock_contexto, mock_datos_anios):
        mock_contexto.return_value = {
            "ultimo_anio": 2023,
            "puntos": [
                {"anio": 2020, "valor": 70.0},
                {"anio": 2021, "valor": 72.0},
                {"anio": 2022, "valor": 74.0},
                {"anio": 2023, "valor": 76.0},
            ],
        }
        mock_datos_anios.return_value = {}

        solicitud = SolicitudProyeccion(
            codigo_municipio="05001", indicador="cobertura_alcantarillado", anio_fin=2026
        )
        respuesta = servicio.calcular_serie_proyectada(solicitud)
        self.assertEqual(respuesta["indicador"], "cobertura_alcantarillado")
        self.assertTrue(len(respuesta["serie_proyectada"]) > 0)


if __name__ == "__main__":
    unittest.main()
