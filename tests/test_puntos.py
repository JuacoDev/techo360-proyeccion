"""Pruebas de app/services/puntos.py (proyeccion bajo demanda de un punto), sin base de datos:
el repositorio se sustituye por un doble. Casos 1-8, 11-13 de claude/fase3-4-umbrales-y-metodos.md
seccion 5, mas compatibilidad del RPC (los tipos existentes siguen funcionando) y no-escritura.
(Los casos 9 -score- y 10 -cache- son del backend.)

Se corre con: python -m unittest discover -s tests
"""
import os
import unittest
from unittest.mock import patch

from app import rabbitmq_consumer as consumidor
from app.origen import es_proyeccion
from app.services import puntos as p


def fila(anio, valor, origen="TerriData"):
    return {"anio": anio, "valor": valor, "origen": origen}


class RepoFalso:
    def __init__(self, series, universo=("68001", "05001")):
        self.series, self.universo, self.consultas = series, set(universo), 0

    def en_universo(self, codigo):
        return codigo in self.universo

    def obtener_serie(self, indicador, codigo):
        self.consultas += 1
        return self.series.get((indicador, codigo), [])


def calcular(indicador, anio, serie, **kw):
    return p.calcular_punto(indicador, "68001", anio, serie, True, **kw)


class Coberturas(unittest.TestCase):
    def test_1_hueco_interior_interpola_exacto(self):
        r = calcular("cobertura_acueducto", 2012, [fila(2010, 50.0), fila(2014, 70.0)])
        self.assertEqual((r["estado"], r["valor"], r["metodo"]), ("calculado", 60.0, "interpolacion-lineal"))
        self.assertEqual(r["origen"], "proyeccion-techo360-interpolacion")

    def test_2_solo_real_anterior_usa_ultimo_valor(self):
        r = calcular("cobertura_alcantarillado", 2013, [fila(2009, 40.0), fila(2011, 55.5)])
        self.assertEqual((r["estado"], r["valor"], r["metodo"]), ("calculado", 55.5, "ultimo-valor-real"))
        self.assertEqual(r["origen"], "proyeccion-techo360-ultimo-valor")

    def test_2b_interpolacion_no_permitida_usa_ultimo_valor(self):
        r = calcular("cobertura_acueducto", 2012, [fila(2010, 50.0), fila(2014, 70.0)],
                     interpolacion_permitida={"cobertura_acueducto": False})
        self.assertEqual((r["valor"], r["metodo"]), (50.0, "ultimo-valor-real"))

    def test_3_sin_real_anterior_es_sin_historia(self):
        r = calcular("cobertura_acueducto", 2008, [fila(2010, 50.0)])
        self.assertEqual((r["estado"], r["motivo"], r["valor"]), ("sin_dato", "sin_historia", None))
        r = calcular("cobertura_acueducto", 2008, [])
        self.assertEqual(r["motivo"], "sin_historia")

    def test_las_proyecciones_no_cuentan_como_reales(self):
        serie = [fila(2010, 50.0), fila(2011, 99.0, "proyeccion-techo360-ultimo-valor")]
        r = calcular("cobertura_acueducto", 2013, serie)
        self.assertEqual(r["valor"], 50.0)


class Deficit(unittest.TestCase):
    SERIE = [fila(2018, 10.0, "DANE-CNPV2018"), fila(2024, 22.0, "proyeccion-tendencia-nacional-2018-2024")]

    def test_4_interpolacion_2018_2024(self):
        for anio, esperado in ((2019, 12.0), (2020, 14.0), (2021, 16.0), (2022, 18.0), (2023, 20.0)):
            with self.subTest(anio):
                r = calcular("deficit_habitacional_cuantitativo", anio, self.SERIE)
                self.assertEqual((r["estado"], r["valor"]), ("calculado", esperado))
                self.assertEqual(r["origen"], "proyeccion-techo360-interpolacion-deficit")

    def test_5_falta_un_extremo(self):
        for serie in ([fila(2024, 22.0, "proyeccion-x")], [fila(2018, 10.0, "DANE-CNPV2018")], []):
            with self.subTest(len(serie)):
                r = calcular("deficit_habitacional_cualitativo", 2021, serie)
                self.assertEqual((r["estado"], r["motivo"]), ("sin_dato", "sin_extremo"))

    def test_5b_2018_real_debe_ser_real(self):
        serie = [fila(2018, 10.0, "proyeccion-x"), fila(2024, 22.0, "proyeccion-y")]
        self.assertEqual(calcular("deficit_habitacional_cuantitativo", 2021, serie)["motivo"], "sin_extremo")

    def test_2018_y_anios_fuera_de_la_ventana_son_sin_dato(self):
        r = calcular("deficit_habitacional_cuantitativo", 2018, [fila(2024, 22.0, "proyeccion-y")])
        self.assertEqual((r["estado"], r["motivo"]), ("sin_dato", "sin_historia"))

    def test_6_recorte_mayor_a_5pp_es_valor_sospechoso(self):
        serie = [fila(2018, 10.0, "DANE-CNPV2018"), fila(2024, 300.0, "proyeccion-y")]
        r = calcular("deficit_habitacional_cuantitativo", 2021, serie)  # 155 -> recorta 55 pp
        self.assertEqual((r["estado"], r["motivo"], r["valor"]), ("sin_dato", "valor_sospechoso", None))

    def test_6b_recorte_de_hasta_5pp_se_acepta(self):
        serie = [fila(2018, 100.0, "DANE-CNPV2018"), fila(2024, 108.0, "proyeccion-y")]
        r = calcular("deficit_habitacional_cuantitativo", 2021, serie)  # 104 -> 100 (4 pp)
        self.assertEqual((r["estado"], r["valor"]), ("calculado", 100.0))

    def test_6c_densidad_negativa_es_sospechosa(self):
        serie = [fila(2010, 10.0), fila(2011, 8.0), fila(2012, 6.0), fila(2013, 4.0)]
        r = calcular("densidad_poblacional", 2020, serie, interpolacion_permitida={})
        self.assertEqual((r["estado"], r["motivo"]), ("sin_dato", "valor_sospechoso"))


class FueraDeAlcance(unittest.TestCase):
    def test_7_codigo_fuera_del_universo_y_anio_fuera_de_rango(self):
        repo = RepoFalso({("deficit_habitacional_cuantitativo", "91263"): [fila(2018, 1.0)]})
        r = p.completar_punto("deficit_habitacional_cuantitativo", "91263", 2021, repo=repo)
        self.assertEqual((r["estado"], r["motivo"]), ("sin_dato", "fuera_de_universo"))
        self.assertEqual(repo.consultas, 0)
        for indicador, anio in (("cobertura_acueducto", 2007), ("cobertura_acueducto", 2043),
                                ("deficit_habitacional_cuantitativo", 2017), ("densidad_poblacional", 1999)):
            with self.subTest((indicador, anio)):
                self.assertEqual(calcular(indicador, anio, [])["motivo"], "fuera_de_rango")

    def test_indicador_desconocido_lanza_error(self):
        with self.assertRaises(ValueError):
            calcular("tenencia_informal_terrenos", 2021, [])


class Densidad(unittest.TestCase):
    def test_8_cuatro_reales_regresion(self):
        serie = [fila(2010, 100.0), fila(2011, 110.0), fila(2012, 120.0), fila(2013, 130.0)]
        r = calcular("densidad_poblacional", 2014, serie)
        self.assertEqual((r["estado"], r["metodo"], r["valor"]), ("calculado", "regresion-lineal-4puntos", 140.0))
        self.assertEqual(r["origen"], "proyeccion-regresion-lineal-4puntos")

    def test_8b_dos_reales_es_sin_historia(self):
        r = calcular("densidad_poblacional", 2014, [fila(2012, 120.0), fila(2013, 130.0)])
        self.assertEqual((r["estado"], r["motivo"]), ("sin_dato", "sin_historia"))

    def test_8c_reales_a_ambos_lados_interpola(self):
        r = calcular("densidad_poblacional", 2004, [fila(2003, 100.0), fila(2007, 120.0)])
        self.assertEqual((r["valor"], r["metodo"]), (105.0, "interpolacion-lineal"))

    def test_8d_regresion_usa_los_4_ultimos_reales(self):
        serie = [fila(2005, 1000.0), fila(2010, 100.0), fila(2011, 110.0), fila(2012, 120.0), fila(2013, 130.0)]
        self.assertEqual(calcular("densidad_poblacional", 2014, serie)["valor"], 140.0)


class Invariantes(unittest.TestCase):
    CASOS = [
        ("cobertura_acueducto", 2012, [fila(2010, 50.0), fila(2014, 70.0)]),
        ("cobertura_acueducto", 2013, [fila(2011, 55.5)]),
        ("deficit_habitacional_cuantitativo", 2021, Deficit.SERIE),
        ("densidad_poblacional", 2014, [fila(2010, 100.0), fila(2011, 110.0), fila(2012, 120.0), fila(2013, 130.0)]),
    ]

    def test_12_origen_proyeccion_es_real_false_y_fiabilidad_30(self):
        for indicador, anio, serie in self.CASOS:
            with self.subTest(indicador):
                r = calcular(indicador, anio, serie)
                self.assertEqual(r["estado"], "calculado")
                self.assertTrue(es_proyeccion(r["origen"]))
                self.assertIs(r["es_real"], False)
                self.assertEqual(r["fiabilidad"], 30.0)

    def test_13_un_valor_existente_nunca_se_reemplaza(self):
        serie = [fila(2010, 50.0), fila(2012, 61.0), fila(2014, 70.0)]
        r = calcular("cobertura_acueducto", 2012, serie)
        self.assertEqual((r["estado"], r["motivo"], r["valor"]), ("sin_dato", "fila_existente", None))
        r = calcular("cobertura_acueducto", 2012, [fila(2012, 61.0, "proyeccion-techo360-ultimo-valor")])
        self.assertEqual(r["motivo"], "fila_existente")

    def test_idempotencia(self):
        for indicador, anio, serie in self.CASOS:
            self.assertEqual(calcular(indicador, anio, serie), calcular(indicador, anio, serie))

    def test_sin_valor_en_la_fila_se_trata_como_hueco(self):
        r = calcular("cobertura_acueducto", 2012, [fila(2010, 50.0), {"anio": 2012, "valor": None, "origen": "TerriData"}])
        self.assertEqual((r["estado"], r["valor"]), ("calculado", 50.0))


class Lote(unittest.TestCase):
    def test_11_lote_de_41_es_error_de_validacion(self):
        repo = RepoFalso({})
        pedidos = [{"indicador": "cobertura_acueducto", "codigo": "68001", "anio": 2012}] * 41
        with self.assertRaises(ValueError):
            p.completar_puntos(pedidos, repo=repo)
        self.assertEqual(repo.consultas, 0)
        self.assertEqual(len(p.completar_puntos(pedidos[:40], repo=repo)), 40)
        self.assertEqual(repo.consultas, 1)  # una consulta por (indicador, codigo)

    def test_lote_vacio_o_no_lista(self):
        for malo in ([], None, "x"):
            with self.assertRaises(ValueError):
                p.completar_puntos(malo, repo=RepoFalso({}))

    def test_lote_conserva_el_orden(self):
        repo = RepoFalso({("cobertura_acueducto", "68001"): [fila(2010, 50.0), fila(2014, 70.0)]})
        pedidos = [{"indicador": "cobertura_acueducto", "codigo": "68001", "anio": a} for a in (2013, 2011, 2012)]
        self.assertEqual([r["anio"] for r in p.completar_puntos(pedidos, repo=repo)], [2013, 2011, 2012])


class RpcYNoEscritura(unittest.TestCase):
    def test_tipo_puntos_del_consumer(self):
        with patch.object(consumidor, "completar_puntos", return_value=[{"estado": "sin_dato"}]) as m:
            salida = consumidor._calcular({"tipo": "puntos", "puntos": [{"indicador": "x", "codigo": "1", "anio": 2012}]})
        self.assertEqual(salida, {"puntos": [{"estado": "sin_dato"}]})
        m.assert_called_once()

    def test_compatibilidad_tipos_existentes(self):
        with patch.object(consumidor, "calcular_zonas", return_value=[{"codigo_municipio": "68001"}]) as z:
            salida = consumidor._calcular({"tipo": "zonas", "indicador": "densidad_poblacional", "anio": 2026})
        z.assert_called_once_with("densidad_poblacional", 2026)
        self.assertEqual(salida["zonas"], [{"codigo_municipio": "68001"}])
        self.assertEqual(salida["modelo"], "regresion-lineal-4puntos")
        with patch.object(consumidor, "calcular_serie_proyectada", return_value={"serie_proyectada": []}) as s:
            salida = consumidor._calcular({"codigo_municipio": "68001", "indicador": "densidad_poblacional", "anio_fin": 2030})
        s.assert_called_once()
        self.assertEqual(salida, {"serie_proyectada": []})

    def test_el_modulo_no_escribe_en_la_base(self):
        raiz = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app")
        for ruta in ("services/puntos.py", "repositories/puntos.py"):
            with open(os.path.join(raiz, ruta), encoding="utf-8") as f:
                codigo = f.read().lower()
            for prohibido in ("guardar_indicador", "insert ", "update ", "delete ", "drop ", "truncate", ".commit("):
                with self.subTest((ruta, prohibido)):
                    # los docstrings pueden nombrar "guardar_indicador"; no debe haber importacion ni SQL de escritura
                    lineas = [l for l in codigo.splitlines() if prohibido in l and not l.lstrip().startswith(('"', "#", "*"))]
                    if prohibido == "guardar_indicador":
                        lineas = [l for l in lineas if "import" in l]
                    self.assertEqual(lineas, [])


if __name__ == "__main__":
    unittest.main()
