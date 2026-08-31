"""
Verificacion de construir_panel() con datos sinteticos -- no hay acceso
a la base real (ver aviso en extraer_panel.py), asi que esto simula lo
que leer_indicadores_largo()/leer_zonas_municipio() devolverian, sin
tocar ninguna base de datos. El proyecto no tiene pytest en
requirements.txt ni una carpeta tests/ existente, asi que esto usa
unittest (stdlib).

Casos cubiertos, elegidos para probar exactamente el cruce de dos
niveles agregado en este cambio (codigo_divipola primero, nombre
normalizado como fallback):

1. Municipio con codigo_divipola presente y valido -- debe cruzar por
   codigo (metodo principal), sin necesitar el fallback.
2. Municipio con codigo_divipola NULL (historico viejo, cargado antes
   de que MapCalorDB guardara el codigo) pero cuyo nombre SI normaliza
   igual que en zonas_municipio -- debe cruzar por el fallback de
   nombre.
3. Municipio con codigo_divipola que no aparece en zonas_municipio Y
   nombre que tampoco matchea -- debe quedar SIN match (NaN en
   codigo_municipio/codigo_departamento), pero la fila NO se descarta
   del panel.
4. El caso real que motivo pasar a cruzar por codigo en primer lugar:
   dos municipios REALES distintos que comparten el mismo nombre
   oficial normalizado ("El Carmen" en Choco y en Santander, con
   codigos y departamentos distintos) -- deben quedar cada uno con SU
   codigo_departamento correcto, no colapsados en uno solo. Un cruce
   solo por nombre no podria distinguirlos.

Uso:
    python -m scripts.tests_extraer_panel
"""
import unittest

import pandas as pd

from scripts.extraer_panel import construir_panel


class TestConstruirPanelCruceDosNiveles(unittest.TestCase):
    def setUp(self):
        # zonas_municipio sintetica -- incluye el caso real "El Carmen"
        # repetido en dos departamentos distintos.
        self.df_zonas = pd.DataFrame([
            {"codigo_municipio": "68001", "codigo_departamento": "68", "municipio_nombre": "Bucaramanga"},
            {"codigo_municipio": "76001", "codigo_departamento": "76", "municipio_nombre": "Santiago De Cali"},
            {"codigo_municipio": "27245", "codigo_departamento": "27", "municipio_nombre": "El Carmen"},
            {"codigo_municipio": "68235", "codigo_departamento": "68", "municipio_nombre": "El Carmen"},
        ])

    def test_cruce_por_codigo_divipola_metodo_principal(self):
        df_largo = pd.DataFrame([
            {"municipio_nombre": "Bucaramanga", "variable": "densidad_poblacional", "anio": 2020, "valor": 1.0, "codigo_divipola": "68001"},
        ])
        panel = construir_panel(df_largo, self.df_zonas)
        fila = panel.iloc[0]
        self.assertEqual(str(fila["codigo_municipio"]), "68001")
        self.assertEqual(str(fila["codigo_departamento"]), "68")

    def test_codigo_null_con_nombre_distinto_al_oficial_queda_sin_match(self):
        # "Cali" (nombre corto de TerriData/DANE) no normaliza igual que
        # "Santiago De Cali" (nombre oficial IGAC) -- este es exactamente
        # el caso 30 del diagnostico: sin codigo_divipola, el fallback
        # por nombre tampoco lo resuelve, y la fila queda sin match (no
        # descartada, ver test_sin_match_no_se_descarta_la_fila).
        df_largo = pd.DataFrame([
            {"municipio_nombre": "Cali", "variable": "densidad_poblacional", "anio": 2020, "valor": 1.0, "codigo_divipola": None},
        ])
        panel = construir_panel(df_largo, self.df_zonas)
        fila = panel.iloc[0]
        self.assertTrue(pd.isna(fila["codigo_municipio"]))

    def test_fallback_matchea_cuando_nombre_normalizado_coincide(self):
        df_largo = pd.DataFrame([
            {"municipio_nombre": "El Carmen", "variable": "densidad_poblacional", "anio": 2020, "valor": 1.0, "codigo_divipola": None},
        ])
        # Con dos "El Carmen" en zonas_municipio, el fallback por nombre
        # SOLO puede quedarse con uno (drop_duplicates keep="first") --
        # exactamente la ambigüedad real que motiva preferir el codigo.
        panel = construir_panel(df_largo, self.df_zonas)
        fila = panel.iloc[0]
        self.assertIn(str(fila["codigo_municipio"]), {"27245", "68235"})

    def test_sin_match_no_se_descarta_la_fila(self):
        df_largo = pd.DataFrame([
            {"municipio_nombre": "Municipio Inventado", "variable": "densidad_poblacional", "anio": 2020, "valor": 1.0, "codigo_divipola": "99999"},
        ])
        panel = construir_panel(df_largo, self.df_zonas)
        self.assertEqual(len(panel), 1)  # la fila sigue ahi
        fila = panel.iloc[0]
        self.assertTrue(pd.isna(fila["codigo_municipio"]))
        self.assertTrue(pd.isna(fila["codigo_departamento"]))

    def test_codigo_desambigua_dos_municipios_con_mismo_nombre_oficial(self):
        # El caso real confirmado en el diagnostico: "El Carmen" existe
        # en Choco (27245) Y en Santander (68235). Con codigo_divipola
        # presente para AMBAS filas, cada una debe quedar con SU
        # departamento correcto -- no colapsadas en una sola.
        df_largo = pd.DataFrame([
            {"municipio_nombre": "El Carmen de Atrato", "variable": "densidad_poblacional", "anio": 2020, "valor": 1.0, "codigo_divipola": "27245"},
            {"municipio_nombre": "El Carmen de Chucuri", "variable": "densidad_poblacional", "anio": 2020, "valor": 2.0, "codigo_divipola": "68235"},
        ])
        panel = construir_panel(df_largo, self.df_zonas).set_index("municipio_nombre")
        self.assertEqual(str(panel.loc["El Carmen de Atrato", "codigo_departamento"]), "27")
        self.assertEqual(str(panel.loc["El Carmen de Chucuri", "codigo_departamento"]), "68")


    def test_lote_mixto_realista_codigo_y_null_juntos(self):
        # Forma mas realista de un lote real: varias variables, varios
        # años, algunas filas con codigo_divipola (post-migracion) y
        # otras sin el (historico viejo), todo junto en el mismo
        # DataFrame -- para que el fix de dtype (ver aviso en
        # construir_panel) se pruebe tambien en la mezcla, no solo en
        # los casos 100% NaN o 100% con codigo por separado.
        filas = []
        for anio in (2019, 2020):
            filas.append({"municipio_nombre": "Bucaramanga", "variable": "densidad_poblacional", "anio": anio, "valor": 1.0, "codigo_divipola": "68001"})
            filas.append({"municipio_nombre": "Bucaramanga", "variable": "cobertura_acueducto", "anio": anio, "valor": 90.0, "codigo_divipola": "68001"})
            filas.append({"municipio_nombre": "Cali", "variable": "densidad_poblacional", "anio": anio, "valor": 2.0, "codigo_divipola": None})
        df_largo = pd.DataFrame(filas)
        panel = construir_panel(df_largo, self.df_zonas).set_index(["municipio_nombre", "anio"])
        self.assertEqual(str(panel.loc[("Bucaramanga", 2020), "codigo_municipio"]), "68001")
        self.assertTrue(pd.isna(panel.loc[("Cali", 2020), "codigo_municipio"]))
        # Features de rezago (lag) siguen funcionando sobre el resultado
        self.assertEqual(panel.loc[("Bucaramanga", 2020), "cobertura_acueducto_lag_1"], 90.0)


if __name__ == "__main__":
    unittest.main()
