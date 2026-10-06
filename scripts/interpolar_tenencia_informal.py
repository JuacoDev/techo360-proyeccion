"""
>>> REVERTIDO POR COMPLETO (31-ago-2026) -- NO VOLVER A CORRER ESTE
>>> SCRIPT TAL CUAL, VER AVISO ABAJO <<<
La premisa de este script (2 años reales, 2023 y 2026) resulto ser
falsa: yearm=2023 era un dato fantasma (mismo snapshot en vivo de IGAC
Catastro que 2026, cargado entre el 17 y el 23-ago-2026 con --anio 2023
pasado a mano, nunca un dato real de 2023 -- confirmado con evidencia de
codigo + estadistica de diferencia = 0.000000 en 1,031/1,031 municipios,
ver techo360-backend/ingestion/db.py). Se borraron en produccion, en
este orden: las 2,062 filas que este script inserto, y despues las
1,033 filas fantasma de yearm=2023 mismas. tenencia_informal_terrenos
quedo con una sola observacion real (2026). Este archivo se conserva
como registro reproducible del metodo (valido en si mismo, para el dia
en que la variable tenga de verdad 2+ años reales) y como advertencia
explicita de este caso -- no correrlo contra el estado actual de la
base (fallaria de todas formas: ya no existe yearm=2023 para esta
variable).

Interpolacion lineal de tenencia_informal_terrenos para 2024/2025, entre
los 2 unicos años reales disponibles (2023 sin codigo DIVIPOLA, 2026 con
codigo). Cruce por nombre normalizado (misma funcion que
scripts/extraer_panel.py::normalizar_nombre, copiada aqui tal cual --
ver aviso mas abajo sobre por que copiada y no importada) -- SOLO se
interpola cuando el nombre de 2023 matchea exactamente UN municipio de
2026; match ambiguo (nombre duplicado entre departamentos, ej. Chima
vs. Chimá) o sin match se EXCLUYEN y se reportan, nunca se asignan por
adivinanza.

>>> POR QUE normalizar_nombre() ESTA COPIADA, NO IMPORTADA <<<
Al momento de escribir este script, scripts/extraer_panel.py vive en las
ramas development/main de este repo pero NO en Testing (divergencia de
ramas detectada en vivo, no corregida aqui por estar fuera del alcance
de esta tarea) -- importarlo habria roto este script en Testing. La
funcion se copio literal (mismo NFKD + strip de tildes + minusculas +
sin puntuacion), no se reimplemento con logica distinta.

>>> HALLAZGO IMPORTANTE (30-ago-2026), reportado al usuario, no resuelto
>>> aqui <<<
Los 1,031 municipios con match unico dieron el MISMO valor exacto en
2023 y 2026, sin excepcion -- ambos años vienen de originvar='IGAC
Catastro', que es una fuente de snapshot EN VIVO (capa ArcGIS
actualizada bimestralmente, ver techo360-backend/ingestion/
igac_catastro_adapter.py), no una serie historica real por año. Esto
sugiere que las filas "2023" y "2026" probablemente se cargaron en
momentos cercanos en el tiempo real, con los años elegidos para acotar
el rango -- no porque reflejen el estado real de la tenencia informal en
esos años especificos. La interpolacion es matematicamente correcta
dado el insumo, pero como ambos extremos son identicos, los valores de
2024/2025 resultantes son planos (sin señal temporal real). Decision
pendiente del usuario sobre como tratar esto a futuro -- no se toco la
carga original de 2023/2026 en esta tarea.

>>> YA EJECUTADO CONTRA PRODUCCION (30-ago-2026), Y YA REVERTIDO
>>> (31-ago-2026) -- ver aviso al inicio del modulo <<<
Corrida real completada y verificada: 2,062 filas insertadas (1,031
municipios x 2 años), 2 municipios excluidos por ambiguedad (Chima/
Chimá), 0 sin match. intervaloconfianza = NULL en todas (no hay error
historico calculable con solo 2 puntos de interpolacion -- se prefiere
NULL a un numero inventado). Las 2,062 filas se borraron al dia
siguiente, junto con la fila fantasma yearm=2023 que las origino.
"""
import re
import unicodedata
from collections import defaultdict

import psycopg2
from psycopg2.extras import execute_values


def normalizar_nombre(nombre) -> str:
    """Copiado tal cual de scripts/extraer_panel.py (rama development)
    -- misma clave de cruce ya usada en el proyecto, no una
    normalizacion nueva. Ver aviso del modulo sobre por que copiada."""
    if nombre is None:
        return ""
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def obtener_url():
    with open(r"C:\Users\Eridanus_2\OneDrive\Documentos\Visual Studio\techo360\techo360-backend\.env", encoding="utf-8") as fh:
        contenido = fh.read()
    m = re.search(r"^DATABASE_URL=(.+)$", contenido, re.MULTILINE)
    return m.group(1).strip()


VARIABLE = "tenencia_informal_terrenos"
ANIO_BASE_1 = 2023
ANIO_BASE_2 = 2026
ANIOS_INTERPOLAR = [2024, 2025]
ORIGIN_INTERPOLACION = "interpolacion-lineal"


def main():
    conn = psycopg2.connect(obtener_url(), connect_timeout=15)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT municipios, valuevar FROM public.mapcalordb WHERE namevar = %s AND yearm = %s",
                (VARIABLE, ANIO_BASE_1),
            )
            filas_2023 = cur.fetchall()
            cur.execute(
                "SELECT municipios, valuevar, codigodivipola FROM public.mapcalordb WHERE namevar = %s AND yearm = %s",
                (VARIABLE, ANIO_BASE_2),
            )
            filas_2026 = cur.fetchall()

        print(f"2023: {len(filas_2023)} filas (sin codigo) | 2026: {len(filas_2026)} filas (con codigo)")

        # Agrupar 2026 por nombre normalizado -> lista de (codigo, nombre_original, valor)
        indice_2026 = defaultdict(list)
        for nombre, valor, codigo in filas_2026:
            indice_2026[normalizar_nombre(nombre)].append((codigo, nombre, valor))

        match_unico = []      # (nombre_2023, valor_2023, codigo_2026, nombre_2026, valor_2026)
        match_ambiguo = []    # (nombre_2023, [candidatos])
        sin_match = []        # nombre_2023

        for nombre_2023, valor_2023 in filas_2023:
            clave = normalizar_nombre(nombre_2023)
            candidatos = indice_2026.get(clave, [])
            if len(candidatos) == 0:
                sin_match.append(nombre_2023)
            elif len(candidatos) == 1:
                codigo, nombre_2026, valor_2026 = candidatos[0]
                match_unico.append((nombre_2023, float(valor_2023), codigo, nombre_2026, float(valor_2026)))
            else:
                match_ambiguo.append((nombre_2023, candidatos))

        print()
        print("=" * 70)
        print(f"Match unico (interpolable):  {len(match_unico)}")
        print(f"Match ambiguo (excluido):    {len(match_ambiguo)}")
        print(f"Sin match (excluido):        {len(sin_match)}")
        print(f"Total 2023:                  {len(filas_2023)}  (verificacion: {len(match_unico)+len(match_ambiguo)+len(sin_match)})")
        print("=" * 70)

        print("\n--- Lista completa: MATCH AMBIGUO (nombre 2023 -> candidatos 2026) ---")
        for nombre_2023, candidatos in sorted(match_ambiguo):
            candidatos_fmt = ", ".join(f"{c[1]}({c[0]})" for c in candidatos)
            print(f"  {nombre_2023!r} -> {candidatos_fmt}")

        print("\n--- Lista completa: SIN MATCH (nombre 2023 sin ningun candidato en 2026) ---")
        for nombre_2023 in sorted(sin_match):
            print(f"  {nombre_2023!r}")

        # --- Interpolacion lineal, solo grupo match_unico ---
        filas_a_insertar = []
        muestra = []
        for nombre_2023, v2023, codigo, nombre_2026, v2026 in match_unico:
            for anio in ANIOS_INTERPOLAR:
                valor_interp = v2023 + (v2026 - v2023) * (anio - ANIO_BASE_1) / (ANIO_BASE_2 - ANIO_BASE_1)
                filas_a_insertar.append((nombre_2026, VARIABLE, valor_interp, anio, ORIGIN_INTERPOLACION, codigo))
            muestra.append((nombre_2026, codigo, v2023, v2026))

        print(f"\nFilas a insertar (2 x {len(match_unico)} municipios match unico): {len(filas_a_insertar)}")

        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM public.mapcalordb WHERE namevar = %s AND yearm = ANY(%s)",
                (VARIABLE, ANIOS_INTERPOLAR),
            )
            preexistentes = cur.fetchone()[0]
        print(f"Verificacion pre-insercion, filas ya existentes en 2024/2025 (debe ser 0): {preexistentes}")
        if preexistentes > 0:
            print("DETENIDO: ya existen filas en 2024/2025 para esta variable -- no se inserta nada, revisar a mano.")
            return

        with conn.cursor() as cur:
            execute_values(
                cur,
                """
                INSERT INTO public.mapcalordb (municipios, namevar, valuevar, yearm, originvar, codigodivipola)
                VALUES %s
                """,
                filas_a_insertar,
            )
        conn.commit()
        print("Commit realizado.")

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.mapcalordb;")
            print(f"\nTotal mapcalordb ahora: {cur.fetchone()[0]}")

        print("\n--- Muestra de verificacion (primeros 5 del grupo match unico) ---")
        for nombre_2026, codigo, v2023, v2026 in sorted(muestra)[:5]:
            v2024 = v2023 + (v2026 - v2023) * (2024 - ANIO_BASE_1) / (ANIO_BASE_2 - ANIO_BASE_1)
            v2025 = v2023 + (v2026 - v2023) * (2025 - ANIO_BASE_1) / (ANIO_BASE_2 - ANIO_BASE_1)
            print(f"  {nombre_2026} ({codigo}): 2023={v2023:.4f}  2024={v2024:.4f}  2025={v2025:.4f}  2026={v2026:.4f}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
