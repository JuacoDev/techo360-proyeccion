"""Paso 0 -- extrae un snapshot de SOLO LECTURA de produccion a CSV local.

Sesion con default_transaction_read_only=on (se confirma con SHOW y se
aborta si no esta en 'on'). No imprime la URL. Deja un MANIFIESTO.sha256
del snapshot para poder demostrar que los pasos siguientes (deterministas)
trabajaron sobre exactamente estos datos.

Vecinos: ST_Touches daba solo 200 pares (los limites IGAC no coinciden
exactamente), asi que se usa ST_DWithin con 0.0005 grados (~55 m).

Uso: python scripts/proyeccion_2024_2042/00_extraer.py
"""
import hashlib
import re
import sys

import pandas as pd
import psycopg2

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from _comun import DIR_ENV_BACKEND, DIR_SNAPSHOT, EXCLUIDOS, asegurar_dirs  # noqa: E402


def conectar():
    env = DIR_ENV_BACKEND.read_text(encoding="utf-8")
    url = re.search(r"^DATABASE_URL\s*=\s*(.+)$", env, re.M).group(1).strip().strip("\"'")
    conn = psycopg2.connect(url, connect_timeout=30,
                            options="-c default_transaction_read_only=on -c statement_timeout=300000")
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SHOW default_transaction_read_only")
    if cur.fetchone()[0] != "on":
        raise SystemExit("La sesion NO es de solo lectura; abortando.")
    return conn


def consulta(conn, sql, params=None):
    return pd.read_sql_query(sql, conn, params=params)


def main():
    asegurar_dirs()
    conn = conectar()
    zonas = consulta(conn, """
        SELECT codigo_municipio AS codigo, codigo_departamento AS depto, nombre,
               ST_Area(geom::geography)/1e6 AS area_km2,
               ST_X(ST_Centroid(geom)) AS lon, ST_Y(ST_Centroid(geom)) AS lat
        FROM zonas_municipio WHERE codigo_municipio <> ALL(%(ex)s) ORDER BY codigo_municipio
    """, {"ex": list(EXCLUIDOS)})
    filas = consulta(conn, """
        SELECT m.codigodivipola, m.municipios, m.namevar, m.yearm, m.valuevar, m.originvar, m.intervaloconfianza
        FROM mapcalordb m JOIN zonas_municipio z ON z.codigo_municipio = m.codigodivipola
        WHERE m.codigodivipola <> ALL(%(ex)s)
        ORDER BY m.codigodivipola, m.namevar, m.yearm
    """, {"ex": list(EXCLUIDOS)})
    vecinos = consulta(conn, """
        SELECT a.codigo_municipio AS a, b.codigo_municipio AS b
        FROM zonas_municipio a JOIN zonas_municipio b
          ON a.codigo_municipio < b.codigo_municipio AND ST_DWithin(a.geom, b.geom, 0.0005)
        WHERE a.codigo_municipio <> ALL(%(ex)s) AND b.codigo_municipio <> ALL(%(ex)s)
        ORDER BY 1, 2
    """, {"ex": list(EXCLUIDOS)})
    # Filas 2024 de las 3 variables del score, TODAS (incluye codigos fuera de
    # zonas_municipio, p.ej. corregimientos): calcular_limites_base() del
    # backend toma min/max sobre todo lo que no sea 01001, no solo el universo.
    limites = consulta(conn, """
        SELECT codigodivipola, namevar, valuevar, originvar FROM mapcalordb
        WHERE yearm = 2024 AND codigodivipola IS NOT NULL AND codigodivipola <> '01001'
          AND namevar IN ('cobertura_acueducto','cobertura_alcantarillado','deficit_habitacional_cuantitativo')
        ORDER BY 1, 2
    """)
    conn.close()
    limites.to_csv(DIR_SNAPSHOT / "score_2024_todos_los_codigos.csv", index=False)

    zonas.to_csv(DIR_SNAPSHOT / "zonas.csv", index=False)
    filas.to_csv(DIR_SNAPSHOT / "mapcalordb.csv", index=False)
    vecinos.to_csv(DIR_SNAPSHOT / "vecinos.csv", index=False)
    with open(DIR_SNAPSHOT / "MANIFIESTO.sha256", "w") as f:
        for nombre in ("zonas.csv", "mapcalordb.csv", "vecinos.csv", "score_2024_todos_los_codigos.csv"):
            h = hashlib.sha256((DIR_SNAPSHOT / nombre).read_bytes()).hexdigest()
            f.write(f"{h}  {nombre}\n")
    print(f"zonas={len(zonas)} filas={len(filas)} pares_vecinos={len(vecinos)}")


if __name__ == "__main__":
    main()
