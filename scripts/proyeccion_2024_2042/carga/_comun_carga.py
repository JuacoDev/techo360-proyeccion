"""Utilidades compartidas por preimagen.py, post_carga_checks.py y reversa_carga.py.
Todas las consultas de estos tres scripts funcionan en una sesion de SOLO LECTURA (sin tablas temporales:
las claves del CSV viajan como arreglos)."""
import hashlib
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import psycopg2

SHA_ESPERADO = "3a74f2ff095f34802c9de7601bf7962165a4d20696f723b5fe4bd55d8ac19ff5"
ES_LOCAL = {"127.0.0.1", "localhost", "::1"}
VARIABLES = ["cobertura_acueducto", "cobertura_alcantarillado", "deficit_habitacional_cuantitativo",
             "deficit_habitacional_cualitativo", "deficit_habitacional_total", "tenencia_informal_terrenos"]
# misma regla que app/origen.py (es_proyeccion), en SQL y SIN '%' para poder usarla junto a parametros
SQL_PROY = "(starts_with(m.originvar, 'proyeccion-') OR right(m.originvar, 11) = '-proyeccion')"
COLS_FILA = "concat_ws('|', m.id, m.municipios, m.namevar, m.valuevar, m.yearm, m.originvar, m.codigodivipola, m.intervaloconfianza)"
CLAVES_CTE = "unnest(%(cods)s::text[], %(vars)s::text[], %(anios)s::int[]) AS k(codigodivipola, namevar, yearm)"
EN_CSV = "EXISTS (SELECT 1 FROM " + CLAVES_CTE + " WHERE k.codigodivipola = m.codigodivipola AND k.namevar = m.namevar AND k.yearm = m.yearm)"


def sha256_archivo(ruta) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cargar_csv(ruta, sha_esperado=SHA_ESPERADO):
    sha = sha256_archivo(ruta)
    if sha != sha_esperado:
        sys.exit(f"SHA256 del CSV {sha} != esperado {sha_esperado}. No se hace nada.")
    df = pd.read_csv(ruta, dtype={"codigodivipola": str})
    return df, sha


def params_claves(df):
    return {"cods": df.codigodivipola.tolist(), "vars": df.nameVar.tolist(), "anios": [int(a) for a in df.yearM]}


def conectar(var_entorno: str, solo_lectura: bool):
    dsn = os.environ.get(var_entorno)
    if not dsn:
        sys.exit(f"Falta la variable de entorno {var_entorno}")
    opciones = "-c default_transaction_read_only=on" if solo_lectura else None
    conn = psycopg2.connect(dsn, connect_timeout=30, options=opciones) if opciones else psycopg2.connect(dsn, connect_timeout=30)
    conn.autocommit = False
    return conn, urlparse(dsn).hostname


def checksums(cur, df):
    """Checksums de la tabla: completa, solo reales, y filas FUERA del CSV. md5 sobre id ordenado."""
    out = {}
    cur.execute(f"SELECT count(*), md5(coalesce(string_agg({COLS_FILA}, E'\\n' ORDER BY m.id), '')) FROM public.mapcalordb m")
    out["completa"] = list(cur.fetchone())
    cur.execute(f"SELECT count(*), md5(coalesce(string_agg({COLS_FILA}, E'\\n' ORDER BY m.id), '')) FROM public.mapcalordb m WHERE NOT {SQL_PROY}")
    out["reales"] = list(cur.fetchone())
    cur.execute(f"SELECT count(*), md5(coalesce(string_agg({COLS_FILA}, E'\\n' ORDER BY m.id), '')) FROM public.mapcalordb m WHERE NOT {EN_CSV}", params_claves(df))
    out["fuera_del_csv"] = list(cur.fetchone())
    return out
