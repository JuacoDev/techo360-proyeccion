"""Carga del CSV de proyecciones (M4) a mapcalordb: UNA transaccion, set-based, con pre y post chequeos.

NO se ha ejecutado contra produccion. Dos modos:
  * por defecto (DRY-RUN): hace TODO dentro de una transaccion y termina con ROLLBACK. Solo contra una
    copia LOCAL (127.0.0.1 / localhost): el dry-run ESCRIBE dentro de la transaccion, y la conexion de
    solo lectura de produccion lo impediria (y no se debe intentar escribir alli ni siquiera para revertir).
  * --ejecutar: hace COMMIT. Contra un host que NO sea local exige ademas --confirmar-sha <sha256 completo
    del CSV> (una confirmacion explicita y consciente de QUE se carga) y la variable TECHO360_AUTORIZO_CARGA=1.

La cadena de conexion va SIEMPRE por la variable de entorno CARGA_DATABASE_URL (nunca por argumento ni archivo).

Mapeo de columnas (CSV -> mapcalordb):
  codigodivipola            -> codigodivipola
  nameVar                   -> namevar
  yearM                     -> yearm
  valuevar                  -> valuevar
  originvar                 -> originvar
  fiabilidad                -> intervaloconfianza   (unica columna disponible; reusada como % 0-100)
  intervalo_confianza ("lo;hi") -> NO se guarda (no hay columna; seguiria solo en el CSV)
  (municipios)              -> zonas_municipio.nombre del codigo (columna NOT NULL; solo en los INSERT)
  id                        -> serial (default de la secuencia mapcalordb_id_seq)
  accion = update  -> UPDATE de valuevar/originvar/intervaloconfianza, y SOLO si la fila existente es proyeccion-*
  accion = insert  -> INSERT (el trigger BEFORE INSERT bloquea duplicados; por eso nunca ON CONFLICT)

Garantias (se verifican dentro de la transaccion y se aborta si alguna falla):
  - claves unicas en el CSV; todos los codigos existen en zonas_municipio;
  - cada update coincide con EXACTAMENTE una fila existente y esa fila es proyeccion-* (ninguna real);
  - ningun insert choca con una clave existente;
  - tras la carga: las filas que NO estan en el CSV (incluidas todas las reales) tienen el mismo checksum
    que antes; cada fila del CSV esta en la tabla con los mismos valores; filas = antes + inserts.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse

import psycopg2

SHA_ESPERADO = "3a74f2ff095f34802c9de7601bf7962165a4d20696f723b5fe4bd55d8ac19ff5"
ES_LOCAL = {"127.0.0.1", "localhost", "::1"}
COLS_FILA = "concat_ws('|', m.id, m.municipios, m.namevar, m.valuevar, m.yearm, m.originvar, m.codigodivipola, m.intervaloconfianza)"
CLAVE = "m.codigodivipola = c.codigodivipola AND m.namevar = c.namevar AND m.yearm = c.yearm"
ES_PROY = "(m.originvar LIKE 'proyeccion-%' OR m.originvar LIKE '%-proyeccion')"
VARIABLES = ("cobertura_acueducto", "cobertura_alcantarillado", "deficit_habitacional_cuantitativo",
             "deficit_habitacional_cualitativo", "deficit_habitacional_total", "tenencia_informal_terrenos")


class Abortar(Exception):
    pass


def sha256_archivo(ruta: Path) -> str:
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def uno(cur, sql, params=None):
    cur.execute(sql, params)
    return cur.fetchone()[0]


def checksum_no_objetivo(cur):
    """(filas, md5) de las filas de mapcalordb cuya clave NO esta en el CSV (incluye TODAS las reales)."""
    cur.execute(f"""SELECT count(*), md5(coalesce(string_agg({COLS_FILA}, E'\\n' ORDER BY m.id), ''))
                    FROM public.mapcalordb m
                    WHERE NOT EXISTS (SELECT 1 FROM carga_csv c WHERE {CLAVE})""")
    return cur.fetchone()


def checksum_reales(cur):
    cur.execute(f"""SELECT count(*), md5(coalesce(string_agg({COLS_FILA}, E'\\n' ORDER BY m.id), ''))
                    FROM public.mapcalordb m WHERE NOT {ES_PROY}""")
    return cur.fetchone()


def ejecutar(csv: Path, dsn: str, commit: bool, informe: Path | None):
    informe_d: dict = {"csv": str(csv), "sha256_csv": sha256_archivo(csv), "modo": "COMMIT" if commit else "DRY-RUN (ROLLBACK)"}
    conn = psycopg2.connect(dsn, connect_timeout=30)
    conn.autocommit = False
    cur = conn.cursor()
    try:
        cur.execute("SHOW transaction_read_only")
        if cur.fetchone()[0] != "off":
            raise Abortar("la sesion es de SOLO LECTURA: no se puede ni simular la carga (esto es lo esperado en produccion).")
        informe_d["base"] = uno(cur, "SELECT current_database()")
        # --- carga del CSV a una tabla temporal (se descarta sola al terminar la transaccion)
        cur.execute("""CREATE TEMP TABLE carga_csv (codigodivipola varchar(5), namevar text, yearm int, valuevar numeric,
                       originvar text, fiabilidad numeric, intervalo text, accion text) ON COMMIT DROP""")
        with open(csv, "r", encoding="utf-8", newline="") as f:
            cur.copy_expert("COPY carga_csv (codigodivipola, namevar, yearm, valuevar, originvar, fiabilidad, intervalo, accion) "
                            "FROM STDIN WITH (FORMAT csv, HEADER true)", f)
        n_csv = uno(cur, "SELECT count(*) FROM carga_csv")
        n_upd = uno(cur, "SELECT count(*) FROM carga_csv WHERE accion = 'update'")
        n_ins = uno(cur, "SELECT count(*) FROM carga_csv WHERE accion = 'insert'")
        informe_d.update(filas_csv=n_csv, updates_csv=n_upd, inserts_csv=n_ins)

        # --- PRE-CHEQUEOS (todos deben dar el valor esperado)
        pre = {
            "claves duplicadas en el CSV": (uno(cur, "SELECT count(*) - count(DISTINCT (codigodivipola, namevar, yearm)) FROM carga_csv"), 0),
            "accion distinta de insert/update": (uno(cur, "SELECT count(*) FROM carga_csv WHERE accion NOT IN ('insert','update') OR accion IS NULL"), 0),
            "codigos que no estan en zonas_municipio": (uno(cur, "SELECT count(*) FROM carga_csv c LEFT JOIN public.zonas_municipio z ON z.codigo_municipio = c.codigodivipola WHERE z.codigo_municipio IS NULL"), 0),
            "variables fuera de alcance": (uno(cur, "SELECT count(*) FROM carga_csv WHERE namevar <> ALL(%s)", (list(VARIABLES),)), 0),
            "anios fuera de 2024-2042": (uno(cur, "SELECT count(*) FROM carga_csv WHERE yearm NOT BETWEEN 2024 AND 2042"), 0),
            "valor o fiabilidad fuera de [0,100] o nulos": (uno(cur, "SELECT count(*) FROM carga_csv WHERE valuevar IS NULL OR valuevar NOT BETWEEN 0 AND 100 OR fiabilidad IS NULL OR fiabilidad NOT BETWEEN 0 AND 100"), 0),
            "originvar sin prefijo proyeccion-techo360-": (uno(cur, "SELECT count(*) FROM carga_csv WHERE originvar NOT LIKE 'proyeccion-techo360-%'"), 0),
            "updates que NO coinciden con una fila existente": (uno(cur, f"SELECT count(*) FROM carga_csv c WHERE c.accion='update' AND NOT EXISTS (SELECT 1 FROM public.mapcalordb m WHERE {CLAVE})"), 0),
            "updates sobre una fila REAL (no proyeccion-*)": (uno(cur, f"SELECT count(*) FROM carga_csv c JOIN public.mapcalordb m ON {CLAVE} WHERE c.accion='update' AND NOT {ES_PROY}"), 0),
            "inserts cuya clave YA existe": (uno(cur, f"SELECT count(*) FROM carga_csv c JOIN public.mapcalordb m ON {CLAVE} WHERE c.accion='insert'"), 0),
        }
        informe_d["pre_chequeos"] = {k: {"valor": v, "esperado": e, "ok": v == e} for k, (v, e) in pre.items()}
        malos = [k for k, (v, e) in pre.items() if v != e]
        if malos:
            raise Abortar("pre-chequeos fallidos: " + "; ".join(f"{k}={pre[k][0]}" for k in malos))

        # --- estado ANTES
        filas_antes = uno(cur, "SELECT count(*) FROM public.mapcalordb")
        no_obj_antes = checksum_no_objetivo(cur)
        reales_antes = checksum_reales(cur)
        cur.execute(f"""CREATE TEMP TABLE pre_imagen ON COMMIT DROP AS
                        SELECT m.id, m.codigodivipola, m.namevar, m.yearm, m.municipios, m.valuevar, m.originvar, m.intervaloconfianza
                        FROM public.mapcalordb m JOIN carga_csv c ON {CLAVE} WHERE c.accion = 'update'""")
        informe_d["antes"] = {"filas": filas_antes, "no_objetivo": list(no_obj_antes), "reales": list(reales_antes)}

        # --- CARGA
        cur.execute(f"""UPDATE public.mapcalordb m SET valuevar = c.valuevar, originvar = c.originvar, intervaloconfianza = c.fiabilidad
                        FROM carga_csv c WHERE c.accion = 'update' AND {CLAVE} AND {ES_PROY}""")
        upd = cur.rowcount
        cur.execute("""INSERT INTO public.mapcalordb (municipios, namevar, valuevar, yearm, originvar, codigodivipola, intervaloconfianza)
                       SELECT z.nombre, c.namevar, c.valuevar, c.yearm, c.originvar, c.codigodivipola, c.fiabilidad
                       FROM carga_csv c JOIN public.zonas_municipio z ON z.codigo_municipio = c.codigodivipola
                       WHERE c.accion = 'insert' ORDER BY c.namevar, c.codigodivipola, c.yearm""")
        ins = cur.rowcount
        informe_d["aplicado"] = {"updates": upd, "inserts": ins}
        if upd != n_upd or ins != n_ins:
            raise Abortar(f"filas afectadas distintas de lo esperado: updates {upd}/{n_upd}, inserts {ins}/{n_ins}")

        # --- POST-CHEQUEOS (dentro de la transaccion, antes de decidir COMMIT/ROLLBACK)
        no_obj_desp = checksum_no_objetivo(cur)
        reales_desp = checksum_reales(cur)
        filas_desp = uno(cur, "SELECT count(*) FROM public.mapcalordb")
        iguales = uno(cur, f"""SELECT count(*) FROM carga_csv c JOIN public.mapcalordb m ON {CLAVE}
                               WHERE m.valuevar = c.valuevar AND m.originvar = c.originvar AND m.intervaloconfianza = c.fiabilidad""")
        sin_nombre = uno(cur, f"SELECT count(*) FROM carga_csv c JOIN public.mapcalordb m ON {CLAVE} WHERE c.accion='insert' AND (m.municipios IS NULL OR m.municipios = '')")
        post = {
            "filas = antes + inserts": (filas_desp, filas_antes + n_ins),
            "checksum de las filas FUERA del CSV (incluye todas las reales) sin cambio": (list(no_obj_desp), list(no_obj_antes)),
            "checksum de TODAS las filas reales sin cambio": (list(reales_desp), list(reales_antes)),
            "filas del CSV presentes con los mismos valores": (iguales, n_csv),
            "inserts con nombre de municipio vacio": (sin_nombre, 0),
        }
        informe_d["post_chequeos"] = {k: {"valor": v, "esperado": e, "ok": v == e} for k, (v, e) in post.items()}
        malos = [k for k, (v, e) in post.items() if v != e]
        if malos:
            raise Abortar("post-chequeos fallidos: " + "; ".join(malos))

        # --- DESGLOSE
        cur.execute("""SELECT namevar, yearm, count(*) FILTER (WHERE accion='insert'), count(*) FILTER (WHERE accion='update')
                       FROM carga_csv GROUP BY 1, 2 ORDER BY 1, 2""")
        informe_d["por_variable_anio"] = [list(r) for r in cur.fetchall()]
        cur.execute(f"""SELECT c.namevar, p.originvar AS origen_actual, c.originvar AS origen_nuevo, min(c.yearm), max(c.yearm), count(*)
                        FROM carga_csv c JOIN pre_imagen p ON p.codigodivipola = c.codigodivipola AND p.namevar = c.namevar AND p.yearm = c.yearm
                        WHERE c.accion = 'update' GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""")
        informe_d["updates_por_categoria"] = [list(r) for r in cur.fetchall()]
        cur.execute("""SELECT c.namevar, c.originvar, count(*) FROM carga_csv c WHERE c.accion='insert' GROUP BY 1,2 ORDER BY 1,2""")
        informe_d["inserts_por_origen"] = [list(r) for r in cur.fetchall()]
        if commit:
            conn.commit()
            informe_d["resultado"] = "COMMIT"
        else:
            conn.rollback()
            informe_d["resultado"] = "ROLLBACK (dry-run): la base queda como estaba"
    except Abortar as e:
        conn.rollback()
        informe_d["resultado"] = f"ABORTADO y ROLLBACK: {e}"
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        informe_d["resultado"] = f"ERROR y ROLLBACK: {type(e).__name__}: {e}"
    finally:
        conn.close()
    if informe:
        informe.write_text(json.dumps(informe_d, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return informe_d


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--sha-esperado", default=SHA_ESPERADO, help="SHA256 que debe tener el CSV")
    ap.add_argument("--ejecutar", action="store_true", help="COMMIT (sin esto: dry-run con ROLLBACK)")
    ap.add_argument("--confirmar-sha", default=None, help="obligatorio con --ejecutar sobre un host NO local")
    ap.add_argument("--informe", type=Path, default=None)
    args = ap.parse_args(argv)

    dsn = os.environ.get("CARGA_DATABASE_URL")
    if not dsn:
        sys.exit("Falta la variable de entorno CARGA_DATABASE_URL")
    host = urlparse(dsn).hostname
    sha = sha256_archivo(args.csv)
    if sha != args.sha_esperado:
        sys.exit(f"El SHA256 del CSV ({sha}) no es el esperado ({args.sha_esperado}). No se hace nada.")
    if host not in ES_LOCAL:
        if not args.ejecutar:
            sys.exit("El dry-run (ROLLBACK) solo se permite contra una copia LOCAL: escribe dentro de la transaccion.")
        if args.confirmar_sha != sha or os.environ.get("TECHO360_AUTORIZO_CARGA") != "1":
            sys.exit("Carga a un host NO local: exige --confirmar-sha <sha256 completo del CSV> y TECHO360_AUTORIZO_CARGA=1 "
                     "(solo con la autorizacion explicita de Juan).")
    res = ejecutar(args.csv, dsn, args.ejecutar, args.informe)
    print(json.dumps({k: res[k] for k in ("modo", "base", "filas_csv", "updates_csv", "inserts_csv", "aplicado", "resultado") if k in res}, ensure_ascii=False, indent=2))
    return 0 if res["resultado"].startswith(("COMMIT", "ROLLBACK (dry-run)")) else 1


if __name__ == "__main__":
    sys.exit(main())
