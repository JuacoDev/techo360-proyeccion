"""REVERSA de la carga: restaura los valores de la pre-imagen y borra las claves insertadas. UNA transaccion.

NO se ha ejecutado contra produccion. Se probo en una copia local (carga -> reversa -> checksum igual).

  1. UPDATE de las filas de pre_imagen_updates.csv (por id): valuevar, originvar, intervaloconfianza vuelven a su
     valor previo. Solo se toca una fila si SIGUE en el estado posterior a la carga (origen
     proyeccion-techo360-*); si alguien la cambio despues, se aborta (usa --forzar solo sabiendo por que).
  2. DELETE de las claves de claves_insertadas.csv (solo si su origen es proyeccion-techo360-*).
  3. Verifica: filas = antes; checksum de la tabla COMPLETA = el de checksums_antes.json (los ids de las filas
     preexistentes no cambian; los ids de las insertadas se pierden y la secuencia NO retrocede: inocuo).
Por defecto es un DRY-RUN (ROLLBACK) y solo contra una copia local; con --ejecutar hace COMMIT (en un host no local
exige --confirmar-sha <sha256 del CSV> y TECHO360_AUTORIZO_CARGA=1).

    REVERSA_DATABASE_URL=... python reversa_carga.py --csv <csv> --antes <carpeta de preimagen.py> [--ejecutar]
"""
import argparse
import io
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun_carga import COLS_FILA, ES_LOCAL, cargar_csv, conectar  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--antes", required=True, type=Path)
    ap.add_argument("--ejecutar", action="store_true")
    ap.add_argument("--confirmar-sha", default=None)
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--informe", type=Path, default=None)
    a = ap.parse_args(argv)
    df, sha = cargar_csv(a.csv)
    antes = json.loads((a.antes / "checksums_antes.json").read_text(encoding="utf-8"))
    if antes["sha256_csv"] != sha:
        sys.exit("checksums_antes.json corresponde a OTRO CSV")
    conn, host = conectar("REVERSA_DATABASE_URL", solo_lectura=False)
    if host not in ES_LOCAL:
        if not a.ejecutar:
            sys.exit("El dry-run solo se permite contra una copia LOCAL (escribe dentro de la transaccion).")
        if a.confirmar_sha != sha or os.environ.get("TECHO360_AUTORIZO_CARGA") != "1":
            sys.exit("Reversa en un host NO local: exige --confirmar-sha <sha256 del CSV> y TECHO360_AUTORIZO_CARGA=1.")
    cur = conn.cursor()
    informe = {"modo": "COMMIT" if a.ejecutar else "DRY-RUN (ROLLBACK)"}
    try:
        cur.execute("SHOW transaction_read_only")
        if cur.fetchone()[0] != "off":
            raise RuntimeError("sesion de solo lectura")
        pre = a.antes / "pre_imagen_updates.csv"
        ins = a.antes / "claves_insertadas.csv"
        cur.execute("CREATE TEMP TABLE pre (id int, codigodivipola varchar(5), namevar text, yearm int, municipios text, valuevar numeric, originvar text, intervaloconfianza numeric) ON COMMIT DROP")
        with open(pre, "r", encoding="utf-8", newline="") as f:
            cur.copy_expert("COPY pre FROM STDIN WITH (FORMAT csv, HEADER true)", f)
        cur.execute("CREATE TEMP TABLE claves_ins (codigodivipola varchar(5), namevar text, yearm int) ON COMMIT DROP")
        with open(ins, "r", encoding="utf-8", newline="") as f:
            cur.copy_expert("COPY claves_ins FROM STDIN WITH (FORMAT csv, HEADER true)", f)
        cur.execute("SELECT (SELECT count(*) FROM pre), (SELECT count(*) FROM claves_ins)")
        n_pre, n_ins = cur.fetchone()
        informe.update(pre_imagen=n_pre, claves_insertadas=n_ins)
        if (n_pre, n_ins) != (antes["updates"], antes["inserts"]):
            raise RuntimeError(f"pre-imagen/claves no coinciden con checksums_antes.json: {n_pre}/{n_ins}")
        # guarda: cada fila a restaurar / borrar sigue en el estado posterior a la carga
        cur.execute("SELECT count(*) FROM pre p JOIN public.mapcalordb m ON m.id = p.id WHERE m.originvar LIKE 'proyeccion-techo360-%'")
        en_estado_post = cur.fetchone()[0]
        cur.execute("""SELECT count(*) FROM claves_ins c JOIN public.mapcalordb m ON m.codigodivipola = c.codigodivipola AND m.namevar = c.namevar AND m.yearm = c.yearm
                       WHERE m.originvar LIKE 'proyeccion-techo360-%'""")
        ins_en_estado_post = cur.fetchone()[0]
        informe["guarda"] = {"updates_en_estado_posterior": [en_estado_post, n_pre], "inserts_en_estado_posterior": [ins_en_estado_post, n_ins]}
        if not a.forzar and (en_estado_post != n_pre or ins_en_estado_post != n_ins):
            raise RuntimeError("hay filas que ya NO estan en el estado posterior a la carga (alguien las cambio): no se revierte sin --forzar")
        cur.execute("""UPDATE public.mapcalordb m SET valuevar = p.valuevar, originvar = p.originvar, intervaloconfianza = p.intervaloconfianza
                       FROM pre p WHERE m.id = p.id""")
        restauradas = cur.rowcount
        cur.execute("""DELETE FROM public.mapcalordb m USING claves_ins c
                       WHERE m.codigodivipola = c.codigodivipola AND m.namevar = c.namevar AND m.yearm = c.yearm AND m.originvar LIKE 'proyeccion-techo360-%'""")
        borradas = cur.rowcount
        informe["aplicado"] = {"restauradas": restauradas, "borradas": borradas}
        if (restauradas, borradas) != (n_pre, n_ins):
            raise RuntimeError(f"filas afectadas distintas de lo esperado: {restauradas}/{n_pre}, {borradas}/{n_ins}")
        cur.execute(f"SELECT count(*), md5(coalesce(string_agg({COLS_FILA}, E'\\n' ORDER BY m.id), '')) FROM public.mapcalordb m")
        despues = list(cur.fetchone())
        informe["checksum_tabla_completa"] = {"despues_de_la_reversa": despues, "antes_de_la_carga": antes["checksums"]["completa"], "ok": despues == antes["checksums"]["completa"]}
        if despues != antes["checksums"]["completa"]:
            raise RuntimeError("la tabla NO quedo igual a antes de la carga")
        if a.ejecutar:
            conn.commit(); informe["resultado"] = "COMMIT: reversa aplicada"
        else:
            conn.rollback(); informe["resultado"] = "ROLLBACK (dry-run): verificado que la reversa deja la tabla como antes"
    except Exception as e:  # noqa: BLE001
        conn.rollback()
        informe["resultado"] = f"ABORTADO y ROLLBACK: {e}"
    finally:
        conn.close()
    if a.informe:
        a.informe.write_text(json.dumps(informe, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(json.dumps(informe, indent=2, ensure_ascii=False, default=str))
    return 0 if informe["resultado"].startswith(("COMMIT", "ROLLBACK (dry-run)")) else 1


if __name__ == "__main__":
    sys.exit(main())
