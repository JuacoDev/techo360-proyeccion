"""Pre-imagen de la carga (SOLO LECTURA; segura contra produccion).

Exporta, en la carpeta indicada:
  pre_imagen_updates.csv   valores ACTUALES (id, clave, municipios, valuevar, originvar, intervaloconfianza) de las
                           filas que la carga va a ACTUALIZAR (accion = update)
  claves_insertadas.csv    claves (codigodivipola, namevar, yearm) que la carga va a INSERTAR
  checksums_antes.json     filas y md5 de la tabla completa, de las filas reales y de las filas fuera del CSV,
                           mas fecha/hora y SHA256 del CSV
Y verifica que las filas a actualizar existen y son proyeccion-* (si no, sale con error).

    PREIMAGEN_DATABASE_URL=...  python preimagen.py --csv <csv> --salida <carpeta>
Se ejecuta JUSTO ANTES de la carga (despues del pg_dump nuevo): reversa_carga.py depende de estos archivos.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun_carga import CLAVES_CTE, SQL_PROY, checksums, cargar_csv, conectar, params_claves  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--salida", required=True, type=Path)
    ap.add_argument("--permitir-escritura", action="store_true", help="solo para la copia local de pruebas (sesion normal)")
    a = ap.parse_args(argv)
    df, sha = cargar_csv(a.csv)
    a.salida.mkdir(parents=True, exist_ok=True)
    conn, host = conectar("PREIMAGEN_DATABASE_URL", solo_lectura=not a.permitir_escritura)
    cur = conn.cursor()
    if not a.permitir_escritura:
        cur.execute("SHOW default_transaction_read_only")
        assert cur.fetchone()[0] == "on", "la sesion NO es de solo lectura"
    cur.execute("SELECT current_database()")
    base = cur.fetchone()[0]

    upd = df[df.accion == "update"]
    ins = df[df.accion == "insert"]
    cur.execute(f"""SELECT m.id, m.codigodivipola, m.namevar, m.yearm, m.municipios, m.valuevar, m.originvar, m.intervaloconfianza
                    FROM public.mapcalordb m, {CLAVES_CTE}
                    WHERE m.codigodivipola = k.codigodivipola AND m.namevar = k.namevar AND m.yearm = k.yearm ORDER BY m.id""",
                params_claves(upd))
    cols = ["id", "codigodivipola", "namevar", "yearm", "municipios", "valuevar", "originvar", "intervaloconfianza"]
    pre = pd.DataFrame(cur.fetchall(), columns=cols)
    problemas = []
    if len(pre) != len(upd):
        problemas.append(f"{len(upd) - len(pre)} updates del CSV no existen en la tabla")
    reales = pre[~(pre.originvar.fillna("").str.startswith("proyeccion-") | pre.originvar.fillna("").str.endswith("-proyeccion"))]
    if len(reales):
        problemas.append(f"{len(reales)} filas a actualizar son REALES (no proyeccion-*)")
    cur.execute(f"SELECT count(*) FROM public.mapcalordb m, {CLAVES_CTE} WHERE m.codigodivipola = k.codigodivipola AND m.namevar = k.namevar AND m.yearm = k.yearm",
                params_claves(ins))
    choques = cur.fetchone()[0]
    if choques:
        problemas.append(f"{choques} inserts del CSV ya existen")
    if problemas:
        sys.exit("PRE-IMAGEN INVALIDA: " + "; ".join(problemas))

    pre.to_csv(a.salida / "pre_imagen_updates.csv", index=False, lineterminator="\n")
    ins[["codigodivipola", "nameVar", "yearM"]].rename(columns={"nameVar": "namevar", "yearM": "yearm"}).to_csv(a.salida / "claves_insertadas.csv", index=False, lineterminator="\n")
    cs = checksums(cur, df)
    meta = {"base": base, "host_local": host in ("127.0.0.1", "localhost"), "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "sha256_csv": sha, "filas_csv": len(df), "updates": len(upd), "inserts": len(ins), "checksums": cs,
            "ultimo_id": None}
    cur.execute("SELECT max(id) FROM public.mapcalordb")
    meta["ultimo_id"] = cur.fetchone()[0]
    (a.salida / "checksums_antes.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    conn.rollback()
    conn.close()
    print(json.dumps({"base": base, "pre_imagen_filas": len(pre), "claves_insertadas": len(ins), "checksums": cs, "ultimo_id": meta["ultimo_id"]}, indent=2))


if __name__ == "__main__":
    main()
