"""Chequeos POSTERIORES a la carga (SOLO LECTURA; funcionan contra produccion con sesion de solo lectura).

NO se han ejecutado contra produccion (la carga no se ha hecho). Se validaron sobre una copia local
restaurada despues de cargar el CSV en ella.

Compara contra checksums_antes.json (de preimagen.py, tomado justo antes de la carga):
  1. filas = antes + inserts
  2. checksum de TODAS las filas reales sin cambio  (ninguna fila real modificada)
  3. checksum de las filas FUERA del CSV sin cambio (nada que no este en el CSV se tocó)
  4. cada fila del CSV esta en la tabla con los mismos valor / originvar / intervaloconfianza
  5. nombre de municipio no vacio en las filas insertadas
  6. sin claves duplicadas
  7. CERO HUECOS: los 1,104 municipios del universo tienen fila (cualquier origen) en las 6 variables y los 19 años
  8. matriz de cobertura variable x año x origen (CSV + resumen markdown)
Sale con codigo 1 si cualquier chequeo falla.

    CHECK_DATABASE_URL=... python post_carga_checks.py --csv <csv> --antes <carpeta con checksums_antes.json> --salida <carpeta>
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun_carga import CLAVES_CTE, SQL_PROY, VARIABLES, checksums, cargar_csv, conectar, params_claves  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--antes", required=True, type=Path, help="carpeta con checksums_antes.json")
    ap.add_argument("--salida", required=True, type=Path)
    a = ap.parse_args(argv)
    df, sha = cargar_csv(a.csv)
    antes = json.loads((a.antes / "checksums_antes.json").read_text(encoding="utf-8"))
    if antes["sha256_csv"] != sha:
        sys.exit("checksums_antes.json corresponde a OTRO CSV")
    a.salida.mkdir(parents=True, exist_ok=True)
    conn, host = conectar("CHECK_DATABASE_URL", solo_lectura=True)
    cur = conn.cursor()
    cur.execute("SHOW default_transaction_read_only")
    assert cur.fetchone()[0] == "on"
    cur.execute("SELECT current_database()")
    base = cur.fetchone()[0]
    res = {}
    cs = checksums(cur, df)
    n_ins = int((df.accion == "insert").sum())

    res["1. filas = antes + inserts"] = (cs["completa"][0] == antes["checksums"]["completa"][0] + n_ins, [cs["completa"][0], antes["checksums"]["completa"][0] + n_ins])
    res["2. ninguna fila REAL modificada (checksum igual)"] = (cs["reales"] == antes["checksums"]["reales"], [cs["reales"], antes["checksums"]["reales"]])
    res["3. filas fuera del CSV sin cambio (checksum igual)"] = (cs["fuera_del_csv"] == antes["checksums"]["fuera_del_csv"], [cs["fuera_del_csv"], antes["checksums"]["fuera_del_csv"]])

    p = params_claves(df)
    p["vals"] = [float(v) for v in df.valuevar]
    p["orig"] = df.originvar.tolist()
    p["fiab"] = [float(v) for v in df.fiabilidad]
    cur.execute("""SELECT count(*) FROM public.mapcalordb m,
                   unnest(%(cods)s::text[], %(vars)s::text[], %(anios)s::int[], %(vals)s::numeric[], %(orig)s::text[], %(fiab)s::numeric[]) AS k(c, n, y, v, o, f)
                   WHERE m.codigodivipola = k.c AND m.namevar = k.n AND m.yearm = k.y AND m.valuevar = k.v AND m.originvar = k.o AND m.intervaloconfianza = k.f""", p)
    iguales = cur.fetchone()[0]
    res["4. filas del CSV presentes con los mismos valores"] = (iguales == len(df), [iguales, len(df)])
    cur.execute("SELECT count(*) FROM public.mapcalordb WHERE originvar LIKE 'proyeccion-techo360-%'")
    n_techo = cur.fetchone()[0]
    res["4b. filas con origen proyeccion-techo360-* = filas del CSV"] = (n_techo == len(df), [n_techo, len(df)])
    ins = df[df.accion == "insert"]
    cur.execute("""SELECT count(*) FROM public.mapcalordb m, unnest(%(cods)s::text[], %(vars)s::text[], %(anios)s::int[]) AS k(c, n, y)
                   WHERE m.codigodivipola = k.c AND m.namevar = k.n AND m.yearm = k.y AND (m.municipios IS NULL OR m.municipios = '')""", params_claves(ins))
    res["5. inserts sin nombre de municipio"] = (cur.fetchone()[0] == 0, 0)
    cur.execute("SELECT count(*) - count(DISTINCT (codigodivipola, namevar, yearm)) FROM public.mapcalordb WHERE codigodivipola IS NOT NULL")
    res["6. claves duplicadas"] = (cur.fetchone()[0] == 0, 0)

    # --- 7 y 8: matriz de cobertura y cero huecos (universo = zonas_municipio sin 00000/01001)
    cur.execute("SELECT count(*) FROM public.zonas_municipio WHERE codigo_municipio NOT IN ('00000','01001')")
    universo = cur.fetchone()[0]
    cur.execute(f"""SELECT m.namevar, m.yearm,
                    CASE WHEN {SQL_PROY} THEN 'proy:' || m.originvar ELSE 'real:' || CASE WHEN m.originvar LIKE 'DANE-proyeccion-poblacion-%%' THEN 'DANE-proyeccion-poblacion' ELSE m.originvar END END AS origen,
                    count(*) AS n
                    FROM public.mapcalordb m JOIN public.zonas_municipio z ON z.codigo_municipio = m.codigodivipola
                    WHERE m.namevar = ANY(%(v)s) AND m.yearm BETWEEN 2024 AND 2042 AND z.codigo_municipio NOT IN ('00000','01001')
                    GROUP BY 1, 2, 3 ORDER BY 1, 2, 3""", {"v": VARIABLES})
    matriz = pd.DataFrame(cur.fetchall(), columns=["variable", "anio", "origen", "municipios"])
    matriz.to_csv(a.salida / "matriz_cobertura_variable_anio_origen.csv", index=False)
    por_celda = matriz.groupby(["variable", "anio"]).municipios.sum().unstack("anio").reindex(VARIABLES)
    huecos = [(v, int(y), int(universo - n)) for v, fila in por_celda.iterrows() for y, n in fila.items() if int(n) != universo]
    res[f"7. CERO HUECOS: {universo} municipios con fila en las 6 variables x 19 anios"] = (not huecos and por_celda.shape == (6, 19), huecos[:10])
    conn.rollback()
    conn.close()

    ok_todo = all(v[0] for v in res.values())
    md = [f"# Chequeos posteriores a la carga -- base `{base}` ({'LOCAL' if host in ('127.0.0.1', 'localhost') else host})\n",
          f"CSV SHA256 `{sha}`\n", "| chequeo | resultado | detalle |", "|---|---|---|"]
    for k, (ok, det) in res.items():
        md.append(f"| {k} | {'OK' if ok else '**FALLA**'} | {det} |")
    md.append(f"\n**RESULTADO GLOBAL: {'TODO OK' if ok_todo else 'HAY FALLAS'}**\n")
    md.append("## Matriz: municipios del universo por variable x año (cualquier origen; esperado 1,104 en todas)\n")
    md.append(por_celda.to_markdown() if hasattr(por_celda, "to_markdown") and _tiene_tabulate() else por_celda.to_string())
    md.append("\n## Origen en 2024 / 2030 / 2042\n")
    sub = matriz[matriz.anio.isin([2024, 2030, 2042])].pivot_table(index=["variable", "origen"], columns="anio", values="municipios", fill_value=0)
    md.append(sub.to_string())
    (a.salida / "chequeos_posteriores.md").write_text("\n".join(md), encoding="utf-8")
    (a.salida / "chequeos_posteriores.json").write_text(json.dumps({k: {"ok": v[0], "detalle": v[1]} for k, v in res.items()}, indent=2, default=str), encoding="utf-8")
    print("\n".join(md[:14]).encode("ascii", "replace").decode())
    return 0 if ok_todo else 1


def _tiene_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    sys.exit(main())
