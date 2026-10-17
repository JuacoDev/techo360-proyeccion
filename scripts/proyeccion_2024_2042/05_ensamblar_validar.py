"""Parte F -- ensambla el CSV final y valida (obligatorio antes de reportar).

Salida: data/proyeccion_2024_2042/proyeccion_techo360_2024_2042.csv
Informes: F_validacion.md, F_cobertura_final.csv, F_expected_vs_generado.csv,
F_score_2024_2042.csv
No escribe nada en la BD.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun import (ANIOS, COLUMNAS_CSV, DIR_INFORMES, DIR_SALIDA, DIR_SNAPSHOT, V_ACU, V_ALC, V_DENS, V_DL,  # noqa: E402
                    V_DQ, V_DT, V_TEN, VARIABLES_ALCANCE, asegurar_dirs, cargar_snapshot, combinar, md_tabla, panel)

# TECHO360_ANCLA = "actual" (por defecto) | "m4": usa parcial/deficit_m4.csv (ancla 2024 cuantitativo con
# saturacion suave, ver 03_deficit.py), escribe un CSV con nombre distinto y compara contra el anterior.
ANCLA = os.environ.get("TECHO360_ANCLA", "actual")
SUFIJO = "_m4" if ANCLA == "m4" else ""
SALIDA = DIR_SALIDA / f"proyeccion_techo360_2024_2042{SUFIJO}.csv"
SALIDA_ANTERIOR = DIR_SALIDA / "proyeccion_techo360_2024_2042.csv"
INSUMOS_SCORE = [V_ACU, V_ALC, V_DQ]


def clase_origen(o: str) -> str:
    if o.startswith("proyeccion-techo360-"):
        return "techo360:" + o[len("proyeccion-techo360-"):]
    if o.startswith("proyeccion-"):
        return "legado:" + o
    return "real:" + ("DANE-proyeccion-poblacion" if o.startswith("DANE-proyeccion-poblacion") else o)


def score(p_acu, p_alc, p_dq, lim, anio):
    def n(v, k):
        lo, hi = lim[k]
        return (v - lo) / (hi - lo)
    a, b, d = n(p_acu[anio], V_ACU), n(p_alc[anio], V_ALC), n(p_dq[anio], V_DQ)
    return ((1 - a) / 3 + (1 - b) / 3 + d / 3).round(4)


def esperados_update_m4(filas, universo):
    """Recalculo INDEPENDIENTE del 03: municipios del universo cuyo ancla 2024 del cuantitativo cambia
    con M4 (v18*1.36 > 60 y el valor M4 redondeado difiere del ancla de la BD)."""
    v18 = filas[(filas.namevar == V_DQ) & (filas.yearm == 2018) & filas.real].set_index("codigodivipola").valuevar.reindex(universo)
    a24 = filas[(filas.namevar == V_DQ) & (filas.yearm == 2024)].set_index("codigodivipola").valuevar.reindex(universo)
    x = v18 * (6.8 / 5.0)
    m4 = np.where(x <= 60.0, x, 60.0 + 40.0 * np.tanh((x - 60.0) / 40.0))
    cambia = (x > 60.0) & (pd.Series(m4, index=universo).round(2).sub(a24).abs() > 0.004)
    return int(cambia.sum())


def comparar_con_anterior(df, filas, universo, estado, sc, lim, P, v_dq):
    """Comparacion M4 vs CSV anterior: diff de filas, 2024 cuantitativo, score y ranking."""
    out = ["\n## Comparacion con el CSV anterior (ancla vigente)\n"]
    ant = pd.read_csv(SALIDA_ANTERIOR, dtype={"codigodivipola": str})
    sha_ant = hashlib.sha256(SALIDA_ANTERIOR.read_bytes()).hexdigest()
    out.append(f"CSV anterior: `{SALIDA_ANTERIOR.name}` SHA256 `{sha_ant}` ({len(ant)} filas; sin modificar).\n")
    clave = ["codigodivipola", "nameVar", "yearM"]
    m = ant.merge(df, on=clave, how="outer", suffixes=("_ant", "_nuevo"), indicator=True)
    solo_ant, solo_nue = m[m._merge == "left_only"], m[m._merge == "right_only"]
    ambos = m[m._merge == "both"].copy()
    ambos["d_valor"] = (ambos.valuevar_nuevo - ambos.valuevar_ant)
    cambia_valor = ambos.d_valor.abs() > 0.004
    otros = ["originvar", "fiabilidad", "intervalo_confianza", "accion"]
    cambia_otro = pd.Series(False, index=ambos.index)
    for c in otros:
        cambia_otro |= ambos[c + "_ant"] != ambos[c + "_nuevo"]
    ambos["cambio"] = cambia_valor | cambia_otro
    out.append(f"- Filas: anterior {len(ant)}, nuevo {len(df)}; solo en el anterior: {len(solo_ant)}; solo en el nuevo: {len(solo_nue)} "
               f"(nuevas filas 2024 cuantitativo = update de las que cambian); en ambos: {len(ambos)}, identicas: {int((~ambos.cambio).sum())}, con algun cambio: {int(ambos.cambio.sum())}.")
    por_var = pd.concat([
        ambos.groupby("nameVar").agg(en_ambos=("cambio", "size"), con_cambio=("cambio", "sum"),
                                     cambio_valor=("d_valor", lambda x: int((x.abs() > 0.004).sum())), max_abs_dvalor=("d_valor", lambda x: float(x.abs().max()))),
        solo_nue.groupby("nameVar").size().rename("solo_nuevo").to_frame(),
        solo_ant.groupby("nameVar").size().rename("solo_anterior").to_frame(),
    ], axis=1).fillna(0).reset_index().rename(columns={"index": "nameVar"})
    out.append("\n" + md_tabla(por_var, 3))
    cob_ten = ambos[ambos.nameVar.isin([V_ACU, V_ALC, V_TEN])]
    out.append(f"\n- **Cobertura y tenencia (partes B y D): filas con cambio = {int(cob_ten.cambio.sum())}** de {len(cob_ten)} (esperado 0). "
               f"Cualitativo: {int(ambos[ambos.nameVar == V_DL].cambio.sum())} con cambio (esperado 0).")
    # --- 2024 cuantitativo
    n24 = solo_nue[(solo_nue.nameVar == v_dq) & (solo_nue.yearM == 2024)]
    out.append(f"\n### Filas 2024 del cuantitativo que cambian\n\n- Filas: **{len(n24)}** (municipios: {n24.codigodivipola.nunique()}), todas accion = `{', '.join(sorted(n24.accion_nuevo.dropna().unique()))}`, "
               f"originvar = `{', '.join(sorted(n24.originvar_nuevo.dropna().unique()))}`, fiabilidad {sorted(n24.fiabilidad_nuevo.unique())}.")
    d24 = filas[(filas.namevar == v_dq) & (filas.yearm == 2024)].set_index("codigodivipola").valuevar.reindex(universo)
    nuevo24 = estado[(estado.namevar == v_dq) & (estado.yearm == 2024)].set_index("codigodivipola").valuevar.reindex(universo)
    en100_ant = d24[d24 >= 99.995].index
    out.append(f"- Municipios del universo en 100 con el ancla anterior: {len(en100_ant)}; **con M4: {int((nuevo24 >= 99.995).sum())}** (esperado 0); max 2024 = {nuevo24.max():.2f}. "
               f"Valores distintos entre esos {len(en100_ant)}: **{int(nuevo24.reindex(en100_ant).round(2).nunique())}** (antes: {int(d24.reindex(en100_ant).round(2).nunique())}); "
               f"rango {nuevo24.reindex(en100_ant).min():.2f}-{nuevo24.reindex(en100_ant).max():.2f}.")
    cam = n24.set_index("codigodivipola")
    dif24 = (nuevo24.reindex(cam.index) - d24.reindex(cam.index))
    out.append(f"- Cambio del valor 2024 en los {len(cam)} municipios: media {dif24.mean():.2f} pp, min {dif24.min():.2f}, max {dif24.max():.2f}.")
    # --- score
    sc_ant = pd.read_csv(DIR_INFORMES / "F_score_2024_2042.csv", index_col=0, dtype={0: str})
    sc_ant.index = sc_ant.index.astype(str).str.zfill(5)
    sc_ant.columns = [int(c) for c in sc_ant.columns]
    sc_ant = sc_ant.reindex(universo)
    r_ant, r_nue = sc_ant[2024].rank(ascending=False, method="first"), sc[2024].rank(ascending=False, method="first")
    mov = (r_ant - r_nue).abs()
    d = sc[2024] - sc_ant[2024]
    out.append("\n### Score 2024: M4 vs CSV anterior (1,104 municipios)\n")
    out.append(md_tabla(pd.DataFrame([{"score_medio_anterior": sc_ant[2024].mean(), "score_medio_M4": sc[2024].mean(), "cambio_medio": d.mean(),
                                       "max_abs_cambio": d.abs().max(), "municipios_con_cambio>0.0005": int((d.abs() > 0.0005).sum()),
                                       "Spearman": float(r_ant.corr(r_nue, method="spearman")), "mueven>=1_puesto": int((mov > 0).sum()),
                                       "mueven>=10": int((mov >= 10).sum()), "mueven>=50": int((mov >= 50).sum()), "max_puestos": int(mov.max()),
                                       "top50_comun": len(set(r_ant[r_ant <= 50].index) & set(r_nue[r_nue <= 50].index))}]), 4))
    out.append(f"- Score medio 2024 -> 2042: anterior {sc_ant[2024].mean():.4f} -> {sc_ant[2042].mean():.4f}; **M4 {sc[2024].mean():.4f} -> {sc[2042].mean():.4f}**.")
    r42 = sc[2042].rank(ascending=False, method="first")
    m42 = (r_nue - r42).abs()
    r42a = sc_ant[2042].rank(ascending=False, method="first")
    m42a = (r_ant - r42a).abs()
    out.append(f"- Ranking 2042 vs 2024 (M4): cambian de puesto {int((m42 > 0).sum())}; >=10: {int((m42 >= 10).sum())}; >=50: {int((m42 >= 50).sum())}; max {int(m42.max())}; "
               f"Spearman {float(r_nue.corr(r42, method='spearman')):.4f}. (Anterior: {int((m42a > 0).sum())} / {int((m42a >= 10).sum())} / {int((m42a >= 50).sum())}; max {int(m42a.max())}.)")
    # --- sensibilidad: max del min-max en el universo (si los huerfanos tambien dejaran de saturar)
    insumos = {V_ACU: P[V_ACU][2024], V_ALC: P[V_ALC][2024], v_dq: P[v_dq][2024]}
    lim_alt = dict(lim)
    lim_alt[v_dq] = (float(insumos[v_dq].min()), float(insumos[v_dq].max()))
    def sc24(l):
        n = lambda v, k: (v - l[k][0]) / (l[k][1] - l[k][0])  # noqa: E731
        return ((1 - n(insumos[V_ACU], V_ACU)) / 3 + (1 - n(insumos[V_ALC], V_ALC)) / 3 + n(insumos[v_dq], v_dq) / 3)
    base = sc24(lim)
    alt = sc24(lim_alt)
    ra, rb = base.rank(ascending=False, method="first"), alt.rank(ascending=False, method="first")
    out.append(f"\n### Sensibilidad (no incluida en el CSV)\n\nLos 19 codigos huerfanos (areas no municipalizadas) NO estan en el CSV y 16 tienen cuantitativo 2024 = 100: en produccion el maximo del min-max "
               f"seguiria en {lim[v_dq][1]:.2f} aunque el universo ya no llegue a 100. Si esos huerfanos tambien se re-anclaran y el maximo bajara a {lim_alt[v_dq][1]:.2f}, "
               f"el score 2024 subiria en media {float((alt - base).mean()):.4f} y {int(((ra - rb).abs() >= 10).sum())} municipios moverian >=10 puestos (Spearman {float(ra.corr(rb, method='spearman')):.4f}).")
    return out


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    universo = list(zonas.codigo)
    md = ["# Parte F -- Validacion\n"]

    partes = [pd.read_csv(DIR_SALIDA / "parcial" / f"{n}.csv", dtype={"codigodivipola": str}) for n in ("cobertura", "deficit" + SUFIJO, "tenencia")]
    df = pd.concat(partes, ignore_index=True)[COLUMNAS_CSV]
    df = df.sort_values(["nameVar", "codigodivipola", "yearM"]).reset_index(drop=True)
    df.to_csv(SALIDA, index=False, lineterminator="\n")
    sha = hashlib.sha256(SALIDA.read_bytes()).hexdigest()
    md.append(f"CSV: `{SALIDA}`  \nSHA256: `{sha}`  \nFilas: {len(df)}\n")

    # ------------------ integridad ------------------
    chk = {
        "duplicados (codigo,var,anio)": int(df.duplicated(["codigodivipola", "nameVar", "yearM"]).sum()),
        "NaN en cualquier columna": int(df.isna().sum().sum()),
        "valuevar fuera de [0,100]": int(((df.valuevar < 0) | (df.valuevar > 100)).sum()),
        "fiabilidad fuera de [0,100]": int(((df.fiabilidad < 0) | (df.fiabilidad > 100)).sum()),
        "codigos fuera del universo": int((~df.codigodivipola.isin(universo)).sum()),
        "codigos 00000/01001": int(df.codigodivipola.isin(["00000", "01001"]).sum()),
        "variables fuera de alcance": int((~df.nameVar.isin(VARIABLES_ALCANCE)).sum()),
        "anios fuera de 2024-2042": int((~df.yearM.between(2024, 2042)).sum()),
        "originvar sin prefijo proyeccion-techo360-": int((~df.originvar.str.startswith("proyeccion-techo360-")).sum()),
        "accion distinta de insert/update": int((~df.accion.isin(["insert", "update"])).sum()),
    }
    lo_hi = df.intervalo_confianza.str.split(";", expand=True).astype(float)
    chk["intervalo mal formado (lo>valor o hi<valor)"] = int(((lo_hi[0] > df.valuevar + 1e-9) | (lo_hi[1] < df.valuevar - 1e-9)).sum())

    # ------------------ ninguna fila real pisada ------------------
    reales = filas[filas.real][["codigodivipola", "namevar", "yearm"]].rename(columns={"namevar": "nameVar", "yearm": "yearM"})
    pisadas = df.merge(reales, on=["codigodivipola", "nameVar", "yearM"])
    chk["filas REALES de la BD pisadas"] = len(pisadas)
    # update debe apuntar a fila proyeccion-* existente; insert a clave inexistente
    ex = filas[["codigodivipola", "namevar", "yearm", "originvar", "real"]].rename(columns={"namevar": "nameVar", "yearm": "yearM"})
    j = df.merge(ex, on=["codigodivipola", "nameVar", "yearM"], how="left", suffixes=("", "_db"))
    chk["update sin fila proyeccion-* existente"] = int(((j.accion == "update") & (j.originvar_db.isna() | j.real.fillna(True).astype(bool))).sum())
    chk["insert sobre clave ya existente"] = int(((j.accion == "insert") & j.originvar_db.notna()).sum())
    md.append("## Integridad (todo debe ser 0)\n")
    md.append(md_tabla(pd.DataFrame({"chequeo": list(chk), "valor": list(chk.values())}), 0))
    resumen_acc = df.groupby(["nameVar", "accion"]).size().unstack(fill_value=0)
    resumen_acc.loc["TOTAL"] = resumen_acc.sum()
    md.append("\n### insert / update por variable\n")
    md.append(md_tabla(resumen_acc.reset_index(), 0))
    ori = df.groupby(["nameVar", "originvar"]).size().reset_index(name="filas")
    md.append("\n### Filas por variable y originvar\n")
    md.append(md_tabla(ori, 0))

    # ------------------ esperado vs generado ------------------
    reales_set = {(c, v, int(y)) for c, v, y in zip(filas[filas.real].codigodivipola, filas[filas.real].namevar, filas[filas.real].yearm)}
    huecos_serie = 0
    rows = []
    gen = df.groupby(["nameVar", "yearM"]).size()
    for var in VARIABLES_ALCANCE:
        for y in ANIOS:
            if var in (V_DQ, V_DL, V_DT) and y == 2024:
                esperado = 1  # solo 27493 (el resto tiene ancla en BD que no se toca)
                if var == V_DQ and ANCLA == "m4":
                    esperado += esperados_update_m4(filas, universo)  # + los que cambian de ancla
            else:
                esperado = sum(1 for c in universo if (c, var, y) not in reales_set)
            g = int(gen.get((var, y), 0))
            rows.append({"variable": var, "anio": y, "esperado": esperado, "generado": g, "dif": g - esperado})
    ev = pd.DataFrame(rows)
    ev.to_csv(DIR_INFORMES / f"F_expected_vs_generado{SUFIJO}.csv", index=False)
    malos = ev[ev["dif"] != 0]
    md.append(f"\n## Esperado vs generado: {len(ev)} celdas variable x anio, con diferencia: {len(malos)}\n")
    if len(malos):
        md.append(md_tabla(malos, 0))
    md.append("(esperado = municipios del universo sin fila REAL ese anio; deficit 2024: solo 27493, el resto conserva su ancla)\n")
    resumen_ev = ev.groupby("variable")[["esperado", "generado"]].sum().reset_index()
    md.append(md_tabla(resumen_ev, 0))

    # ------------------ tabla final de cobertura (BD + CSV) ------------------
    estado = combinar(filas, df.drop(columns=["accion"]))
    estado = estado[estado.codigodivipola.isin(universo) & estado.yearm.between(2024, 2042) & estado.namevar.isin(VARIABLES_ALCANCE)].copy()
    estado["clase"] = estado.originvar.map(clase_origen)
    tabla = estado.groupby(["namevar", "yearm", "clase"]).size().reset_index(name="n")
    tot = estado.groupby(["namevar", "yearm"]).codigodivipola.nunique().reset_index(name="con_valor")
    tot["sin_fila"] = len(universo) - tot.con_valor
    tabla.to_csv(DIR_INFORMES / f"F_cobertura_final_por_origen{SUFIJO}.csv", index=False)
    tot.to_csv(DIR_INFORMES / f"F_cobertura_final_totales{SUFIJO}.csv", index=False)
    pivot = tot.pivot(index="yearm", columns="namevar", values="sin_fila")
    md.append("\n## Huecos restantes tras cargar el CSV: municipios SIN fila por variable y anio (universo 1,104)\n")
    md.append(md_tabla(pivot.reset_index(), 0))
    md.append("\nEsperado: todo 0. Distribucion por origen (variable, origen) en 2024 / 2030 / 2042:\n")
    sub = tabla[tabla.yearm.isin([2024, 2030, 2042])].pivot_table(index=["namevar", "clase"], columns="yearm", values="n", fill_value=0).reset_index()
    md.append(md_tabla(sub, 0))

    # ------------------ score de privacion 2024-2042 ------------------
    s_all = pd.read_csv(DIR_SNAPSHOT / "score_2024_todos_los_codigos.csv", dtype={"codigodivipola": str})
    lim_antes = {v: (float(g.valuevar.min()), float(g.valuevar.max())) for v, g in s_all.groupby("namevar")}
    nuevos24 = df[(df.yearM == 2024) & df.nameVar.isin(INSUMOS_SCORE)]
    despues = pd.concat([s_all.rename(columns={"namevar": "nameVar"}).merge(
        nuevos24[["codigodivipola", "nameVar"]], on=["codigodivipola", "nameVar"], how="left", indicator=True).query("_merge=='left_only'")
        [["codigodivipola", "nameVar", "valuevar"]], nuevos24[["codigodivipola", "nameVar", "valuevar"]]])
    lim_despues = {v: (float(g.valuevar.min()), float(g.valuevar.max())) for v, g in despues.groupby("nameVar")}
    md.append("\n## Score de privacion 2024-2042 con el CSV\n")
    tl = pd.DataFrame([{"variable": v, "min_antes": lim_antes[v][0], "max_antes": lim_antes[v][1],
                        "min_despues": lim_despues[v][0], "max_despues": lim_despues[v][1],
                        "cambia": lim_antes[v] != lim_despues[v]} for v in INSUMOS_SCORE])
    md.append("Min/max de 2024 (todos los origenes y codigos, igual que `calcular_limites_base`), antes y despues de cargar el CSV:\n")
    md.append(md_tabla(tl, 2))

    P = {v: panel(estado, v, ANIOS).reindex(universo) for v in INSUMOS_SCORE}
    sc = pd.DataFrame({y: score(P[V_ACU], P[V_ALC], P[V_DQ], lim_despues, y) for y in ANIOS})
    con_score = sc.notna().sum()
    sc.to_csv(DIR_INFORMES / f"F_score_2024_2042{SUFIJO}.csv")
    md.append("\nMunicipios con score por anio (esperado 1,104): " + ", ".join(f"{y}: {int(con_score[y])}" for y in ANIOS))
    sin = sc[sc.isna().any(axis=1)]
    md.append(f"\nExcepciones (municipios con algun anio sin score): {len(sin)}" + (f" -> {list(sin.index)[:20]}" if len(sin) else ""))
    fuera = int(((sc < 0) | (sc > 1)).sum().sum())
    md.append(f"- Scores fuera de [0,1]: {fuera}. Con limites de 2024 los anios futuros solo pueden salirse por valores fuera del rango 2024 (ver arriba).")

    # ranking 2024: antes (BD actual, 1,052) vs despues (CSV cargado)
    base24 = {v: s_all[s_all.namevar == v].set_index("codigodivipola").valuevar.reindex(universo) for v in INSUMOS_SCORE}
    antes24 = score({2024: base24[V_ACU]}, {2024: base24[V_ALC]}, {2024: base24[V_DQ]}, lim_antes, 2024)
    comunes = antes24.dropna().index
    r_antes = antes24.loc[comunes].rank(ascending=False, method="first")
    r_desp_com = sc.loc[comunes, 2024].rank(ascending=False, method="first")
    mov = (r_antes - r_desp_com).abs()
    md.append("\n### Ranking 2024: score actual de BD (1,052 municipios) vs con el CSV cargado (mismos 1,052, orden relativo)\n")
    md.append(f"- Cambian de puesto relativo: {int((mov > 0).sum())} de {len(comunes)}; |cambio| >= 10 puestos: {int((mov >= 10).sum())}; >= 50: {int((mov >= 50).sum())}; max {int(mov.max())}.")
    rho = float(r_antes.corr(r_desp_com, method="spearman"))
    md.append(f"- Spearman entre ambos ordenes: {rho:.4f}.")
    nuevos = [c for c in universo if c not in set(comunes)]
    r_all24 = sc[2024].rank(ascending=False, method="first")
    md.append(f"- Los {len(nuevos)} municipios nuevos con score entran en el ranking 2024 con puestos entre {int(r_all24[nuevos].min())} y {int(r_all24[nuevos].max())} (de 1,104); "
              f"score medio de los nuevos {sc.loc[nuevos, 2024].mean():.3f} vs {sc.loc[comunes, 2024].mean():.3f} de los demas.")
    r42 = sc[2042].rank(ascending=False, method="first")
    mov42 = (r_all24 - r42).abs()
    md.append("\n### Ranking 2042 vs 2024 (con el CSV, 1,104)\n")
    md.append(f"- Cambian de puesto: {int((mov42 > 0).sum())}; >= 10: {int((mov42 >= 10).sum())}; >= 50: {int((mov42 >= 50).sum())}; max {int(mov42.max())}; "
              f"Spearman {float(r_all24.corr(r42, method='spearman')):.4f}.")
    medias = pd.DataFrame({"anio": ANIOS, "score_medio": [float(sc[y].mean()) for y in ANIOS], "min": [float(sc[y].min()) for y in ANIOS], "max": [float(sc[y].max()) for y in ANIOS]})
    md.append("\n" + md_tabla(medias.iloc[[0, 1, 2, 6, 12, 18]], 4))

    if ANCLA == "m4":
        md.extend(comparar_con_anterior(df, filas, universo, estado, sc, lim_despues, P, V_DQ))
    (DIR_INFORMES / f"F_validacion{SUFIJO}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
