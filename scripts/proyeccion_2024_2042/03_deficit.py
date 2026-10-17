"""Parte C -- deficit habitacional (cuantitativo, cualitativo, total) 2025-2042.

Ancla: el valor 2024 existente en BD (proyeccion-tendencia-nacional-2018-2024)
se conserva y NO se emite de nuevo (excepto 27493, que no tiene deficit).
C1  identidad total = cuant + cual (2018 real y ancla 2024).
C2  ridge (covariables + departamento como efecto fijo) ajustado al corte
    transversal 2018; cuant: densidad (+dept); cual: densidad + acueducto +
    alcantarillado (+dept). Debe superar a la mediana departamental en CV.
C3  deficit(m,t) = ancla(m) + [f(X(m,t)) - f(X(m,2024))]
C4  tope de cambio anual = 2x |cambio anual nacional 2018->2024 en BD|
C5  fiabilidad max(30, 55-3*(t-2025)); sin backtest temporal (1 punto real).
C6  27493: nivel = f(X) con departamento Choco, fiabilidad 30.

Salidas: parcial/deficit.csv, informes/C_deficit.md y C_cambios_mayores_15pp.csv
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun import (ANIOS, DIR_INFORMES, DIR_SALIDA, V_ACU, V_ALC, V_DENS, V_DL, V_DQ, V_DT, accion,  # noqa: E402
                    asegurar_dirs, cargar_snapshot, combinar, fila, indice_existente, md_tabla, panel)
from _modelos import dif_rmse_bootstrap, evaluar, pipeline_ridge  # noqa: E402

# TECHO360_ANCLA = "actual" (por defecto, comportamiento original) | "m4": el ancla 2024 del
# cuantitativo pasa de min(100, v18*1.36) a una saturacion suave: x = v18*1.36; x si x <= 60,
# y 60 + 40*tanh((x-60)/40) por encima (monotona, sin empates, nunca llega a 100). Solo cambia
# a los municipios con x > 60. Cualitativo y total conservan su ancla.
ANCLA = os.environ.get("TECHO360_ANCLA", "actual")
if ANCLA not in ("actual", "m4"):
    raise SystemExit("TECHO360_ANCLA debe ser 'actual' o 'm4'")
SUFIJO = "_m4" if ANCLA == "m4" else ""
UMBRAL_M4 = 60.0
ORIGEN_ANCLA_M4 = "proyeccion-techo360-tendencia-saturacion-suave-2018-2024"
FIABILIDAD_ANCLA_M4 = 70.0  # la misma del ancla original (parte de un dato municipal real 2018)

ORIGEN_DELTA = "proyeccion-techo360-delta-ridge-deficit"
ORIGEN_CONST = "proyeccion-techo360-valor-2024-constante"
ORIGEN_NIVEL = "proyeccion-techo360-modelo-deficit-nivel"
FACTOR_TOPE = 2.0
TASAS_ANCLA = {V_DQ: 6.8 / 5.0, V_DL: 20.0 / 26.9, V_DT: 26.8 / 31.9}  # las de ingestion/proyeccion_2024.py
COD_NUEVO_BELEN = "27493"


def cercano(serie: pd.Series, anio: int):
    """Valor del anio mas cercano con dato (empate -> el anterior)."""
    s = serie.dropna()
    if s.empty:
        return np.nan
    return float(s.iloc[(np.abs(s.index.values - anio) + (s.index.values > anio) * 0.1).argmin()])


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    cob = pd.read_csv(DIR_SALIDA / "parcial" / "cobertura.csv", dtype={"codigodivipola": str})
    estado = combinar(filas, cob)
    existente = indice_existente(filas)
    z = zonas.set_index("codigo")
    reales = filas[filas.real]
    P = lambda v, a=None: panel(estado, v, a)  # noqa: E731
    dq, dl, dt = (panel(reales, v) for v in (V_DQ, V_DL, V_DT))
    md = ["# Parte C -- Deficit habitacional\n"]

    # ------------------------------ C1 ------------------------------
    c18 = pd.DataFrame({"dq": dq[2018], "dl": dl[2018], "dt": dt[2018]}).dropna()
    d18 = (c18.dt - c18.dq - c18.dl)
    anc = pd.DataFrame({"dq": estado[(estado.namevar == V_DQ) & (estado.yearm == 2024)].set_index("codigodivipola").valuevar,
                        "dl": estado[(estado.namevar == V_DL) & (estado.yearm == 2024)].set_index("codigodivipola").valuevar,
                        "dt": estado[(estado.namevar == V_DT) & (estado.yearm == 2024)].set_index("codigodivipola").valuevar}).reindex(z.index)
    d24 = (anc.dt - anc.dq - anc.dl).dropna()
    cambia_m4 = pd.Series(False, index=anc.index)
    if ANCLA == "m4":
        x = (dq[2018] * TASAS_ANCLA[V_DQ]).reindex(anc.index)
        m4 = np.where(x <= UMBRAL_M4, x, UMBRAL_M4 + (100 - UMBRAL_M4) * np.tanh((x - UMBRAL_M4) / (100 - UMBRAL_M4)))
        nuevo = pd.Series(m4, index=anc.index).round(2)
        cambia_m4 = (x > UMBRAL_M4) & (nuevo.sub(anc.dq).abs() > 0.004)
        anc_dq_bd = anc.dq.copy()
        anc.loc[cambia_m4, "dq"] = nuevo[cambia_m4]
        md.append(f"## Ancla 2024 del cuantitativo con M4 (saturacion suave sobre {UMBRAL_M4:.0f})\n")
        md.append(f"- Municipios con v18*1.36 > {UMBRAL_M4:.0f}: {int((x > UMBRAL_M4).sum())}; cambian de valor (redondeado a 2 decimales): {int(cambia_m4.sum())}. "
                  f"Los demas ({int((~cambia_m4).sum())}) quedan con el ancla de la BD, intactos.")
        md.append(f"- Con el ancla vigente: {int((anc_dq_bd >= 99.995).sum())} en 100. Con M4: {int((anc.dq >= 99.995).sum())} en 100; "
                  f"max {anc.dq.max():.2f}; valores distintos entre los {int((anc_dq_bd >= 99.995).sum())} que estaban en 100: "
                  f"{int(anc.dq[anc_dq_bd >= 99.995].round(2).nunique())}.\n")
    md.append("## C1. Identidad total = cuantitativo + cualitativo\n")
    md.append(f"- 2018 real (n={len(c18)}): |dif| max = {d18.abs().max():.4f}, mediana = {d18.abs().median():.5f}, "
              f"municipios con |dif|>0.01: {(d18.abs() > 0.01).sum()}. Identidad CUMPLIDA en 2018 real.")
    md.append(f"- Ancla 2024 (n={len(d24)}): dif (total - cuant - cual): media {d24.mean():.3f}, mediana {d24.median():.3f}, "
              f"min {d24.min():.2f}, max {d24.max():.2f}; |dif|>1 pp en {(d24.abs() > 1).sum()} municipios. "
              "Las anclas 2024 NO cumplen la identidad (cada componente se extrapolo con su propia tasa nacional).")
    md.append("- Decision: total(m,t) = ancla_total(m) + [cuant(m,t)-ancla_cuant(m)] + [cual(m,t)-ancla_cual(m)], es decir "
              "la suma de los deltas sobre su propia ancla. Asi se conserva el ancla total tal cual esta en BD "
              "(que no cambia) y la identidad se cumple en los CAMBIOS, no en el nivel 2024.\n")

    # ------------------------- datos de entrenamiento -------------------------
    dens = panel(filas[filas.real], V_DENS)
    acu = panel(filas[filas.real], V_ACU)
    alc = panel(filas[filas.real], V_ALC)
    tr = pd.DataFrame(index=c18.index)
    tr["dq"], tr["dl"], tr["dt"] = c18.dq, c18.dl, c18.dt
    tr["depto"] = z.depto.reindex(tr.index)
    tr["ldens"] = [np.log1p(cercano(dens.loc[c], 2018)) if c in dens.index else np.nan for c in tr.index]
    tr["acu"] = [cercano(acu.loc[c], 2018) if c in acu.index else np.nan for c in tr.index]
    tr["alc"] = [cercano(alc.loc[c], 2018) if c in alc.index else np.nan for c in tr.index]
    md.append(f"## Entrenamiento 2018 transversal: {len(tr)} municipios, {tr.depto.nunique()} departamentos; "
              f"sin densidad cercana: {tr.ldens.isna().sum()}, sin cobertura acu/alc: {tr.acu.isna().sum()}/{tr.alc.isna().sum()} "
              "(se excluyen del modelo que las usa).\n")

    # ------------------------------ C2 ------------------------------
    especificaciones = {V_DQ: ("dq", ["ldens"]), V_DL: ("dl", ["ldens", "acu", "alc"])}
    modelos, veredictos, tablas = {}, {}, []
    for var, (ycol, cols) in especificaciones.items():
        d = tr.dropna(subset=cols + [ycol])
        tabla, err = evaluar(d, ycol, cols)
        tabla.insert(0, "variable", var)
        tablas.append(tabla)
        e_r, e_d = err[("cv_repetida", "ridge_dept")], err[("cv_repetida", "depto")]
        dif, lo, hi = dif_rmse_bootstrap(e_r, e_d)
        r_rep = tabla[(tabla.validacion == "cv_repetida_5x5")].set_index("predictor")
        supera = bool(r_rep.loc["ridge_dept", "RMSE"] < r_rep.loc["depto", "RMSE"] and hi < 0)
        veredictos[var] = {"supera_mediana_depto": supera, "dif_rmse": dif, "ic": (lo, hi), "n": len(d), "cols": cols,
                           "rmse_cv": float(r_rep.loc["ridge_dept", "RMSE"])}
        if supera:
            modelos[var] = pipeline_ridge(cols, True).fit(d[cols + ["depto"]], d[ycol])
        # utilidad marginal de la covariable (ridge solo-depto no existe: se mide contra la mediana depto)
    tab_all = pd.concat(tablas)
    tab_all.to_csv(DIR_INFORMES / f"C_validacion_modelos{SUFIJO}.csv", index=False)
    md.append("## C2. Validacion transversal\n")
    md.append("Nota de metodo: el modelo lleva departamento como efecto fijo, asi que la CV agrupada por departamento "
              "(que deja fuera todo el departamento) no puede evaluarlo contra la mediana departamental (no existe para "
              "un departamento sin datos). Por eso la comparacion de decision es la CV repetida 5x5 (el departamento SI tiene "
              "datos de entrenamiento: es el caso real, 1,103 municipios en 33 departamentos), y la CV agrupada se reporta "
              "como prueba de que las covariables generalizan a departamentos nuevos (ridge sin departamento vs mediana nacional).\n")
    md.append(md_tabla(tab_all, 3))
    for var, v in veredictos.items():
        md.append(f"\n- **{var}**: RMSE(ridge+dept) - RMSE(mediana depto) = {v['dif_rmse']:.3f} pp "
                  f"(IC95% bootstrap {v['ic'][0]:.3f} .. {v['ic'][1]:.3f}); n={v['n']}; "
                  f"{'SUPERA la mediana departamental' if v['supera_mediana_depto'] else 'NO supera la mediana departamental -> valor 2024 constante'}.")

    # --------------------- tendencia nacional y topes (C4) ---------------------
    nacional = {}
    for var, comp in ((V_DQ, "dq"), (V_DL, "dl")):
        ya = estado[(estado.namevar == var) & (estado.yearm == 2024)].set_index("codigodivipola").valuevar
        yb = reales[(reales.namevar == var) & (reales.yearm == 2018)].set_index("codigodivipola").valuevar
        cambio = (ya - yb).dropna()
        cambio = cambio[cambio.index.isin(z.index)]
        nacional[var] = {"cambio_anual_medio": float(cambio.mean() / 6), "cambio_anual_mediano": float(cambio.median() / 6),
                         "n": int(len(cambio))}
        nacional[var]["tope"] = FACTOR_TOPE * abs(nacional[var]["cambio_anual_medio"])
    md.append("\n## C4. Tendencia nacional 2018->2024 en BD y tope anual (2x)\n")
    md.append(md_tabla(pd.DataFrame(nacional).T.reset_index().rename(columns={"index": "variable"}), 4))

    # ------------------------------ C3 ------------------------------
    anios_f = [a for a in ANIOS if a >= 2025]
    dens_f = P(V_DENS, ANIOS)
    acu_f, alc_f = P(V_ACU, ANIOS), P(V_ALC, ANIOS)
    salto = (dens_f[2026] / ((dens_f[2025] + dens_f[2027]) / 2)).dropna()
    md.append(f"\n- Continuidad de la densidad usada como covariable: razon dens(2026)/prom(2025,2027): mediana {salto.median():.4f}, "
              f"p5 {salto.quantile(.05):.4f}, p95 {salto.quantile(.95):.4f} (2026 viene de TerriData; resto de DANE).\n")

    filas_csv, resumen, topes_activados = [], [], 0
    caminos = {}  # (codigo, var) -> serie 2024..2042 (valor sin recortar a total)

    def predictor(var, codigo, anio):
        """f(X(m,t)) sin departamento (el efecto fijo se cancela en el delta)."""
        cols = veredictos[var]["cols"]
        x = {"ldens": np.log1p(dens_f.at[codigo, anio]), "acu": acu_f.at[codigo, anio], "alc": alc_f.at[codigo, anio]}
        fila_x = pd.DataFrame([{**{c: x[c] for c in cols}, "depto": z.at[codigo, "depto"]}])
        return float(modelos[var].predict(fila_x)[0])

    for var, comp in ((V_DQ, "dq"), (V_DL, "dl")):
        tope = nacional[var]["tope"]
        for codigo in z.index:
            if codigo == COD_NUEVO_BELEN:
                continue
            a = float(anc.at[codigo, comp]) if codigo in anc.index and not np.isnan(anc.at[codigo, comp]) else np.nan
            if np.isnan(a):
                raise SystemExit(f"{codigo}: sin ancla {var}")
            camino = {2024: a}
            if var in modelos:
                f0 = predictor(var, codigo, 2024)
                prev_modelo, prev = f0, a
                for t in anios_f:
                    ft = predictor(var, codigo, t)
                    paso = ft - prev_modelo
                    paso_cap = float(np.clip(paso, -tope, tope))
                    topes_activados += int(abs(paso) > tope)
                    prev = float(np.clip(prev + paso_cap, 0, 100))
                    camino[t] = prev
                    prev_modelo = ft
            else:
                for t in anios_f:
                    camino[t] = a
            caminos[(codigo, var)] = camino

    # total = ancla_total + deltas (C1)
    for codigo in z.index:
        if codigo == COD_NUEVO_BELEN:
            continue
        a_t = float(anc.at[codigo, "dt"])
        cq, cl = caminos[(codigo, V_DQ)], caminos[(codigo, V_DL)]
        caminos[(codigo, V_DT)] = {t: float(np.clip(a_t + (cq[t] - cq[2024]) + (cl[t] - cl[2024]), 0, 100)) for t in [2024] + anios_f}

    # ------------------------------ C6: 27493 ------------------------------
    def nivel_nuevo_belen(var, anio):
        cols = veredictos[var]["cols"]
        x = {"ldens": np.log1p(dens_f.at[COD_NUEVO_BELEN, anio]), "acu": acu_f.at[COD_NUEVO_BELEN, anio], "alc": alc_f.at[COD_NUEVO_BELEN, anio]}
        modelo = modelos.get(var)
        fila_x = pd.DataFrame([{**{c: x[c] for c in cols}, "depto": z.at[COD_NUEVO_BELEN, "depto"]}])
        if modelo is not None:
            return float(modelo.predict(fila_x)[0])
        # el modelo no supero a la mediana departamental: nivel = mediana departamental 2018
        d = tr[tr.depto == z.at[COD_NUEVO_BELEN, "depto"]]
        return float(d["dq" if var == V_DQ else "dl"].median())

    for var in (V_DQ, V_DL):
        tope = nacional[var]["tope"]
        camino, prev = {}, None
        for t in ANIOS:
            ft = float(np.clip(nivel_nuevo_belen(var, t), 0, 100))
            if prev is None:
                prev = ft
            else:
                prev = float(np.clip(prev + np.clip(ft - camino[t - 1], -tope, tope), 0, 100))
            camino[t] = ft if t == 2024 else prev
        caminos[(COD_NUEVO_BELEN, var)] = camino
    caminos[(COD_NUEVO_BELEN, V_DT)] = {t: float(np.clip(caminos[(COD_NUEVO_BELEN, V_DQ)][t] + caminos[(COD_NUEVO_BELEN, V_DL)][t], 0, 100)) for t in ANIOS}

    # ------------------------------ filas CSV ------------------------------
    origen_gral = ORIGEN_DELTA if modelos else ORIGEN_CONST
    for (codigo, var), camino in sorted(caminos.items()):
        for t in sorted(camino):
            if codigo != COD_NUEVO_BELEN and t == 2024:
                continue  # ancla en BD: no se re-emite
            acc = accion(existente, codigo, var, t)
            if acc is None:
                continue
            if codigo == COD_NUEVO_BELEN:
                fiab, origen = 30.0, ORIGEN_NIVEL
                sigma = veredictos[V_DQ if var != V_DL else V_DL]["rmse_cv"] if var != V_DT else float(np.hypot(veredictos[V_DQ]["rmse_cv"], veredictos[V_DL]["rmse_cv"]))
            else:
                fiab = max(30.0, 55.0 - 3.0 * (t - 2025))
                origen = (ORIGEN_DELTA if var in modelos or (var == V_DT and modelos) else ORIGEN_CONST)
                sigma = (veredictos[V_DQ]["rmse_cv"] if var == V_DQ else veredictos[V_DL]["rmse_cv"] if var == V_DL
                         else float(np.hypot(veredictos[V_DQ]["rmse_cv"], veredictos[V_DL]["rmse_cv"])))
            half = 1.96 * sigma * np.sqrt((t - 2018) / 6.0)
            v = camino[t]
            filas_csv.append(fila(codigo, var, t, v, origen, fiab, v - half, v + half, acc))

    if ANCLA == "m4":
        sigma_dq = veredictos[V_DQ]["rmse_cv"]
        for codigo in cambia_m4[cambia_m4].index:
            acc = accion(existente, codigo, V_DQ, 2024)
            assert acc == "update", (codigo, acc)  # la fila 2024 ya existe como proyeccion-*
            v = float(anc.at[codigo, "dq"])
            half = 1.96 * sigma_dq * np.sqrt((2024 - 2018) / 6.0)
            filas_csv.append(fila(codigo, V_DQ, 2024, v, ORIGEN_ANCLA_M4, FIABILIDAD_ANCLA_M4, v - half, v + half, acc))
    df = pd.DataFrame(filas_csv).sort_values(["nameVar", "codigodivipola", "yearM"]).reset_index(drop=True)
    df.to_csv(DIR_SALIDA / "parcial" / f"deficit{SUFIJO}.csv", index=False)

    # ------------------------------ C4: distribucion de cambios ------------------------------
    cambios = pd.DataFrame([{"codigo": c, "variable": v, "ancla2024": cam[2024], "v2042": cam[2042], "delta": cam[2042] - cam[2024]}
                            for (c, v), cam in caminos.items() if c != COD_NUEVO_BELEN])
    md.append("\n## C4. Distribucion de (2042 - 2024), pp\n")
    desc = cambios.groupby("variable").delta.describe(percentiles=[.05, .25, .5, .75, .95]).reset_index()
    md.append(md_tabla(desc, 3))
    mayores = cambios[cambios.delta.abs() > 15].copy()
    mayores["nombre"] = mayores.codigo.map(z.nombre)
    mayores.to_csv(DIR_INFORMES / f"C_cambios_mayores_15pp{SUFIJO}.csv", index=False)
    md.append(f"\nMunicipios-variable con |cambio 2024->2042| > 15 pp: {len(mayores)} "
              f"({mayores.codigo.nunique()} municipios distintos); detalle en C_cambios_mayores_15pp.csv. "
              f"Pasos anuales topados por el limite: {topes_activados}.")
    if len(mayores):
        md.append("\n" + md_tabla(mayores.sort_values("delta").head(12)[["codigo", "nombre", "variable", "ancla2024", "v2042", "delta"]], 2))

    # ------------------------------ C6 ------------------------------
    nb = {v: caminos[(COD_NUEVO_BELEN, v)] for v in (V_DQ, V_DL, V_DT)}
    md.append("\n## C6. 27493 Nuevo Belen de Bajira (sin deficit real)\n")
    md.append(md_tabla(pd.DataFrame([{"variable": v, "2024": nb[v][2024], "2030": nb[v][2030], "2042": nb[v][2042]} for v in nb]), 2))
    md.append("\nNota: usa densidad DANE y las coberturas semilla de la Parte E (acueducto 20, alcantarillado 5); "
              "el efecto es el del departamento Choco. Fiabilidad 30%.")
    # ------------------------------ C5 ------------------------------
    md.append("\n## C5. Fiabilidad\n- max(30, 55 - 3*(anio-2025)): 55 en 2025, 31 en 2033, 30 desde 2034. "
              "**No hubo backtest temporal**: solo existe 1 punto real municipal (CNPV 2018); la validacion fue transversal "
              "(corte 2018). El intervalo usa 1.96 x RMSE_cv x sqrt((t-2018)/6): es una heuristica de ensanchamiento, "
              "no un intervalo estadisticamente derivado.")
    md.append(f"\n- originvar general de los municipios con ancla: `{origen_gral}`. Filas emitidas: {len(df)} "
              f"(insert {int((df.accion=='insert').sum())}, update {int((df.accion=='update').sum())}).")
    (DIR_INFORMES / f"C_deficit{SUFIJO}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
