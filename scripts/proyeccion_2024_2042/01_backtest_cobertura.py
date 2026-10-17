"""Parte A -- backtest de coberturas (acueducto y alcantarillado).

Principal: entrena con datos <= 2021, evalua contra reales 2022/2023/2024
(horizontes 1,2,3) en municipios con >=4 puntos reales <= 2021.
Complemento: backtest de origen movil (cortes 2012..2021) para decidir el
horizonte maximo de la pendiente y calibrar la anchura de los intervalos.
Determinista (el bootstrap por municipios usa semilla fija).

Salidas en data/proyeccion_2024_2042/informes/: A_*.csv, A_backtest.md,
params_cobertura.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun import DIR_INFORMES, SEMILLA, V_ACU, V_ALC, asegurar_dirs, cargar_snapshot, md_tabla, panel_reales  # noqa: E402
from _metodos_cobertura import predecir  # noqa: E402

METODOS = ["ultimo", "ols4", "ols6", "ols7", "ols4_clip90", "ols4_clip95"]
CORTE = 2021
OBJETIVOS = [2022, 2023, 2024]


def serie_hasta(panel, codigo, corte):
    s = panel.loc[codigo].dropna()
    s = s[s.index <= corte]
    return list(map(int, s.index)), list(map(float, s.values))


def topes_pendiente(panel, corte):
    """|pendiente| p90/p95 de ajustes de 4 puntos moviles con datos <= corte
    (solo datos de entrenamiento, sin fuga)."""
    pend = []
    for codigo in panel.index:
        a, v = serie_hasta(panel, codigo, corte)
        for i in range(4, len(a) + 1):
            pend.append(abs(np.polyfit(a[i - 4:i], v[i - 4:i], 1)[0]))
    pend = np.array(pend)
    return {"p90": float(np.quantile(pend, 0.90)), "p95": float(np.quantile(pend, 0.95))}


def pred_metodo(m, a, v, y, topes, hmax=None):
    if m.startswith("ols4_clip"):
        return predecir("ols4_clip", a, v, y, tope_pend=topes["p" + m[-2:]], horizonte_max=hmax)
    return predecir(m, a, v, y, horizonte_max=hmax)


def evaluar_corte(panel, corte, objetivos, topes, h_desde_corte=True, hmax=None, metodos=METODOS):
    filas = []
    for codigo in panel.index:
        a, v = serie_hasta(panel, codigo, corte)
        if len(a) < 4:
            continue
        for y in objetivos:
            if y <= corte or y not in panel.columns or pd.isna(panel.at[codigo, y]):
                continue
            real = float(panel.at[codigo, y])
            h = y - (corte if h_desde_corte else a[-1])
            for m in metodos:
                cruda = pred_metodo(m, a, v, y, topes, hmax)
                filas.append((codigo, y, h, m, cruda, float(np.clip(cruda, 0, 100)), real, a[-1]))
    return pd.DataFrame(filas, columns=["codigo", "anio", "h", "metodo", "cruda", "pred", "real", "ult_real"])


def metricas(df):
    e = df.pred - df.real
    return pd.Series({"n": len(df), "RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(e.abs().mean()),
                      "fuera_0_100": int(((df.cruda < 0) | (df.cruda > 100)).sum())})


def bootstrap_dif_mse(df, m_a, m_b, reps=2000):
    """IC95% (bootstrap por municipio) de MSE(m_a) - MSE(m_b); negativo = m_a mejor."""
    a = df[df.metodo == m_a].set_index(["codigo", "anio"])
    b = df[df.metodo == m_b].set_index(["codigo", "anio"])
    d = ((a.pred - a.real) ** 2 - (b.pred - b.real) ** 2).groupby(level=0).mean()
    rng = np.random.default_rng(SEMILLA)
    idx = rng.integers(0, len(d), size=(reps, len(d)))
    medias = d.values[idx].mean(axis=1)
    return float(d.mean()), float(np.quantile(medias, 0.025)), float(np.quantile(medias, 0.975))


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    resultados, rolling_all = {}, []
    md = ["# Parte A -- Backtest de cobertura\n"]
    paneles = {v: panel_reales(filas, v, 2024) for v in (V_ACU, V_ALC)}
    for var, panel in paneles.items():
        topes = topes_pendiente(panel, CORTE)
        df = evaluar_corte(panel, CORTE, OBJETIVOS, topes)
        completos = df.groupby("codigo").anio.nunique()
        completos = completos[completos == 3].index  # misma muestra en los 3 horizontes
        dfc = df[df.codigo.isin(completos)]
        tab = dfc.groupby(["metodo", "h"]).apply(metricas, include_groups=False).reset_index()
        tab["metodo"] = pd.Categorical(tab.metodo, METODOS, ordered=True)
        tab = tab.sort_values(["h", "metodo"])
        tab.insert(0, "variable", var)
        resultados[var] = {"tab": tab, "topes": topes, "n_muni": len(completos), "df": dfc}
        tab.to_csv(DIR_INFORMES / f"A_backtest_corte2021_{var}.csv", index=False)

        rl = []
        for corte in range(2012, 2022):
            tp = topes_pendiente(panel, corte)
            r = evaluar_corte(panel, corte, list(range(corte + 1, 2025)), tp, h_desde_corte=False)
            r["corte"] = corte
            rl.append(r)
        rl = pd.concat(rl)
        rl["variable"] = var
        rolling_all.append(rl)

    comb = pd.concat([r["tab"] for r in resultados.values()])
    medio = comb.groupby("metodo", observed=True).RMSE.mean().sort_values()
    medio_mae = comb.groupby("metodo", observed=True).MAE.mean()
    base = medio["ultimo"]
    ganador = str(medio.index[0])

    # Regla de decision (mas estricta que solo RMSE del corte 2021): un metodo
    # solo reemplaza al baseline "ultimo valor" si gana a la vez en (i) RMSE
    # medio h=1..3, (ii) MAE medio h=1..3 y (iii) RMSE del origen movil
    # (cortes 2012-2021, h=1..3 propios). Es un criterio de consistencia,
    # no solo una diferencia puntual en un unico corte.
    rolling = pd.concat(rolling_all)
    rol_tab = rolling.groupby(["variable", "h", "metodo"]).apply(metricas, include_groups=False).reset_index()
    rol_tab.to_csv(DIR_INFORMES / "A_origen_movil_por_horizonte.csv", index=False)
    rol13 = rol_tab[rol_tab.h <= 3].groupby("metodo").RMSE.mean()
    criterios = {}
    if ganador != "ultimo":
        criterios = {"rmse_corte2021": bool(medio[ganador] < base),
                     "mae_corte2021": bool(medio_mae[ganador] < medio_mae["ultimo"]),
                     "rmse_origen_movil_h1a3": bool(rol13[ganador] < rol13["ultimo"])}
    supera = bool(criterios) and all(criterios.values())
    decision = ganador if supera else "ultimo"

    dif = {}
    if decision != "ultimo":
        for var, r in resultados.items():
            dif[var] = bootstrap_dif_mse(r["df"], decision, "ultimo")

    metodo_cap = decision if decision != "ultimo" else "ols4"
    pivote = rol_tab[rol_tab.metodo.isin(["ultimo", metodo_cap])].pivot_table(
        index=["variable", "h"], columns="metodo", values="RMSE").reset_index()
    pivote["regresion_gana"] = pivote[metodo_cap] < pivote["ultimo"]
    nn = rol_tab[rol_tab.metodo == "ultimo"].set_index(["variable", "h"]).n
    pivote["n"] = [int(nn[(v, h)]) for v, h in zip(pivote.variable, pivote.h)]
    pivote.to_csv(DIR_INFORMES / "A_horizonte_regresion_vs_ultimo.csv", index=False)

    hs_ok = sorted(pivote[pivote.n >= 200].h.unique())
    h_star = 0
    for h in hs_ok:
        if pivote[(pivote.h == h) & (pivote.n >= 200)].regresion_gana.all():
            h_star = int(h)
        else:
            break

    rmse_h = {}
    metodo_int = decision if (decision != "ultimo" and h_star > 0) else "ultimo"
    for var, panel in paneles.items():
        rl = []
        for corte in range(2012, 2022):
            tp = topes_pendiente(panel, corte)
            rl.append(evaluar_corte(panel, corte, list(range(corte + 1, 2025)), tp, h_desde_corte=False,
                                    hmax=(h_star if metodo_int != "ultimo" else None), metodos=[metodo_int]))
        rl = pd.concat(rl)
        por_h = rl.groupby("h").apply(lambda d: pd.Series({"n": len(d), "rmse": float(np.sqrt(((d.pred - d.real) ** 2).mean()))}),
                                      include_groups=False)
        por_h = por_h[(por_h.n >= 100) & (por_h.index >= 1)]
        hh = por_h.index.values.astype(float)
        b, a = np.polyfit(np.log(hh), np.log(por_h.rmse.values), 1)  # rmse(h) = exp(a) * h^b
        rmse_h[var] = {"empirico": {int(k): round(float(v), 3) for k, v in por_h.rmse.items()},
                       "n": {int(k): int(v) for k, v in por_h.n.items()},
                       "potencia_a": float(a), "potencia_b": float(b)}

    params = {"metodo": decision, "metodo_ganador_bruto": ganador, "criterios_ganador_vs_baseline": criterios, "supera_baseline": supera,
              "horizonte_max_pendiente": int(h_star), "metodo_intervalos": metodo_int,
              "tope_pendiente_corte2021": {v: resultados[v]["topes"] for v in resultados},
              "rmse_por_horizonte": rmse_h,
              "rmse_medio_h1a3": {str(k): round(float(v), 4) for k, v in medio.items()}}
    (DIR_INFORMES / "params_cobertura.json").write_text(json.dumps(params, indent=2), encoding="utf-8")

    md.append(f"Corte de entrenamiento: <= {CORTE}; objetivos {OBJETIVOS} (h=1,2,3). Municipios elegibles con los 3 objetivos: "
              + ", ".join(f"{v}: {r['n_muni']}" for v, r in resultados.items()) + ".\n")
    for var, r in resultados.items():
        md.append(f"\n## {var} (topes de pendiente de entrenamiento: p90={r['topes']['p90']:.2f}, p95={r['topes']['p95']:.2f} pp/anio)\n")
        md.append(md_tabla(r["tab"]))
    md.append("\n## RMSE medio sobre h=1..3 y ambas variables\n")
    md.append(md_tabla(medio.to_frame("RMSE"), 4))
    md.append(f"\nGanador bruto: **{ganador}**; baseline (ultimo valor) = {base:.4f}. Decision: **{decision}** (supera baseline: {supera}).\n")
    for var, (d, lo, hi) in dif.items():
        md.append(f"- {var}: MSE({decision}) - MSE(ultimo) = {d:.3f} (IC95% bootstrap por municipio {lo:.3f} .. {hi:.3f}).")
    md.append("\n## Origen movil (cortes 2012-2021): RMSE por horizonte propio, regresion vs ultimo valor\n")
    md.append(md_tabla(pivote))
    md.append(f"\nHorizonte maximo de la pendiente (regresion gana en todos los h consecutivos desde 1, n>=200): **H={h_star}**.\n")
    md.append("\n## RMSE por horizonte del metodo final (para intervalos)\n```\n" + json.dumps(rmse_h, indent=1) + "\n```")
    (DIR_INFORMES / "A_backtest.md").write_text("\n".join(md), encoding="utf-8")
    print((DIR_INFORMES / "A_backtest.md").read_text(encoding="utf-8").encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
