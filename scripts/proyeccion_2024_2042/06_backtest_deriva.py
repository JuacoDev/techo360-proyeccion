"""Parte 1 (tanda 3) -- backtest 1b: "ultimo valor + deriva" para cobertura.

Metodos (todos recortados a [0,100]):
  ultimo                      baseline
  {dep|nac}_W{4|6}_f{1|0.5}   ultimo + f * deriva_grupo * h
deriva_grupo = mediana, sobre los municipios del grupo (departamento o pais),
de la deriva anual de cada municipio; la deriva anual de un municipio = mediana
de las diferencias entre observaciones reales consecutivas (dividida por la
separacion en anios) dentro de los ultimos W anios del corte (robusta a saltos
y a valores arrastrados). h = anio_objetivo - ultimo anio real del municipio.

Evaluacion: municipios con >=4 puntos reales hasta el corte; cortes moviles
2012-2021; objetivos hasta 2024; horizonte de reporte = anio - corte (1..8).
Criterio de adopcion (por variable): el metodo debe ganar al baseline en
RMSE y MAE tanto en el corte 2021 (h=1..3) como en el origen movil (h=1..3),
con IC95% bootstrap (por municipio, semilla fija) del cambio estrictamente <0.
Salidas: informes/G_backtest_deriva.md, G_*.csv, params_deriva.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun import DIR_INFORMES, SEMILLA, V_ACU, V_ALC, asegurar_dirs, cargar_snapshot, md_tabla, panel_reales  # noqa: E402

CORTES = list(range(2012, 2022))
VENTANAS = (4, 6)
FACTORES = (1.0, 0.5)
GRUPOS = ("dep", "nac")
H_MAX = 8
REPS = 2000
BASE = "ultimo"


def nombre(g, w, f):
    return f"{g}_W{w}_f{f:g}"


METODOS = [BASE] + [nombre(g, w, f) for g in GRUPOS for w in VENTANAS for f in FACTORES]


def deriva_municipio(anios, valores, corte, ventana):
    """Mediana de (dif/separacion) entre observaciones consecutivas en (corte-W, corte]."""
    sel = [(a, v) for a, v in zip(anios, valores) if corte - ventana < a <= corte]
    if len(sel) < 2:
        return np.nan
    d = [(sel[i + 1][1] - sel[i][1]) / (sel[i + 1][0] - sel[i][0]) for i in range(len(sel) - 1)]
    return float(np.median(d))


def construir_obs(panel, depto):
    filas = []
    for corte in CORTES:
        series = {}
        for c in panel.index:
            s = panel.loc[c].dropna()
            s = s[s.index <= corte]
            if len(s) >= 4:
                series[c] = (list(map(int, s.index)), list(map(float, s.values)))
        drift = {}
        for w in VENTANAS:
            m = {c: deriva_municipio(a, v, corte, w) for c, (a, v) in series.items()}
            ser = pd.Series(m)
            nac = float(ser.median())
            dep = ser.groupby(depto.reindex(ser.index)).median()
            drift[w] = (m, nac, dep)
        for c, (a, v) in series.items():
            ultimo_a, ultimo_v = a[-1], v[-1]
            for y in range(corte + 1, 2025):
                if y not in panel.columns or pd.isna(panel.at[c, y]):
                    continue
                h_rep = y - corte
                if h_rep > H_MAX:
                    continue
                h = y - ultimo_a
                fila = {"corte": corte, "codigo": c, "anio": y, "h": h_rep, "real": float(panel.at[c, y])}
                fila[BASE] = ultimo_v
                for w in VENTANAS:
                    m, nac, dep = drift[w]
                    d_dep = dep.get(depto[c], np.nan)
                    d_dep = nac if np.isnan(d_dep) else d_dep
                    for g, dg in (("dep", d_dep), ("nac", nac)):
                        for f in FACTORES:
                            fila[nombre(g, w, f)] = float(np.clip(ultimo_v + f * dg * h, 0, 100))
                filas.append(fila)
    return pd.DataFrame(filas)


def metricas_y_ic(df, metodos, reps=REPS):
    """RMSE, MAE de cada metodo y IC95% bootstrap (por municipio) de la diferencia con BASE."""
    cod = df.codigo.astype("category")
    k = cod.cat.codes.values
    n_m = len(cod.cat.categories)
    n = np.bincount(k, minlength=n_m).astype(float)
    S, A = {}, {}
    for m in metodos:
        e = df[m].values - df.real.values
        S[m] = np.bincount(k, weights=e ** 2, minlength=n_m)
        A[m] = np.bincount(k, weights=np.abs(e), minlength=n_m)
    rng = np.random.default_rng(SEMILLA)
    idx = rng.integers(0, n_m, size=(reps, n_m))
    N = n[idx].sum(axis=1)
    rm_b = {m: np.sqrt(S[m][idx].sum(axis=1) / N) for m in metodos}
    ma_b = {m: A[m][idx].sum(axis=1) / N for m in metodos}
    filas = []
    for m in metodos:
        fila = {"metodo": m, "n": int(n.sum()), "RMSE": float(np.sqrt(S[m].sum() / n.sum())), "MAE": float(A[m].sum() / n.sum())}
        if m != BASE:
            for etq, boot, real in (("dRMSE", rm_b[m] - rm_b[BASE], fila["RMSE"] - float(np.sqrt(S[BASE].sum() / n.sum()))),
                                    ("dMAE", ma_b[m] - ma_b[BASE], fila["MAE"] - float(A[BASE].sum() / n.sum()))):
                fila[etq] = real
                fila[etq + "_lo"], fila[etq + "_hi"] = float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        filas.append(fila)
    return pd.DataFrame(filas)


def gana(t):
    """True si dRMSE y dMAE tienen IC95% estrictamente < 0."""
    return (t.dRMSE_hi < 0) & (t.dMAE_hi < 0)


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    depto = zonas.set_index("codigo").depto
    md = ["# Backtest 1b -- ultimo valor + deriva\n",
          f"Metodos: {', '.join(METODOS)}. Cortes {CORTES[0]}-{CORTES[-1]}, ventanas {VENTANAS}, factores {FACTORES}. "
          f"Bootstrap por municipio, {REPS} remuestreos, semilla {SEMILLA}.\n"]
    veredicto, params = {}, {"metodos": METODOS}
    todos = {}
    for var in (V_ACU, V_ALC):
        panel = panel_reales(filas, var, 2024)
        obs = construir_obs(panel, depto)
        todos[var] = obs
        md.append(f"\n## {var}: {len(obs)} observaciones, {obs.codigo.nunique()} municipios\n")
        # (i) corte 2021, h=1..3
        c21 = obs[(obs.corte == 2021) & (obs.h <= 3)]
        t21 = metricas_y_ic(c21, METODOS)
        # (ii) origen movil h=1..3
        rol = obs[obs.h <= 3]
        trol = metricas_y_ic(rol, METODOS)
        for etq, t in (("corte2021_h1a3", t21), ("origen_movil_h1a3", trol)):
            t.insert(0, "variable", var)
            t.insert(1, "evaluacion", etq)
            t.to_csv(DIR_INFORMES / f"G_{etq}_{var}.csv", index=False)
            md.append(f"\n### {etq} (n={int(t.n.iloc[0])})\n")
            md.append(md_tabla(t.drop(columns=["variable", "evaluacion"]), 3))
        win21, winrol = gana(t21.iloc[1:]), gana(trol.iloc[1:])
        ok = set(t21.iloc[1:][win21].metodo) & set(trol.iloc[1:][winrol].metodo)
        veredicto[var] = sorted(ok)
        # (iii) por horizonte (origen movil, h=1..8)
        ph = []
        for h in range(1, H_MAX + 1):
            sub = obs[obs.h == h]
            if sub.codigo.nunique() < 100:
                continue
            t = metricas_y_ic(sub, METODOS, reps=500)
            t.insert(0, "h", h)
            ph.append(t)
        ph = pd.concat(ph)
        ph.insert(0, "variable", var)
        ph.to_csv(DIR_INFORMES / f"G_por_horizonte_{var}.csv", index=False)
        resumen = ph[ph.metodo != BASE].copy()
        resumen["gana_h"] = gana(resumen)
        piv = resumen.pivot(index="metodo", columns="h", values="gana_h")
        md.append("\n### Gana (RMSE y MAE con IC95%<0) por horizonte, origen movil\n")
        md.append(md_tabla(piv.reset_index().astype({c: str for c in piv.columns}), 0))
        rb = ph[ph.metodo == BASE][["h", "n", "RMSE", "MAE"]]
        md.append("\nBaseline por horizonte:\n\n" + md_tabla(rb, 3))
        md.append(f"\n**Metodos que cumplen el criterio de adopcion en {var}: {veredicto[var] or 'ninguno'}**")

    comunes = sorted(set(veredicto[V_ACU]) & set(veredicto[V_ALC]))
    params["adoptados_ambas_variables"] = comunes
    params["veredicto_por_variable"] = veredicto
    md.append(f"\n## Decision\n\nMetodos que ganan en AMBAS variables: **{comunes or 'ninguno'}**. "
              + ("Se mantiene el ULTIMO VALOR; no se regenera nada." if not comunes else "Candidatos para regenerar."))
    (DIR_INFORMES / "params_deriva.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    (DIR_INFORMES / "G_backtest_deriva.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
