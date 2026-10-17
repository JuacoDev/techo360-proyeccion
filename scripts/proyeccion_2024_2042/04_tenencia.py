"""Parte D -- tenencia informal 2024-2042.

D1  355 municipios con valor real (IGAC Catastro 2026): ultimo valor constante.
D2  749 restantes: escalera mediana departamental -> ridge -> LightGBM
    (LightGBM solo si mejora >10% de RMSE con CV agrupada por departamento).
D3  sesgo de seleccion: los 355 vs los 749 (KS, diferencia estandarizada de
    medias, AUC de un clasificador 355-vs-749, departamentos sin reales).
D4  valor constante, fiabilidad 30, intervalo +-1.96*RMSE_cv (inflado por
    el sesgo), recorte 0-100.
Salidas: parcial/tenencia.csv, informes/D_tenencia.md, D_sesgo_covariables.csv
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _comun import (ANIOS, DIR_INFORMES, DIR_SALIDA, V_ACU, V_ALC, V_DENS, V_DL, V_DQ, V_TEN, accion,  # noqa: E402
                    asegurar_dirs, cargar_snapshot, combinar, fiabilidad_antiguedad, fila, indice_existente, md_tabla,
                    panel)
from _modelos import SEMILLA, dif_rmse_bootstrap, evaluar, metricas, pipeline_ridge  # noqa: E402

ORIGEN_IGAC = "proyeccion-techo360-ultimo-valor-igac-2026"
ORIGEN_MODELO = "proyeccion-techo360-modelo-tenencia"
ANIO_REAL = 2026
COLS = ["ldens", "acu", "alc", "dl18"]
AUC_FUERTE = 0.70
MIN_REALES_DEPTO = 3


def lgbm():
    return lgb.LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=7, min_child_samples=10,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=SEMILLA, n_jobs=1, verbose=-1, deterministic=True, force_row_wise=True)


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    cob = pd.read_csv(DIR_SALIDA / "parcial" / "cobertura.csv", dtype={"codigodivipola": str})
    dfc = pd.read_csv(DIR_SALIDA / "parcial" / "deficit.csv", dtype={"codigodivipola": str})
    estado = combinar(filas, pd.concat([cob, dfc]))
    existente = indice_existente(filas)
    z = zonas.set_index("codigo")
    md = ["# Parte D -- Tenencia informal\n"]

    ten = filas[(filas.namevar == V_TEN)]
    real26 = ten[(ten.yearm == ANIO_REAL) & ten.real].set_index("codigodivipola").valuevar
    copia24 = ten[(ten.yearm == 2024) & ~ten.real].set_index("codigodivipola").valuevar
    ids_real = sorted(real26.index)
    ids_modelo = sorted(set(z.index) - set(ids_real))
    md.append(f"- Reales 2026 (IGAC Catastro): {len(ids_real)}; a modelar: {len(ids_modelo)}; copias 2024 existentes: {len(copia24)} "
              f"(de las cuales {int((copia24.reindex(real26.index).dropna() != real26.reindex(copia24.index).dropna()).sum())} difieren del real 2026).")
    dif_snap = (real26.reindex(copia24.index) - copia24).dropna()
    rmse_snap = float(np.sqrt((dif_snap ** 2).mean()))
    md.append(f"- Diferencia real2026 - copia2024 (snapshots IGAC de fechas distintas, n={len(dif_snap)}): RMSE {rmse_snap:.2f} pp, "
              f"|dif| max {dif_snap.abs().max():.2f}. Se usa como piso del intervalo de D1.\n")

    # ------------------------- covariables -------------------------
    dens = panel(estado, V_DENS)
    acu, alc = panel(estado, V_ACU), panel(estado, V_ALC)
    dl18 = panel(filas[filas.real], V_DL)[2018]
    X = pd.DataFrame(index=z.index)
    X["depto"] = z.depto
    X["ldens"] = np.log1p(dens[2024].reindex(X.index))
    X["acu"] = acu[2024].reindex(X.index)
    X["alc"] = alc[2024].reindex(X.index)
    X["dl18"] = dl18.reindex(X.index)
    # 27493 no tiene deficit 2018: mediana departamental (Choco) de 2018
    for c in X.index[X.dl18.isna()]:
        X.at[c, "dl18"] = float(dl18[[i for i in dl18.index if i in z.index and z.at[i, "depto"] == z.at[c, "depto"]]].median())
    assert X[COLS].notna().all().all(), X[COLS].isna().sum()
    X["y"] = real26.reindex(X.index)
    X["real"] = X.y.notna()
    train = X[X.real].copy()
    pred = X[~X.real].copy()
    md.append(f"- Covariables: log1p(densidad 2024), acueducto 2024, alcantarillado 2024 (coberturas finales de la Parte B/E), "
              f"deficit cualitativo 2018 real, departamento. Entrenamiento n={len(train)}.\n")

    # ------------------------- D3: sesgo de seleccion -------------------------
    filas_s = []
    for c in COLS + ["y_dq18"]:
        if c == "y_dq18":
            dq18 = panel(filas[filas.real], V_DQ)[2018]
            a, b = dq18.reindex(train.index).dropna(), dq18.reindex(pred.index).dropna()
            nombre = "deficit_cuantitativo_2018"
        else:
            a, b, nombre = train[c], pred[c], c
        ks = ks_2samp(a, b)
        sd = np.sqrt((a.var() + b.var()) / 2)
        filas_s.append({"covariable": nombre, "media_355": a.mean(), "media_749": b.mean(), "mediana_355": a.median(),
                        "mediana_749": b.median(), "dif_estandarizada": (a.mean() - b.mean()) / sd, "KS": ks.statistic, "p_KS": ks.pvalue})
    sesgo = pd.DataFrame(filas_s)
    sesgo.to_csv(DIR_INFORMES / "D_sesgo_covariables.csv", index=False)
    # clasificador 355-vs-749 (sin departamento): AUC por validacion cruzada
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pr = cross_val_predict(clf, X[COLS], X.real, cv=KFold(5, shuffle=True, random_state=SEMILLA), method="predict_proba")[:, 1]
    auc = float(roc_auc_score(X.real, pr))
    por_depto = X.groupby("depto").real.agg(["sum", "count"])
    sin_reales = por_depto[por_depto["sum"] == 0]
    n749_sin = int(pred.depto.isin(sin_reales.index).sum())
    n749_poco = int(pred.depto.isin(por_depto[por_depto["sum"] < MIN_REALES_DEPTO].index).sum())
    top = por_depto.assign(cuota=lambda d: d["sum"] / d["count"]).sort_values("sum", ascending=False).head(6)
    md.append("## D3. Sesgo de seleccion (355 con valor real vs 749 sin valor)\n")
    md.append(md_tabla(sesgo, 3))
    md.append(f"\n- AUC del clasificador 355-vs-749 con las covariables (CV 5 pliegues): **{auc:.3f}** (0.5 = sin desfase; >= {AUC_FUERTE} = desfase fuerte).")
    md.append(f"- Departamentos sin ningun municipio real: {len(sin_reales)} de {len(por_depto)}; afectan a {n749_sin} de los 749; "
              f"con menos de {MIN_REALES_DEPTO} reales: {n749_poco} de los 749.")
    md.append("- Departamentos con mas reales (reales / municipios): " + ", ".join(f"{d} {int(r['sum'])}/{int(r['count'])}" for d, r in top.iterrows()) + ".\n")

    # factor de desfase (heuristico documentado): w = 2*(1-AUC) en [0.3, 1]
    w_modelo = float(np.clip(2 * (1 - auc), 0.3, 1.0)) if auc >= AUC_FUERTE else 1.0
    inflar = 1.0 / w_modelo
    md.append(f"- {'Desfase FUERTE' if auc >= AUC_FUERTE else 'Desfase no fuerte'}: peso del modelo w = {w_modelo:.2f} "
              f"(prediccion = w*modelo + (1-w)*mediana departamental), intervalo x{inflar:.2f}. "
              "Regla heuristica fijada a priori, no validada (no hay etiqueta para los 749).\n")

    # ------------------------- D2: escalera -------------------------
    tabla, err = evaluar(train, "y", COLS)
    tabla.to_csv(DIR_INFORMES / "D_validacion_modelos.csv", index=False)
    # LightGBM con los mismos pliegues agrupados que ridge_sin_dept
    ytr = train.y.reset_index(drop=True)
    Xtr = train[COLS].reset_index(drop=True)
    gtr = train.depto.reset_index(drop=True)
    p_l, idx_l, p_r = [], [], []
    for a, b in GroupKFold(n_splits=min(5, gtr.nunique())).split(Xtr, ytr, groups=gtr):
        p_l.append(lgbm().fit(Xtr.iloc[a], ytr.iloc[a]).predict(Xtr.iloc[b]))
        idx_l.append(b)
    idx_l = np.concatenate(idx_l)
    m_l = metricas(ytr.values[idx_l], np.clip(np.concatenate(p_l), 0, 100))
    tabla = pd.concat([tabla, pd.DataFrame([{"validacion": "cv_agrupada_depto", "predictor": "lightgbm", **m_l}])], ignore_index=True)
    tabla.to_csv(DIR_INFORMES / "D_validacion_modelos.csv", index=False)
    md.append("## D2. Validacion (355 reales)\n")
    md.append(md_tabla(tabla, 3))

    rep = tabla[tabla.validacion == "cv_repetida_5x5"].set_index("predictor")
    gru = tabla[tabla.validacion == "cv_agrupada_depto"].set_index("predictor")
    dif, lo, hi = dif_rmse_bootstrap(err[("cv_repetida", "ridge_dept")], err[("cv_repetida", "depto")])
    ridge_gana = bool(rep.loc["ridge_dept", "RMSE"] < rep.loc["depto", "RMSE"] and hi < 0)
    mejora_lgbm = 1 - gru.loc["lightgbm", "RMSE"] / gru.loc["ridge_sin_dept", "RMSE"]
    lgbm_gana = bool(mejora_lgbm > 0.10)
    ridge_vs_nac_grup = 1 - gru.loc["ridge_sin_dept", "RMSE"] / gru.loc["nacional", "RMSE"]
    md.append(f"\n- ridge+dept vs mediana departamental (CV repetida): dif RMSE {dif:.3f} (IC95% {lo:.3f} .. {hi:.3f}) -> "
              f"{'ridge gana' if ridge_gana else 'ridge NO supera a la mediana departamental'}.")
    md.append(f"- Generalizacion a departamentos sin datos (CV agrupada): ridge sin dept mejora a la mediana nacional en {100*ridge_vs_nac_grup:.1f}% de RMSE.")
    md.append(f"- LightGBM vs ridge (CV agrupada por depto): mejora {100*mejora_lgbm:.1f}% de RMSE -> {'SE USA LightGBM (>10%)' if lgbm_gana else 'NO se usa (exige >10%)'}.\n")

    # ------------------------- ajuste final y prediccion -------------------------
    if lgbm_gana:
        rung, modelo_g = "lightgbm", lgbm().fit(train[COLS], train.y)
    elif ridge_gana:
        rung = "ridge"
        modelo_g = pipeline_ridge(COLS, True).fit(train[COLS + ["depto"]], train.y)
    else:
        rung, modelo_g = "mediana_depto", None
    # winsorizar covariables al rango del entrenamiento (guarda anti-extrapolacion)
    lim = {c: (train[c].min(), train[c].max()) for c in COLS}
    pc = pred.copy()
    n_fuera = {c: int(((pc[c] < lim[c][0]) | (pc[c] > lim[c][1])).sum()) for c in COLS}
    for c in COLS:
        pc[c] = pc[c].clip(*lim[c])
    nac = float(train.y.median())
    med_d = train.groupby("depto").y.agg(["median", "count"])
    base = pd.Series([float(med_d.at[d, "median"]) if d in med_d.index and med_d.at[d, "count"] >= MIN_REALES_DEPTO else nac for d in pc.depto], index=pc.index)
    if rung == "mediana_depto":
        crudo = base.copy()
    elif rung == "ridge":
        crudo = pd.Series(modelo_g.predict(pc[COLS + ["depto"]]), index=pc.index)
    else:
        crudo = pd.Series(modelo_g.predict(pc[COLS]), index=pc.index)
    final = (w_modelo * crudo + (1 - w_modelo) * base).clip(0, 100)
    md.append(f"## Escalon elegido: **{rung}**\n")
    md.append(f"- Covariables de los 749 fuera del rango de entrenamiento (winsorizadas): {n_fuera}.")
    md.append(f"- Prediccion de los 749 (tras encoger): media {final.mean():.2f}, mediana {final.median():.2f}, min {final.min():.2f}, max {final.max():.2f}; "
              f"los 355 reales: media {train.y.mean():.2f}, mediana {train.y.median():.2f}.")

    # RMSE_cv por municipio: dept con reales -> CV repetida; sin reales -> CV agrupada
    clave = {"ridge": ("ridge_dept", "ridge_sin_dept"), "mediana_depto": ("depto", "nacional"), "lightgbm": ("ridge_dept", "lightgbm")}[rung]
    rm_inf = float(rep.loc[clave[0], "RMSE"])
    rm_gru = float(gru.loc[clave[1], "RMSE"])
    rmse_i = pd.Series([rm_inf if d in med_d.index and med_d.at[d, "count"] >= MIN_REALES_DEPTO else rm_gru for d in pc.depto], index=pc.index)

    # ------------------------- filas -------------------------
    out = []
    for c in ids_real:
        v = float(real26[c])
        for t in ANIOS:
            if t == ANIO_REAL:
                continue
            acc = accion(existente, c, V_TEN, t)
            if acc is None:
                continue
            h = abs(t - ANIO_REAL)
            half = 1.96 * rmse_snap * np.sqrt(1 + h)
            out.append(fila(c, V_TEN, t, v, ORIGEN_IGAC, fiabilidad_antiguedad(h), v - half, v + half, acc))
    for c in ids_modelo:
        v = float(final[c])
        half = 1.96 * float(rmse_i[c]) * inflar
        for t in ANIOS:
            acc = accion(existente, c, V_TEN, t)
            if acc is None:
                continue
            out.append(fila(c, V_TEN, t, v, ORIGEN_MODELO, 30.0, v - half, v + half, acc))
    df = pd.DataFrame(out).sort_values(["codigodivipola", "yearM"]).reset_index(drop=True)
    df.to_csv(DIR_SALIDA / "parcial" / "tenencia.csv", index=False)
    md.append(f"\n- Intervalo D4: +-1.96*RMSE_cv ({rm_inf:.2f} con departamento observado; {rm_gru:.2f} sin) x {inflar:.2f}. "
              f"Para D1: +-1.96*{rmse_snap:.2f}*sqrt(1+|t-2026|) (heuristica; no hay serie temporal).")
    md.append(f"- Filas emitidas: {len(df)} (insert {int((df.accion=='insert').sum())}, update {int((df.accion=='update').sum())}).")
    (DIR_INFORMES / "D_tenencia.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
