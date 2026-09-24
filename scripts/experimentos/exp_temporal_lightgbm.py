"""
Experimentos 1, 2 y 3 -- version VECTORIZADA (mismo calculo que
exp_temporal.py, pero prediciendo todos los municipios de una vez en cada
paso recursivo en lugar de fila por fila).

Backtesting de cortes multiples (2014/2016/2018/2020), evaluacion
RECURSIVA multi-año.

ANTI-FUGA: para el corte C, el entrenamiento usa SOLO filas con año <= C,
y las features de una fila objetivo del año t se calculan con historia
estrictamente anterior a t. En inferencia el ancla es siempre C, y cada
paso realimenta la propia prediccion (nunca el valor real).
"""
import sys
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from scripts.extraer_panel import RUTA_SALIDA

CORTES = [2014, 2016, 2018, 2020]
ANIO_FIN = 2024
VARIABLES = ["densidad_poblacional", "cobertura_acueducto", "cobertura_alcantarillado"]
LIMITES = {"densidad_poblacional": (0.0, np.inf),
           "cobertura_acueducto": (0.0, 100.0),
           "cobertura_alcantarillado": (0.0, 100.0)}
PARAMS = dict(objective="regression", metric="rmse", num_leaves=15, learning_rate=0.05,
              min_data_in_leaf=20, feature_fraction=0.9, num_threads=-1, verbose=-1)
ROUNDS = 400
FEATS = ["lag1", "lag2", "lag3", "d1", "d2", "dep", "mun"]
CATEG = ["dep", "mun"]

panel = pd.read_csv(RUTA_SALIDA,
                    dtype={"codigo_municipio": "string", "codigo_departamento": "string"})
panel = panel.dropna(subset=["codigo_municipio"])
CATS_MUN = sorted(panel["codigo_municipio"].unique().tolist())
CATS_DEP = sorted(panel["codigo_departamento"].dropna().unique().tolist())
DEPTO = panel.drop_duplicates("codigo_municipio").set_index("codigo_municipio")["codigo_departamento"].to_dict()

# matriz municipio x año por variable. TODAS comparten el mismo indice de
# municipios y las mismas columnas de año: si no, los filtros booleanos
# calculados sobre una variable no alinean con otra (algunos municipios no
# tienen ninguna fila de alguna variable).
MAT = {}
for v in VARIABLES:
    m = panel.pivot_table(index="codigo_municipio", columns="anio", values=v, aggfunc="first")
    MAT[v] = m.reindex(index=CATS_MUN, columns=range(2008, ANIO_FIN + 1))


def armar_X(lag1, lag2, lag3, muns, feats, anio=None):
    d = {"lag1": lag1, "lag2": lag2, "lag3": lag3, "d1": lag1 - lag2, "d2": lag2 - lag3}
    X = pd.DataFrame(d)
    if anio is not None:
        X["anio"] = anio
    X["dep"] = pd.Categorical([DEPTO.get(c) for c in muns], categories=CATS_DEP)
    X["mun"] = pd.Categorical(list(muns), categories=CATS_MUN)
    return X[feats]


def entrenar(variable, corte, objetivo, con_anio, linear_tree):
    M = MAT[variable]
    anios = [a for a in M.columns if a <= corte]
    filas_l1, filas_l2, filas_l3, filas_mun, filas_anio, ys = [], [], [], [], [], []
    for t in anios:
        if (t - 1) not in M.columns or (t - 2) not in M.columns or (t - 3) not in M.columns:
            continue
        v_t, l1, l2, l3 = M[t].to_numpy(), M[t - 1].to_numpy(), M[t - 2].to_numpy(), M[t - 3].to_numpy()
        ok = ~(np.isnan(v_t) | np.isnan(l1) | np.isnan(l2) | np.isnan(l3))
        if objetivo == "nivel":
            y = v_t
        elif objetivo == "diff":
            y = v_t - l1
        else:
            ok &= (v_t > 0) & (l1 > 0)
            y = np.where(ok, np.log(np.where(ok, v_t, 1) / np.where(ok, l1, 1)), np.nan)
        filas_l1.append(l1[ok]); filas_l2.append(l2[ok]); filas_l3.append(l3[ok])
        filas_mun.append(M.index.to_numpy()[ok]); filas_anio.append(np.full(ok.sum(), t)); ys.append(y[ok])
    if not ys or sum(len(y) for y in ys) < 50:
        return None
    lag1, lag2, lag3 = np.concatenate(filas_l1), np.concatenate(filas_l2), np.concatenate(filas_l3)
    muns, anio_arr, y = np.concatenate(filas_mun), np.concatenate(filas_anio), np.concatenate(ys)
    feats = FEATS + (["anio"] if con_anio else [])
    X = armar_X(lag1, lag2, lag3, muns, feats, anio_arr if con_anio else None)
    p = dict(PARAMS)
    if linear_tree:
        p["linear_tree"] = True
    return lgb.train(p, lgb.Dataset(X, label=y, categorical_feature=CATEG, free_raw_data=False), ROUNDS), feats


def predecir(modelo_feats, variable, muns, corte, objetivo, con_anio):
    modelo, feats = modelo_feats
    M = MAT[variable].loc[muns]
    lo, hi = LIMITES[variable]
    l1, l2, l3 = M[corte].to_numpy(), M[corte - 1].to_numpy(), M[corte - 2].to_numpy()
    out = {}
    for t in range(corte + 1, ANIO_FIN + 1):
        X = armar_X(l1, l2, l3, muns, feats, np.full(len(muns), t) if con_anio else None)
        bruto = modelo.predict(X)
        if objetivo == "nivel":
            val = bruto
        elif objetivo == "diff":
            val = l1 + bruto
        else:
            val = l1 * np.exp(bruto)
        val = np.clip(val, lo, hi)
        out[t] = val
        l3, l2, l1 = l2, l1, val
    return out


def reg_lineal_lote(variable, muns, corte, n, anios_obj):
    M = MAT[variable].loc[muns]
    lo, hi = LIMITES[variable]
    cols = [a for a in M.columns if a <= corte]
    pend = np.full(len(muns), np.nan); inter = np.full(len(muns), np.nan)
    arr = M[cols].to_numpy()
    for i in range(len(muns)):
        disp = [(a, v) for a, v in zip(cols, arr[i]) if not np.isnan(v)][-n:]
        if len(disp) >= 2:
            xs = [a for a, _ in disp]; ys = [v for _, v in disp]
            if len(set(xs)) >= 2:
                pend[i], inter[i] = np.polyfit(xs, ys, 1)
    return {t: np.clip(pend * t + inter, lo, hi) for t in anios_obj}


def ultimo_valor_lote(variable, muns, corte, anios_obj):
    M = MAT[variable].loc[muns]
    cols = [a for a in M.columns if a <= corte]
    arr = M[cols].to_numpy()
    ult = np.array([next((v for v in fila[::-1] if not np.isnan(v)), np.nan) for fila in arr])
    return {t: ult.copy() for t in anios_obj}


def metricas(pred, variable, muns, anios_obj):
    M = MAT[variable].loc[muns]
    errs, apes = [], []
    for t in anios_obj:
        real = M[t].to_numpy(); p = pred[t]
        ok = ~(np.isnan(real) | np.isnan(p))
        e = p[ok] - real[ok]
        errs.append(e)
        r = real[ok]
        apes.append(np.abs(e[r > 0]) / r[r > 0] * 100)
    e = np.concatenate(errs); a = np.concatenate(apes)
    if len(e) == 0:
        return None
    return {"n": len(e), "rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
            "ape_med": float(np.median(a)), "ape_p90": float(np.percentile(a, 90))}


def entrenar_hibrido(variable, corte, muns):
    """Base = regresion lineal N=4; LightGBM aprende su RESIDUO usando las
    otras variables del municipio, departamento y tamaño."""
    M = MAT[variable]
    filas, ys = [], []
    anclas = [a for a in M.columns if a <= corte and a >= 2011]
    for a in anclas:
        objetivos = [t for t in M.columns if a < t <= corte]
        if not objetivos:
            continue
        base = reg_lineal_lote(variable, M.index, a, 4, objetivos)
        for t in objetivos:
            real = M[t].to_numpy(); bp = base[t]; anc = M[a].to_numpy()
            ok = ~(np.isnan(real) | np.isnan(bp) | np.isnan(anc))
            if ok.sum() == 0:
                continue
            fila = {"h": np.full(ok.sum(), t - a), "base": bp[ok], "nivel_ancla": anc[ok]}
            for v2 in VARIABLES:
                if v2 != variable:
                    fila[f"otra_{v2[:10]}"] = MAT[v2][a].to_numpy()[ok]
            fila["_mun"] = M.index.to_numpy()[ok]
            filas.append(pd.DataFrame(fila)); ys.append(real[ok] - bp[ok])
    if not filas:
        return None
    D = pd.concat(filas, ignore_index=True); y = np.concatenate(ys)
    muns_col = D.pop("_mun")
    D["dep"] = pd.Categorical([DEPTO.get(c) for c in muns_col], categories=CATS_DEP)
    D["mun"] = pd.Categorical(list(muns_col), categories=CATS_MUN)
    feats = list(D.columns)
    return lgb.train(PARAMS, lgb.Dataset(D, label=y, categorical_feature=CATEG, free_raw_data=False), ROUNDS), feats


def main():
    for variable in VARIABLES:
        print(f"\n{'='*80}\nVARIABLE: {variable}\n{'='*80}")
        acum = {}
        for corte in CORTES:
            anios_obj = list(range(corte + 1, ANIO_FIN + 1))
            M = MAT[variable]
            hist_ok = M[[a for a in M.columns if a <= corte]].notna().sum(axis=1) >= 5
            fut_ok = M[anios_obj].notna().all(axis=1)
            muns = M.index[hist_ok & fut_ok].tolist()
            if len(muns) < 50:
                continue

            metodos = {"VIG_reglineal_N4": reg_lineal_lote(variable, muns, corte, 4, anios_obj),
                       "VIG_ultimo_valor": ultimo_valor_lote(variable, muns, corte, anios_obj)}
            for nombre, obj, con_anio, lt in [
                ("E1_diff", "diff", False, False),
                ("E1_logratio", "logratio", False, False),
                ("E2_nivel_lineartree", "nivel", True, True),
                ("E2_diff_lineartree", "diff", False, True),
                ("REF_nivel_lgbm", "nivel", True, False),
            ]:
                mf = entrenar(variable, corte, obj, con_anio, lt)
                if mf:
                    metodos[nombre] = predecir(mf, variable, muns, corte, obj, con_anio)

            hib = entrenar_hibrido(variable, corte, muns)
            if hib:
                modelo_h, feats_h = hib
                base = reg_lineal_lote(variable, muns, corte, 4, anios_obj)
                pred_h = {}
                Msub = MAT[variable].loc[muns]
                for t in anios_obj:
                    d = {"h": np.full(len(muns), t - corte), "base": base[t],
                         "nivel_ancla": Msub[corte].to_numpy()}
                    for v2 in VARIABLES:
                        if v2 != variable:
                            d[f"otra_{v2[:10]}"] = MAT[v2].loc[muns][corte].to_numpy()
                    D = pd.DataFrame(d)
                    D["dep"] = pd.Categorical([DEPTO.get(c) for c in muns], categories=CATS_DEP)
                    D["mun"] = pd.Categorical(list(muns), categories=CATS_MUN)
                    lo, hi = LIMITES[variable]
                    pred_h[t] = np.clip(base[t] + modelo_h.predict(D[feats_h]), lo, hi)
                metodos["E3_hibrido_residuos"] = pred_h

            print(f"\n--- corte {corte} ({len(muns)} municipios, horizontes 1-{ANIO_FIN-corte}) ---")
            print(f"  {'metodo':<26}{'RMSE':>10}{'MAE':>10}{'%med':>9}{'%p90':>9}")
            for nombre, pred in metodos.items():
                m = metricas(pred, variable, muns, anios_obj)
                if m:
                    acum.setdefault(nombre, []).append(m)
                    print(f"  {nombre:<26}{m['rmse']:>10.3f}{m['mae']:>10.3f}{m['ape_med']:>9.3f}{m['ape_p90']:>9.3f}")
            sys.stdout.flush()

        print(f"\n### PROMEDIO DE LOS {len(CORTES)} CORTES -- {variable} ###")
        print(f"  {'metodo':<26}{'RMSE':>10}{'MAE':>10}{'%med':>9}{'%p90':>9}")
        for nombre, ms in sorted(acum.items(), key=lambda kv: np.mean([m['rmse'] for m in kv[1]])):
            pr = {k: float(np.mean([m[k] for m in ms])) for k in ("rmse", "mae", "ape_med", "ape_p90")}
            print(f"  {nombre:<26}{pr['rmse']:>10.3f}{pr['mae']:>10.3f}{pr['ape_med']:>9.3f}{pr['ape_p90']:>9.3f}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
