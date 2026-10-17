"""Modelos de relacion transversal compartidos por deficit (03) y tenencia (04).

Predictores comparados siempre en el mismo esquema de validacion:
  * nacional      : mediana nacional del conjunto de entrenamiento.
  * depto         : mediana departamental del entrenamiento (nacional si el
                    departamento no tiene datos en el entrenamiento).
  * ridge_dept    : ridge con covariables + departamento como efecto fijo.
  * ridge_sin_dept: ridge solo con covariables (sirve para CV agrupada).
Validaciones:
  * CV repetida (KFold 5x5, semilla fija): el departamento SI tiene datos de
    entrenamiento -> es el escenario de prediccion de un municipio cuyo
    departamento ya tiene observaciones.
  * CV agrupada por departamento (GroupKFold 5): el departamento completo se
    deja fuera -> mide cuanto generalizan las covariables a departamentos
    sin observaciones (aqui la mediana departamental no existe: se compara
    contra la mediana nacional).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEMILLA = 42
ALPHAS = np.logspace(-2, 3, 21)


def pipeline_ridge(num_cols, con_dept):
    pre = [("num", StandardScaler(), list(num_cols))]
    if con_dept:
        pre.append(("dept", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["depto"]))
    return Pipeline([("pre", ColumnTransformer(pre)), ("ridge", RidgeCV(alphas=ALPHAS))])


def _mediana_depto(train_y, train_d, d_test):
    med = train_y.groupby(train_d).median()
    nac = float(np.median(train_y))
    return np.array([med.get(d, nac) for d in d_test])


def metricas(y, p):
    e = np.asarray(p) - np.asarray(y)
    ss_tot = ((y - np.mean(y)) ** 2).sum()
    return {"RMSE": float(np.sqrt((e ** 2).mean())), "MAE": float(np.abs(e).mean()),
            "R2": float(1 - (e ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")}


def evaluar(df, ycol, num_cols, extra=None):
    """df con columnas: ycol, num_cols, depto. Devuelve (tabla, errores) con
    metricas por predictor y esquema, y los errores por fila para pruebas
    pareadas."""
    y = df[ycol].reset_index(drop=True)
    d = df["depto"].reset_index(drop=True)
    X = df[list(num_cols) + ["depto"]].reset_index(drop=True)
    filas, errores = [], {}

    # ---- CV repetida (departamento con datos de entrenamiento) ----
    acum = {k: [] for k in ("nacional", "depto", "ridge_dept")}
    idx_all = []
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=5, random_state=SEMILLA).split(X):
        ytr, yte = y.iloc[tr], y.iloc[te]
        acum["nacional"].append(np.full(len(te), np.median(ytr)))
        acum["depto"].append(_mediana_depto(ytr, d.iloc[tr], d.iloc[te]))
        m = pipeline_ridge(num_cols, True).fit(X.iloc[tr], ytr)
        acum["ridge_dept"].append(m.predict(X.iloc[te]))
        idx_all.append(te)
    idx = np.concatenate(idx_all)
    for k, v in acum.items():
        p = np.clip(np.concatenate(v), 0, 100)
        errores[("cv_repetida", k)] = pd.Series(p - y.values[idx], index=idx)
        filas.append({"validacion": "cv_repetida_5x5", "predictor": k, **metricas(y.values[idx], p)})

    # ---- CV agrupada por departamento ----
    n_grupos = d.nunique()
    gkf = GroupKFold(n_splits=min(5, n_grupos))
    acum = {"nacional": [], "ridge_sin_dept": []}
    idx_all = []
    for tr, te in gkf.split(X, y, groups=d):
        ytr = y.iloc[tr]
        acum["nacional"].append(np.full(len(te), np.median(ytr)))
        m = pipeline_ridge(num_cols, False).fit(X.iloc[tr], ytr)
        acum["ridge_sin_dept"].append(m.predict(X.iloc[te]))
        idx_all.append(te)
    idx = np.concatenate(idx_all)
    for k, v in acum.items():
        p = np.clip(np.concatenate(v), 0, 100)
        errores[("cv_agrupada_depto", k)] = pd.Series(p - y.values[idx], index=idx)
        filas.append({"validacion": "cv_agrupada_depto", "predictor": k, **metricas(y.values[idx], p)})
    return pd.DataFrame(filas), errores


def dif_rmse_bootstrap(e_a, e_b, reps=2000):
    """IC95% bootstrap de RMSE(a)-RMSE(b) con remuestreo de filas (pareado)."""
    # Ambos vectores salen de los mismos pliegues en el mismo orden (alineacion posicional).
    a, b = e_a.values, e_b.values
    assert len(a) == len(b)
    rng = np.random.default_rng(SEMILLA)
    n = len(a)
    difs = []
    for _ in range(reps):
        s = rng.integers(0, n, n)
        difs.append(np.sqrt((a[s] ** 2).mean()) - np.sqrt((b[s] ** 2).mean()))
    return float(np.sqrt((a ** 2).mean()) - np.sqrt((b ** 2).mean())), float(np.quantile(difs, 0.025)), float(np.quantile(difs, 0.975))
