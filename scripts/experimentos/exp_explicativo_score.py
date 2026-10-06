"""
Experimento 5: uso EXPLICATIVO de LightGBM sobre el Score de Privacion,
con importancia por SHAP.

SHAP se calcula con `pred_contrib=True` de LightGBM, que implementa
TreeSHAP nativamente (el paquete `shap` se instalo pero su DLL esta
bloqueada por una directiva de Control de Aplicaciones de Windows en esta
maquina; el algoritmo es el mismo).

>>> ADVERTENCIA DE CIRCULARIDAD <<<
El Score de Privacion NO es un fenomeno observado: es una formula
determinista y lineal definida por el propio proyecto,
    score = (1-acueducto_n)/3 + (1-alcantarillado_n)/3 + deficit_cuanti_n/3
sobre normalizaciones min-max. Entrenar un modelo para "explicar" el
score a partir de esas MISMAS 3 variables es tautologico: el modelo solo
puede recuperar los pesos de 1/3 que ya estan escritos en el codigo, y
SHAP devolveria esa misma particion. Peor aun, la atribucion por
municipio se puede calcular EXACTA y analiticamente desde la formula, sin
ningun modelo. Por eso aqui se corren dos modelos:
  A (control): con las 3 variables de la formula -> demuestra la
     circularidad (R2 ~ 1, aportes ~ 1/3 cada una).
  B (informativo): EXCLUYE las 3 variables de la formula y usa solo
     caracteristicas observables independientes -> esto si responde algo
     no trivial: que rasgos de un municipio acompañan a la privacion.
"""
import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg2

warnings.filterwarnings("ignore")

DB = os.environ.get("DATABASE_URL")
if not DB:
    raise SystemExit(
        "Falta la variable de entorno DATABASE_URL, ej.:\n"
        "  DATABASE_URL=postgresql://... python -m scripts.experimentos.exp_explicativo_score"
    )

REGION = {}
for dep, reg in [(["08","13","20","23","44","47","70","88"], "Caribe"),
                 (["05","11","15","17","25","41","54","63","66","68","73"], "Andina"),
                 (["19","27","52","76"], "Pacifica"),
                 (["50","81","85","99"], "Orinoquia"),
                 (["18","86","91","94","95","97"], "Amazonia")]:
    for d in dep:
        REGION[d] = reg

conn = psycopg2.connect(DB)
zonas = pd.read_sql("SELECT codigo_municipio AS cod, codigo_departamento, nombre FROM zonas_municipio", conn)
ind = pd.read_sql("""SELECT codigodivipola AS cod, namevar, valuevar::float AS valor FROM mapcalordb
                     WHERE yearm=2024 AND codigodivipola IS NOT NULL AND codigodivipola<>'01001'""", conn)
areas = pd.read_sql("SELECT codigo_municipio AS cod, ST_Area(geom::geography)/1000000.0 AS area_km2 FROM zonas_municipio", conn)
conn.close()

df = zonas.merge(ind.pivot_table(index="cod", columns="namevar", values="valor", aggfunc="first"), on="cod", how="left").merge(areas, on="cod", how="left")
df["region"] = df["codigo_departamento"].map(REGION).fillna("Otra")


def norm(s):
    return (s - s.min()) / (s.max() - s.min())


ACU, ALC, CUANTI = "cobertura_acueducto", "cobertura_alcantarillado", "deficit_habitacional_cuantitativo"
df["score_privacion"] = ((1 - norm(df[ACU])) / 3 + (1 - norm(df[ALC])) / 3 + norm(df[CUANTI]) / 3).round(4)
datos = df[df["score_privacion"].notna()].copy()
print(f"Municipios con score calculable: {len(datos)} de {len(df)}")

PARAMS = dict(objective="regression", metric="rmse", num_leaves=15, learning_rate=0.05,
              min_data_in_leaf=20, feature_fraction=0.9, num_threads=-1, verbose=-1)


def correr(nombre, feats, cats):
    X = datos[feats].copy()
    for c in cats:
        X[c] = X[c].astype("category")
    y = datos["score_privacion"].to_numpy()

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(datos))
    folds = np.array_split(idx, 5)
    pred = np.empty(len(datos)); pred[:] = np.nan
    for k in range(5):
        te = folds[k]; tr = np.concatenate([folds[j] for j in range(5) if j != k])
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], label=y[tr], categorical_feature=cats), 400)
        pred[te] = m.predict(X.iloc[te])
    err = pred - y
    ss_res = float(np.sum(err ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    print(f"\n{'='*70}\n{nombre}\n{'='*70}")
    print(f"  features: {feats}")
    print(f"  R2 (validacion cruzada 5 folds) = {1 - ss_res/ss_tot:.4f}")
    print(f"  RMSE={np.sqrt(np.mean(err**2)):.4f}  MAE={np.mean(np.abs(err)):.4f}")

    modelo = lgb.train(PARAMS, lgb.Dataset(X, label=y, categorical_feature=cats), 400)
    contrib = modelo.predict(X, pred_contrib=True)   # TreeSHAP nativo
    shap_vals = contrib[:, :-1]
    imp = np.abs(shap_vals).mean(axis=0)
    orden = np.argsort(imp)[::-1]
    print(f"\n  Importancia SHAP global (|contribucion| media al score):")
    total = imp.sum()
    for i in orden:
        print(f"    {feats[i]:<38}{imp[i]:.5f}   ({100*imp[i]/total:5.1f}% del total)")

    print(f"\n  Importancia SHAP por REGION (% del total dentro de cada region):")
    print("    " + f"{'region':<12}" + "".join(f"{feats[i][:17]:>19}" for i in orden[:4]))
    for reg in ["Caribe", "Andina", "Pacifica", "Orinoquia", "Amazonia"]:
        mask = (datos["region"] == reg).to_numpy()
        if mask.sum() < 10:
            continue
        imp_r = np.abs(shap_vals[mask]).mean(axis=0)
        tot_r = imp_r.sum()
        print("    " + f"{reg:<12}({mask.sum():>4})" + "".join(f"{100*imp_r[i]/tot_r:>14.1f}%" for i in orden[:4]))
    return modelo


# --- A: control circular ---
correr("MODELO A (CONTROL CIRCULAR) -- las 3 variables de la propia formula",
       [ACU, ALC, CUANTI], [])

# --- B: informativo, excluye las 3 de la formula ---
correr("MODELO B (INFORMATIVO) -- excluye las 3 variables de la formula",
       ["densidad_poblacional", "deficit_habitacional_cualitativo", "area_km2", "codigo_departamento"],
       ["codigo_departamento"])

print(f"\n{'='*70}\nATRIBUCION EXACTA (sin modelo), directo de la formula\n{'='*70}")
aporte = pd.DataFrame({
    "acueducto":      (1 - norm(datos[ACU])) / 3,
    "alcantarillado": (1 - norm(datos[ALC])) / 3,
    "deficit_cuanti": norm(datos[CUANTI]) / 3,
})
print("  Aporte medio de cada componente al score (suma = score medio):")
for c in aporte.columns:
    print(f"    {c:<20}{aporte[c].mean():.4f}   (desviacion entre municipios: {aporte[c].std():.4f})")
print(f"    {'score medio':<20}{datos['score_privacion'].mean():.4f}")
print("\n  Cuanto explica cada componente de la VARIANZA del score entre municipios:")
var_total = datos["score_privacion"].var()
for c in aporte.columns:
    print(f"    {c:<20}cov/var = {100*np.cov(aporte[c], datos['score_privacion'])[0,1]/var_total:5.1f}%")
