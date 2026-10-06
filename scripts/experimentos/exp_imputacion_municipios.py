"""
Experimento 4: imputacion ENTRE MUNICIPIOS para cobertura_acueducto y
cobertura_alcantarillado.

Validacion cruzada por GRUPO: se dejan fuera MUNICIPIOS completos (no
años), que es la unica forma honesta de simular un municipio sin dato.
Se compara contra alternativas simples: mediana departamental y mediana
nacional.

NOTA sobre tenencia_informal_terrenos: NO se imputa. Los 302 municipios
con dato no son una muestra aleatoria -- el IGAC eligio donde correr la
clasificacion de informalidad, asi que el "faltante" depende de la
decision del IGAC y muy probablemente tambien del nivel real de
informalidad (faltante no aleatorio). Un modelo entrenado sobre los 302
aprenderia el patron de los municipios que el IGAC SI evaluo y lo
extrapolaria a los que decidio no evaluar, sin ninguna forma de medir ese
sesgo con los datos disponibles.
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
        "  DATABASE_URL=postgresql://... python -m scripts.experimentos.exp_imputacion_municipios"
    )
ANIO = 2024
OBJETIVOS = ["cobertura_acueducto", "cobertura_alcantarillado"]

conn = psycopg2.connect(DB)
zonas = pd.read_sql("SELECT codigo_municipio, codigo_departamento, nombre FROM zonas_municipio", conn)
ind = pd.read_sql(f"""
    SELECT codigodivipola AS cod, namevar, valuevar::float AS valor
    FROM mapcalordb WHERE yearm={ANIO} AND codigodivipola IS NOT NULL AND codigodivipola <> '01001'
""", conn)
# Area real (PostGIS) como proxy de tamaño del municipio
areas = pd.read_sql("""
    SELECT codigo_municipio AS cod, ST_Area(geom::geography)/1000000.0 AS area_km2
    FROM zonas_municipio
""", conn)
conn.close()

ancho = ind.pivot_table(index="cod", columns="namevar", values="valor", aggfunc="first")
df = zonas.rename(columns={"codigo_municipio": "cod"}).merge(ancho, on="cod", how="left").merge(areas, on="cod", how="left")
print(f"Municipios en zonas_municipio: {len(df)}")

print("\n=== 1) Cuantos gaps hay y si se SOLAPAN entre las dos variables ===")
falta_acu = df["cobertura_acueducto"].isna()
falta_alc = df["cobertura_alcantarillado"].isna()
print(f"  sin cobertura_acueducto:      {falta_acu.sum()}")
print(f"  sin cobertura_alcantarillado: {falta_alc.sum()}")
print(f"  sin AMBAS (solapamiento):     {(falta_acu & falta_alc).sum()}")
print(f"  sin acueducto pero CON alcantarillado: {(falta_acu & ~falta_alc).sum()}")
print(f"  sin alcantarillado pero CON acueducto: {(falta_alc & ~falta_acu).sum()}")

print("\n=== 2) Que features tienen disponibles los municipios con gap ===")
gaps = df[falta_acu | falta_alc]
for col in ["densidad_poblacional", "deficit_habitacional_total", "deficit_habitacional_cuantitativo",
            "deficit_habitacional_cualitativo", "area_km2"]:
    disp = gaps[col].notna().sum() if col in gaps.columns else 0
    print(f"  {col:36} disponible en {disp}/{len(gaps)} municipios con gap")

FEATURES = ["densidad_poblacional", "deficit_habitacional_total", "deficit_habitacional_cuantitativo",
            "deficit_habitacional_cualitativo", "area_km2", "codigo_departamento"]


def evaluar(objetivo, incluir_otra_cobertura):
    otra = [o for o in OBJETIVOS if o != objetivo][0]
    feats = list(FEATURES) + ([otra] if incluir_otra_cobertura else [])
    datos = df[df[objetivo].notna()].copy()
    datos = datos.dropna(subset=[f for f in feats if f != "codigo_departamento"], how="all")
    X = datos[feats].copy()
    X["codigo_departamento"] = X["codigo_departamento"].astype("category")
    y = datos[objetivo].to_numpy()

    # CV por grupo: municipios completos fuera (cada municipio es 1 fila en
    # este corte de un solo año, asi que un KFold barajado equivale a dejar
    # municipios fuera; se fija semilla para reproducibilidad).
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(datos))
    folds = np.array_split(idx, 5)

    pred_lgbm = np.empty(len(datos)); pred_lgbm[:] = np.nan
    pred_medep = np.empty(len(datos)); pred_medep[:] = np.nan
    pred_menac = np.empty(len(datos)); pred_menac[:] = np.nan

    for k in range(5):
        test_i = folds[k]
        train_i = np.concatenate([folds[j] for j in range(5) if j != k])
        Xtr, ytr = X.iloc[train_i], y[train_i]
        modelo = lgb.train(dict(objective="regression", metric="rmse", num_leaves=15,
                                learning_rate=0.05, min_data_in_leaf=20, feature_fraction=0.9,
                                num_threads=-1, verbose=-1),
                           lgb.Dataset(Xtr, label=ytr, categorical_feature=["codigo_departamento"]),
                           400)
        pred_lgbm[test_i] = np.clip(modelo.predict(X.iloc[test_i]), 0, 100)

        # Alternativas simples, calculadas SOLO con el fold de entrenamiento
        tr = datos.iloc[train_i]
        med_dep = tr.groupby("codigo_departamento")[objetivo].median()
        med_nac = float(tr[objetivo].median())
        dep_test = datos.iloc[test_i]["codigo_departamento"]
        pred_medep[test_i] = dep_test.map(med_dep).fillna(med_nac).to_numpy()
        pred_menac[test_i] = med_nac

    def m(p):
        e = p - y
        ape = np.abs(e[y > 0]) / y[y > 0] * 100
        return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                "ape_med": float(np.median(ape)), "ape_p90": float(np.percentile(ape, 90))}

    return {"n": len(datos), "LightGBM": m(pred_lgbm), "mediana_departamental": m(pred_medep),
            "mediana_nacional": m(pred_menac)}


for objetivo in OBJETIVOS:
    for incluir in (True, False):
        etq = "CON la otra cobertura como feature" if incluir else "SIN la otra cobertura (caso real de los gaps)"
        r = evaluar(objetivo, incluir)
        print(f"\n=== {objetivo} -- {etq} (n={r['n']}, CV 5 folds por municipio) ===")
        print(f"  {'metodo':<26}{'RMSE':>10}{'MAE':>10}{'%med':>9}{'%p90':>9}")
        for nombre in ("LightGBM", "mediana_departamental", "mediana_nacional"):
            mm = r[nombre]
            print(f"  {nombre:<26}{mm['rmse']:>10.3f}{mm['mae']:>10.3f}{mm['ape_med']:>9.3f}{mm['ape_p90']:>9.3f}")


# ============================================================
# ¿Los municipios SIN dato son comparables a los que SI tienen?
# Si no lo son, el error medido por validacion cruzada (que se mide sobre
# los que SI tienen dato) SUBESTIMA el error real al imputarlos.
# ============================================================
print("\n" + "=" * 70)
print("SESGO: comparacion de los municipios CON gap vs SIN gap")
print("=" * 70)
gap = df["cobertura_acueducto"].isna()
print(f"{'variable':<36}{'con gap':>26}{'sin gap':>26}")
for col in ["densidad_poblacional", "area_km2", "deficit_habitacional_total",
            "deficit_habitacional_cuantitativo", "deficit_habitacional_cualitativo"]:
    if col not in df.columns:
        continue
    a, b = df.loc[gap, col].dropna(), df.loc[~gap, col].dropna()
    if len(a) == 0 or len(b) == 0:
        continue
    print(f"{col:<36}{f'mediana={a.median():.2f} (n={len(a)})':>26}{f'mediana={b.median():.2f} (n={len(b)})':>26}")

print("\nDistribucion por departamento de los municipios con gap:")
por_dep = df.loc[gap].groupby("codigo_departamento").size().sort_values(ascending=False)
total_dep = df.groupby("codigo_departamento").size()
for dep, n in por_dep.items():
    print(f"  depto {dep}: {n} de {total_dep[dep]} ({100*n/total_dep[dep]:.0f}% del departamento)")
print(f"\n  departamentos afectados: {len(por_dep)} de {df['codigo_departamento'].nunique()}")
print("\nMunicipios con gap:")
print("  " + ", ".join(sorted(df.loc[gap, "nombre"].tolist())))
print("\nLECTURA: si el grupo con gap tiene deficit sistematicamente mas alto,")
print("la cobertura imputada quedaria SOBREESTIMADA (deficit alto <-> cobertura baja),")
print("y el error real de imputar seria MAYOR que el que mide la validacion cruzada.")
