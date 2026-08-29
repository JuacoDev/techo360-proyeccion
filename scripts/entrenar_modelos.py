"""
Entrenamiento de los 3 modelos LightGBM independientes (uno por variable
objetivo: densidad_poblacional, cobertura_acueducto, cobertura_alcantarillado)
sobre el panel ya extraido por extraer_panel.py.

LightGBM no soporta regresion multi-target de forma nativa -- se entrenan
3 boosters completamente independientes, con las mismas categorias de
feature (lags de su propia variable, codigo_departamento, codigo_municipio,
año) pero ajustados y evaluados por separado. No se usa MultiOutputRegressor
ni ningun wrapper multi-output: al ser 3 modelos de todas formas, es mas
simple y transparente entrenarlos explicitamente.

>>> LIMITACION CONFIRMADA EN EL PANEL, MANEJADA AQUI (no en extraer_panel.py) <<<
extraer_panel.py::VARIABLES_CON_LAG = ["cobertura_acueducto",
"cobertura_alcantarillado"] -- densidad_poblacional NUNCA recibe columnas
de lag en el panel guardado en disco (confirmado leyendo ese modulo, no
supuesto). Tocar extraer_panel.py esta fuera del alcance de este cambio,
asi que este script calcula densidad_poblacional_lag_1/_lag_2 aqui mismo,
en memoria, con la misma tecnica (groupby por municipio, shift 1/2,
ordenado por año) -- ver asegurar_lags(). El CSV en disco no se modifica.

Diferencia deliberada respecto al baseline ingenuo: calcular_baseline_ingenuo
descarta filas sin lag_1 (el baseline NO PUEDE predecir sin el valor del año
anterior). LightGBM si puede: maneja NaN nativamente en las features (las
manda a la rama que minimiza la perdida durante el entrenamiento), asi que
aqui solo se descartan filas con el TARGET faltante -- no se tira informacion
valida de más solo porque falte un lag.

>>> AJUSTE DE HIPERPARAMETROS (cobertura_acueducto / cobertura_alcantarillado) <<<
Resultado real de la corrida inicial (config A, panel real 17,333 filas):
ambas variables mejoraban sobre el baseline ingenuo en validacion
(2022-2023) pero perdian contra el en test (2024), con <variable>_lag_1
dominando el gain de importancia por ordenes de magnitud sobre el resto --
señal de que el arbol arma nodos muy especificos que ajustan ruido del
historico 2008-2021 en vez de un patron que generalice al salto a 2024.

La documentacion oficial de LightGBM es explicita sobre esto:
min_data_in_leaf "previene que el modelo agregue nodos que describen muy
pocas observaciones... es muy improbable que esto generalice bien, y
probablemente es señal de overfitting", y la recomendacion estandar ante
una brecha val/test es "reducir num_leaves, aumentar min_data_in_leaf, o
agregar regularizacion". Por eso densidad_poblacional (que SI generaliza
bien) usa la config unica original (PARAMS_BASE, sin tocar), y
cobertura_acueducto/cobertura_alcantarillado comparan 3 configs
(CONFIGS_COMPARACION) antes de decidir cual guardar -- ver
entrenar_variable_con_comparacion() y elegir_mejor_config() para la regla
de seleccion documentada.

Uso:
    python -m scripts.entrenar_modelos
"""
import json
import logging
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.extraer_panel import RUTA_SALIDA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIR_MODELOS = os.path.join("models")
DIR_REPORTES = os.path.join("models", "reportes")

FEATURES_COMUNES = ["codigo_departamento", "codigo_municipio", "anio"]
FEATURES_CATEGORICAS = ["codigo_departamento", "codigo_municipio"]

# Config de arranque -- sigue siendo la unica config usada para
# densidad_poblacional (fuera de alcance del ajuste de hiperparametros,
# ya generaliza bien). num_leaves bajado de 31 (default) por el tamaño de
# train (~14K filas totales, bastante menos por variable tras filtrar
# NaN de target).
PARAMS_BASE = {
    "num_leaves": 15,
    "learning_rate": 0.05,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.9,
}

# Las 3 configs a comparar para cobertura_acueducto/cobertura_alcantarillado
# (ver aviso del modulo). Config A = PARAMS_BASE (la ya corrida contra el
# panel real). B sube min_data_in_leaf. C suma num_leaves mas bajo, para
# ver si el efecto de ambos ajustes se refuerza.
CONFIGS_COMPARACION = {
    "A": {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 20, "feature_fraction": 0.9},
    "B": {"num_leaves": 15, "learning_rate": 0.05, "min_data_in_leaf": 40, "feature_fraction": 0.9},
    "C": {"num_leaves": 8, "learning_rate": 0.05, "min_data_in_leaf": 40, "feature_fraction": 0.9},
}
VARIABLES_CON_COMPARACION = ["cobertura_acueducto", "cobertura_alcantarillado"]

JUSTIFICACION_AJUSTE = (
    "La documentacion oficial de LightGBM indica que min_data_in_leaf previene "
    "directamente que el modelo agregue nodos que describen muy pocas "
    "observaciones ('es muy improbable que esto generalice bien, y probablemente "
    "es señal de overfitting'), y que ante una brecha val/test la recomendacion "
    "estandar es reducir num_leaves, aumentar min_data_in_leaf, o agregar "
    "regularizacion. Config B prueba el primer ajuste solo; Config C suma el "
    "segundo para ver si el efecto se refuerza."
)

NUM_BOOST_ROUND = 2000       # techo alto; early stopping corta antes
EARLY_STOPPING_ROUNDS = 50

# Aviso (no bloqueo) si una particion queda con menos municipios distintos
# que esto, para el ajuste de particion mencionado en el prompt original.
UMBRAL_MUNICIPIOS_POCOS = 50


def cargar_panel(ruta: str = RUTA_SALIDA) -> pd.DataFrame:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontro el panel en '{ruta}'. Correr primero "
            f"'python -m scripts.extraer_panel' contra la base real "
            f"(requiere DATABASE_URL) para generarlo."
        )
    panel = pd.read_csv(ruta)
    panel["codigo_departamento"] = panel["codigo_departamento"].astype("category")
    panel["codigo_municipio"] = panel["codigo_municipio"].astype("category")
    logger.info(f"Panel cargado: {len(panel)} filas, {len(panel.columns)} columnas, desde '{ruta}'.")
    return panel


def asegurar_lags(panel: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Si el panel ya trae {variable}_lag_1/_lag_2 (variables en
    VARIABLES_CON_LAG de extraer_panel.py), los deja tal cual. Si no (caso
    confirmado de densidad_poblacional), los calcula aqui, en memoria, con
    la misma tecnica que usa extraer_panel.py::construir_panel -- sin
    modificar el CSV en disco."""
    col_lag1, col_lag2 = f"{variable}_lag_1", f"{variable}_lag_2"
    if col_lag1 in panel.columns and col_lag2 in panel.columns:
        return panel

    logger.warning(
        f"'{variable}' no trae columnas de lag en el panel (no esta en "
        f"VARIABLES_CON_LAG de extraer_panel.py) -- calculando {col_lag1}/"
        f"{col_lag2} aqui mismo, en memoria, solo para esta corrida de "
        f"entrenamiento."
    )
    panel = panel.sort_values(["municipio_nombre", "anio"]).reset_index(drop=True)
    panel[col_lag1] = panel.groupby("municipio_nombre")[variable].shift(1)
    panel[col_lag2] = panel.groupby("municipio_nombre")[variable].shift(2)
    return panel


def calcular_baseline_ingenuo(panel: pd.DataFrame, variable: str) -> dict:
    """Misma formula que extraer_panel.py::calcular_baseline_ingenuo
    (prediccion = lag_1), reimplementada aqui para no depender de que esa
    funcion asuma que la columna de lag ya existe en el CSV (ver
    asegurar_lags -- para densidad_poblacional se calcula en este mismo
    script, no en extraer_panel.py)."""
    columna_lag = f"{variable}_lag_1"
    resultados = {}
    for particion in ("val", "test"):
        sub = panel.loc[
            (panel["particion"] == particion) & panel[variable].notna() & panel[columna_lag].notna()
        ]
        if len(sub) == 0:
            logger.warning(f"Baseline ingenuo ({variable}, {particion}): 0 filas evaluables (sin lag_1 disponible).")
            resultados[particion] = {"n": 0, "rmse": None, "mae": None}
            continue
        error = sub[variable] - sub[columna_lag]
        rmse_val = float((error**2).mean() ** 0.5)
        mae_val = float(error.abs().mean())
        resultados[particion] = {"n": int(len(sub)), "rmse": rmse_val, "mae": mae_val}
        logger.info(f"Baseline ingenuo ({variable}, {particion}): n={len(sub)}, RMSE={rmse_val:.4f}, MAE={mae_val:.4f}")
    return resultados


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _entrenar_una_config(panel: pd.DataFrame, variable: str, params: dict, nombre_config: str) -> tuple:
    """Entrena UN booster con UNA config de hiperparametros -- sin guardar
    nada en disco (eso lo decide quien llama, una vez comparadas todas las
    configs candidatas). Devuelve (booster, reporte)."""
    panel = asegurar_lags(panel, variable)
    features = [f"{variable}_lag_1", f"{variable}_lag_2"] + FEATURES_COMUNES

    # Solo se exige que el TARGET no sea nulo -- a diferencia del baseline
    # ingenuo, LightGBM maneja NaN nativamente en las features, asi que no
    # hay que descartar filas por lag faltante (ver aviso del modulo).
    datos = panel.loc[panel[variable].notna(), ["particion", variable] + features].copy()

    conteos_municipio = (
        panel.loc[panel[variable].notna()].groupby("particion")["codigo_municipio"].nunique()
    )
    for particion in ("train", "val", "test"):
        n_municipios = int(conteos_municipio.get(particion, 0))
        if n_municipios < UMBRAL_MUNICIPIOS_POCOS:
            logger.warning(
                f"'{variable}' [{nombre_config}]: particion '{particion}' queda con solo {n_municipios} "
                f"municipio(s) distinto(s) (umbral de aviso: {UMBRAL_MUNICIPIOS_POCOS}) -- revisar la "
                f"particion temporal antes de confiar en esta evaluacion."
            )

    train = datos[datos["particion"] == "train"]
    val = datos[datos["particion"] == "val"]
    test = datos[datos["particion"] == "test"]

    dtrain = lgb.Dataset(
        train[features], label=train[variable],
        categorical_feature=FEATURES_CATEGORICAS, free_raw_data=False,
    )
    dval = lgb.Dataset(
        val[features], label=val[variable], reference=dtrain,
        categorical_feature=FEATURES_CATEGORICAS, free_raw_data=False,
    )

    params_completos = {
        **params,
        "objective": "regression",
        "metric": "rmse",
        "num_threads": -1,
        "verbose": -1,
    }

    evals_result = {}
    booster = lgb.train(
        params_completos,
        dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dtrain, dval],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.record_evaluation(evals_result),
        ],
    )

    pred_val = booster.predict(val[features], num_iteration=booster.best_iteration)
    pred_test = booster.predict(test[features], num_iteration=booster.best_iteration)

    metricas_modelo = {
        "val": {"n": int(len(val)), "rmse": _rmse(val[variable], pred_val), "mae": _mae(val[variable], pred_val)},
        "test": {"n": int(len(test)), "rmse": _rmse(test[variable], pred_test), "mae": _mae(test[variable], pred_test)},
    }

    baseline = calcular_baseline_ingenuo(panel, variable)

    importancia = dict(zip(features, booster.feature_importance(importance_type="gain").tolist()))
    importancia = dict(sorted(importancia.items(), key=lambda kv: kv[1], reverse=True))

    def _supera(particion):
        rmse_base = baseline[particion]["rmse"]
        if rmse_base is None:
            return None
        return metricas_modelo[particion]["rmse"] < rmse_base

    reporte = {
        "variable": variable,
        "nombre_config": nombre_config,
        "params": params,
        "features": features,
        "num_arboles_reales": int(booster.best_iteration),
        "metricas_modelo": metricas_modelo,
        "baseline_ingenuo": baseline,
        "supera_baseline": {"val": _supera("val"), "test": _supera("test")},
        "importancia_features_gain": importancia,
        "curva_rmse_val_por_ronda": evals_result["val"]["rmse"],
    }

    logger.info(
        f"'{variable}' [{nombre_config}] {params}: {reporte['num_arboles_reales']} arbol(es) -- "
        f"val RMSE={metricas_modelo['val']['rmse']:.4f} (baseline {baseline['val']['rmse']}), "
        f"test RMSE={metricas_modelo['test']['rmse']:.4f} (baseline {baseline['test']['rmse']})."
    )

    return booster, reporte


def guardar_modelo_y_reporte(booster: lgb.Booster, reporte: dict) -> dict:
    variable = reporte["variable"]
    os.makedirs(DIR_MODELOS, exist_ok=True)
    ruta_modelo = os.path.join(DIR_MODELOS, f"lightgbm_{variable}.txt")
    booster.save_model(ruta_modelo)

    reporte = {**reporte, "ruta_modelo": ruta_modelo}
    os.makedirs(DIR_REPORTES, exist_ok=True)
    ruta_reporte = os.path.join(DIR_REPORTES, f"reporte_{variable}.json")
    with open(ruta_reporte, "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)

    logger.info(f"'{variable}': modelo guardado en '{ruta_modelo}', reporte en '{ruta_reporte}'.")
    return reporte


def entrenar_modelo_variable(panel: pd.DataFrame, variable: str, params: dict = None) -> dict:
    """Entrena UNA sola configuracion y la guarda en disco de una vez --
    usado para densidad_poblacional (fuera de alcance del ajuste de
    hiperparametros de esta ronda, sin comparacion de configs)."""
    params = params or PARAMS_BASE
    booster, reporte = _entrenar_una_config(panel, variable, params, nombre_config="unica")
    return guardar_modelo_y_reporte(booster, reporte)


def elegir_mejor_config(reportes_por_config: dict) -> tuple:
    """Regla de seleccion (documentada aqui, no solo en el prompt que la
    origino):
    1. Entre las configs que SI superan el baseline ingenuo en test, se
       elige la de menor RMSE de test (desempate: menor RMSE de val).
    2. Si NINGUNA config supera el baseline en test, se elige la de menor
       RMSE de test de todas formas (la "menos mala" de las 3) -- pero se
       marca explicitamente para que no se adopte en produccion sin
       revision humana.
    Devuelve (nombre_config_elegida, ninguna_supera_baseline_en_test)."""
    candidatas = {k: r for k, r in reportes_por_config.items() if r["supera_baseline"]["test"]}
    ninguna_supera = not candidatas
    universo = candidatas if candidatas else reportes_por_config

    nombre_elegido = min(
        universo,
        key=lambda k: (universo[k]["metricas_modelo"]["test"]["rmse"], universo[k]["metricas_modelo"]["val"]["rmse"]),
    )
    return nombre_elegido, ninguna_supera


def entrenar_variable_con_comparacion(panel: pd.DataFrame, variable: str) -> dict:
    """Entrena las 3 configs de CONFIGS_COMPARACION para `variable`, elige
    una segun elegir_mejor_config(), y SOLO esa se guarda en disco (las
    otras 2 quedan solo en el reporte de comparacion, no se persisten)."""
    reportes_por_config = {}
    boosters_por_config = {}

    for nombre_config, params in CONFIGS_COMPARACION.items():
        logger.info(f"--- '{variable}' / config {nombre_config}: {params} ---")
        booster, reporte = _entrenar_una_config(panel, variable, params, nombre_config)
        reportes_por_config[nombre_config] = reporte
        boosters_por_config[nombre_config] = booster

    nombre_ganador, ninguna_supera = elegir_mejor_config(reportes_por_config)
    if ninguna_supera:
        logger.warning(
            f"'{variable}': NINGUNA de las 3 configs supera el baseline ingenuo en test -- se elige "
            f"'{nombre_ganador}' por tener el menor RMSE de test de las 3, pero revisar antes de usar "
            f"en produccion (ver 'ninguna_config_supera_baseline_test' en comparacion_hiperparametros.json)."
        )

    reporte_ganador = guardar_modelo_y_reporte(boosters_por_config[nombre_ganador], reportes_por_config[nombre_ganador])
    reportes_por_config[nombre_ganador] = reporte_ganador

    logger.info(f"'{variable}': config elegida = '{nombre_ganador}' (params={CONFIGS_COMPARACION[nombre_ganador]}).")

    return {
        "variable": variable,
        "justificacion_ajuste": JUSTIFICACION_AJUSTE,
        "configs_probadas": CONFIGS_COMPARACION,
        "resultados_por_config": reportes_por_config,
        "config_elegida": nombre_ganador,
        "ninguna_config_supera_baseline_test": ninguna_supera,
        "regla_de_seleccion": (
            "Entre las configs que superan el baseline ingenuo en test, la de menor RMSE de test "
            "(desempate: menor RMSE de val). Si ninguna supera el baseline, se elige la de menor "
            "RMSE de test de todas formas, marcada para revision humana."
        ),
    }


def main():
    panel = cargar_panel()

    logger.info("=== Entrenando modelo LightGBM para 'densidad_poblacional' (config unica, sin cambios) ===")
    reporte_densidad = entrenar_modelo_variable(panel, "densidad_poblacional")

    comparaciones = {}
    for variable in VARIABLES_CON_COMPARACION:
        logger.info(f"=== Comparando 3 configs de hiperparametros para '{variable}' ===")
        comparaciones[variable] = entrenar_variable_con_comparacion(panel, variable)

    os.makedirs(DIR_REPORTES, exist_ok=True)
    ruta_comparacion = os.path.join(DIR_REPORTES, "comparacion_hiperparametros.json")
    with open(ruta_comparacion, "w", encoding="utf-8") as fh:
        json.dump(comparaciones, fh, indent=2, ensure_ascii=False)
    logger.info(f"Comparacion de hiperparametros guardada en '{ruta_comparacion}'.")

    print("\n" + "=" * 70)
    print("RESUMEN -- densidad_poblacional (sin cambios)")
    print("=" * 70)
    m, b = reporte_densidad["metricas_modelo"], reporte_densidad["baseline_ingenuo"]
    print(f"densidad_poblacional: {reporte_densidad['num_arboles_reales']} arbol(es)")
    print(f"  Val  -> RMSE={m['val']['rmse']:.4f}  (baseline {b['val']['rmse']})")
    print(f"  Test -> RMSE={m['test']['rmse']:.4f}  (baseline {b['test']['rmse']})")

    print("\n" + "=" * 70)
    print("COMPARACION DE HIPERPARAMETROS -- cobertura_acueducto / cobertura_alcantarillado")
    print("=" * 70)
    for variable, comp in comparaciones.items():
        print(f"\n{variable}  (config elegida: {comp['config_elegida']}"
              f"{'  -- NINGUNA supero el baseline en test' if comp['ninguna_config_supera_baseline_test'] else ''})")
        print(f"  {'Config':<8}{'Arboles':<10}{'Val RMSE':<12}{'Test RMSE':<12}{'Supera baseline (test)':<24}")
        for nombre_config, r in comp["resultados_por_config"].items():
            marca = "  <-- elegida" if nombre_config == comp["config_elegida"] else ""
            print(
                f"  {nombre_config:<8}{r['num_arboles_reales']:<10}"
                f"{r['metricas_modelo']['val']['rmse']:<12.4f}{r['metricas_modelo']['test']['rmse']:<12.4f}"
                f"{str(r['supera_baseline']['test']):<24}{marca}"
            )

    return {"densidad_poblacional": reporte_densidad, **comparaciones}


if __name__ == "__main__":
    main()
