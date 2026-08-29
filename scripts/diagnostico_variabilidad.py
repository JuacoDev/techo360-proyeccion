"""
Diagnostico de variabilidad real -- SOLO lectura y calculo estadistico
sobre el panel ya existente (data/processed/panel_entrenamiento.csv), sin
entrenar ningun modelo. No modifica entrenar_modelos.py, extraer_panel.py
ni los modelos guardados en models/.

Contexto: tras probar 3 configuraciones de hiperparametros de LightGBM
(ver entrenar_modelos.py) ninguna supero el baseline ingenuo (prediccion =
valor del año anterior) en test 2024 para cobertura_acueducto ni
cobertura_alcantarillado. Este script responde, con datos reales, si eso
se explica porque esas dos variables tienen poco margen real de cambio
año a año (techo de cobertura ya alcanzado en la mayoria de municipios) --
en cuyo caso el baseline ingenuo es un piso genuinamente dificil de vencer,
no una limitacion del modelo -- comparando contra densidad_poblacional
(que si supero el baseline) como punto de referencia de "cuanto margen de
cambio tiene una variable donde el modelo si gana".

Uso:
    python -m scripts.diagnostico_variabilidad
"""
import json
import logging
import os

import pandas as pd

from scripts.extraer_panel import RUTA_SALIDA

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VARIABLES_COBERTURA = ["cobertura_acueducto", "cobertura_alcantarillado"]
VARIABLE_REFERENCIA = "densidad_poblacional"  # supero el baseline -- punto de comparacion
ANIO_MAS_RECIENTE = 2024
ANIO_TRAIN_FIN = 2021
UMBRAL_DELTA_ESTABLE = 1.0  # puntos porcentuales, para "practicamente no cambio"

DIR_REPORTES = os.path.join("models", "reportes")


def cargar_panel(ruta: str = RUTA_SALIDA) -> pd.DataFrame:
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontro el panel en '{ruta}'. Correr primero "
            f"'python -m scripts.extraer_panel' contra la base real "
            f"(requiere DATABASE_URL) para generarlo."
        )
    panel = pd.read_csv(ruta)
    logger.info(f"Panel cargado: {len(panel)} filas, desde '{ruta}'.")
    return panel


def distribucion_anio(panel: pd.DataFrame, variable: str, anio: int) -> dict:
    """Punto 1: percentiles y % de municipios por encima de 90/95, para
    UN año dado (pensado para el año mas reciente, 2024)."""
    valores = panel.loc[panel["anio"] == anio, variable].dropna()
    if len(valores) == 0:
        return {"n": 0}
    percentiles = valores.quantile([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        "n": int(len(valores)),
        "percentiles": {f"p{int(q * 100)}": float(v) for q, v in percentiles.items()},
        "pct_municipios_gte_90": float((valores >= 90).mean() * 100),
        "pct_municipios_gte_95": float((valores >= 95).mean() * 100),
    }


def delta_anual(panel: pd.DataFrame, variable: str, umbral_estable: float = UMBRAL_DELTA_ESTABLE) -> dict:
    """Punto 2: variacion año a año (valor_t - valor_t-1) por municipio,
    sobre todo el historico disponible -- media, desviacion estandar, y
    que fraccion de esos deltas es practicamente cero (dentro de
    +/-umbral_estable)."""
    panel_ordenado = panel.sort_values(["municipio_nombre", "anio"])
    delta = panel_ordenado.groupby("municipio_nombre")[variable].diff().dropna()
    if len(delta) == 0:
        return {"n_deltas": 0}
    return {
        "n_deltas": int(len(delta)),
        "media": float(delta.mean()),
        "std": float(delta.std()),
        f"pct_deltas_dentro_de_{umbral_estable:g}pp": float((delta.abs() <= umbral_estable).mean() * 100),
    }


def concentracion_techo(panel: pd.DataFrame, variable: str) -> dict:
    """Punto 3: cuantos municipio-año tienen valores >= 95 y >= 99, sobre
    TODO el panel (no solo un año), y que fraccion del total representan."""
    valores = panel[variable].dropna()
    n_total = len(valores)
    if n_total == 0:
        return {"n_total_municipio_anio": 0}
    n_95 = int((valores >= 95).sum())
    n_99 = int((valores >= 99).sum())
    return {
        "n_total_municipio_anio": int(n_total),
        "n_gte_95": n_95,
        "frac_gte_95": n_95 / n_total,
        "n_gte_99": n_99,
        "frac_gte_99": n_99 / n_total,
    }


def _resumen_periodo(valores: pd.Series) -> dict:
    if len(valores) == 0:
        return {"n": 0}
    return {
        "n": int(len(valores)),
        "media": float(valores.mean()),
        "std": float(valores.std()),
        "p50": float(valores.median()),
        "pct_gte_95": float((valores >= 95).mean() * 100),
        "pct_gte_99": float((valores >= 99).mean() * 100),
    }


def comparacion_train_vs_test(panel: pd.DataFrame, variable: str) -> dict:
    """Punto 4: ¿2024 (test) tiene una distribucion notablemente distinta
    a 2008-2021 (train)? En particular, ¿mas municipios llegaron al techo
    en 2024 que en train?"""
    train = panel.loc[panel["anio"] <= ANIO_TRAIN_FIN, variable].dropna()
    test = panel.loc[panel["anio"] == ANIO_MAS_RECIENTE, variable].dropna()
    return {"train_2008_2021": _resumen_periodo(train), "test_2024": _resumen_periodo(test)}


def diagnosticar_variable_cobertura(panel: pd.DataFrame, variable: str) -> dict:
    return {
        "variable": variable,
        "distribucion_2024": distribucion_anio(panel, variable, ANIO_MAS_RECIENTE),
        "delta_anual_2008_2024": delta_anual(panel, variable),
        "concentracion_techo_todo_el_panel": concentracion_techo(panel, variable),
        "train_vs_test": comparacion_train_vs_test(panel, variable),
    }


def main() -> dict:
    panel = cargar_panel()

    reporte = {
        "cobertura_acueducto": diagnosticar_variable_cobertura(panel, "cobertura_acueducto"),
        "cobertura_alcantarillado": diagnosticar_variable_cobertura(panel, "cobertura_alcantarillado"),
        "densidad_poblacional_referencia": {
            "variable": VARIABLE_REFERENCIA,
            "delta_anual_2008_2024": delta_anual(panel, VARIABLE_REFERENCIA),
        },
    }

    os.makedirs(DIR_REPORTES, exist_ok=True)
    ruta_reporte = os.path.join(DIR_REPORTES, "diagnostico_variabilidad.json")
    with open(ruta_reporte, "w", encoding="utf-8") as fh:
        json.dump(reporte, fh, indent=2, ensure_ascii=False)
    logger.info(f"Diagnostico guardado en '{ruta_reporte}'.")

    print("\n" + "=" * 70)
    print("DIAGNOSTICO DE VARIABILIDAD REAL")
    print("=" * 70)
    for variable in ("cobertura_acueducto", "cobertura_alcantarillado"):
        d = reporte[variable]
        print(f"\n--- {variable} ---")
        print(f"1) Distribucion 2024 (n={d['distribucion_2024']['n']}): {d['distribucion_2024']['percentiles']}")
        print(f"   >=90%: {d['distribucion_2024']['pct_municipios_gte_90']:.1f}%   "
              f">=95%: {d['distribucion_2024']['pct_municipios_gte_95']:.1f}%")
        da = d["delta_anual_2008_2024"]
        print(f"2) Delta año a año (n={da['n_deltas']}): media={da['media']:.3f}, std={da['std']:.3f}, "
              f"% dentro de +/-{UMBRAL_DELTA_ESTABLE}pp = {da[f'pct_deltas_dentro_de_{UMBRAL_DELTA_ESTABLE:g}pp']:.1f}%")
        ct = d["concentracion_techo_todo_el_panel"]
        print(f"3) Concentracion en el techo (todo el panel, n={ct['n_total_municipio_anio']}): "
              f">=95%: {ct['n_gte_95']} ({ct['frac_gte_95']*100:.1f}%)   "
              f">=99%: {ct['n_gte_99']} ({ct['frac_gte_99']*100:.1f}%)")
        tt = d["train_vs_test"]
        print(f"4) Train (2008-2021) vs Test (2024): "
              f"media {tt['train_2008_2021']['media']:.2f} -> {tt['test_2024']['media']:.2f}, "
              f"%>=95 {tt['train_2008_2021']['pct_gte_95']:.1f}% -> {tt['test_2024']['pct_gte_95']:.1f}%, "
              f"%>=99 {tt['train_2008_2021']['pct_gte_99']:.1f}% -> {tt['test_2024']['pct_gte_99']:.1f}%")

    da_ref = reporte["densidad_poblacional_referencia"]["delta_anual_2008_2024"]
    print(f"\n--- densidad_poblacional (referencia, supero el baseline) ---")
    print(f"2) Delta año a año (n={da_ref['n_deltas']}): media={da_ref['media']:.3f}, std={da_ref['std']:.3f}, "
          f"% dentro de +/-{UMBRAL_DELTA_ESTABLE}pp = {da_ref[f'pct_deltas_dentro_de_{UMBRAL_DELTA_ESTABLE:g}pp']:.1f}%")

    return reporte


if __name__ == "__main__":
    main()
