"""
Extraccion del panel de entrenamiento (Postgres -> CSV) para el primer
modelo LightGBM de cobertura de acueducto/alcantarillado/densidad
poblacional, a nivel municipio, 2008-2024.

>>> ESQUEMA CONFIRMADO -- de donde sale, y por que no se asumio de mas <<<
No hay acceso directo a la base real desde donde se escribio este script
(sin SSH/VPN a la VPS de produccion) -- el esquema de MapCalorDB se tomo
de techo360-backend/ingestion/db.py, el modulo que EN LA PRACTICA escribe
esa tabla (confirmado contra produccion en varios commits), no de una
suposicion:

    MapCalorDB(municipios, nameVar, valueVar, yearM, originVar)
    UNIQUE(municipios, nameVar, yearM)

IMPORTANTE, encontrado al revisar ese modulo antes de escribir nada aqui:
MapCalorDB NO tiene ninguna columna de codigo DIVIPOLA. La columna
"municipios" guarda el NOMBRE del municipio (viene de
IndicadorRecord.entidad_territorial en el pipeline de ingesta) -- el
campo IndicadorRecord.codigo_divipola existe en el objeto Python pero
NUNCA se guarda en MapCalorDB, se descarta en el guardado. Por eso este
script cruza por NOMBRE contra zonas_municipio/zonas_departamento (las
tablas que si tienen codigo_municipio/codigo_departamento, construidas
por ingestion/zonas_igac_adapter.py, misma base postgres-principal) para
obtener las columnas categoricas que pide el panel.

Cruzar por nombre es fragil (mayusculas, acentos, puntuacion --
"Bogota, D.C." en zonas_municipio vs. lo que TerriData haya puesto en
MapCalorDB.municipios pueden no ser identicos) -- exactamente el tipo de
suposicion que ya causo un bug real en este proyecto (entidad_territorial
vacio colapsando ~1100 municipios en 1 fila, en el adaptador de catastro
IGAC). Para no repetir ese patron: el cruce se hace sobre un nombre
NORMALIZADO (minusculas, sin tildes, sin puntuacion), y el script
REPORTA explicitamente cualquier nombre de MapCalorDB que no encuentre
match en zonas_municipio -- nunca los descarta en silencio, los deja en
el panel con codigo_municipio/codigo_departamento en blanco y avisa por
consola cuantos/cuales quedaron asi, para que se pueda revisar a mano.

Uso:
    DATABASE_URL=postgresql://... python -m scripts.extraer_panel
"""
import logging
import os
import re
import unicodedata

import pandas as pd
from sqlalchemy import bindparam, create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

VARIABLES_PANEL = ["densidad_poblacional", "cobertura_acueducto", "cobertura_alcantarillado"]
VARIABLES_CON_LAG = ["cobertura_acueducto", "cobertura_alcantarillado"]

ANIO_MIN = 2008
ANIO_MAX = 2024
ANIOS_ESPERADOS = ANIO_MAX - ANIO_MIN + 1  # 17

# Particion por AÑO (no aleatoria) -- para no filtrar informacion futura
# hacia el pasado. Ajustar aqui si al inspeccionar los datos reales
# alguna particion queda con muy pocos municipios.
ANIO_TRAIN_FIN = 2021
ANIO_VAL_INICIO = 2022
ANIO_VAL_FIN = 2023
ANIO_TEST_INICIO = 2024

RUTA_SALIDA = os.path.join("data", "processed", "panel_entrenamiento.csv")


def normalizar_nombre(nombre) -> str:
    """Minusculas, sin tildes/diacriticos, sin puntuacion, espacios
    colapsados -- clave de cruce entre MapCalorDB.municipios (nombre
    tal cual lo puso TerriData) y zonas_municipio.nombre (nombre tal
    cual lo puso el IGAC), que NO se puede asumir que vengan escritos
    identicos."""
    if nombre is None or (isinstance(nombre, float) and pd.isna(nombre)):
        return ""
    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def leer_indicadores_largo(engine) -> pd.DataFrame:
    """Formato largo: una fila por (municipio, variable, año)."""
    query = text(
        """
        SELECT municipios AS municipio_nombre, nameVar AS variable, yearM AS anio, valueVar AS valor
        FROM MapCalorDB
        WHERE nameVar IN :variables AND yearM BETWEEN :anio_min AND :anio_max
        """
    ).bindparams(bindparam("variables", expanding=True))
    df = pd.read_sql(query, engine, params={"variables": VARIABLES_PANEL, "anio_min": ANIO_MIN, "anio_max": ANIO_MAX})
    logger.info(f"Leidos de MapCalorDB: {len(df)} registros ({df['variable'].nunique()} variables)")
    return df


def leer_zonas_municipio(engine) -> pd.DataFrame:
    query = text("SELECT codigo_municipio, codigo_departamento, nombre AS municipio_nombre FROM zonas_municipio")
    df = pd.read_sql(query, engine)
    logger.info(f"Leidos de zonas_municipio: {len(df)} municipios")
    return df


def verificar_cobertura_de_anios(df_largo: pd.DataFrame) -> None:
    """Chequeo pedido explicitamente: si el backfill historico 2008-2024
    no corrio todavia (o corrio parcial), avisar ANTES de seguir --
    aqui no se puede consultar la base con antelacion (sin acceso
    directo), asi que la verificacion se hace en tiempo de ejecucion del
    propio script, contra lo que de verdad vino en la consulta."""
    for variable in VARIABLES_PANEL:
        anios_presentes = sorted(df_largo.loc[df_largo["variable"] == variable, "anio"].unique())
        faltantes = sorted(set(range(ANIO_MIN, ANIO_MAX + 1)) - set(anios_presentes))
        if faltantes:
            logger.warning(
                f"'{variable}': faltan {len(faltantes)}/{ANIOS_ESPERADOS} años "
                f"({faltantes}) -- el backfill historico puede no haber corrido completo. "
                f"Revisar antes de confiar en el panel/baseline resultante."
            )
        else:
            logger.info(f"'{variable}': los {ANIOS_ESPERADOS} años (2008-2024) estan completos.")


def construir_panel(df_largo: pd.DataFrame, df_zonas: pd.DataFrame) -> pd.DataFrame:
    # --- Pivot largo -> ancho (una fila por municipio-año, una columna por variable) ---
    panel = df_largo.pivot_table(
        index=["municipio_nombre", "anio"], columns="variable", values="valor", aggfunc="first",
    ).reset_index()
    panel.columns.name = None

    # --- Cruce con zonas_municipio, por nombre normalizado -- NUNCA se descartan filas sin match ---
    panel["_clave_join"] = panel["municipio_nombre"].map(normalizar_nombre)
    df_zonas = df_zonas.copy()
    df_zonas["_clave_join"] = df_zonas["municipio_nombre"].map(normalizar_nombre)
    df_zonas_dedup = df_zonas.drop_duplicates(subset="_clave_join", keep="first")

    panel = panel.merge(
        df_zonas_dedup[["_clave_join", "codigo_municipio", "codigo_departamento"]],
        on="_clave_join", how="left",
    )

    sin_match = panel.loc[panel["codigo_municipio"].isna(), "municipio_nombre"].drop_duplicates()
    if len(sin_match) > 0:
        logger.warning(
            f"{len(sin_match)} nombre(s) de municipio en MapCalorDB SIN match en zonas_municipio "
            f"(quedan con codigo_municipio/codigo_departamento en blanco, NO se descartan del panel): "
            f"{sorted(sin_match.tolist())}"
        )
    else:
        logger.info("Todos los nombres de municipio de MapCalorDB encontraron match en zonas_municipio.")

    panel = panel.drop(columns=["_clave_join"])
    panel["codigo_municipio"] = panel["codigo_municipio"].astype("category")
    panel["codigo_departamento"] = panel["codigo_departamento"].astype("category")

    # --- Features de rezago, por municipio, ordenado por año ---
    panel = panel.sort_values(["municipio_nombre", "anio"]).reset_index(drop=True)
    for variable in VARIABLES_CON_LAG:
        if variable not in panel.columns:
            continue
        panel[f"{variable}_lag_1"] = panel.groupby("municipio_nombre")[variable].shift(1)
        panel[f"{variable}_lag_2"] = panel.groupby("municipio_nombre")[variable].shift(2)

    # --- Particion train/val/test por año (NO aleatoria) ---
    def _particion(anio: int) -> str:
        if anio <= ANIO_TRAIN_FIN:
            return "train"
        if ANIO_VAL_INICIO <= anio <= ANIO_VAL_FIN:
            return "val"
        return "test"

    panel["particion"] = panel["anio"].map(_particion)

    for particion, grupo in panel.groupby("particion"):
        logger.info(
            f"Particion '{particion}': {len(grupo)} filas, "
            f"{grupo['municipio_nombre'].nunique()} municipios distintos, "
            f"años {sorted(grupo['anio'].unique())}"
        )

    return panel


def calcular_baseline_ingenuo(panel: pd.DataFrame, variable: str = "cobertura_acueducto") -> dict:
    """Baseline: prediccion = lag_1 (valor del año anterior del mismo
    municipio). RMSE/MAE sobre validacion y test -- el piso minimo que
    el LightGBM real tiene que superar."""
    columna_lag = f"{variable}_lag_1"
    if columna_lag not in panel.columns or variable not in panel.columns:
        raise ValueError(f"'{variable}' no tiene columna de lag_1 en el panel -- revisar VARIABLES_CON_LAG.")

    resultados = {}
    for particion in ("val", "test"):
        sub = panel.loc[(panel["particion"] == particion) & panel[variable].notna() & panel[columna_lag].notna()]
        if len(sub) == 0:
            logger.warning(f"Baseline ingenuo ({variable}, {particion}): 0 filas evaluables (sin lag_1 disponible).")
            resultados[particion] = {"n": 0, "rmse": None, "mae": None}
            continue

        error = sub[variable] - sub[columna_lag]
        rmse = (error**2).mean() ** 0.5
        mae = error.abs().mean()
        resultados[particion] = {"n": len(sub), "rmse": rmse, "mae": mae}
        logger.info(
            f"Baseline ingenuo ({variable}, {particion}): n={len(sub)}, RMSE={rmse:.4f}, MAE={mae:.4f}"
        )

    return resultados


def main():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit(
            "Falta la variable de entorno DATABASE_URL -- correr esto dentro "
            "del contenedor (o con la URL de postgres-principal a mano), ej.:\n"
            "  DATABASE_URL=postgresql://... python -m scripts.extraer_panel"
        )

    engine = create_engine(database_url)

    df_largo = leer_indicadores_largo(engine)
    if df_largo.empty:
        raise SystemExit(
            "MapCalorDB no devolvio NINGUN registro para las 3 variables/rango de años pedido -- "
            "el backfill historico probablemente no ha corrido todavia. Avisar antes de continuar."
        )
    verificar_cobertura_de_anios(df_largo)

    df_zonas = leer_zonas_municipio(engine)
    if df_zonas.empty:
        logger.warning(
            "zonas_municipio esta vacia -- el panel va a quedar SIN codigo_municipio/codigo_departamento "
            "en ninguna fila (correr ingestion/zonas_igac_adapter.py en techo360-backend primero)."
        )

    panel = construir_panel(df_largo, df_zonas)

    os.makedirs(os.path.dirname(RUTA_SALIDA), exist_ok=True)
    panel.to_csv(RUTA_SALIDA, index=False)
    logger.info(f"Panel guardado en {RUTA_SALIDA} ({len(panel)} filas, {len(panel.columns)} columnas)")

    print()
    calcular_baseline_ingenuo(panel, variable="cobertura_acueducto")


if __name__ == "__main__":
    main()
