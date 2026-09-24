"""
Extraccion del panel municipal (Postgres -> CSV) de cobertura de
acueducto/alcantarillado/densidad poblacional, 2008-2024.

Nacio como panel de ENTRENAMIENTO del primer modelo LightGBM. Ese modelo
ya no existe (ver scripts/rellenar_baseline_2025_2026.py), pero el panel
sigue siendo la base de evidencia del proyecto: es lo que usan
scripts/diagnostico_variabilidad.py y los backtestings con los que se
eligieron los metodos que SI estan en produccion -- incluido el baseline
ingenuo que calcula este mismo modulo al final.

>>> ESQUEMA CONFIRMADO -- de donde sale, y por que no se asumio de mas <<<
No hay acceso directo a la base real desde donde se escribio este script
(sin SSH/VPN a la VPS de produccion) -- el esquema de MapCalorDB se tomo
de techo360-backend/ingestion/db.py, el modulo que EN LA PRACTICA escribe
esa tabla (confirmado contra produccion en varios commits), no de una
suposicion:

    MapCalorDB(municipios, nameVar, valueVar, yearM, originVar, codigoDivipola)
    UNIQUE(municipios, nameVar, yearM)

>>> HISTORIAL: MapCalorDB SI tiene codigo DIVIPOLA ahora, pero no todo
>>> el historico lo tiene <<<
Hasta el commit que agrego migrar_columna_codigo_divipola() +
guardar_indicador() guardando codigoDivipola (techo360-backend), esta
tabla NO tenia ninguna columna de codigo DIVIPOLA -- la columna
"municipios" guardaba solo el NOMBRE (IndicadorRecord.entidad_territorial),
y record.codigo_divipola se descartaba en el guardado. Esto se detecto
diagnosticando en vivo (sin acceso a la base) los 54 nombres que
extraer_panel.py reportaba sin match: resultaron ser dos problemas
distintos --
  (a) ~29 eran agregados departamentales de TerriData (filas con
      codigo terminado en '000', ej. Antioquia=05000) que no son
      municipios en absoluto -- ya se filtran en la fuente, en
      terridata_adapter.py::fetch(), asi que no deberian volver a
      aparecer en corridas nuevas.
  (b) ~30 eran municipios REALES con nomenclatura distinta entre
      TerriData/DANE (nombres cortos, ej. "Cali", "Cartagena") e IGAC
      (nombres oficiales completos, ej. "Santiago De Cali",
      "Cartagena De Indias") -- confirmado ademas que el mismo nombre
      oficial IGAC puede repetirse en dos municipios reales distintos
      (ej. "El Carmen" existe en Choco Y en Santander, codigos
      distintos), asi que NINGUN cruce por nombre, por mas tabla de
      equivalencias que tenga, puede distinguir ese caso -- solo el
      codigo puede.

Por (b), el cruce por NOMBRE nunca deja de ser necesario del todo: las
filas del historico cargadas ANTES del cambio en techo360-backend se
quedan con codigoDivipola en NULL para siempre (no es seguro
reconstruirlo desde el nombre ya guardado -- exactamente el mismo tipo
de suposicion fragil que se esta tratando de dejar de hacer). Por eso el
cruce contra zonas_municipio/zonas_departamento (las tablas que tienen
codigo_municipio/codigo_departamento, construidas por
ingestion/zonas_igac_adapter.py, misma base postgres-principal) ahora es
de DOS niveles:

  1. PRINCIPAL: por codigo_divipola == codigo_municipio (exacto, no
     depende de que los nombres coincidan entre fuentes).
  2. FALLBACK: por nombre NORMALIZADO (minusculas, sin tildes, sin
     puntuacion) -- SOLO para las filas que no tenian codigo_divipola o
     cuyo codigo no aparecio en zonas_municipio.

El script REPORTA explicitamente cuantas filas cruzaron por cada
metodo, y cualquier nombre de MapCalorDB que no encuentre match por
NINGUNO de los dos -- nunca los descarta en silencio, los deja en el
panel con codigo_municipio/codigo_departamento en blanco y avisa por
consola cuantos/cuales quedaron asi, para que se pueda revisar a mano.

>>> AVISO: los codigos DIVIPOLA van SIEMPRE como string con zfill
>>> (municipio=5, departamento=2) -- bug real, encontrado 23-sep-2026 <<<
El panel se guarda en CSV y se vuelve a leer despues. Un CSV no lleva
tipos: pandas INFIERE. Como los codigos son solo digitos, `pd.read_csv`
los parseaba como int64, convirtiendo "05001" en 5001 y "05" en 5 (cero
inicial perdido). Esas eran entonces las CATEGORIAS con las que se
entrenaba LightGBM, y quedaban grabadas en el .txt del modelo (bloque
`pandas_categorical`).

En inferencia, app/repositories/proyeccion.py devuelve los codigos como
STRING ("05001", "19"), que es como los guarda la base. Al predecir,
LightGBM re-mapea la columna categorica contra las categorias del
entrenamiento: string vs int nunca coincide, asi que TODOS los
municipios caian en "categoria desconocida" (-1), en silencio.

Consecuencia medida contra el modelo real de densidad_poblacional:
6 municipios distintos devolvian la MISMA prediccion exacta
(66.474966 para todos). Con los codigos como int (los del
entrenamiento) devolvian 6 valores distintos y coherentes -- Medellin y
Bogota recuperaban densidades ~2593 y ~2264. Es decir:
codigo_municipio (622 splits en el modelo) y codigo_departamento (200
splits) estaban ANULADOS en produccion para todos los municipios.

Por eso aqui los codigos se normalizan con normalizar_codigo() antes del
astype("category"), y CUALQUIER lector de este CSV debe pasar dtype
explicito para esas dos columnas:

    pd.read_csv(RUTA_SALIDA, dtype={"codigo_municipio": "string",
                                    "codigo_departamento": "string"})

Nota (23-sep-2026): el modelo LightGBM que motivo este aviso ya no
existe -- se retiro del pipeline por rendir peor que los metodos simples
(ver scripts/rellenar_baseline_2025_2026.py y, para densidad,
app/services/proyeccion.py). El aviso se conserva porque el panel sigue
usandose para diagnostico y backtesting, y el mismo error de tipos
volveria a falsear cualquier analisis que cruce por codigo de municipio.

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


def normalizar_codigo(valor, ancho: int):
    """Codigo DIVIPOLA SIEMPRE como string con zfill (municipio=5,
    departamento=2) -- ver el aviso del modulo sobre el round-trip del CSV.
    Devuelve None si el valor es nulo (no se inventa un codigo)."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return None
    texto = str(valor).strip()
    if texto == "" or texto.lower() == "nan":
        return None
    # Excel/pandas pueden entregar el codigo como numero (ej. 5001.0 ->
    # "5001"), perdiendo el cero inicial -- el zfill lo restituye.
    if texto.endswith(".0"):
        texto = texto[:-2]
    return texto.zfill(ancho)


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
    """Formato largo: una fila por (municipio, variable, año). Incluye
    codigo_divipola (columna codigoDivipola, ver aviso del modulo sobre
    por que puede venir NULL en filas del historico viejo)."""
    query = text(
        """
        SELECT municipios AS municipio_nombre, nameVar AS variable, yearM AS anio,
               valueVar AS valor, codigoDivipola AS codigo_divipola
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
    # --- Codigo DIVIPOLA por (municipio_nombre, anio) -- se toma de
    # cualquier variable de esa fila que lo tenga (deberia ser el mismo
    # entre variables del mismo municipio/año; si ninguna lo tiene, ej.
    # historico cargado antes de que MapCalorDB guardara codigoDivipola,
    # queda NaN y esa fila cae al fallback por nombre mas abajo) ---
    codigos_por_fila = (
        df_largo.dropna(subset=["codigo_divipola"])
        .drop_duplicates(subset=["municipio_nombre", "anio"], keep="first")
        .set_index(["municipio_nombre", "anio"])["codigo_divipola"]
    )

    # --- Pivot largo -> ancho (una fila por municipio-año, una columna por variable) ---
    panel = df_largo.pivot_table(
        index=["municipio_nombre", "anio"], columns="variable", values="valor", aggfunc="first",
    ).reset_index()
    panel.columns.name = None
    # dtype="object" explicito -- si TODO el lote quedara sin codigo_divipola
    # (ej. una corrida restringida solo a historico viejo), pandas
    # infiere float64 para una columna 100% NaN, y el fallback de abajo
    # (que asigna strings de codigo) revienta al intentar escribir un
    # string en una columna float64. Con object desde el arranque, tanto
    # NaN como strings conviven sin problema sin importar la mezcla.
    panel["codigo_divipola"] = pd.array(
        panel.set_index(["municipio_nombre", "anio"]).index.map(codigos_por_fila), dtype="object",
    )

    # --- Cruce con zonas_municipio, en DOS niveles (ver aviso del modulo) ---
    # 1. PRINCIPAL: por codigo_divipola == codigo_municipio -- exacto, no
    #    depende de que los nombres de TerriData e IGAC coincidan.
    codigos_municipio_validos = set(df_zonas["codigo_municipio"])
    mapa_departamento_por_municipio = (
        df_zonas.drop_duplicates(subset="codigo_municipio", keep="first")
        .set_index("codigo_municipio")["codigo_departamento"]
    )
    coincide_por_codigo = panel["codigo_divipola"].isin(codigos_municipio_validos)
    panel["codigo_municipio"] = panel["codigo_divipola"].where(coincide_por_codigo).astype("object")
    cruzados_por_codigo = int(coincide_por_codigo.sum())

    # 2. FALLBACK: por nombre normalizado -- SOLO para lo que el cruce por
    #    codigo no resolvio (fragil, ver aviso del modulo -- por eso no es
    #    el metodo principal, solo cubre lo que el codigo no puede).
    pendiente = ~coincide_por_codigo
    cruzados_por_fallback = 0
    if pendiente.any():
        df_zonas_nombre = df_zonas.assign(_clave_join=df_zonas["municipio_nombre"].map(normalizar_nombre))
        mapa_municipio_por_nombre = (
            df_zonas_nombre.drop_duplicates(subset="_clave_join", keep="first")
            .set_index("_clave_join")["codigo_municipio"]
        )
        clave_join = panel["municipio_nombre"].map(normalizar_nombre)
        codigo_municipio_por_nombre = clave_join.map(mapa_municipio_por_nombre)

        cruzados_por_fallback = int((pendiente & codigo_municipio_por_nombre.notna()).sum())
        panel.loc[pendiente, "codigo_municipio"] = codigo_municipio_por_nombre[pendiente]

    panel["codigo_departamento"] = panel["codigo_municipio"].map(mapa_departamento_por_municipio)
    panel = panel.drop(columns=["codigo_divipola"])

    logger.info(
        f"Cruce contra zonas_municipio: {cruzados_por_codigo} fila(s) por codigo_divipola (metodo "
        f"principal), {cruzados_por_fallback} fila(s) adicionales por nombre normalizado (fallback, "
        f"historico sin codigo_divipola cargado)."
    )

    sin_match = panel.loc[panel["codigo_municipio"].isna(), "municipio_nombre"].drop_duplicates()
    if len(sin_match) > 0:
        logger.warning(
            f"{len(sin_match)} nombre(s) de municipio en MapCalorDB SIN match en zonas_municipio "
            f"(ni por codigo_divipola ni por nombre normalizado -- quedan con codigo_municipio/"
            f"codigo_departamento en blanco, NO se descartan del panel): {sorted(sin_match.tolist())}"
        )
    else:
        logger.info("Todos los municipios de MapCalorDB encontraron match en zonas_municipio (por codigo o por nombre).")

    # >>> AVISO: los codigos van SIEMPRE como string con zfill <<<
    # Ver el aviso del modulo. Esto se aplica justo antes del astype
    # ("category") para que las CATEGORIAS que ve LightGBM en
    # entrenamiento sean exactamente las mismas strings que manda el
    # servicio de inferencia ("05001", no 5001).
    panel["codigo_municipio"] = panel["codigo_municipio"].map(lambda v: normalizar_codigo(v, 5))
    panel["codigo_departamento"] = panel["codigo_departamento"].map(lambda v: normalizar_codigo(v, 2))

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
