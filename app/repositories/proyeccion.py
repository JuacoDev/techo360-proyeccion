import os
import re
import unicodedata

import pandas as pd
from sqlalchemy import create_engine, text

# >>> AVISO -- engine reutilizado a nivel de modulo (25-sep-2026) <<<
# Antes cada llamada a obtener_contexto_municipio() hacia
# create_engine(database_url) + dispose() -- abrir la conexion sola media
# ~1.8s (medido en vivo contra el VPS real). Un engine de SQLAlchemy ya
# trae su propio pool de conexiones pensado para vivir durante todo el
# proceso, no para crearse y destruirse por request -- create_engine()
# NO abre una conexion en si (es perezoso), asi que crearlo una sola vez
# aqui no mantiene nada abierto de forma innecesaria mientras el proceso
# esta inactivo.
_engine = None


def _obtener_engine():
    global _engine
    if _engine is None:
        database_url = os.environ.get("DATABASE_URL")
        if not database_url:
            raise RuntimeError("Falta la variable de entorno DATABASE_URL")
        _engine = create_engine(database_url)
    return _engine


def _normalizar_nombre(nombre) -> str:
    if nombre is None or (isinstance(nombre, float) and pd.isna(nombre)):
        return ""

    texto = unicodedata.normalize("NFKD", str(nombre))
    texto = "".join(caracter for caracter in texto if not unicodedata.combining(caracter))
    texto = texto.lower()
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def obtener_contexto_municipio(codigo_municipio: str, indicador: str, n_puntos: int = 4) -> dict:
    """Contexto para proyectar: los ultimos `n_puntos` puntos REALES del
    municipio, con su año. El default (4) es el que eligio el backtesting
    documentado en app/services/proyeccion.py -- quien llama puede pedir
    otro, pero el valor de produccion se define alla (PUNTOS_REGRESION),
    no aqui."""
    engine = _obtener_engine()
    zonas = pd.read_sql(
        text(
            """
            SELECT codigo_municipio, codigo_departamento, nombre AS municipio_nombre
            FROM zonas_municipio
            WHERE codigo_municipio = :codigo_municipio
            """
        ),
        engine,
        params={"codigo_municipio": codigo_municipio},
    )
    if zonas.empty:
        raise LookupError(f"No existe el municipio '{codigo_municipio}'")

    zona = zonas.iloc[0]
    anio_actual = pd.Timestamp.now().year

    # >>> AVISO: solo puntos REALES, no rellenos ni proyecciones <<<
    # La extrapolacion lineal se ancla en los 2 ultimos puntos REALES;
    # si se colara una fila proyectada, se estaria extrapolando sobre
    # una extrapolacion. Origenes observados hoy para
    # densidad_poblacional en MapCalorDB:
    #   "TerriData"                       2000-2026  -> REAL
    #   "DANE-proyeccion-poblacion-2024"  2024       -> REAL (es el
    #       nombre de la fuente oficial del DANE, no un relleno de
    #       este pipeline -- ver ingestion/db.py en techo360-backend)
    #   "TerriData-proyeccion"            2027-2042  -> PROYECTADO
    # Por eso el filtro NO puede ser un simple '%proyeccion%': eso
    # descartaria el dato real del DANE (el mismo error que ya se
    # cometio una vez con el filtro de entrenamiento). La regla
    # exacta es: descartar los origenes que EMPIEZAN por
    # 'proyeccion-' (convencion de relleno del proyecto) o que
    # TERMINAN en '-proyeccion' (caso TerriData-proyeccion).
    #
    # El filtro por anio se mantiene, pero ya no es el que protege:
    # hoy excluye 2027-2042 solo porque estamos en 2026 -- en 2027
    # dejaria entrar TerriData-proyeccion como si fuera real.
    #
    # >>> AVISO -- filtro por codigo_divipola en SQL (25-sep-2026) <<<
    # Antes esta query traia LAS FILAS DE TODOS LOS MUNICIPIOS (29,732
    # para densidad_poblacional, medido en vivo) y filtraba a un solo
    # municipio DESPUES, en pandas -- ~3.1s de los ~4.6s totales de cada
    # llamada. Filtrar por codigoDivipola en el WHERE es el camino rapido
    # (99% de los casos, medido en vivo: el join por codigo funciona para
    # los 1105 municipios reales). Si no hay coincidencia por codigo (el
    # caso raro que el fallback de mas abajo cubria), se hace una segunda
    # consulta SIN ese filtro para reintentar por nombre -- pagando el
    # costo completo solo en ese caso excepcional, no en el camino normal.
    registros = pd.read_sql(
        text(
            """
            SELECT municipios AS municipio_nombre, yearM AS anio,
                   valueVar AS valor, codigoDivipola AS codigo_divipola,
                   originVar AS origen
            FROM MapCalorDB
            WHERE nameVar = :indicador
              AND yearM <= :anio_actual
              AND originVar NOT LIKE 'proyeccion-%'
              AND originVar NOT LIKE '%-proyeccion'
              AND codigoDivipola = :codigo_municipio
            """
        ),
        engine,
        params={"indicador": indicador, "anio_actual": anio_actual, "codigo_municipio": codigo_municipio},
    )

    if registros.empty:
        # Fallback raro: el codigo no matcheo (dato historico con
        # codigoDivipola nulo/distinto) -- se reintenta por nombre
        # normalizado, pagando el costo de traer todo el indicador solo
        # en este caso excepcional.
        registros = pd.read_sql(
            text(
                """
                SELECT municipios AS municipio_nombre, yearM AS anio,
                       valueVar AS valor, codigoDivipola AS codigo_divipola,
                       originVar AS origen
                FROM MapCalorDB
                WHERE nameVar = :indicador
                  AND yearM <= :anio_actual
                  AND originVar NOT LIKE 'proyeccion-%'
                  AND originVar NOT LIKE '%-proyeccion'
                """
            ),
            engine,
            params={"indicador": indicador, "anio_actual": anio_actual},
        )
        if registros.empty:
            raise LookupError(
                f"No hay datos historicos para '{codigo_municipio}' y '{indicador}'"
            )
        clave_zona = _normalizar_nombre(zona["municipio_nombre"])
        registros = registros[registros["municipio_nombre"].map(_normalizar_nombre) == clave_zona]
        if registros.empty:
            raise LookupError(
                f"No hay datos historicos para '{codigo_municipio}' y '{indicador}'"
            )

    coincidencias = registros.dropna(subset=["valor"]).sort_values("anio")
    if len(coincidencias) < 2:
        raise LookupError(
            f"Se requieren al menos dos valores historicos para '{codigo_municipio}'"
        )

    # Se devuelven los ultimos `n_puntos` puntos reales CON SU AÑO, no solo
    # sus valores: la regresion se ajusta contra el año real de cada punto,
    # y la serie de un municipio puede tener huecos (los ultimos puntos no
    # tienen por que ser años consecutivos). Si hay menos de `n_puntos`
    # disponibles se devuelven todos los que existan -- con 2 la regresion
    # degenera exactamente en la recta por esos 2 puntos, que era el
    # metodo anterior.
    ultimos = coincidencias.tail(n_puntos)
    puntos = [
        {"anio": int(fila["anio"]), "valor": float(fila["valor"]), "origen": str(fila["origen"])}
        for _, fila in ultimos.iterrows()
    ]

    return {
        "codigo_municipio": str(codigo_municipio).strip().zfill(5),
        "codigo_departamento": str(zona["codigo_departamento"]).strip().zfill(2),
        "ultimo_anio": puntos[-1]["anio"],
        "puntos": puntos,
    }


def obtener_datos_municipio_para_anios(
    codigo_municipio: str, indicador: str, anio_inicio: int, anio_fin: int
) -> dict[int, float]:
    """Devuelve los valores disponibles por municipio y año, sin límite por año actual."""
    engine = _obtener_engine()
    registros = pd.read_sql(
        text(
            """
            SELECT yearM AS anio, valueVar AS valor
            FROM MapCalorDB
            WHERE nameVar = :indicador
              AND yearM BETWEEN :anio_inicio AND :anio_fin
              AND originVar NOT LIKE 'proyeccion-%'
              AND originVar NOT LIKE '%-proyeccion'
              AND codigoDivipola = :codigo_municipio
            ORDER BY yearM
            """
        ),
        engine,
        params={
            "indicador": indicador,
            "anio_inicio": anio_inicio,
            "anio_fin": anio_fin,
            "codigo_municipio": codigo_municipio,
        },
    )
    registros = registros.dropna(subset=["valor"])
    return {int(fila.anio): float(fila.valor) for fila in registros.itertuples(index=False)}


def obtener_datos_todos_municipios_para_anio(indicador: str, anio: int) -> dict[str, float]:
    """Devuelve en una consulta los valores disponibles de todos los municipios para un año."""
    engine = _obtener_engine()
    registros = pd.read_sql(
        text(
            """
            SELECT valueVar AS valor, codigoDivipola AS codigo_divipola
            FROM MapCalorDB
            WHERE nameVar = :indicador
              AND yearM = :anio
              AND originVar NOT LIKE 'proyeccion-%'
              AND originVar NOT LIKE '%-proyeccion'
              AND codigoDivipola IS NOT NULL
            """
        ),
        engine,
        params={"indicador": indicador, "anio": anio},
    )
    registros = registros.dropna(subset=["valor"])
    return {
        str(fila.codigo_divipola).strip().zfill(5): float(fila.valor)
        for fila in registros.itertuples(index=False)
    }


def obtener_puntos_todos_los_municipios(indicador: str, n_puntos: int = 4) -> dict[str, list[dict]]:
    """T-104: version "todos los municipios" de obtener_contexto_municipio()
    -- UNA sola consulta SQL (sin filtrar por codigo, a proposito: aqui SI
    se quieren todas las filas) en vez de una consulta por municipio.

    No hace el join contra zonas_municipio (no hace falta codigo_departamento
    para un choropleth, ver schemas/proyecciones.py en techo360-backend:
    ProyeccionResponse no lo expone) ni el fallback de matching por nombre
    de obtener_contexto_municipio() (ese fallback cubre un caso raro de UN
    municipio especifico con codigo_divipola nulo/inconsistente; para el
    batch completo, un municipio individual sin codigo simplemente se
    excluye del choropleth en vez de intentar reconciliarlo -- pagar el
    costo de un segundo fetch completo por cada municipio raro anularia la
    ganancia entera del batch).

    Devuelve {codigo_divipola: [{"anio", "valor"}, ...ultimos n_puntos]},
    solo para municipios con 2+ puntos reales distintos (los que no
    alcanzan quedan fuera, mismo criterio que LookupError en la version
    atomica). NOTA: sigue con el tope `yearM <= anio_actual` (ver AVISO en
    obtener_contexto_municipio) -- ya no es el camino que decide si un año
    es "real" para calcular_zonas()/calcular_serie_proyectada() (eso lo
    resuelven obtener_datos_todos_municipios_para_anio()/
    obtener_datos_municipio_para_anios() de arriba, sin ese tope), esta
    funcion solo sigue dando la base de puntos para la regresion cuando de
    verdad no hay dato real."""
    engine = _obtener_engine()
    anio_actual = pd.Timestamp.now().year
    registros = pd.read_sql(
        text(
            """
            SELECT yearM AS anio, valueVar AS valor, codigoDivipola AS codigo_divipola
            FROM MapCalorDB
            WHERE nameVar = :indicador
              AND yearM <= :anio_actual
              AND originVar NOT LIKE 'proyeccion-%'
              AND originVar NOT LIKE '%-proyeccion'
              AND codigoDivipola IS NOT NULL
            """
        ),
        engine,
        params={"indicador": indicador, "anio_actual": anio_actual},
    )
    registros = registros.dropna(subset=["valor"]).sort_values(["codigo_divipola", "anio"])

    resultado: dict[str, list[dict]] = {}
    for codigo, grupo in registros.groupby("codigo_divipola"):
        ultimos = grupo.tail(n_puntos)
        if len(ultimos["anio"].unique()) < 2:
            continue
        resultado[str(codigo).strip().zfill(5)] = [
            {"anio": int(fila["anio"]), "valor": float(fila["valor"])} for _, fila in ultimos.iterrows()
        ]
    return resultado