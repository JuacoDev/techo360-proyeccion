import os
import re
import unicodedata

import pandas as pd
from sqlalchemy import create_engine, text


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
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("Falta la variable de entorno DATABASE_URL")

    engine = create_engine(database_url)
    try:
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
            params={"indicador": indicador, "anio_actual": pd.Timestamp.now().year},
        )
    finally:
        engine.dispose()

    if registros.empty:
        raise LookupError(
            f"No hay datos historicos para '{codigo_municipio}' y '{indicador}'"
        )

    coincidencias = registros[registros["codigo_divipola"].astype("string") == codigo_municipio]
    if coincidencias.empty:
        clave_zona = _normalizar_nombre(zona["municipio_nombre"])
        coincidencias = registros[
            registros["municipio_nombre"].map(_normalizar_nombre) == clave_zona
        ]

    coincidencias = coincidencias.dropna(subset=["valor"]).sort_values("anio")
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