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


def obtener_contexto_municipio(codigo_municipio: str, indicador: str) -> dict:
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
        registros = pd.read_sql(
            text(
                """
                SELECT municipios AS municipio_nombre, yearM AS anio,
                       valueVar AS valor, codigoDivipola AS codigo_divipola
                FROM MapCalorDB
                WHERE nameVar = :indicador
                  AND yearM <= :anio_actual
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

    ultimos = coincidencias.tail(2)
    return {
        "codigo_municipio": codigo_municipio,
        "codigo_departamento": str(zona["codigo_departamento"]),
        "ultimo_anio": int(ultimos.iloc[-1]["anio"]),
        "lag_1": float(ultimos.iloc[-1]["valor"]),
        "lag_2": float(ultimos.iloc[-2]["valor"]),
    }