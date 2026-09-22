import os
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import pandas as pd

from app.repositories.proyeccion import obtener_contexto_municipio
from app.schemas import SolicitudProyeccion

INDICADOR_VIGENTE = "densidad_poblacional"
RUTA_MODELO_POR_DEFECTO = (
    Path(__file__).resolve().parents[2] / "models" / "lightgbm_densidad_poblacional.txt"
)


@lru_cache(maxsize=1)
def cargar_modelo() -> lgb.Booster:
    ruta_modelo = Path(os.environ.get("DENSIDAD_MODEL_PATH", RUTA_MODELO_POR_DEFECTO))
    if not ruta_modelo.is_file():
        raise FileNotFoundError(
            f"No se encontro el modelo LightGBM en '{ruta_modelo}'"
        )
    return lgb.Booster(model_file=str(ruta_modelo))


def _proyectar_serie(modelo: lgb.Booster, contexto: dict, anio_fin: int) -> list[dict]:
    anio_actual = contexto["ultimo_anio"]
    if anio_fin <= anio_actual:
        raise ValueError(
            f"anio_fin debe ser mayor que el ultimo año disponible ({anio_actual})"
        )

    lag_1 = contexto["lag_1"]
    lag_2 = contexto["lag_2"]
    serie = []

    for anio in range(anio_actual + 1, anio_fin + 1):
        entrada = pd.DataFrame(
            [{
                "densidad_poblacional_lag_1": lag_1,
                "densidad_poblacional_lag_2": lag_2,
                "codigo_departamento": contexto["codigo_departamento"],
                "codigo_municipio": contexto["codigo_municipio"],
                "anio": anio,
            }]
        )
        entrada["codigo_departamento"] = entrada["codigo_departamento"].astype("category")
        entrada["codigo_municipio"] = entrada["codigo_municipio"].astype("category")

        valor = float(modelo.predict(entrada)[0])
        serie.append({"anio": anio, "valor": valor})
        lag_2, lag_1 = lag_1, valor

    return serie


def calcular_serie_proyectada(solicitud: SolicitudProyeccion) -> dict:
    if solicitud.indicador != INDICADOR_VIGENTE:
        raise ValueError(
            f"El unico indicador disponible es '{INDICADOR_VIGENTE}'"
        )

    contexto = obtener_contexto_municipio(
        solicitud.codigo_municipio,
        solicitud.indicador,
    )
    serie = _proyectar_serie(cargar_modelo(), contexto, solicitud.anio_fin)

    return {
        "codigo_municipio": solicitud.codigo_municipio,
        "indicador": solicitud.indicador,
        "modelo": "lightgbm_densidad_poblacional",
        "anio_inicio": serie[0]["anio"],
        "anio_fin": solicitud.anio_fin,
        "serie_proyectada": serie,
    }