from pydantic import BaseModel, Field


class SolicitudProyeccion(BaseModel):
    codigo_municipio: str = Field(min_length=5, max_length=5)
    indicador: str
    anio_fin: int = Field(default=2029, ge=2025)


class SolicitudProyeccionZonas(BaseModel):
    """T-104: choropleth completo -- un indicador, todos los municipios,
    un solo año (ver app/services/proyeccion.py::calcular_zonas). Distinta
    de SolicitudProyeccion (que sigue siendo la atomica, un municipio) --
    el consumer (app/rabbitmq_consumer.py) distingue cual usar por el
    campo "tipo" del payload, no por la forma del schema."""

    indicador: str
    anio: int = Field(ge=2025)