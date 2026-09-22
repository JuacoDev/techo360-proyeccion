from pydantic import BaseModel, Field


class SolicitudProyeccion(BaseModel):
    codigo_municipio: str = Field(min_length=5, max_length=5)
    indicador: str
    anio_fin: int = Field(default=2029, ge=2025)