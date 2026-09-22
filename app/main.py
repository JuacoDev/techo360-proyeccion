from fastapi import FastAPI, HTTPException

from app.schemas import SolicitudProyeccion
from app.services.proyeccion import calcular_serie_proyectada

app = FastAPI(title="Techo 360 - Servicio IA Proyeccion")


@app.get("/health")
def health():
    """Prueba de humo: confirma que el contenedor levanto correctamente."""
    return {"status": "ok", "service": "proyeccion"}


@app.post("/proyeccion/calcular")
def calcular_proyeccion(solicitud: SolicitudProyeccion):
    try:
        return calcular_serie_proyectada(solicitud)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


