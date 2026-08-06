from fastapi import FastAPI

app = FastAPI(title="Techo 360 - Servicio IA Proyeccion")


@app.get("/health")
def health():
    """Prueba de humo: confirma que el contenedor levanto correctamente."""
    return {"status": "ok", "service": "proyeccion"}


# prueba CI/CD


# TODO: aqui van los endpoints reales:
#   POST /proyeccion/calcular (encola la tarea en RabbitMQ)
#   GET /proyeccion/resultado/{job_id}
#   modelo LightGBM (entrenamiento + inferencia)
