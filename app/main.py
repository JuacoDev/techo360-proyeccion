import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.rabbitmq_consumer import iniciar_consumidor

RABBITMQ_URL = os.environ.get("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")


@asynccontextmanager
async def lifespan(app: FastAPI):
    connection = await iniciar_consumidor(RABBITMQ_URL)
    yield
    await connection.close()


app = FastAPI(title="Techo 360 - Servicio IA Proyeccion", lifespan=lifespan)


@app.get("/health")
def health():
    """Prueba de humo: confirma que el contenedor levanto correctamente."""
    return {"status": "ok", "service": "proyeccion"}
