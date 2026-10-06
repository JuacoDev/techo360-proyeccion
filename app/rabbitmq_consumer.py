import json
import logging

import aio_pika
from aio_pika import DeliveryMode, Message
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage, AbstractRobustConnection

from app.schemas import SolicitudProyeccion, SolicitudProyeccionZonas
from app.services.proyeccion import calcular_serie_proyectada, calcular_zonas

logger = logging.getLogger(__name__)

REQUEST_QUEUE = "proyeccion_requests"

# Mapeo de las excepciones que ya lanza app/services/proyeccion.py (mismas
# que su version HTTP original mapeaba a status codes) a un "tipo" de
# error que el backend, del otro lado del RPC, vuelve a traducir a HTTP.
_TIPO_POR_EXCEPCION = (
    (ValueError, "rango_invalido"),
    (LookupError, "sin_datos"),
    (FileNotFoundError, "modelo_no_disponible"),
    (RuntimeError, "error_interno"),
)


def _calcular(payload: dict) -> dict:
    """T-104: el payload distingue atomico (un municipio, produccion
    actual) de batch/zonas (todos los municipios, T-104) por el campo
    "tipo" -- default "atomico" para no romper a quien publique sin ese
    campo (el backend de hoy no lo manda, y no deberia hacer falta que lo
    mande para seguir funcionando igual que antes)."""
    tipo = payload.get("tipo", "atomico")

    if tipo == "zonas":
        solicitud = SolicitudProyeccionZonas(indicador=payload["indicador"], anio=payload["anio"])
        zonas = calcular_zonas(solicitud.indicador, solicitud.anio)
        return {
            "indicador": solicitud.indicador,
            "anio": solicitud.anio,
            "modelo": "regresion-lineal-4puntos",
            "zonas": zonas,
        }

    solicitud = SolicitudProyeccion(
        codigo_municipio=payload["codigo_municipio"],
        indicador=payload["indicador"],
        anio_fin=payload["anio_fin"],
    )
    return calcular_serie_proyectada(solicitud)


async def _procesar_mensaje(message: AbstractIncomingMessage, channel: AbstractChannel) -> None:
    async with message.process():
        try:
            payload = json.loads(message.body)
            resultado = _calcular(payload)
            respuesta = {"status": "ok", **resultado}
        except Exception as error:  # noqa: BLE001 -- limite del consumer, no debe tumbar el proceso
            tipo = "error_interno"
            for tipo_excepcion, nombre_tipo in _TIPO_POR_EXCEPCION:
                if isinstance(error, tipo_excepcion):
                    tipo = nombre_tipo
                    break
            else:
                logger.exception("Error inesperado calculando la proyeccion")
            respuesta = {"status": "error", "tipo": tipo, "detail": str(error)}

        if not message.reply_to:
            logger.warning("Mensaje sin reply_to, no se puede responder: %s", message.correlation_id)
            return

        await channel.default_exchange.publish(
            Message(
                body=json.dumps(respuesta).encode(),
                content_type="application/json",
                correlation_id=message.correlation_id,
                delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=message.reply_to,
        )


async def iniciar_consumidor(rabbitmq_url: str) -> AbstractRobustConnection:
    connection = await aio_pika.connect_robust(rabbitmq_url)
    channel = await connection.channel()
    await channel.set_qos(prefetch_count=4)

    # durable=True y con los mismos parametros que la declaracion del
    # lado del backend (services/proyeccion_rpc_client.py) -- si este
    # consumer arranca despues de que el backend ya publico, encuentra
    # los mensajes esperando en la cola en vez de perderlos.
    queue = await channel.declare_queue(REQUEST_QUEUE, durable=True)

    async def on_message(message: AbstractIncomingMessage) -> None:
        await _procesar_mensaje(message, channel)

    await queue.consume(on_message)
    logger.info("Consumidor de '%s' arrancado.", REQUEST_QUEUE)
    return connection
