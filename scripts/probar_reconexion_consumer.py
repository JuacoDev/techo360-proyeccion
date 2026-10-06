"""Prueba de reconexion del consumer REAL de app/rabbitmq_consumer.py (T-126).

Verifica que, tras recrear o reiniciar RabbitMQ o cortarle la red, la cola
'proyeccion_requests' reaparece con consumer_count=1 y un mensaje RPC publicado
se procesa. Levanta un RabbitMQ DESECHABLE en Docker (contenedor
'rmq-reconexion-proy', puerto local 5675); nunca toca el RabbitMQ del VPS.
El calculo y la BD se simulan: app.services.proyeccion se sustituye por un stub
ANTES de importar el consumer (asi no hacen falta pandas, numpy, sqlalchemy ni
psycopg2); la validacion del payload (SolicitudProyeccion, pydantic) y el
contrato RPC son los reales.

Contexto: aio-pika guarda las colas del canal en un WeakSet; si nada retiene
el objeto queue (ver _QUEUES_VIVAS), tras reconectar la cola/consumer no se
restauran (consumer_count=0, sin error ni log). Sin el fix este script falla.

Requiere: Docker en marcha y `pip install aio-pika==9.4.3 pydantic`.
Uso:
    python scripts/probar_reconexion_consumer.py                 # los 3 escenarios x3
    python scripts/probar_reconexion_consumer.py recreate 1      # un escenario, 1 ciclo
Escenarios: recreate (broker nuevo y vacio, como el VPS), restart
(docker restart, la cola durable persiste), netcut (docker network
disconnect/connect, con heartbeat corto para detectar el corte).
"""
import asyncio
import gc
import json
import os
import subprocess
import sys
import time
import uuid

# Permite `python scripts/...` desde la raiz del repo
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types

import aio_pika
from aio_pika import DeliveryMode, Message

# Stub del calculo/BD: debe quedar en sys.modules antes de importar el consumer
_stub = types.ModuleType("app.services.proyeccion")
_stub.calcular_serie_proyectada = lambda _solicitud: {"serie": []}
_stub.calcular_zonas = lambda _indicador, _anio: []
sys.modules["app.services.proyeccion"] = _stub

import app.rabbitmq_consumer as consumer  # noqa: E402 -- tras el stub, a proposito

CONTENEDOR = "rmq-reconexion-proy"
PUERTO = 5675
RED = "rmq-reconexion-proy-net"
COLA = consumer.REQUEST_QUEUE
# heartbeat corto: sin el, un corte de red tarda ~minutos en detectarse
URL = f"amqp://guest:guest@localhost:{PUERTO}/?heartbeat=5"
ESPERA_MAX_S = 60


def docker(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, check=check)


def levantar_broker() -> None:
    docker("rm", "-f", CONTENEDOR)
    docker("network", "create", RED)
    docker("run", "-d", "--name", CONTENEDOR, "--hostname", CONTENEDOR, "--network", RED,
           "-e", "RABBITMQ_ERLANG_COOKIE=reprocookie", "-p", f"{PUERTO}:5672",
           "rabbitmq:3.13-management-alpine", check=True)
    esperar_broker()


def esperar_broker() -> None:
    time.sleep(10)  # un `exec` como root en el arranque deja la cookie de Erlang ilegible
    t0 = time.time()
    while time.time() - t0 < 120:
        r = docker("exec", "-u", "rabbitmq", CONTENEDOR, "rabbitmq-diagnostics", "-q", "check_port_connectivity")
        if r.returncode == 0:
            return
        time.sleep(2)
    raise RuntimeError("el broker no quedo listo")


def consumers_de_la_cola() -> int | None:
    r = docker("exec", "-u", "rabbitmq", CONTENEDOR, "rabbitmqctl", "list_queues", "--no-table-headers", "name", "consumers")
    for linea in r.stdout.splitlines():
        partes = linea.split()
        if len(partes) == 2 and partes[0] == COLA:
            return int(partes[1])
    return None  # la cola no existe


def accion(escenario: str) -> None:
    if escenario == "recreate":
        docker("rm", "-f", CONTENEDOR)
        docker("run", "-d", "--name", CONTENEDOR, "--hostname", CONTENEDOR, "--network", RED,
               "-e", "RABBITMQ_ERLANG_COOKIE=reprocookie", "-p", f"{PUERTO}:5672",
               "rabbitmq:3.13-management-alpine", check=True)
    elif escenario == "restart":
        docker("restart", CONTENEDOR, check=True)
    elif escenario == "netcut":
        docker("network", "disconnect", RED, CONTENEDOR, check=True)
        time.sleep(20)  # > 2 heartbeats: el cliente detecta el corte
        docker("network", "connect", RED, CONTENEDOR, check=True)
    else:
        raise SystemExit(f"escenario desconocido: {escenario}")
    esperar_broker()


async def mensaje_se_procesa() -> bool:
    """Publica una peticion RPC (mismo contrato que el backend) y espera la respuesta."""
    connection = await aio_pika.connect_robust(URL)
    try:
        channel = await connection.channel()
        cola_respuesta = await channel.declare_queue(exclusive=True)
        futuro: asyncio.Future = asyncio.get_running_loop().create_future()
        correlation_id = str(uuid.uuid4())

        async def on_reply(message) -> None:
            async with message.process():
                if message.correlation_id == correlation_id and not futuro.done():
                    futuro.set_result(json.loads(message.body))

        await cola_respuesta.consume(on_reply)
        await channel.default_exchange.publish(
            Message(
                body=json.dumps({"codigo_municipio": "68001",
                                 "indicador": "densidad_poblacional", "anio_fin": 2027}).encode(),
                content_type="application/json", correlation_id=correlation_id,
                reply_to=cola_respuesta.name, delivery_mode=DeliveryMode.PERSISTENT,
            ),
            routing_key=COLA,
        )
        respuesta = await asyncio.wait_for(futuro, timeout=15)
        return respuesta.get("status") == "ok"
    except Exception:  # noqa: BLE001 -- aqui solo importa si hubo respuesta o no
        return False
    finally:
        await connection.close()


async def main(escenarios: list[str], ciclos: int) -> int:
    await asyncio.to_thread(levantar_broker)
    conexion = await consumer.iniciar_consumidor(URL)  # codigo real, como en el lifespan
    gc.collect()  # fuerza el escenario que dispara el bug (colas sin retener se pierden aqui)

    fallos = 0
    resultados = [("inicial", await asyncio.to_thread(consumers_de_la_cola), await mensaje_se_procesa(), 0)]
    for escenario in escenarios:
        for i in range(1, ciclos + 1):
            await asyncio.to_thread(accion, escenario)
            gc.collect()
            t0 = time.time()
            n = await asyncio.to_thread(consumers_de_la_cola)
            while n != 1 and time.time() - t0 < ESPERA_MAX_S:
                await asyncio.sleep(2)
                n = await asyncio.to_thread(consumers_de_la_cola)
            procesado = (n == 1) and await mensaje_se_procesa()
            resultados.append((f"{escenario}{i}", n, procesado, round(time.time() - t0)))

    print(f"{'ciclo':<10} {'consumer_count':<15} {'mensaje procesado':<18} segundos")
    for nombre, n, ok, seg in resultados:
        print(f"{nombre:<10} {str(n):<15} {str(ok):<18} {seg}")
        fallos += 0 if (n == 1 and ok) else 1
    await conexion.close()
    docker("rm", "-f", CONTENEDOR)
    docker("network", "rm", RED)
    print("RESULTADO:", "OK" if fallos == 0 else f"{fallos} ciclo(s) fallaron")
    return 1 if fallos else 0


if __name__ == "__main__":
    escenarios = [sys.argv[1]] if len(sys.argv) > 1 else ["recreate", "restart", "netcut"]
    ciclos = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    sys.exit(asyncio.run(main(escenarios, ciclos)))
