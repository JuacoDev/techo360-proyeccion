# techo360-proyeccion

Servicio IA - Proyeccion sobre Techo 360.

> **Nota (23-sep-2026):** este servicio ya NO usa LightGBM. Los modelos se
> retiraron tras medir que metodos mas simples los superaban: regresion
> lineal sobre los ultimos puntos reales para `densidad_poblacional` (ver
> `app/services/proyeccion.py`) y baseline de ultimo valor observado para
> cobertura de acueducto/alcantarillado (ver
> `scripts/rellenar_baseline_2025_2026.py`). Ambos avisos incluyen las
> tablas de backtesting que respaldan la decision.

Parte de la arquitectura multi-repo de Techo 360. Se despliega desde
`techo360-infra` — ver ese repo para el `docker-compose.yml` completo.

## Desarrollo local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Estado en producción (5-oct-2026)

- **Sin cambios de código** el 4–5-oct. `Testing` y `development` están en el mismo SHA (`5add66e`, el último cambio de código, del 3-oct: soporte de `cobertura_acueducto`/`cobertura_alcantarillado` para el score de privación proyectado).
- El backend la consulta por RabbitMQ (cola `proyeccion_requests`, con `prefetch 4` en el consumer de este servicio).
- **Se recrea en un deploy de infra si cambian las variables de entorno compartidas del compose** (toma todas las del archivo común): ocurrió el 5-oct a las 03:32 UTC (I1), cuando ese conjunto ya incluía la variable del interruptor del chatbot. El `docker compose up -d` del deploy de infra solo recrea los servicios cuya configuración cambió; `docker compose up -d --dry-run` lo predice sin ejecutarlo. No se midió la ventana sin consumer de ese recreado.
- Fuera de ese deploy, un despliegue del backend, del frontend o del chatbot no la toca.
- No verificado: el comportamiento de las peticiones de proyección que estén en vuelo durante un recreado.
