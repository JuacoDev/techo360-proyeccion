# techo360-proyeccion

## Resumen

Servicio de proyeccion de indicadores territoriales.
Responde solicitudes por RabbitMQ y expone el contrato de proyecciones.
Usa regresion lineal para densidad poblacional cuando falta el dato real.
Usa el ultimo valor real para cobertura de acueducto y alcantarillado.
No usa LightGBM desde septiembre de 2026.

## Uso local

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Variables de entorno

| Nombre | Para que sirve | Valor por defecto |
|---|---|---|
| `DATABASE_URL` | Lectura de datos | TODO: verificar en el codigo |
| `RABBITMQ_URL` | Solicitudes RPC | TODO: verificar en el codigo |
| `TECHO360_ANCLA` | Ancla de proyeccion | TODO: verificar |
| `CARGA_DATABASE_URL` | Base de carga | TODO: verificar |
| `TECHO360_AUTORIZO_CARGA` | Permite carga | apagado |
| `TECHO360_DATOS_REALES` | Activa pruebas reales | apagado |

No se incluyen credenciales ni valores secretos. Los defaults no comprobados
permanecen como `TODO`.

## Pruebas

```bash
python -m unittest discover -s tests
```

Las pruebas con datos reales requieren `TECHO360_DATOS_REALES=1` y una
`DATABASE_URL` de solo lectura; no fueron ejecutadas en esta documentacion.

## Estructura

`app/` API y servicios; `scripts/` cargas y baselines; `tests/` pruebas;
`data/` datos locales.

## Banderas del chatbot

Este repo no implementa banderas del chatbot. Las banderas compartidas deben
estar apagadas por defecto en los servicios que las consumen.

## Documentacion de fases

La documentacion de fases se mantiene fuera de este repositorio.

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

## Contrato de `calcular_zonas` (10-oct-2026)

`GET /api/proyecciones/zonas` (vía este servicio) devuelve por municipio `{codigo_municipio, valor, es_real, origen, fiabilidad}`:

- Si existe fila en `mapcalordb` para (municipio, variable, año) se devuelve **tal cual**, con `es_real = not es_proyeccion(origen)` (`app/origen.py`: empieza por `proyeccion-` o termina en `-proyeccion`; `DANE-proyeccion-poblacion-AAAA` es real). `fiabilidad` = `intervaloConfianza` si es proyección, 100 si es dato real.
- Si no existe, se calcula un respaldo con `es_real=false`: **último valor real** para acueducto/alcantarillado (backtest: gana a la regresión de 4 puntos en RMSE, MAE y origen móvil) y regresión lineal de 4 puntos para `densidad_poblacional`. Fiabilidad `max(30, 60-10*(antigüedad-1))`.
- Las regresiones siguen excluyendo filas proyectadas al ajustar (mismo patrón en SQL, `SQL_SOLO_REALES`).
- Pruebas: `python -m unittest discover -s tests` (con dobles). Las de datos reales (`tests/test_contrato_datos_reales.py`) solo corren con `TECHO360_DATOS_REALES=1` y `DATABASE_URL`, en sesión de solo lectura, desde un contenedor temporal `:verify`.

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
