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
