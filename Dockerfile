FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY scripts ./scripts
# Ya no se copia models/: el directorio se elimino al retirar LightGBM del
# pipeline (ver scripts/rellenar_baseline_2025_2026.py). El servicio calcula
# la proyeccion con una regresion lineal, sin artefactos en disco.
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
