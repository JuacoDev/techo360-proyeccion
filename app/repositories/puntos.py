"""Consultas de SOLO LECTURA para app/services/puntos.py (proyeccion bajo demanda de un punto).

Reutiliza el engine del repositorio existente. No contiene ninguna escritura (INSERT/UPDATE/DELETE):
el calculo bajo demanda nunca modifica mapcalordb.
"""
from sqlalchemy import text

from app.repositories.proyeccion import _obtener_engine

CODIGOS_FUERA_DEL_UNIVERSO = ("00000", "01001")


def en_universo(codigo: str) -> bool:
    """El codigo esta en zonas_municipio y no es 00000 ni 01001."""
    if codigo in CODIGOS_FUERA_DEL_UNIVERSO:
        return False
    with _obtener_engine().connect() as conn:
        fila = conn.execute(
            text("SELECT 1 FROM zonas_municipio WHERE codigo_municipio = :codigo"), {"codigo": codigo}
        ).first()
    return fila is not None


def obtener_serie(indicador: str, codigo: str) -> list[dict]:
    """Todas las filas (años, reales y proyecciones) del municipio para la variable."""
    with _obtener_engine().connect() as conn:
        filas = conn.execute(
            text(
                """
                SELECT yearm, valuevar, originvar
                FROM MapCalorDB
                WHERE codigodivipola = :codigo AND namevar = :indicador
                ORDER BY yearm
                """
            ),
            {"codigo": codigo, "indicador": indicador},
        ).all()
    return [{"anio": int(f[0]), "valor": None if f[1] is None else float(f[1]), "origen": f[2]} for f in filas]
