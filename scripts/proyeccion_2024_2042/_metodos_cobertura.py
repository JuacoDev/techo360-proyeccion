"""Metodos de proyeccion de coberturas (acueducto/alcantarillado) usados
tanto por el backtest (01) como por la generacion (02). Funciones puras."""
import numpy as np


def ultimos_puntos(anios, valores, n):
    """Ultimos n puntos reales (anios ascendentes)."""
    return anios[-n:], valores[-n:]


def pendiente_ols(anios, valores):
    if len(set(anios)) < 2:
        return 0.0, float(valores[-1])
    b, a = np.polyfit(anios, valores, 1)
    return float(b), float(a)


def predecir(metodo, anios, valores, anio_obj, tope_pend=None, horizonte_max=None):
    """Prediccion SIN recortar a [0,100]. `anios`/`valores`: serie real
    ordenada (solo hasta el corte). metodo: 'ultimo' | 'ols4' | 'ols6' |
    'ols7' | 'ols4_clip'. tope_pend: |pendiente| maxima (pp/anio) para
    'ols4_clip'. horizonte_max: si se da, la recta deja de extrapolar mas
    alla de (ultimo_anio_real + horizonte_max) y el valor se mantiene."""
    ultimo_anio = anios[-1]
    if metodo == "ultimo":
        return float(valores[-1])
    n = {"ols4": 4, "ols6": 6, "ols7": 7, "ols4_clip": 4}[metodo]
    xa, xv = ultimos_puntos(anios, valores, n)
    b, a = pendiente_ols(xa, xv)
    if metodo == "ols4_clip" and tope_pend is not None:
        # recorte de la pendiente manteniendo el nivel ajustado en el ultimo anio
        nivel_ult = b * ultimo_anio + a
        b = float(np.clip(b, -tope_pend, tope_pend))
        a = nivel_ult - b * ultimo_anio
    x = anio_obj
    if horizonte_max is not None:
        x = min(anio_obj, ultimo_anio + horizonte_max)
    return float(b * x + a)
