"""Proyeccion bajo demanda de UN punto (municipio, año) que falta en MapCalorDB (paso 3.1/3.2).

Especificacion: claude/fase3-4-umbrales-y-metodos.md (seccion 2). NUNCA escribe en la base: solo
calcula al vuelo y devuelve el resultado; este modulo no importa ninguna funcion de guardado.

completar_punto(indicador, codigo, anio) -> dict
    {indicador, codigo, anio, estado: "calculado"|"sin_dato", valor, origen, fiabilidad,
     es_real: False, metodo, motivo}

Se invoca solo cuando la tabla NO tiene fila para ese municipio-año; si la tiene, el valor
existente jamas se sustituye (estado "sin_dato", motivo "fila_existente").

Metodos por variable:
  * cobertura_acueducto / cobertura_alcantarillado
      - real antes y despues + interpolacion permitida por el backtest -> interpolacion lineal;
      - solo real anterior (o la interpolacion no esta permitida) -> ultimo valor real;
      - sin real anterior -> sin_dato / sin_historia.
  * densidad_poblacional
      - reales a ambos lados + interpolacion permitida -> interpolacion lineal;
      - >= 4 reales anteriores -> regresion lineal de 4 puntos (metodo vigente);
      - si no -> sin_dato / sin_historia.
  * deficit_habitacional_cuantitativo / _cualitativo, años 2019-2023
      - interpolacion lineal entre el real 2018 y el valor 2024:
        v(y) = v2018 + (v2024 - v2018) * (y - 2018) / 6; falta un extremo -> sin_extremo.

Reglas comunes: es_real siempre False y el origen cumple es_proyeccion(); recorte a [0, 100]
(coberturas y deficit) o >= 0 (densidad); si el recorte mueve el valor mas de 5 pp (o la densidad
sale negativa) el resultado es sin_dato / valor_sospechoso; fiabilidad 30 (minimo vigente de las
proyecciones, via fiabilidad_de_origen sin intervalo de confianza).
"""
import numpy as np

from app.origen import es_proyeccion

COBERTURAS = ("cobertura_acueducto", "cobertura_alcantarillado")
DEFICITS = ("deficit_habitacional_cuantitativo", "deficit_habitacional_cualitativo")
DENSIDAD = "densidad_poblacional"
INDICADORES_PUNTOS = frozenset((*COBERTURAS, *DEFICITS, DENSIDAD))

# Rango de años que tiene cada variable en mapcalordb (REPORTADO por 3.1-b).
RANGO_ANIOS = {
    "cobertura_acueducto": (2008, 2042),
    "cobertura_alcantarillado": (2008, 2042),
    "deficit_habitacional_cuantitativo": (2018, 2042),
    "deficit_habitacional_cualitativo": (2018, 2042),
    DENSIDAD: (2000, 2042),
}

MAX_LOTE = 40
PUNTOS_REGRESION = 4
UMBRAL_RECORTE_PP = 5.0
FIABILIDAD_ORIGEN_NUEVO = 30.0  # minimo vigente de las proyecciones

ORIGEN_INTERPOLACION = "proyeccion-techo360-interpolacion"
ORIGEN_INTERPOLACION_DEFICIT = "proyeccion-techo360-interpolacion-deficit"
ORIGEN_ULTIMO_VALOR = "proyeccion-techo360-ultimo-valor"  # existente (mismo nombre que el CSV de proyecciones)
ORIGEN_REGRESION = "proyeccion-regresion-lineal-4puntos"  # existente (origen vigente de densidad)

METODO_INTERPOLACION = "interpolacion-lineal"
METODO_ULTIMO_VALOR = "ultimo-valor-real"
METODO_REGRESION = "regresion-lineal-4puntos"
METODO_INTERPOLACION_DEFICIT = "interpolacion-lineal-2018-2024"

ANIO_REAL_DEFICIT = 2018
ANIO_EXTREMO_DEFICIT = 2024
ANIOS_DEFICIT_INTERPOLADOS = range(2019, 2024)

# Decision del backtest de interpolacion (claude/fase3-4 seccion 4): la interpolacion se usa solo
# donde su RMSE <= el del metodo de comparacion. Resultado en
# data/proyeccion_2024_2042/informes/J_backtest_interpolacion.md.
INTERPOLACION_PERMITIDA = {
    "cobertura_acueducto": True,
    "cobertura_alcantarillado": True,
    DENSIDAD: True,
}


def fiabilidad_de_origen(origen: str) -> float:
    """Fiabilidad de un valor calculado aqui: siempre una proyeccion sin intervalo de confianza,
    asi que es el minimo vigente (30)."""
    return FIABILIDAD_ORIGEN_NUEVO


def _sin_dato(indicador: str, codigo: str, anio: int, motivo: str) -> dict:
    return {
        "indicador": indicador,
        "codigo": codigo,
        "anio": anio,
        "estado": "sin_dato",
        "valor": None,
        "origen": None,
        "fiabilidad": None,
        "es_real": False,
        "metodo": None,
        "motivo": motivo,
    }


def _calculado(indicador: str, codigo: str, anio: int, valor: float, origen: str, metodo: str) -> dict:
    return {
        "indicador": indicador,
        "codigo": codigo,
        "anio": anio,
        "estado": "calculado",
        "valor": round(float(valor), 2),
        "origen": origen,
        "fiabilidad": fiabilidad_de_origen(origen),
        "es_real": False,
        "metodo": metodo,
        "motivo": None,
    }


def _reales(serie: list[dict]) -> list[tuple[int, float]]:
    """(año, valor) de las filas reales con valor, ordenadas por año."""
    return sorted(
        (int(f["anio"]), float(f["valor"]))
        for f in serie
        if f.get("valor") is not None and not es_proyeccion(f.get("origen"))
    )


def _valor_de(serie: list[dict], anio: int):
    for f in serie:
        if int(f["anio"]) == anio and f.get("valor") is not None:
            return float(f["valor"])
    return None


def _interpolar(anio: int, a0: int, v0: float, a1: int, v1: float) -> float:
    return v0 + (v1 - v0) * (anio - a0) / (a1 - a0)


def _recortar(indicador: str, valor: float):
    """(valor recortado, sospechoso). Sospechoso = el recorte movio el valor mas de 5 pp, o la
    densidad es negativa antes de recortar."""
    if indicador == DENSIDAD:
        return max(0.0, valor), valor < 0
    recortado = min(100.0, max(0.0, valor))
    return recortado, abs(recortado - valor) > UMBRAL_RECORTE_PP


def calcular_punto(
    indicador: str,
    codigo: str,
    anio: int,
    serie: list[dict],
    en_universo: bool,
    interpolacion_permitida: dict | None = None,
) -> dict:
    """Nucleo puro (sin base de datos): `serie` = filas de mapcalordb del municipio y la variable,
    cada una {"anio", "valor", "origen"} (reales y proyecciones)."""
    if indicador not in INDICADORES_PUNTOS:
        raise ValueError(f"Los indicadores disponibles son: {', '.join(sorted(INDICADORES_PUNTOS))}")
    permitida = INTERPOLACION_PERMITIDA if interpolacion_permitida is None else interpolacion_permitida

    if not en_universo:
        return _sin_dato(indicador, codigo, anio, "fuera_de_universo")
    desde, hasta = RANGO_ANIOS[indicador]
    if not desde <= anio <= hasta:
        return _sin_dato(indicador, codigo, anio, "fuera_de_rango")
    if _valor_de(serie, anio) is not None:
        return _sin_dato(indicador, codigo, anio, "fila_existente")  # un valor existente no se sustituye

    reales = _reales(serie)
    antes = [(a, v) for a, v in reales if a < anio]
    despues = [(a, v) for a, v in reales if a > anio]

    if indicador in DEFICITS:
        if anio not in ANIOS_DEFICIT_INTERPOLADOS:
            return _sin_dato(indicador, codigo, anio, "sin_historia")
        v2018 = dict(reales).get(ANIO_REAL_DEFICIT)
        v2024 = _valor_de(serie, ANIO_EXTREMO_DEFICIT)
        if v2018 is None or v2024 is None:
            return _sin_dato(indicador, codigo, anio, "sin_extremo")
        valor = _interpolar(anio, ANIO_REAL_DEFICIT, v2018, ANIO_EXTREMO_DEFICIT, v2024)
        origen, metodo = ORIGEN_INTERPOLACION_DEFICIT, METODO_INTERPOLACION_DEFICIT
    else:
        usa_interpolacion = bool(antes and despues and permitida.get(indicador, False))
        if usa_interpolacion:
            (a0, v0), (a1, v1) = antes[-1], despues[0]
            valor = _interpolar(anio, a0, v0, a1, v1)
            origen, metodo = ORIGEN_INTERPOLACION, METODO_INTERPOLACION
        elif indicador in COBERTURAS:
            if not antes:
                return _sin_dato(indicador, codigo, anio, "sin_historia")
            valor = antes[-1][1]
            origen, metodo = ORIGEN_ULTIMO_VALOR, METODO_ULTIMO_VALOR
        else:  # densidad
            if len(antes) < PUNTOS_REGRESION:
                return _sin_dato(indicador, codigo, anio, "sin_historia")
            ultimos = antes[-PUNTOS_REGRESION:]
            pendiente, intercepto = np.polyfit([a for a, _ in ultimos], [v for _, v in ultimos], 1)
            valor = float(pendiente * anio + intercepto)
            origen, metodo = ORIGEN_REGRESION, METODO_REGRESION

    recortado, sospechoso = _recortar(indicador, valor)
    if sospechoso:
        return _sin_dato(indicador, codigo, anio, "valor_sospechoso")
    assert es_proyeccion(origen)  # coherencia con el contrato 2.2: lo calculado nunca se presenta como real
    return _calculado(indicador, codigo, anio, recortado, origen, metodo)


def completar_punto(indicador: str, codigo: str, anio: int, repo=None) -> dict:
    """Un punto. `repo` (inyectable en pruebas) expone obtener_serie(indicador, codigo) y
    en_universo(codigo); por defecto, las consultas de solo lectura de app/repositories/puntos.py."""
    if repo is None:
        from app.repositories import puntos as repo
    if indicador not in INDICADORES_PUNTOS:
        raise ValueError(f"Los indicadores disponibles son: {', '.join(sorted(INDICADORES_PUNTOS))}")
    if not repo.en_universo(codigo):
        return _sin_dato(indicador, codigo, anio, "fuera_de_universo")
    return calcular_punto(indicador, codigo, anio, repo.obtener_serie(indicador, codigo), True)


def completar_puntos(puntos: list[dict], repo=None) -> list[dict]:
    """Lote de a lo sumo MAX_LOTE puntos [{"indicador", "codigo", "anio"}]; una consulta por
    (indicador, codigo) distinto. El resultado conserva el orden de la entrada."""
    if not isinstance(puntos, list) or not puntos:
        raise ValueError("El lote debe ser una lista no vacia de puntos.")
    if len(puntos) > MAX_LOTE:
        raise ValueError(f"El lote admite como maximo {MAX_LOTE} puntos; recibio {len(puntos)}.")
    if repo is None:
        from app.repositories import puntos as repo
    series: dict = {}
    universo: dict = {}
    resultado = []
    for p in puntos:
        indicador, codigo, anio = p["indicador"], p["codigo"], int(p["anio"])
        if indicador not in INDICADORES_PUNTOS:
            raise ValueError(f"Los indicadores disponibles son: {', '.join(sorted(INDICADORES_PUNTOS))}")
        if codigo not in universo:
            universo[codigo] = repo.en_universo(codigo)
        if not universo[codigo]:
            resultado.append(_sin_dato(indicador, codigo, anio, "fuera_de_universo"))
            continue
        if (indicador, codigo) not in series:
            series[(indicador, codigo)] = repo.obtener_serie(indicador, codigo)
        resultado.append(calcular_punto(indicador, codigo, anio, series[(indicador, codigo)], True))
    return resultado
