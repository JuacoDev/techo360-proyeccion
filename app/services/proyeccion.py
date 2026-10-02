"""
Proyeccion futura de densidad_poblacional a nivel municipio.

>>> AVISO: aqui NO se usa LightGBM, y es una decision con evidencia
>>> (23-sep-2026) -- no una simplificacion por falta de tiempo <<<

Este servicio uso hasta esta fecha el modelo
models/lightgbm_densidad_poblacional.txt. Se reemplazo por una regresion
lineal sobre los ultimos puntos REALES del municipio (ver mas abajo la 2a
ronda de backtesting, que fijo la ventana en 4 puntos), despues de un
backtesting contra los datos reales disponibles.

MONTAJE DEL BACKTESTING: se entreno solo con anio <= 2018 y se evaluo
contra los años 2019-2024 que si existen en el historico, prediciendo de
forma RECURSIVA multi-año (igual que hacia este servicio en produccion,
no a un paso con lags verdaderos), sobre los 956 municipios con serie
completa 2017-2024.

    Variante                                   RMSE global   MAE    corr. con
                                                                    tendencia real
    A  LightGBM con `anio` absoluto (lo previo)   223.89    22.66      -0.212
    B  LightGBM sin `anio`                        226.07    22.72      -0.236
    C  LightGBM multi-horizonte (feature `h`)     100.57    14.22      +0.052
    D  ultimo valor observado (baseline ingenuo)   73.11    12.74        --
    E  extrapolacion lineal de 2 puntos            28.07     5.94      +0.940

Dos resultados decisivos:
  1. La extrapolacion lineal es ~8x mejor en RMSE que el mejor LightGBM
     recursivo (28.07 vs 223.89) y mejor incluso que el baseline ingenuo.
  2. Las variantes con LightGBM recursivo tienen correlacion NEGATIVA
     (-0.212 y -0.236) entre el cambio que predicen y el cambio real:
     no es que no capten la tendencia, es que la invierten.

La causa estructural ya estaba medida: el umbral de split mas alto sobre
`anio` en el modelo entrenado era 2020.5, asi que 2021, 2024, 2027, 2035
y 2042 daban predicciones bit-identicas con los mismos lags. Un arbol no
extrapola fuera del rango visto en entrenamiento. En la practica eso
producia series planas: 4 de 6 capitales probadas devolvian el mismo
valor para 5 años seguidos, y el 25,6% de una muestra de 78 municipios
quedaba completamente plano.

Esto es coherente con la decision que el proyecto ya habia tomado para
cobertura_acueducto/cobertura_alcantarillado (ver ingestion/db.py en
techo360-backend): no es defendible usar a proposito el metodo que ya se
sabe que es peor solo porque tambien es "un modelo".

>>> LIMITACION CONOCIDA del backtesting, no ocultar <<<
Los valores "reales" 2019-2024 contra los que se evaluo son ELLOS MISMOS
proyecciones demograficas de DANE/TerriData, no mediciones
independientes (censo). Son series suaves, casi lineales por
construccion, lo que favorece estructuralmente a la extrapolacion
lineal. La conclusion valida es "la extrapolacion lineal reproduce mejor
la serie oficial de densidad que el LightGBM entrenado", NO "la
extrapolacion lineal predice mejor la densidad real futura del pais".
Ademas, MapCalorDB ya trae filas "TerriData-proyeccion" para 2027-2042
de esta misma variable: si en algun momento se decide servir esas filas
directamente, este servicio podria dejar de hacer falta.

============================================================
2a RONDA DE BACKTESTING (23-sep-2026): ¿cuantos puntos usar?
============================================================
La version inicial ajustaba la recta con SOLO los 2 ultimos puntos, y eso
tenia una debilidad concreta: el ultimo par fijaba la direccion de toda la
serie futura. Verificado en vivo -- Medellin, Cali y Barranquilla venian
de un 2025->2026 a la baja y proyectaban descenso sostenido (-4.15,
-12.99 y -22.68 hab/km2 por año) aunque su serie 2022-2025 venia subiendo.

Se midio, sin asumir que "mas puntos es mejor": para cada año de corte se
toman los ultimos N puntos reales, se ajusta la recta por minimos
cuadrados y se extrapola hasta 2024 contra los valores reales que
quedaron fuera. CUATRO cortes distintos (2014, 2016, 2018, 2020), ~956
municipios por corte, para no decidir con una sola foto.

    N        RMSE prom   MAE prom   % mediano   % p90
    2 (antes)   31.285      5.728      2.221     8.323
    3           29.801      5.584      2.221     8.111
    4  <--      29.822      5.604      2.187     8.009
    5           29.521      5.618      2.265     8.072
    6           29.362      5.722      2.404     8.251
    8           29.878      5.998      2.678     8.630
    10          31.167      6.284      2.869     8.957
    todos       32.660      6.569      3.004     9.249

Se eligio N=4. Las metricas ABSOLUTAS (RMSE/MAE) estan dominadas por un
puñado de ciudades de densidad altisima (Bogota, Medellin: miles de
hab/km2), asi que no dicen que tan bien le va al municipio tipico; por eso
el desempate se hizo con el error PORCENTUAL, donde N=4 es el mejor en
mediana (2.187%) y en la cola p90 (8.009%), quedando ademas a 1,6% del
mejor RMSE y a 0,4% del mejor MAE. Es el unico N que esta en el podio de
las cuatro metricas a la vez.

>>> HONESTIDAD SOBRE LA FUERZA DE ESTA EVIDENCIA <<<
  1. Las diferencias entre N=3, 4 y 5 son de ~2% en todas las metricas:
     la evidencia NO distingue con fuerza entre esos tres. Lo que si
     queda claro es que una ventana de 3-5 puntos le gana al N=2 anterior
     y tambien a las ventanas largas (N>=8 y "todos", que empeoran todo).
  2. El resultado NO es uniforme por periodo: en los cortes 2014 y 2016
     el N=2 era el mejor o casi, y solo en 2018/2020 las ventanas mas
     largas ganan claramente. El promedio favorece a N=4, pero la mejora
     depende del periodo, no es universal.
  3. La mejora de N=4 sobre N=2 es real pero MODERADA: -4,7% de RMSE,
     -1,5% de error porcentual mediano, -3,8% en la cola p90. No resuelve
     del todo el caso Medellin/Cali/Barranquilla: esas tres mejoraban
     mucho mas con N=8, pero N=8 empeora al resto del pais, y no se
     optimiza para tres ciudades a costa de 1.100 municipios.
  4. Sigue aplicando la limitacion de la 1a ronda: los "reales" contra
     los que se mide son proyecciones DANE/TerriData, no mediciones
     independientes.

============================================================
3a RONDA: ¿y si LightGBM sirve en OTRO enfoque? (23-sep-2026)
============================================================
Antes de eliminar LightGBM del repo se evaluaron CINCO enfoques
distintos (objetivo en cambio, linear_tree, hibrido de residuos,
imputacion entre municipios, uso explicativo con SHAP) contra el metodo
vigente de cada variable. Ninguno lo supero de forma robusta: para
densidad_poblacional, la regresion lineal N=4 gana en las CUATRO
metricas frente a todas las variantes probadas.

El registro completo -- metodologia, tablas por variable, veredicto de
cada experimento y los scripts para reproducirlo -- esta en
scripts/experimentos/investigacion_lightgbm.py. No se repite aqui para
no duplicar la fuente de verdad.
"""
import numpy as np

from app.repositories.proyeccion import (
    obtener_contexto_municipio,
    obtener_datos_municipio_para_anios,
    obtener_datos_todos_municipios_para_anio,
    obtener_puntos_todos_los_municipios,
)
from app.schemas import SolicitudProyeccion

INDICADOR_VIGENTE = "densidad_poblacional"

# Score de privacion por proyeccion (T-?, decision de producto): acueducto
# y alcantarillado se proyectan con el MISMO metodo (regresion lineal
# N=4) -- el docstring de este modulo ya documentaba que es la decision
# correcta para esas 2 variables ("Esto es coherente con la decision que
# el proyecto ya habia tomado para cobertura_acueducto/cobertura_alcantarillado").
# deficit_habitacional_cuantitativo NO entra aqui a proposito: viene de un
# censo fijo (CNPV 2018, ver ingestion/dane_deficit_adapter.py en
# techo360-backend), un solo punto real por municipio para siempre -- la
# regresion exige 2+ años distintos y este nunca los tendria. El backend
# lo maneja aparte (reusa el valor del año base 2024 congelado), sin pasar
# por este servicio.
INDICADORES_SOPORTADOS = frozenset({
    INDICADOR_VIGENTE,
    "cobertura_acueducto",
    "cobertura_alcantarillado",
})

# Techo de validez fisica por indicador -- None = sin techo (densidad no
# tiene limite superior natural). cobertura_acueducto/alcantarillado son
# porcentajes [0, 100]: sin este techo, una recta extrapolada a muchos
# años podria devolver, por ejemplo, 107% de cobertura.
VALOR_MAXIMO_POR_INDICADOR = {
    "cobertura_acueducto": 100.0,
    "cobertura_alcantarillado": 100.0,
}

# Tamaño de la ventana de la regresion: los ultimos N puntos REALES del
# municipio. Elegido con la 2a ronda de backtesting (ver aviso del
# modulo), no a ojo. Si un municipio tiene menos de N puntos, se usan
# todos los que tenga; con 2 la regresion degenera en la recta por esos
# 2 puntos (el metodo anterior), asi que esos municipios no cambian.
PUNTOS_REGRESION = 4

# Nombre del metodo que se reporta en la respuesta. Se mantiene el mismo
# CAMPO que antes (`modelo`) para no romper el contrato del consumer RPC
# ni del backend; lo que cambia es su valor, que ahora dice la verdad
# sobre como se calculo la serie.
METODO_VIGENTE = f"regresion-lineal-{PUNTOS_REGRESION}puntos"

# La densidad poblacional no puede ser negativa: en un municipio con
# pendiente negativa, un horizonte largo podria cruzar el cero. Se recorta
# en 0 en vez de devolver un valor imposible (mismo criterio que el
# recorte a [0, 100] de los porcentajes en ingestion/proyeccion_2024.py).
VALOR_MINIMO = 0.0


def _proyectar_serie(contexto: dict, anio_fin: int, indicador: str) -> list[dict]:
    """Regresion lineal por minimos cuadrados sobre los ultimos N puntos
    reales, extrapolada a los años futuros pedidos:

        valor(a) = pendiente * a + intercepto

    La recta se ajusta contra el AÑO REAL de cada punto, no contra su
    posicion en la lista: la serie de un municipio puede tener huecos, y
    ajustar por posicion deformaria la pendiente justo en esos casos.
    """
    anio_actual = contexto["ultimo_anio"]
    if anio_fin <= anio_actual:
        raise ValueError(
            f"anio_fin debe ser mayor que el ultimo año disponible ({anio_actual})"
        )

    anios = [p["anio"] for p in contexto["puntos"]]
    valores = [p["valor"] for p in contexto["puntos"]]

    if len(set(anios)) < 2:
        # Defensivo: el repositorio ya exige 2+ puntos y ordena por año,
        # asi que esto solo pasaria con datos corruptos (varias filas del
        # mismo año para el mismo municipio). Mejor fallar que ajustar una
        # recta sobre un solo año.
        raise RuntimeError(
            f"Se necesitan al menos 2 años distintos para ajustar la recta; "
            f"se recibieron: {anios}"
        )

    pendiente, intercepto = np.polyfit(anios, valores, 1)
    techo = VALOR_MAXIMO_POR_INDICADOR.get(indicador)

    def _recortar(valor: float) -> float:
        valor = max(VALOR_MINIMO, valor)
        return valor if techo is None else min(valor, techo)

    return [
        {"anio": anio, "valor": _recortar(float(pendiente * anio + intercepto))}
        for anio in range(anio_actual + 1, anio_fin + 1)
    ]


def calcular_serie_proyectada(solicitud: SolicitudProyeccion) -> dict:
    if solicitud.indicador not in INDICADORES_SOPORTADOS:
        raise ValueError(
            f"Los indicadores disponibles son: {', '.join(sorted(INDICADORES_SOPORTADOS))}"
        )

    contexto = obtener_contexto_municipio(
        solicitud.codigo_municipio,
        solicitud.indicador,
        n_puntos=PUNTOS_REGRESION,
    )
    serie = _proyectar_serie(contexto, solicitud.anio_fin, solicitud.indicador)
    datos_disponibles = obtener_datos_municipio_para_anios(
        solicitud.codigo_municipio,
        solicitud.indicador,
        serie[0]["anio"],
        solicitud.anio_fin,
    )
    serie = [
        {**punto, "valor": datos_disponibles.get(punto["anio"], punto["valor"])}
        for punto in serie
    ]

    return {
        "codigo_municipio": solicitud.codigo_municipio,
        "indicador": solicitud.indicador,
        "modelo": METODO_VIGENTE,
        "anio_inicio": serie[0]["anio"],
        "anio_fin": solicitud.anio_fin,
        "serie_proyectada": serie,
    }


def calcular_zonas(indicador: str, anio: int) -> list[dict]:
    """T-104: choropleth completo -- UNA consulta SQL para todos los
    municipios (obtener_puntos_todos_los_municipios) + un calculo de
    regresion por grupo, en vez de ~1105 llamadas individuales a
    calcular_serie_proyectada().

    Medido en vivo (26-sep-2026, diagnostico previo a esta funcion):
    1105 llamadas secuenciales via el camino atomico tardaban ~85 min
    (4.6s/municipio, dominado por reabrir conexion + traer la tabla
    COMPLETA en cada llamada) y, incluso despues de arreglar ese cuello
    de botella en obtener_contexto_municipio(), seguian tardando ~13 min
    -- inviable para un endpoint HTTP sincrono de cualquier forma. Esta
    funcion hace 1 sola consulta (no 1105) y calcula la regresion de
    todos los municipios en el mismo proceso.

    No reemplaza a calcular_serie_proyectada() (la RPC atomica de
    produccion sigue igual, sin romper su contrato) -- es una funcion
    nueva para el caso "todos los municipios, un indicador, un año".

    >>> FIX del gap ya anotado como bloqueante (Paso 4,
    >>> techo360-backend/routers/proyecciones.py, resuelto 26/28-sep-2026,
    >>> ver commit 27c02cd de JrGG22) <<<
    Para el año solicitado, consulta los valores disponibles por municipio
    SIN el limite de `yearM <= anio_actual` (obtener_datos_todos_municipios_
    para_anio, en app/repositories/proyeccion.py) -- si hay un valor de
    fuente aceptada para ese año exacto, se devuelve como `es_real: true`
    en vez de la regresion, sin importar que el reloj del servidor todavia
    no haya llegado a ese año (confirmado en vivo con SQL directo contra
    MapCalorDB: DANE real 2024-2042 ya cargado). Solo cuando el año pedido
    NO tiene dato real se usan los ultimos PUNTOS_REGRESION puntos reales
    de obtener_puntos_todos_los_municipios() (esa si sigue con el tope de
    fecha, ver su propio docstring) para la regresion."""
    if indicador not in INDICADORES_SOPORTADOS:
        raise ValueError(
            f"Los indicadores disponibles son: {', '.join(sorted(INDICADORES_SOPORTADOS))}"
        )

    puntos_por_municipio = obtener_puntos_todos_los_municipios(indicador, n_puntos=PUNTOS_REGRESION)
    datos_disponibles = obtener_datos_todos_municipios_para_anio(indicador, anio)

    resultado = []
    codigos = dict.fromkeys((*puntos_por_municipio, *datos_disponibles))
    for codigo in codigos:
        puntos = puntos_por_municipio.get(codigo)
        valor_disponible = datos_disponibles.get(codigo)
        if valor_disponible is not None:
            resultado.append({"codigo_municipio": codigo, "valor": round(valor_disponible, 2), "es_real": True})
            continue

        if puntos is None:
            continue

        anios = [p["anio"] for p in puntos]
        if anio <= anios[-1]:
            # Mismo criterio que _proyectar_serie(): no tiene sentido
            # "proyectar" hacia un año que ya paso el ultimo punto real
            # disponible y no coincidio arriba -- se excluye del
            # choropleth en vez de forzar un numero sin sustento.
            continue

        ultimos = puntos[-PUNTOS_REGRESION:]
        anios_regresion = [p["anio"] for p in ultimos]
        valores_regresion = [p["valor"] for p in ultimos]
        pendiente, intercepto = np.polyfit(anios_regresion, valores_regresion, 1)
        valor = max(VALOR_MINIMO, float(pendiente * anio + intercepto))
        techo = VALOR_MAXIMO_POR_INDICADOR.get(indicador)
        if techo is not None:
            valor = min(valor, techo)
        resultado.append({"codigo_municipio": codigo, "valor": round(valor, 2), "es_real": False})

    return resultado
