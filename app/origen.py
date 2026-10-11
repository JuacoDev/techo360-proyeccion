"""Clasificacion real / proyeccion de una fila de MapCalorDB segun su originVar.

Un unico lugar para la regla (antes estaba repetida como LIKE en cada
consulta del repositorio):

    es_proyeccion(origen) = origen EMPIEZA por "proyeccion-"  (convencion de
                            relleno del proyecto, ver ingestion/db.py en
                            techo360-backend)
                         o TERMINA en "-proyeccion"  (caso TerriData-proyeccion)

El filtro NO puede ser un simple '%proyeccion%': descartaria "DANE-proyeccion-
poblacion-AAAA", que es el NOMBRE de la fuente oficial del DANE (dato real),
no un relleno de este pipeline.

es_real(origen) = not es_proyeccion(origen). Una fila sin originVar (None o "")
no cae en ninguno de los dos patrones, asi que se trata como real; hoy no hay
ninguna asi en MapCalorDB (se verifica en tests_contrato_datos_reales.py).

Los fragmentos SQL de abajo expresan LA MISMA regla en SQL (las consultas
filtran en la base para no traer filas innecesarias); los tests comprueban que
helper y SQL coinciden.
"""

PREFIJO_PROYECCION = "proyeccion-"
SUFIJO_PROYECCION = "-proyeccion"

# Misma regla que es_proyeccion(), para usar dentro de un WHERE (el campo se
# llama originVar en la tabla; Postgres lo pliega a minusculas).
SQL_SOLO_REALES = (
    f"originVar NOT LIKE '{PREFIJO_PROYECCION}%' AND originVar NOT LIKE '%{SUFIJO_PROYECCION}'"
)


def es_proyeccion(originvar) -> bool:
    if not isinstance(originvar, str):
        return False
    return originvar.startswith(PREFIJO_PROYECCION) or originvar.endswith(SUFIJO_PROYECCION)


def es_real(originvar) -> bool:
    return not es_proyeccion(originvar)
