"""
REGISTRO DE LA INVESTIGACION: ¿puede LightGBM aportar valor real en
techo360-proyeccion? (23-sep-2026)

>>> AVISO -- ESTUDIO NEGATIVO DOCUMENTADO, no ocultar <<<
Se evaluaron CINCO enfoques distintos de LightGBM contra el metodo
vigente de cada variable. NINGUNO lo supero de forma robusta. Este
modulo existe para que esa conclusion sea verificable y reproducible, no
solo afirmada: los tres scripts de este paquete vuelven a generar las
tablas de abajo contra la base real.

Contexto de por que se investigo: el entregable oficial del Sprint 4
exige "modelos predictivos ajustados con metricas de precision,
correlaciones e impacto de variables". Antes de eliminar LightGBM del
pipeline convenia descartar que sirviera en ALGUN otro enfoque, no solo
en el que ya habia fallado.

============================================================
METODOLOGIA COMUN
============================================================
  - Backtesting de 4 cortes (2014, 2016, 2018, 2020), evaluando contra
    los años reales posteriores que si existen en el panel, hasta 2024.
  - Proyeccion RECURSIVA multi-año (cada paso realimenta su propia
    prediccion, nunca el valor real), que es como funciona produccion.
  - ANTI-FUGA: para el corte C, el entrenamiento usa SOLO filas con
    año <= C, y las features de una fila objetivo del año t se calculan
    con historia estrictamente anterior a t. En la imputacion, la
    validacion deja fuera MUNICIPIOS completos, no años.
  - Hiperparametros FIJOS e identicos en todos los experimentos
    (num_leaves=15, learning_rate=0.05, min_data_in_leaf=20,
    feature_fraction=0.9, 400 rondas). No se tuneo por experimento: de
    lo contrario la comparacion premiaria al que mas busqueda recibio.
  - Metricas: RMSE, MAE, error porcentual mediano y p90 -- las mismas
    de las rondas anteriores, para que todo sea comparable.
  - Criterio de empate: una mejora dentro de ~2% se reporta como empate,
    no como victoria (es el margen que ya se habia visto entre N=3/4/5).

============================================================
EXPERIMENTOS 1 y 2 -- objetivo en CAMBIO, y linear_tree
============================================================
E1 entrena LightGBM para predecir la variacion anual (diferencia y
log-ratio) y reconstruye el nivel acumulando. E2 repite con
linear_tree=True, con objetivo en nivel y en cambio. E3 (abajo) aparece
en las mismas tablas.

densidad_poblacional -- metodo vigente: regresion lineal N=4
    metodo                        RMSE      MAE     %med    %p90
    VIG_reglineal_N4            29.826    5.605    2.186   8.000   <-- gana todo
    E3_hibrido_residuos         34.475    6.722    2.809  10.208
    E1_diff                     37.724    7.005    2.410   9.156
    VIG_ultimo_valor            78.480   13.313    3.660  11.464
    E2_nivel_lineartree        144.500   20.061    2.645  12.643
    E2_diff_lineartree         207.151   17.134    2.331   9.161
    REF_nivel_lgbm             476.905   65.518    5.067  35.929
    E1_logratio                590.106   53.912    2.253   9.074

cobertura_acueducto -- metodo vigente: ultimo valor observado
    metodo                        RMSE      MAE     %med    %p90
    E1_diff                     23.212   17.101   18.971 101.557
    E3_hibrido_residuos         23.427   15.940   16.970  74.554
    VIG_ultimo_valor            25.644   15.647   12.737  73.898   <-- gana 3 de 4
    VIG_reglineal_N4            32.232   21.929   24.046  98.696

cobertura_alcantarillado -- metodo vigente: ultimo valor observado
    metodo                        RMSE      MAE     %med    %p90
    E1_diff                     20.028   14.162   22.975  97.401
    E3_hibrido_residuos         20.598   13.366   20.075  83.200
    VIG_ultimo_valor            22.260   12.882   15.316  80.968   <-- gana 3 de 4
    VIG_reglineal_N4            28.379   18.865   31.443 105.028

VEREDICTO E1: descartar. En densidad pierde en las CUATRO metricas. En
coberturas gana SOLO RMSE (~10%) y pierde MAE (~10%), error porcentual
mediano (~50% peor) y p90. Como las coberturas son porcentajes en la
misma escala 0-100, no hay dominancia de escala que justifique preferir
RMSE: MAE y % mediano son criterios legitimos ahi, y los pierde. Que
RMSE y p90 (ambas de cola) se contradigan indica que la ventaja de RMSE
no es robusta.

VEREDICTO E2 (linear_tree): descartar. Es la peor familia en densidad
(RMSE 144 y 207 frente a 29.8 del metodo vigente) y no aporta ventaja en
coberturas. La idea era que las hojas lineales permitieran extrapolar;
en la practica explotan en algunos municipios.

Nota sobre E1_logratio: tiene error porcentual mediano razonable (2.253)
pero RMSE catastrofico (590). Al acumular multiplicativamente, unos pocos
municipios se disparan. Es el ejemplo mas claro de por que este proyecto
no decide con una sola metrica.

Hallazgo lateral util: VIG_reglineal_N4 es claramente MALO para
coberturas (RMSE 32.2 y 28.4). Confirma que la decision de usar metodos
distintos por variable no es arbitraria.

============================================================
EXPERIMENTO 3 -- hibrido de residuos
============================================================
Base = regresion lineal N=4; LightGBM entrenado sobre SUS RESIDUOS, con
las otras variables del municipio, departamento y tamaño como features.

VEREDICTO: descartar. En densidad DEGRADA su propia base: parte de
29.826 de RMSE y la correccion la empeora a 34.475. El aporte del modelo
sobre los residuos es negativo, no nulo. En coberturas queda entre medias
sin superar al baseline en MAE ni en % mediano.

============================================================
EXPERIMENTO 4 -- imputacion ENTRE municipios (solo coberturas)
============================================================
Validacion cruzada de 5 folds dejando municipios completos fuera.

    variable         escenario                      LightGBM    mediana depto
    acueducto        sin la otra cobertura (REAL)   RMSE 25.06   RMSE 26.86
                                                    %med 25.1    %med 27.6
    acueducto        con la otra cobertura          RMSE 19.33        --
    alcantarillado   sin la otra cobertura (REAL)   RMSE 24.50   RMSE 27.05
                                                    %med 34.2    %med 39.9
    alcantarillado   con la otra cobertura          RMSE 18.49        --

HALLAZGO ESTRUCTURAL: 52 de los 53 municipios sin dato carecen de AMBAS
coberturas. La feature mas predictiva (la cobertura hermana, que baja el
RMSE de ~25 a ~19) NO existe justo donde se necesitaria.

VEREDICTO: descartar para produccion, conservar como analisis del gap.
LightGBM le gana a la mediana departamental un 7-11%, por encima del
umbral de empate -- pero TODOS los metodos son malos en absoluto: error
porcentual mediano de 25% (acueducto) y 34% (alcantarillado). Publicar
una cobertura municipal con un tercio de error tipico no es defendible.

SESGO ADICIONAL, no ocultar: la falta NO es aleatoria. Los municipios con
gap tienen deficit cuantitativo mediano 49% mas alto (18.70 vs 12.54) y
deficit total mas alto. Como el deficit correlaciona negativamente con la
cobertura, un modelo entrenado en los municipios CON dato (menos
carenciados) SOBREESTIMARIA la cobertura de los gaps, y el error medido
por validacion cruzada SUBESTIMA el error real de imputarlos. Entre los
53 hay capitales (Ibague, Cucuta, Leticia, San Andres), lo que sugiere un
artefacto de reporte de TerriData mas que un patron unico.

POR QUE NO SE IMPUTA tenencia_informal_terrenos: los 302 municipios con
dato no son una muestra aleatoria -- el IGAC eligio donde correr la
clasificacion de informalidad, y esa eleccion probablemente correlaciona
con el nivel real de informalidad (faltante NO aleatorio). Un modelo
entrenado sobre esos 302 aprenderia el patron de los que el IGAC SI
evaluo y lo extrapolaria a los que decidio no evaluar. Para hacerlo sin
sesgo haria falta una variable que explique la decision del IGAC de forma
independiente del resultado, o una submuestra validada externamente;
ninguna existe en los datos disponibles.

============================================================
EXPERIMENTO 5 -- uso explicativo del Score de Privacion (SHAP)
============================================================
SHAP se calcula con `pred_contrib=True` de LightGBM, que implementa
TreeSHAP nativamente (el paquete `shap` se instalo pero su DLL quedo
bloqueada por una directiva de Control de Aplicaciones de Windows en la
maquina de desarrollo; el algoritmo es el mismo).

PROBLEMA DE FONDO: el Score de Privacion NO es un fenomeno observado, es
una formula lineal determinista definida por el propio proyecto
(ver routers/zonas.py en techo360-backend). Explicarlo a partir de sus
propios 3 insumos es tautologico.

    Modelo A (control circular, con las 3 variables de la formula):
        R2 = 0.9956  -- solo recupera la formula
        SHAP: acueducto 39.8%, alcantarillado 33.3%, deficit cuanti 26.8%

    Atribucion EXACTA, analitica, sin ningun modelo (% de la varianza
    del score entre municipios):
        acueducto 39.0%, alcantarillado 39.8%, deficit cuanti 21.3%

    Modelo B (informativo, EXCLUYE las 3 variables de la formula):
        R2 = 0.2056  -- debil
        SHAP: densidad 43.5%, departamento 25.4%,
              deficit cualitativo 16.1%, area 15.0%

VEREDICTO: usar SOLO como analisis, y con matices. El modelo A es
circular y el B explica poco (las caracteristicas observables fuera de la
formula dan cuenta de ~21% de la variacion del score). Para el entregable,
la descomposicion analitica exacta es MAS rigurosa que un modelo ajustado:
es exacta, no aproximada, y no requiere entrenar nada.

Lo unico con valor descriptivo propio: la composicion del score varia por
region -- en Amazonia el deficit pesa 45.3% mientras en Caribe el
alcantarillado pesa 41.5%. Pero eso tambien se obtiene analiticamente por
region, sin modelo.

============================================================
CAVEAT HONESTO SOBRE LA FUERZA DE LA EVIDENCIA
============================================================
Los "reales" 2019-2024 de densidad_poblacional son ELLOS MISMOS
proyecciones demograficas de DANE/TerriData: series suaves, casi lineales
por construccion, lo que favorece ESTRUCTURALMENTE a la regresion lineal.
Esa comparacion no es contra mediciones independientes, y asi hay que
leerla.

Las de cobertura_acueducto/cobertura_alcantarillado, en cambio, son datos
REPORTADOS OBSERVADOS de TerriData. Ese resultado esta en terreno mas
firme -- y ahi el baseline de ultimo valor observado igual gana 3 de las
4 metricas.

============================================================
CONCLUSION
============================================================
Ningun enfoque supera al metodo vigente, asi que LightGBM se elimina del
pipeline. Un estudio negativo documentado y reproducible es mas
defendible que adoptar un modelo peor para poder decir que se uso ML:
lo que queda en produccion (regresion lineal N=4 con N elegido por
backtesting, y baseline de ultimo valor observado) esta respaldado por
metricas de precision sobre 4 cortes, correlaciones con la tendencia real
e impacto de variables medido de forma exacta.

Reproducir:
    DATABASE_URL=postgresql://... python -m scripts.experimentos.exp_temporal_lightgbm
    DATABASE_URL=postgresql://... python -m scripts.experimentos.exp_imputacion_municipios
    DATABASE_URL=postgresql://... python -m scripts.experimentos.exp_explicativo_score

NOTA: estos scripts requieren `lightgbm`, que YA NO esta en
requirements.txt (se quito al eliminarlo del pipeline). Para re-correr la
investigacion hay que instalarlo aparte:  pip install lightgbm==4.5.0
Es deliberado: la dependencia no debe volver a la imagen de produccion
solo para poder reproducir un estudio que concluyo que no se use.
"""
