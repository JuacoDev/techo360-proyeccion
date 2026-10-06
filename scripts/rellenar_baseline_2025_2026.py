"""
Relleno 2025-2026 para cobertura_acueducto / cobertura_alcantarillado
via baseline ingenuo (repetir el ultimo valor REAL observado, 2024).

Decision metodologica (30-ago-2026): NO se usa LightGBM para estas dos
variables porque el modelo entrenado ya demostro (RMSE de test) ser peor
que este mismo baseline -- usar a proposito el metodo que ya se sabe
peor no es defendible academicamente solo porque "es un modelo".

>>> NUMEROS EXACTOS DE ESA COMPARACION, conservados aqui (23-sep-2026) <<<
Se conservan en el codigo porque los reportes JSON que los contenian
(models/reportes/) se borraron junto con los modelos huerfanos -- ver el
aviso de abajo. Siguen estando en el historial de git, pero la conclusion
no deberia depender de ir a buscarlos ahi.

Se probaron 3 configuraciones de hiperparametros por variable (A:
num_leaves=15/min_data_in_leaf=20, B: min_data_in_leaf=40, C:
num_leaves=8/min_data_in_leaf=40), particion temporal train 2008-2021 /
val 2022-2023 / test 2024, contra el baseline ingenuo (= valor del año
anterior):

    cobertura_acueducto        val RMSE   test RMSE
      LightGBM A                 19.880     17.588
      LightGBM B                 19.756     17.564
      LightGBM C (la mejor)      18.938     16.745
      baseline ingenuo           22.284     15.598   <-- gana en TEST

    cobertura_alcantarillado   val RMSE   test RMSE
      LightGBM A (la mejor)      16.410     14.999
      LightGBM B                 16.395     15.027
      LightGBM C                 16.430     15.083
      baseline ingenuo           17.930     13.889   <-- gana en TEST

Patron claro y consistente en las dos variables: el modelo le gana al
baseline en VALIDACION pero pierde en TEST, que es el año mas reciente y
el que de verdad importa para proyectar. NINGUNA de las 3 configs supera
al baseline en test en ninguna de las dos variables.

>>> MODELOS HUERFANOS ELIMINADOS (23-sep-2026) <<<
Hasta esta fecha, scripts/entrenar_modelos.py seguia entrenando y
guardando en disco models/lightgbm_cobertura_acueducto.txt y
models/lightgbm_cobertura_alcantarillado.txt, pese a que la decision de
arriba ya los habia descartado. Se verifico que NADA los cargaba: el
unico codigo que tocaba modelos LightGBM en todo el monorepo era el
propio script de entrenamiento (grep de Booster/model_file/MODEL_PATH
sobre los 5 repos), y app/ no importa lightgbm en ningun punto. El
servicio real (app/main.py -> rabbitmq_consumer -> services/proyeccion.py)
solo expone densidad_poblacional, que ya usa regresion lineal.
Se eliminaron el script de entrenamiento, los dos modelos y sus reportes;
este baseline queda como el unico metodo real para estas dos variables.

>>> CONFIRMADO POR UNA INVESTIGACION MAS AMPLIA (23-sep-2026) <<<
Antes de dar por cerrada la eliminacion se probaron CINCO enfoques
distintos de LightGBM (objetivo en cambio, linear_tree, hibrido de
residuos, imputacion entre municipios, uso explicativo con SHAP) sobre un
backtesting de 4 cortes. Para estas dos variables el resultado fue
consistente: LightGBM gana SOLO en RMSE (~10%) y pierde en MAE (~10%), en
error porcentual mediano (~50% peor) y en p90. Como son porcentajes en la
misma escala 0-100, MAE y % mediano son criterios legitimos aqui, asi que
este baseline sigue siendo mejor para el municipio tipico.

Tambien se evaluo si LightGBM podria IMPUTAR los ~53 municipios sin dato:
le gana a la mediana departamental (7-11%), pero con error porcentual
mediano de 25% (acueducto) y 34% (alcantarillado), y con un sesgo
estructural -- los municipios sin dato tienen deficit sistematicamente mas
alto, asi que la imputacion los SOBREESTIMARIA y la validacion cruzada
subestima ese error. No se adopta.

Tablas completas y scripts reproducibles:
scripts/experimentos/investigacion_lightgbm.py

Metodo:
1. Para cada variable, toma el valor real (originvar='TerriData',
   yearm=2024, codigodivipola no vacio) de cada municipio.
2. Para 2025 y 2026: si YA existe una fila con ese mismo nombre de
   municipio + variable + año (real o de relleno), NO se toca -- se
   reporta y se salta. Si no existe, se inserta con el MISMO valor de
   2024 (no se encadena 2025->2026, ambos usan 2024 directamente).
3. intervaloconfianza = RMSE de TEST del baseline ingenuo ya documentado
   (Fase_IA): 15.56 para acueducto, 13.89 para alcantarillado. Es un
   proxy del error historico del metodo, no un intervalo de prediccion
   riguroso -- se guarda documentado como tal.

Por lotes (no fila por fila) para no golpear con ~4000 round-trips
individuales un Postgres de produccion con CPU/RAM limitados (0.35 CPU,
512MB segun docker-compose.yml) sobre una conexion Tailscale con
latencia real. No usa ON CONFLICT (trigger BEFORE INSERT de Jeremy lo
rompe, ver techo360-backend/ingestion/db.py) -- calcula que filas faltan
ANTES de insertar, en memoria, con una sola consulta de lo ya existente.

>>> YA EJECUTADO CONTRA PRODUCCION (30-ago-2026) <<<
Corrida real completada y verificada: 3,918 filas insertadas (980+980
acueducto 2025/2026, 979+979 alcantarillado 2025/2026). Se queda aqui
como registro reproducible del metodo usado, no como job recurrente --
no esta pensado para volver a correr sin ajustar ANIO_BASE/ANIOS_RELLENO
si el proyecto necesita rellenar años distintos en el futuro.
"""
import re
import psycopg2
from psycopg2.extras import execute_values

RMSE_TEST_BASELINE = {
    "cobertura_acueducto": 15.56,
    "cobertura_alcantarillado": 13.89,
}
ANIO_BASE = 2024
ANIOS_RELLENO = [2025, 2026]
ORIGIN_RELLENO = "ultimo-valor-observado"


def obtener_url():
    with open(r"C:\Users\Eridanus_2\OneDrive\Documentos\Visual Studio\techo360\techo360-backend\.env", encoding="utf-8") as fh:
        contenido = fh.read()
    m = re.search(r"^DATABASE_URL=(.+)$", contenido, re.MULTILINE)
    return m.group(1).strip()


def main():
    conn = psycopg2.connect(obtener_url(), connect_timeout=15)
    resumen = {}

    try:
        for variable in ("cobertura_acueducto", "cobertura_alcantarillado"):
            rmse = RMSE_TEST_BASELINE[variable]

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT municipios, codigodivipola, valuevar
                    FROM public.mapcalordb
                    WHERE namevar = %s AND yearm = %s AND originvar = 'TerriData'
                      AND codigodivipola IS NOT NULL AND codigodivipola <> ''
                    """,
                    (variable, ANIO_BASE),
                )
                base = cur.fetchall()

            print(f"\n=== {variable}: {len(base)} municipios con valor real en {ANIO_BASE} (con codigo) ===")

            # Todo lo que YA existe para esta variable en 2025 o 2026,
            # en UNA sola consulta -- para no preguntar fila por fila.
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT municipios, yearm FROM public.mapcalordb WHERE namevar = %s AND yearm = ANY(%s)",
                    (variable, ANIOS_RELLENO),
                )
                ya_existentes = set(cur.fetchall())

            resumen[variable] = {}
            for anio in ANIOS_RELLENO:
                filas_a_insertar = [
                    (municipio, variable, valor, anio, ORIGIN_RELLENO, codigo, rmse)
                    for municipio, codigo, valor in base
                    if (municipio, anio) not in ya_existentes
                ]
                saltadas = len(base) - len(filas_a_insertar)

                if filas_a_insertar:
                    with conn.cursor() as cur:
                        execute_values(
                            cur,
                            """
                            INSERT INTO public.mapcalordb
                                (municipios, namevar, valuevar, yearm, originvar, codigodivipola, intervaloconfianza)
                            VALUES %s
                            """,
                            filas_a_insertar,
                        )
                    conn.commit()

                resumen[variable][anio] = {"insertadas": len(filas_a_insertar), "saltadas_ya_existe": saltadas}
                print(f"  {anio}: {len(filas_a_insertar)} insertadas, {saltadas} ya existian (no tocadas)")

        print("\n" + "=" * 70)
        print("RESUMEN FINAL")
        print("=" * 70)
        for variable, por_anio in resumen.items():
            for anio, r in por_anio.items():
                print(f"{variable} {anio}: insertadas={r['insertadas']}  ya_existian={r['saltadas_ya_existe']}  "
                      f"originvar='{ORIGIN_RELLENO}'  intervaloconfianza={RMSE_TEST_BASELINE[variable]}")

        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM public.mapcalordb;")
            print(f"\nTotal filas en mapcalordb ahora: {cur.fetchone()[0]}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
