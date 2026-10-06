"""
Relleno 2025-2026 para cobertura_acueducto / cobertura_alcantarillado
via baseline ingenuo (repetir el ultimo valor REAL observado, 2024).

Decision metodologica (30-ago-2026): NO se usa LightGBM para estas dos
variables porque el modelo entrenado ya demostro (RMSE de test) ser peor
que este mismo baseline -- usar a proposito el metodo que ya se sabe
peor no es defendible academicamente solo porque "es un modelo".

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
