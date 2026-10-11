"""Utilidades comunes de los scripts de proyeccion 2024-2042 (solo lectura
contra la BD; todo lo demas trabaja sobre un snapshot local).

Rutas: las salidas viven en data/proyeccion_2024_2042/ (data/ ya esta en
.gitignore). Semilla fija para todo lo estocastico.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMILLA = 42
RAIZ = Path(__file__).resolve().parents[2]
DIR_SALIDA = RAIZ / "data" / "proyeccion_2024_2042"
DIR_SNAPSHOT = DIR_SALIDA / "snapshot"
DIR_INFORMES = DIR_SALIDA / "informes"
DIR_ENV_BACKEND = RAIZ.parent / "techo360-backend" / ".env"

EXCLUIDOS = ("00000", "01001")
ANIO_BASE = 2024
ANIO_FIN = 2042
ANIOS = list(range(2024, 2043))

V_ACU = "cobertura_acueducto"
V_ALC = "cobertura_alcantarillado"
V_DQ = "deficit_habitacional_cuantitativo"
V_DL = "deficit_habitacional_cualitativo"
V_DT = "deficit_habitacional_total"
V_TEN = "tenencia_informal_terrenos"
V_DENS = "densidad_poblacional"
VARIABLES_ALCANCE = [V_ACU, V_ALC, V_DQ, V_DL, V_DT, V_TEN]
TODAS = VARIABLES_ALCANCE + [V_DENS]

PREFIJO_PROY = "proyeccion-"
PREFIJO_PROPIO = "proyeccion-techo360-"


def asegurar_dirs():
    for d in (DIR_SNAPSHOT, DIR_INFORMES):
        d.mkdir(parents=True, exist_ok=True)


def es_real(origen: pd.Series) -> pd.Series:
    """Misma regla que el frontend y app/repositories/proyeccion.py."""
    o = origen.fillna("")
    return ~(o.str.startswith(PREFIJO_PROY) | o.str.endswith("-proyeccion"))


def cargar_snapshot():
    """(filas, zonas, vecinos) del snapshot local, ya restringido al universo."""
    filas = pd.read_csv(DIR_SNAPSHOT / "mapcalordb.csv", dtype={"codigodivipola": str})
    zonas = pd.read_csv(DIR_SNAPSHOT / "zonas.csv", dtype={"codigo": str, "depto": str})
    vecinos = pd.read_csv(DIR_SNAPSHOT / "vecinos.csv", dtype=str)
    filas["real"] = es_real(filas["originvar"])
    return filas, zonas, vecinos


def serie(filas, codigo, var, solo_reales=True):
    f = filas[(filas.codigodivipola == codigo) & (filas.namevar == var)]
    if solo_reales:
        f = f[f.real]
    return f.sort_values("yearm")


def panel_reales(filas, var, anio_max=None):
    """DataFrame municipio x anio solo con filas reales."""
    f = filas[(filas.namevar == var) & filas.real]
    if anio_max is not None:
        f = f[f.yearm <= anio_max]
    return f.pivot_table(index="codigodivipola", columns="yearm", values="valuevar", aggfunc="first")


def fiabilidad_antiguedad(antiguedad: int) -> float:
    """max(30, 60 - 10*(antiguedad-1)), pedida como regla por defecto."""
    return float(max(30, 60 - 10 * (antiguedad - 1)))


def md_tabla(df, decimales=3) -> str:
    """Tabla markdown sin depender de tabulate."""
    df = df.reset_index() if df.index.name or isinstance(df.index, pd.MultiIndex) else df
    def f(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else f"{v:.{decimales}f}"
        return str(v)
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    cuerpo = ["| " + " | ".join(f(v) for v in fila) + " |" for fila in df.itertuples(index=False)]
    return "\n".join([cab, sep] + cuerpo)


COLUMNAS_CSV = ["codigodivipola", "nameVar", "yearM", "valuevar", "originvar", "fiabilidad",
                "intervalo_confianza", "accion"]


def indice_existente(filas):
    """{(codigo, var, anio): es_real} de lo que ya hay en la BD (snapshot)."""
    return {(c, v, int(y)): bool(r) for c, v, y, r in
            zip(filas.codigodivipola, filas.namevar, filas.yearm, filas.real)}


def accion(existente, codigo, var, anio):
    """None si ya hay fila REAL (no se pisa nunca); 'update' si hay una fila
    proyeccion-*; 'insert' si no hay fila."""
    k = (codigo, var, int(anio))
    if k not in existente:
        return "insert"
    return None if existente[k] else "update"


def fila(codigo, var, anio, valor, origen, fiab, lo, hi, acc):
    return {"codigodivipola": codigo, "nameVar": var, "yearM": int(anio),
            "valuevar": round(float(np.clip(valor, 0, 100)), 2), "originvar": origen,
            "fiabilidad": round(float(fiab), 1),
            "intervalo_confianza": f"{max(0.0, lo):.2f};{min(100.0, hi):.2f}", "accion": acc}


def rmse_horizonte(params_var, h):
    """RMSE esperado a h anios: empirico si h<=12 (n suficiente), ley de
    potencia ajustada en el backtest si h>12."""
    emp = {int(k): v for k, v in params_var["empirico"].items()}
    if h in emp:
        return emp[h]
    return float(np.exp(params_var["potencia_a"]) * max(h, 1) ** params_var["potencia_b"])


def combinar(filas, generado: pd.DataFrame) -> pd.DataFrame:
    """Estado resultante de mapcalordb tras cargar `generado` (las filas del
    CSV sustituyen a las existentes con la misma clave). Columnas:
    codigodivipola, namevar, yearm, valuevar, originvar, real."""
    base = filas[["codigodivipola", "namevar", "yearm", "valuevar", "originvar", "real"]].copy()
    if generado is None or generado.empty:
        return base
    g = generado.rename(columns={"nameVar": "namevar", "yearM": "yearm"})[
        ["codigodivipola", "namevar", "yearm", "valuevar", "originvar"]].copy()
    g["real"] = False
    clave = ["codigodivipola", "namevar", "yearm"]
    base = base.merge(g[clave].assign(_x=1), on=clave, how="left")
    base = base[base._x.isna()].drop(columns="_x")
    return pd.concat([base, g], ignore_index=True)


def panel(estado: pd.DataFrame, var: str, anios=None) -> pd.DataFrame:
    f = estado[estado.namevar == var]
    p = f.pivot_table(index="codigodivipola", columns="yearm", values="valuevar", aggfunc="first")
    return p if anios is None else p.reindex(columns=anios)
