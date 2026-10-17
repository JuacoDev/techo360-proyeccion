"""Partes B y E -- generacion de cobertura 2024-2042.

B: 1,101 municipios con serie propia -> metodo unico decidido en la Parte A
   (params_cobertura.json; hoy: ultimo valor real), usando SOLO sus puntos
   reales. Fiabilidad max(30, 60-10*(antiguedad-1)) desde su ultimo punto
   real; intervalo = +-1.96*RMSE(h) del backtest, h = anio - ultimo real.
E: 88001, 94343, 27493 -> semillas_externas.json (cifra documentada o
   semilla), constante 2024-2042, fiabilidad 30, intervalo = banda.

Salidas: data/proyeccion_2024_2042/parcial/cobertura.csv e informes/
B_revision_piojo.md, E_discrepancias.md.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun import (ANIOS, DIR_INFORMES, DIR_SALIDA, V_ACU, V_ALC, accion, asegurar_dirs, cargar_snapshot,  # noqa: E402
                    fiabilidad_antiguedad, fila, indice_existente, md_tabla, panel_reales, rmse_horizonte)

ORIGEN_ULTIMO = "proyeccion-techo360-ultimo-valor"
SEMILLAS = json.loads((Path(__file__).parent / "semillas_externas.json").read_text(encoding="utf-8"))
SIN_SERIE = ["88001", "94343", "27493"]


def generar():
    asegurar_dirs()
    (DIR_SALIDA / "parcial").mkdir(exist_ok=True)
    filas, zonas, vecinos = cargar_snapshot()
    params = json.loads((DIR_INFORMES / "params_cobertura.json").read_text(encoding="utf-8"))
    if params["metodo"] != "ultimo":
        raise SystemExit(f"Este generador implementa el metodo 'ultimo'; params dice '{params['metodo']}'. "
                         "Extender antes de continuar (no se fuerza un metodo no validado).")
    existente = indice_existente(filas)
    salida, ultimo_real = [], {}
    for var in (V_ACU, V_ALC):
        panel = panel_reales(filas, var, 2024)
        rm = params["rmse_por_horizonte"][var]
        for codigo in zonas.codigo:
            if codigo in SIN_SERIE:
                continue
            s = panel.loc[codigo].dropna() if codigo in panel.index else pd.Series(dtype=float)
            if s.empty:
                raise SystemExit(f"{codigo} {var}: sin serie real y no esta en SIN_SERIE")
            ua, uv = int(s.index[-1]), float(s.iloc[-1])
            ultimo_real[(codigo, var)] = (ua, uv, len(s))
            for anio in ANIOS:
                acc = accion(existente, codigo, var, anio)
                if acc is None:
                    continue  # ya hay fila real: no se pisa
                h = anio - ua
                if h < 1:
                    continue  # anio dentro de la serie real sin fila (hueco interno): fuera de alcance
                half = 1.96 * rmse_horizonte(rm, h)
                salida.append(fila(codigo, var, anio, uv, ORIGEN_ULTIMO, fiabilidad_antiguedad(h),
                                   uv - half, uv + half, acc))
    # ---- E: tres municipios sin serie ----
    for codigo in SIN_SERIE:
        for var in (V_ACU, V_ALC):
            cfg = SEMILLAS[codigo][var]
            lo, hi = cfg["banda"]
            for anio in ANIOS:
                acc = accion(existente, codigo, var, anio)
                if acc is None:
                    continue
                salida.append(fila(codigo, var, anio, cfg["valor"], cfg["origen"], 30.0, lo, hi, acc))
    df = pd.DataFrame(salida)
    df = df.sort_values(["nameVar", "codigodivipola", "yearM"]).reset_index(drop=True)
    df.to_csv(DIR_SALIDA / "parcial" / "cobertura.csv", index=False)
    return df, filas, zonas, vecinos, ultimo_real, existente


def informe_piojo(df, filas, ultimo_real):
    c = "08549"
    lineas = ["# Revision manual: Piojo (08549)\n"]
    for var in (V_ACU, V_ALC):
        s = filas[(filas.codigodivipola == c) & (filas.namevar == var)].sort_values("yearm")
        reales = s[s.real][["yearm", "valuevar", "originvar"]]
        prox = s[~s.real][["yearm", "valuevar", "originvar", "intervaloconfianza"]]
        ua, uv, n = ultimo_real[(c, var)]
        lineas.append(f"\n## {var}: {n} puntos reales, ultimo real {ua} = {uv}\n")
        lineas.append("Reales (ultimos 6):\n\n" + md_tabla(reales.tail(6).rename(columns={"yearm": "anio"}), 2))
        lineas.append("\nFilas proyeccion YA en BD:\n\n" + (md_tabla(prox.rename(columns={"yearm": "anio"}), 2) if len(prox) else "(ninguna)"))
        g = df[(df.codigodivipola == c) & (df.nameVar == var)]
        lineas.append("\nGeneradas (primeras 5 y ultima):\n\n" + md_tabla(pd.concat([g.head(5), g.tail(1)])[["yearM", "valuevar", "originvar", "fiabilidad", "intervalo_confianza", "accion"]], 2))
    (DIR_INFORMES / "B_revision_piojo.md").write_text("\n".join(lineas), encoding="utf-8")


def informe_e(filas, zonas, vecinos):
    """E3: semilla/documentado vs mediana de vecinos (contigüidad PostGIS) y del
    departamento, usando el ULTIMO valor real de cada municipio vecino."""
    ult = {}
    for var in (V_ACU, V_ALC):
        p = panel_reales(filas, var, 2024)
        ult[var] = p.apply(lambda r: r.dropna().iloc[-1] if r.notna().any() else np.nan, axis=1)
    dep = zonas.set_index("codigo").depto
    vec = {}
    for a, b in zip(vecinos.a, vecinos.b):
        vec.setdefault(a, set()).add(b)
        vec.setdefault(b, set()).add(a)
    z = zonas.set_index("codigo")
    filas_e = []
    for codigo in SIN_SERIE:
        v = vec.get(codigo, set())
        if not v:  # isla: sin contigüidad -> 3 mas cercanos por centroide
            d = np.hypot(z.lon - z.loc[codigo, "lon"], (z.lat - z.loc[codigo, "lat"]))
            v = set(d.drop(codigo).nsmallest(3).index)
            tipo_v = "3 mas cercanos (sin contiguidad)"
        else:
            tipo_v = f"{len(v)} contiguos"
        for var in (V_ACU, V_ALC):
            cfg = SEMILLAS[codigo][var]
            med_v = float(ult[var].reindex(list(v)).median())
            mismos = [c for c in zonas.codigo if dep[c] == dep[codigo] and c != codigo]
            med_d = float(ult[var].reindex(mismos).median())
            alt = cfg.get("alternativa_semilla", {}).get("valor")
            filas_e.append({"codigo": codigo, "nombre": SEMILLAS[codigo]["nombre"], "variable": var,
                            "modo": cfg["modo"], "valor_usado": cfg["valor"], "alternativa_semilla": alt if alt is not None else "",
                            "mediana_vecinos": med_v, "n_vecinos": len(v), "tipo_vecinos": tipo_v,
                            "mediana_depto": med_d, "dif_vs_vecinos": cfg["valor"] - med_v,
                            "dif_vs_depto": cfg["valor"] - med_d})
    de = pd.DataFrame(filas_e)
    de.to_csv(DIR_INFORMES / "E_discrepancias.csv", index=False)
    (DIR_INFORMES / "E_discrepancias.md").write_text("# Parte E3 -- discrepancias\n\n" + md_tabla(de, 2), encoding="utf-8")
    return de


def main():
    df, filas, zonas, vecinos, ultimo_real, _ = generar()
    informe_piojo(df, filas, ultimo_real)
    de = informe_e(filas, zonas, vecinos)
    print(f"filas cobertura generadas: {len(df)}")
    print(df.groupby(["nameVar", "originvar", "accion"]).size().to_string())
    print(de.to_string(index=False))
    ceros = sum(1 for (c, v), (a, val, n) in ultimo_real.items() if val in (0.0, 100.0))
    print("municipio-variable con ultimo valor real exactamente 0 o 100:", ceros, "de", len(ultimo_real))


if __name__ == "__main__":
    main()
