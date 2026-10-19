"""Backtest de la interpolacion (claude/fase3-4 seccion 4) y verificacion de los huecos (seccion 3).

Solo lectura de un CSV exportado (series de las cinco variables de mapcalordb, con la cabecera
codigodivipola,namevar,yearm,valuevar,originvar) y de la lista de codigos de zonas_municipio.
No se conecta a ninguna base y no escribe nada salvo el informe Markdown que se le pide.

Uso:
    python scripts/proyeccion_2024_2042/10_backtest_interpolacion.py --modo backtest \
        --series <series.csv> --zonas <zonas.csv> --salida <informe.md>
    python scripts/proyeccion_2024_2042/10_backtest_interpolacion.py --modo verificar ...

backtest: para cada municipio con real antes y despues de un año real conocido, se enmascara ese
valor y se estima con (a) ultimo valor, (b) interpolacion lineal y (c) en densidad tambien la
regresion de 4 puntos. RMSE y MAE por variable, global y por distancia al hueco (1, 2, 3+ años).
Regla: la interpolacion se usa donde su RMSE <= el del metodo de comparacion.

verificar: calcula todos los huecos (municipio-año sin fila) con app/services/puntos.py y comprueba
calculado + sin_dato = huecos, los invariantes de la seccion 3 y la idempotencia.
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, RAIZ)

from app.origen import es_proyeccion  # noqa: E402
from app.services import puntos as pt  # noqa: E402

VARIABLES = ["cobertura_acueducto", "cobertura_alcantarillado", "deficit_habitacional_cuantitativo",
             "deficit_habitacional_cualitativo", "densidad_poblacional"]
HUECOS_ESPERADOS = {"cobertura_acueducto": 967, "cobertura_alcantarillado": 968,
                    "deficit_habitacional_cuantitativo": 5521, "deficit_habitacional_cualitativo": 5521,
                    "densidad_poblacional": 103}
EXCLUIDOS = {"00000", "01001"}


def cargar(series_csv, zonas_csv):
    universo = set()
    with open(zonas_csv, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            if fila["codigo_municipio"] not in EXCLUIDOS:
                universo.add(fila["codigo_municipio"])
    series = defaultdict(list)  # (variable, codigo) -> [{"anio","valor","origen"}]
    with open(series_csv, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            valor = float(fila["valuevar"]) if fila["valuevar"] != "" else None
            series[(fila["namevar"], fila["codigodivipola"])].append(
                {"anio": int(fila["yearm"]), "valor": valor, "origen": fila["originvar"]})
    return universo, series


def recortar(variable, valor):
    if variable == "densidad_poblacional":
        return max(0.0, valor)
    return min(100.0, max(0.0, valor))


def metricas(errores):
    if not errores:
        return {"n": 0, "rmse": float("nan"), "mae": float("nan")}
    a = np.asarray(errores, dtype=float)
    return {"n": len(a), "rmse": float(math.sqrt(np.mean(a ** 2))), "mae": float(np.mean(np.abs(a)))}


def backtest(universo, series):
    """{variable: {metodo: {tramo: [errores]}}} con tramo en 'global','d=1','d=2','d=3+'."""
    res = {}
    for variable in ("cobertura_acueducto", "cobertura_alcantarillado", "densidad_poblacional"):
        por_metodo = defaultdict(lambda: defaultdict(list))
        por_metodo_sub = defaultdict(lambda: defaultdict(list))  # subconjunto con regresion disponible
        for codigo in universo:
            reales = pt._reales(series.get((variable, codigo), []))
            for i in range(1, len(reales) - 1):
                (a0, v0), (a, v), (a1, v1) = reales[i - 1], reales[i], reales[i + 1]
                d = a - a0
                tramo = "d=1" if d == 1 else ("d=2" if d == 2 else "d=3+")
                e_ultimo = recortar(variable, v0) - v
                e_interp = recortar(variable, v0 + (v1 - v0) * (a - a0) / (a1 - a0)) - v
                for t in ("global", tramo):
                    por_metodo["ultimo_valor"][t].append(e_ultimo)
                    por_metodo["interpolacion"][t].append(e_interp)
                if variable == "densidad_poblacional" and i >= pt.PUNTOS_REGRESION:
                    ult = reales[i - pt.PUNTOS_REGRESION:i]
                    pend, inter = np.polyfit([x for x, _ in ult], [y for _, y in ult], 1)
                    e_reg = recortar(variable, float(pend * a + inter)) - v
                    for t in ("global", tramo):
                        por_metodo_sub["regresion_4"][t].append(e_reg)
                        por_metodo_sub["interpolacion"][t].append(e_interp)
                        por_metodo_sub["ultimo_valor"][t].append(e_ultimo)
        res[variable] = {"todos": por_metodo, "subconjunto_regresion": por_metodo_sub}
    return res


def decision(res):
    """Interpolacion permitida si su RMSE global <= el del metodo de comparacion (cobertura: ultimo
    valor; densidad: regresion de 4 puntos, sobre el mismo subconjunto donde existe)."""
    out = {}
    for variable, r in res.items():
        if variable == "densidad_poblacional":
            comp, base = "regresion_4", r["subconjunto_regresion"]
        else:
            comp, base = "ultimo_valor", r["todos"]
        i = metricas(base["interpolacion"]["global"])["rmse"]
        c = metricas(base[comp]["global"])["rmse"]
        out[variable] = (i <= c, i, c, comp)
    return out


def escribir_backtest(res, ruta):
    dec = decision(res)
    L = ["# Informe J -- backtest de la interpolacion lineal (3.1/3.2)", "",
         "Regla (claude/fase3-4 seccion 4): la interpolacion se usa donde su RMSE <= el del metodo de comparacion.", "",
         "Metodo: para cada municipio con valor real antes y despues de un año real conocido se enmascara ese valor "
         "y se estima con (a) ultimo valor real, (b) interpolacion lineal entre los reales vecinos y, en densidad, "
         "(c) regresion lineal de 4 puntos sobre los 4 reales anteriores. Todo recortado ([0,100] en coberturas, >=0 en densidad). "
         "`d` = distancia en años entre el valor enmascarado y el real anterior. Datos: series reales de mapcalordb exportadas en solo lectura (17/10/2026); "
         "los 'reales' son los de originvar sin patron de proyeccion.", ""]
    for variable, r in res.items():
        L += [f"## {variable}", "", "| conjunto | metodo | d | n | RMSE | MAE |", "|---|---|---|---|---|---|"]
        for conjunto, nombre in (("todos", "todos los puntos"), ("subconjunto_regresion", "con regresion disponible (>=4 reales previos)")):
            base = r[conjunto]
            if not base:
                continue
            for metodo in ("ultimo_valor", "interpolacion", "regresion_4"):
                if metodo not in base:
                    continue
                for t in ("global", "d=1", "d=2", "d=3+"):
                    m = metricas(base[metodo].get(t, []))
                    L.append(f"| {nombre} | {metodo} | {t} | {m['n']} | {m['rmse']:.3f} | {m['mae']:.3f} |")
        usa, i, c, comp = dec[variable]
        L += ["", f"**Decision:** RMSE interpolacion {i:.3f} vs {comp} {c:.3f} -> interpolacion "
              f"{'PERMITIDA' if usa else 'NO permitida (rige el metodo validado)'}.", ""]
    L += ["## Deficit", "", "Cada municipio tiene un solo punto real (2018); no hay backtest posible. La interpolacion "
          "2018->2024 es decision de Juan (15:10) y se aplica sin validacion estadistica.", ""]
    with open(ruta, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")
    return dec


def verificar(universo, series, permitida):
    filas = {}
    for variable in VARIABLES:
        desde, hasta = pt.RANGO_ANIOS[variable]
        cuenta = Counter()
        motivos = Counter()
        sospechosos = 0
        problemas = []
        huecos = 0
        for codigo in sorted(universo):
            serie = series.get((variable, codigo), [])
            presentes = {f["anio"] for f in serie}
            for anio in range(desde, hasta + 1):
                if anio in presentes:
                    continue
                huecos += 1
                r = pt.calcular_punto(variable, codigo, anio, serie, True, permitida)
                r2 = pt.calcular_punto(variable, codigo, anio, serie, True, permitida)
                if r != r2:
                    problemas.append("no idempotente")
                cuenta[r["estado"]] += 1
                if r["estado"] == "sin_dato":
                    motivos[r["motivo"]] += 1
                    continue
                if r["es_real"] or not es_proyeccion(r["origen"]):
                    problemas.append("calculado presentado como real o sin origen de proyeccion")
                v = r["valor"]
                if (variable == "densidad_poblacional" and v < 0) or (variable != "densidad_poblacional" and not 0 <= v <= 100):
                    problemas.append("fuera de rango")
        filas[variable] = {"huecos": huecos, "calculado": cuenta["calculado"], "sin_dato": cuenta["sin_dato"],
                           "motivos": dict(motivos), "problemas": Counter(problemas)}
    return filas


def escribir_verificacion(filas, ruta, dec):
    L = ["", "## Verificacion de huecos (seccion 3 de claude/fase3-4)", "",
         "Huecos = municipio-año del universo (1.104) dentro del rango de la variable SIN fila en mapcalordb (cualquier origen).", "",
         "| variable | huecos | esperado | calculado | sin_dato | motivos de sin_dato | calculado+sin_dato = huecos | % valor_sospechoso sobre calculados | problemas |",
         "|---|---|---|---|---|---|---|---|---|"]
    for v, r in filas.items():
        sosp = r["motivos"].get("valor_sospechoso", 0)
        pct = 100.0 * sosp / r["calculado"] if r["calculado"] else 0.0
        ok = r["calculado"] + r["sin_dato"] == r["huecos"] == HUECOS_ESPERADOS[v]
        L.append(f"| {v} | {r['huecos']} | {HUECOS_ESPERADOS[v]} | {r['calculado']} | {r['sin_dato']} | {r['motivos']} | "
                 f"{'SI' if ok else 'NO'} | {pct:.2f} % | {dict(r['problemas']) or 'ninguno'} |")
    with open(ruta, "a", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modo", choices=("backtest", "verificar", "ambos"), default="ambos")
    ap.add_argument("--series", required=True)
    ap.add_argument("--zonas", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    universo, series = cargar(a.series, a.zonas)
    print(f"universo={len(universo)} pares_variable_municipio={len(series)}")
    res = backtest(universo, series)
    dec = decision(res)
    for v, (usa, i, c, comp) in dec.items():
        print(f"{v}: interpolacion RMSE {i:.3f} vs {comp} {c:.3f} -> {'PERMITIDA' if usa else 'NO'}")
    if a.modo in ("backtest", "ambos"):
        escribir_backtest(res, a.salida)
    if a.modo in ("verificar", "ambos"):
        permitida = {v: d[0] for v, d in dec.items()}
        filas = verificar(universo, series, permitida)
        for v, r in filas.items():
            print(v, r["huecos"], r["calculado"], r["sin_dato"], r["motivos"], dict(r["problemas"]))
        escribir_verificacion(filas, a.salida, dec)


if __name__ == "__main__":
    main()
