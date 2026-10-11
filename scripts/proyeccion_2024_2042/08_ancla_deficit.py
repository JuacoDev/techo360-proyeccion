"""Parte 4 -- revision del ancla de deficit 2024 (SOLO REPORTE; no cambia filas ni el CSV).

El ancla vigente (ingestion/proyeccion_2024.py) es MULTIPLICATIVA:
    v2024 = min(100, v2018 * (tasa2024/tasa2018))   con tasas DANE nacionales:
    cuantitativo x1.36 (5.0 -> 6.8), cualitativo x0.7435 (26.9 -> 20.0), total x0.8401 (31.9 -> 26.8)
Se prueban alternativas para el cuantitativo:
  M0 actual           : min(100, v18*1.36)
  M1 tope max 2018    : min(max18, v18*1.36)
  M2 odds (logit)     : odds' = k*odds, k calibrado para que la MEDIA municipal suba x1.36
  M3 aditiva          : v18 + (1.36-1)*media18, recortada a [0,100]
  M4 saturacion suave : x1.36 igual hasta 60 y, por encima, 60+40*tanh((x-60)/40)
Salidas: informes/I_ancla_deficit.md y CSV.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from _comun import DIR_INFORMES, DIR_SALIDA, V_ACU, V_ALC, V_DL, V_DQ, V_DT, asegurar_dirs, cargar_snapshot, combinar, md_tabla, panel  # noqa: E402

TASAS = {V_DQ: 6.8 / 5.0, V_DL: 20.0 / 26.9, V_DT: 26.8 / 31.9}


def score(acu, alc, dq, lim):
    n = lambda v, k: (v - lim[k][0]) / (lim[k][1] - lim[k][0])  # noqa: E731
    return ((1 - n(acu, "acu")) / 3 + (1 - n(alc, "alc")) / 3 + n(dq, "dq") / 3)


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    z = zonas.set_index("codigo")
    univ = list(zonas.codigo)
    md = ["# Parte 4 -- Ancla de deficit 2024 (solo reporte)\n"]

    real18 = {v: filas[(filas.namevar == v) & (filas.yearm == 2018) & filas.real].set_index("codigodivipola").valuevar for v in TASAS}
    db24 = {v: filas[(filas.namevar == v) & (filas.yearm == 2024)].set_index("codigodivipola").valuevar for v in TASAS}

    # ---------- hallazgo: el ancla ya es multiplicativa ----------
    md.append("## Como se calculo el ancla vigente\n")
    filas_r = []
    for v, tasa in TASAS.items():
        a, b = real18[v], db24[v]
        idx = a.index.intersection(b.index)
        sin_clip = (a[idx] * tasa).clip(0, 100)
        ok = (sin_clip - b[idx]).abs() <= 0.011
        filas_r.append({"variable": v, "tasa": tasa, "n": len(idx), "max2018": a.max(), "ancla = min(100,v18*tasa) en": f"{int(ok.sum())} de {len(idx)}",
                        "umbral 100/tasa": 100 / tasa, "munis con v18 >= umbral": int((a >= 100 / tasa).sum()),
                        "ancla == 100": int((b >= 99.995).sum()), "2018 == 100": int((a >= 99.995).sum()), "ancla == 0": int((b <= 0.005).sum())})
    md.append(md_tabla(pd.DataFrame(filas_r), 4))
    md.append("\nLa premisa de una suma aditiva no se cumple: el ancla es **multiplicativa** (x1.36 en cuantitativo) y se recorta a 100. "
              "Saturan los municipios con deficit cuantitativo 2018 >= 73.5%. Cualitativo (x0.7435) y total (x0.8401) bajan, por lo que no pueden pasar de 100 "
              "ni saturar a partir de valores <= 100.\n")

    # ---------- los 36 ----------
    a18, a24 = real18[V_DQ], db24[V_DQ]
    s36 = a24[(a24 >= 99.995) & a24.index.isin(z.index)].index
    t36 = pd.DataFrame({"codigo": s36, "municipio": z.nombre.reindex(s36).values, "depto": z.depto.reindex(s36).values,
                        "v2018": a18.reindex(s36).values})
    t36["delta_nacional_x"] = TASAS[V_DQ]
    t36["v2018_x1.36_sin_recorte"] = t36.v2018 * TASAS[V_DQ]
    t36["valor_2024_actual"] = a24.reindex(s36).values
    t36["exceso_recortado"] = t36["v2018_x1.36_sin_recorte"] - 100
    t36 = t36.sort_values("v2018", ascending=False)
    t36.to_csv(DIR_INFORMES / "I_36_municipios_cuantitativo_100.csv", index=False)
    md.append(f"## Los {len(t36)} municipios con cuantitativo 2024 = 100% (universo)\n")
    md.append(f"Con 2018 real en [{t36.v2018.min():.1f}, {t36.v2018.max():.1f}]; ninguno es 100 en 2018 (max 2018 = {a18.max():.2f}). "
              f"Por departamento: {t36.depto.value_counts().to_dict()}.\n")
    md.append(md_tabla(t36.head(12)[["codigo", "municipio", "depto", "v2018", "v2018_x1.36_sin_recorte", "valor_2024_actual"]], 2))
    md.append(f"\n(Los {len(t36)} aparecen en el CSV `proyeccion_techo360_2024_2042.csv` como ancla; el exceso recortado va de {t36.exceso_recortado.min():.1f} a {t36.exceso_recortado.max():.1f} pp.)\n")

    # ---------- alternativas para el cuantitativo ----------
    todos18 = a18.copy()
    r = TASAS[V_DQ]
    mean18 = float(todos18.mean())
    max18 = float(todos18.max())
    p = todos18 / 100.0
    odds = (p / (1 - p)).replace([np.inf], np.nan)

    def f_odds(k):
        pp = k * odds / (1 + k * odds)
        return float(100 * pp.mean()) - r * mean18

    k = brentq(f_odds, 1e-3, 1e3)
    alt = pd.DataFrame(index=todos18.index)
    alt["M0_actual"] = (todos18 * r).clip(0, 100)
    alt["M1_tope_max2018"] = np.minimum(todos18 * r, max18)
    alt["M2_odds"] = 100 * (k * odds / (1 + k * odds))
    alt["M3_aditiva_media"] = (todos18 + (r - 1) * mean18).clip(0, 100)
    # M4: x1.36 intacto hasta T=60 y saturacion suave (tanh) por encima: monotona, sin empates, nunca llega a 100
    T = 60.0
    x = todos18 * r
    alt["M4_saturacion_suave"] = np.where(x <= T, x, T + (100 - T) * np.tanh((x - T) / (100 - T)))
    base_check = (alt.M0_actual - a24.reindex(alt.index)).abs().max()
    md.append("## Alternativas para el cuantitativo 2024\n")
    md.append(f"Media municipal 2018 = {mean18:.2f}; max 2018 = {max18:.2f}; k (odds) = {k:.4f}; la nacional ponderada 5.0 -> 6.8 NO es la media municipal sin ponderar. "
              f"(Chequeo: M0 reproduce el ancla vigente, diferencia max {base_check:.3f}.)\n")
    resumen = []
    for m in alt.columns:
        v = alt[m]
        resumen.append({"metodo": m, "media_2024": v.mean(), "munis ==100": int((v >= 99.995).sum()), "munis >=99": int((v >= 99).sum()),
                        "valores distintos en el top 36": int(v.loc[s36].round(2).nunique()), "cambio medio vs actual": float((v - alt.M0_actual).mean()),
                        "max |cambio|": float((v - alt.M0_actual).abs().max())})
    md.append(md_tabla(pd.DataFrame(resumen), 3))

    # ---------- efecto en el score 2024 ----------
    cob = pd.read_csv(DIR_SALIDA / "parcial" / "cobertura.csv", dtype={"codigodivipola": str})
    dfi = pd.read_csv(DIR_SALIDA / "parcial" / "deficit.csv", dtype={"codigodivipola": str})
    est = combinar(filas, pd.concat([cob, dfi]))
    acu = panel(est, V_ACU, [2024])[2024].reindex(univ)
    alc = panel(est, V_ALC, [2024])[2024].reindex(univ)
    dq_act = panel(est, V_DQ, [2024])[2024].reindex(univ)  # incluye 27493 (modelo)
    out = []
    # limites: todos los codigos con ancla (universo + huerfanos), como calcular_limites_base
    s_all = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "proyeccion_2024_2042" / "snapshot" / "score_2024_todos_los_codigos.csv", dtype={"codigodivipola": str})
    lim_ac = (float(acu.min()), float(acu.max()))
    lim_al = (float(alc.min()), float(alc.max()))
    sc0 = None
    for m in alt.columns:
        dq = dq_act.copy()
        mapeo = alt[m].reindex(univ)
        dq.loc[mapeo.dropna().index] = mapeo.dropna()
        lim_dq = (float(min(dq.min(), alt[m].min())), float(max(dq.max(), alt[m].max())))
        sc = score(acu, alc, dq, {"acu": lim_ac, "alc": lim_al, "dq": lim_dq})
        if sc0 is None:
            sc0 = sc.copy()
        rk0 = sc0.rank(ascending=False, method="first")
        rk = sc.rank(ascending=False, method="first")
        mov = (rk0 - rk).abs()
        top50_0 = set(rk0[rk0 <= 50].index)
        top50 = set(rk[rk <= 50].index)
        d = sc - sc0
        out.append({"metodo": m, "score_medio": sc.mean(), "cambio_medio": d.mean(), "max|cambio|": d.abs().max(),
                    "cambio_medio_en_36": d.reindex(s36).mean(), "munis mueven >=10 puestos": int((mov >= 10).sum()),
                    "mueven >=50": int((mov >= 50).sum()), "max_puestos": int(mov.max()), "Spearman vs actual": float(spearmanr(sc0, sc).statistic),
                    "top50 comun con actual": len(top50 & top50_0)})
    md.append("\n## Efecto en el score 2024 (1,104 municipios; coberturas y 27493 del CSV vigente; min-max recalculado)\n")
    md.append(md_tabla(pd.DataFrame(out), 4))
    md.append("\nNota: M0 es la referencia (cambio 0 por construccion).\n")
    pd.DataFrame(out).to_csv(DIR_INFORMES / "I_efecto_score_2024.csv", index=False)
    alt.to_csv(DIR_INFORMES / "I_alternativas_cuantitativo_2024.csv")
    (DIR_INFORMES / "I_ancla_deficit.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
