"""Parte 3 -- homogeneidad de la cobertura (solo lectura sobre el snapshot local).

(a) originvar por anio de las coberturas para los municipios con serie propia.
(b) contraste con cifras SSPD 2023 (SUI, urbanas) citadas por nombre en el
    Informe Nacional de Coberturas AAA 2023, contra el valor de la BD 2023.
(c) comparacion de distribuciones: municipios con cobertura >= 90%.
No modifica nada; conserva 88001 = 7.73 / 2.43 (decision ya tomada).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _comun import DIR_INFORMES, V_ACU, V_ALC, asegurar_dirs, cargar_snapshot, md_tabla  # noqa: E402

SIN_SERIE = ["88001", "94343", "27493"]
# Cifras SSPD 2023 (urbanas, SUI) leidas del PDF; (codigo, nombre, acueducto, alcantarillado). None = no citada.
SSPD = [
    ("05480", "Mutata", 0.87, 0.87), ("13160", "Cantagallo", 0.43, None), ("19318", "Guapi", 14.85, None),
    ("88001", "San Andres", 7.73, 2.43), ("88564", "Providencia", None, 0.90), ("47030", "Algarrobo", None, 5.92),
    ("18860", "Valparaiso", None, 8.60), ("47570", "Nueva Granada", None, 99.93), ("47720", "Zapayan", None, 95.68),
    ("70230", "Chalan", None, 94.82), ("44560", "Manaure", None, 80.78), ("44847", "Uribia", None, 74.94),
]


def main():
    asegurar_dirs()
    filas, zonas, _ = cargar_snapshot()
    z = set(zonas.codigo) - set(SIN_SERIE)
    md = ["# Parte 3 -- Homogeneidad de la cobertura (88001)\n"]
    f = filas[filas.namevar.isin([V_ACU, V_ALC]) & filas.codigodivipola.isin(z)]
    t = f.groupby(["namevar", "yearm", "originvar"]).size().reset_index(name="n")
    t.to_csv(DIR_INFORMES / "H_originvar_por_anio.csv", index=False)
    md.append("## (a) originvar por anio (municipios con serie propia, 1,101)\n")
    for var in (V_ACU, V_ALC):
        p = t[t.namevar == var].pivot_table(index="yearm", columns="originvar", values="n", fill_value=0)
        p["total"] = p.sum(axis=1)
        md.append(f"\n**{var}**\n\n" + md_tabla(p.reset_index(), 0))
    real = f[f.real]
    md.append(f"\nOrigenes reales distintos en cobertura: {sorted(real.originvar.unique())}; "
              f"el 100% de las filas reales ({len(real)}) es 'TerriData'.\n")

    # (b) contraste con SSPD urbano 2023
    r23 = f[(f.yearm == 2023) & f.real].pivot_table(index="codigodivipola", columns="namevar", values="valuevar")
    r24 = f[(f.yearm == 2024) & f.real].pivot_table(index="codigodivipola", columns="namevar", values="valuevar")
    filas_c = []
    for c, n, a, l in SSPD:
        if c in SIN_SERIE:
            continue
        filas_c.append({"codigo": c, "municipio": n,
                        "SSPD_urb_acu": a, "BD_acu_2023": r23[V_ACU].get(c) if c in r23.index else np.nan,
                        "SSPD_urb_alc": l, "BD_alc_2023": r23[V_ALC].get(c) if c in r23.index else np.nan})
    c = pd.DataFrame(filas_c)
    c["dif_acu"] = c.BD_acu_2023 - c.SSPD_urb_acu
    c["dif_alc"] = c.BD_alc_2023 - c.SSPD_urb_alc
    c.to_csv(DIR_INFORMES / "H_contraste_sspd_2023.csv", index=False)
    md.append("\n## (b) Contraste puntual: SSPD 2023 urbano vs BD (TerriData REC) 2023\n")
    md.append(md_tabla(c, 2))
    dd = pd.concat([c.dif_acu.dropna(), c.dif_alc.dropna()])
    md.append(f"\nDiferencia BD - SSPD en {len(dd)} comparaciones: mediana {dd.median():.1f} pp, |dif| mediana {dd.abs().median():.1f} pp, "
              f"max |dif| {dd.abs().max():.1f} pp; coinciden (<=3 pp) en {(dd.abs() <= 3).sum()} de {len(dd)}.\n")

    # (c) distribucion: >=90 %
    n_acu = r23[V_ACU].dropna()
    n_alc = r23[V_ALC].dropna()
    md.append("\n## (c) Distribucion 2023: BD vs SSPD urbano\n")
    comp = pd.DataFrame([
        {"medida": "acueducto >= 90%", "SSPD_urbano_2023": "668 de 1,102 (60.6%)", "BD_2023": f"{int((n_acu >= 90).sum())} de {len(n_acu)} ({100 * (n_acu >= 90).mean():.1f}%)"},
        {"medida": "alcantarillado >= 90% ", "SSPD_urbano_2023": "514 de 1,102 (46.6%)", "BD_2023": f"{int((n_alc >= 90).sum())} de {len(n_alc)} ({100 * (n_alc >= 90).mean():.1f}%)"},
        {"medida": "acueducto mediana", "SSPD_urbano_2023": "no publicada en el texto", "BD_2023": f"{n_acu.median():.1f}"},
    ])
    md.append(md_tabla(comp, 1))
    md.append("\nRural (SSPD 2023): 400 municipios con cobertura rural de acueducto <= 30% (36.3%); "
              f"en la BD, acueducto <= 30%: {int((n_acu <= 30).sum())} de {len(n_acu)} ({100 * (n_acu <= 30).mean():.1f}%).\n")
    (DIR_INFORMES / "H_homogeneidad.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
