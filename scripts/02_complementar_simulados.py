"""
PASO 2 · Complemento con datos simulados identificados (SISMO AI PERÚ)
=======================================================================
Problema: en los reportes del IGP (2020-2026) casi todas las intensidades
observadas son bajas. Las clases V–VI (76 registros) y VII–XII (3 registros)
no tienen datos suficientes para que KNN aprenda.

Solución (permitida por la guía del reto): agregar registros SIMULADOS,
marcados en la columna 'origen', solo para las clases con pocos datos.

Cómo se simulan:
  * Epicentros y profundidades: se toman de sismos REALES del catálogo IGP (M ≥ 4.5).
  * Localidades: centros poblados urbanos REALES del INEI a menos de 300 km.
  * Magnitud: hipotética, entre 5.0 y 8.5 Mw.
  * Intensidad: ecuación de predicción de intensidad (IPE) de
      Allen, Wald & Worden (2012), versión con distancia hipocentral,
    + una corrección regional de +0.55 grados calibrada con los
      4,544 registros observados del IGP (ver salida del script),
    + variabilidad aleatoria con la desviación estándar de la propia IPE.

Ejecutar:
    python scripts/02_complementar_simulados.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RNG = np.random.default_rng(2026)

META_POR_CLASE = {"V–VI": 900, "VII–XII": 900}   # registros por clase al final
NOMBRE_CLASE = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte"}


def ipe_allen_2012(magnitud, r_hipo):
    """Allen, Wald & Worden (2012), IPE de intensidad Mercalli con distancia hipocentral."""
    c0, c1, c2, c4 = 2.085, 1.428, -1.402, 0.078
    media = c0 + c1 * magnitud + c2 * np.log(np.sqrt(r_hipo ** 2 + 1.0))
    return media + np.where(r_hipo > 50, c4 * np.log(r_hipo / 50.0), 0.0)


def sigma_allen_2012(r_hipo):
    s1, s2, s3 = 0.82, 0.37, 22.9
    return s1 + s2 / (1.0 + (r_hipo / s3) ** 2)


def clase(valor):
    return "I–IV" if valor <= 4 else ("V–VI" if valor <= 6 else "VII–XII")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def main():
    real = pd.read_csv(DATA / "dataset_sismos_intensidad.csv")
    real["origen"] = "IGP observado"
    eventos = pd.read_csv(DATA / "eventos_igp.csv")
    ccpp = pd.read_csv(DATA / "raw" / "inei_centros_poblados.csv")
    urbanos = ccpp[ccpp["tipo"] == "Urban"].dropna(subset=["latitude", "longitude"])

    # 1. Calibración regional: ¿cuánto se desvía la IPE de lo observado por el IGP?
    r_hipo = np.sqrt(real["distancia_km"] ** 2 + real["profundidad_km"] ** 2)
    sesgo = float((real["intensidad_mm"] - ipe_allen_2012(real["magnitud"], r_hipo)).mean())
    pred_cal = np.clip(np.round(ipe_allen_2012(real["magnitud"], r_hipo) + sesgo), 1, 12)
    acierto = (pred_cal.map(clase) == real["intensidad"]).mean()
    print(f"Corrección regional (IGP − IPE): {sesgo:+.2f} grados")
    print(f"Acierto de clase de la IPE calibrada sobre datos reales: {acierto:.1%}")

    # 2. Epicentros reales (M >= 4.5, dentro del Perú o su mar)
    base_ev = eventos[(eventos["magnitud"] >= 4.5) &
                      (eventos["departamento"] != "FUERA DEL PERÚ")].reset_index(drop=True)

    faltan = {c: int(max(0, m - (real["intensidad"] == c).sum())) for c, m in META_POR_CLASE.items()}
    print(f"Registros simulados necesarios: {faltan}")

    lat_u, lon_u = urbanos["latitude"].to_numpy(), urbanos["longitude"].to_numpy()
    nom_u = urbanos["ccpp"].str.title().to_numpy()
    dep_u = urbanos["departamento"].to_numpy()

    simulados, n_evento = [], 0
    while any(v > 0 for v in faltan.values()):
        ev = base_ev.iloc[RNG.integers(len(base_ev))]
        magnitud = round(float(RNG.uniform(5.0, 8.5)), 1)
        dist = haversine_km(ev["latitud"], ev["longitud"], lat_u, lon_u)
        cerca = np.where(dist <= 300)[0]
        if len(cerca) == 0:
            continue
        n_evento += 1
        evento_id = f"SIM-{n_evento:04d}"
        elegidas = RNG.choice(cerca, size=min(6, len(cerca)), replace=False)
        for j in elegidas:
            rh = float(np.sqrt(dist[j] ** 2 + ev["profundidad_km"] ** 2))
            valor = ipe_allen_2012(magnitud, rh) + sesgo + RNG.normal(0, sigma_allen_2012(rh))
            valor = int(np.clip(np.round(valor), 1, 12))
            c = clase(valor)
            if faltan.get(c, 0) <= 0:
                continue
            faltan[c] -= 1
            simulados.append({
                "evento_id": evento_id, "fecha": "simulado",
                "latitud": ev["latitud"], "longitud": ev["longitud"],
                "localidad": nom_u[j], "departamento_localidad": dep_u[j],
                "lat_localidad": round(lat_u[j], 5), "lon_localidad": round(lon_u[j], 5),
                "magnitud": magnitud, "profundidad_km": ev["profundidad_km"],
                "distancia_km": round(float(dist[j]), 1),
                "intensidad_mm": valor, "intensidad": c, "nivel": NOMBRE_CLASE[c],
                "origen": "Simulado (IPE Allen 2012 calibrada)",
            })

    sim = pd.DataFrame(simulados)
    final = pd.concat([real, sim], ignore_index=True)
    final.to_csv(DATA / "dataset_final.csv", index=False)

    print(f"\nEventos simulados: {sim['evento_id'].nunique()} · registros: {len(sim)}")
    print("\nDataset final por clase y origen:")
    print(pd.crosstab(final["intensidad"], final["origen"], margins=True,
                      margins_name="Total").to_string())
    print(f"\nGuardado -> data/dataset_final.csv ({len(final):,} filas)")


if __name__ == "__main__":
    main()
