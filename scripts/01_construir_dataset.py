"""
PASO 1 · Construcción del dataset evento-localidad (SISMO AI PERÚ)
====================================================================
Fuente principal: reportes sísmicos del Instituto Geofísico del Perú (IGP / CENSIS).
Cada reporte del IGP indica la intensidad observada (escala de Mercalli
modificada) en una o varias localidades, por ejemplo:

    "III-IV Ambo, Huánuco, Cerro de Pasco; II Panao, Acomayo"

Este script:
  1. Lee el catálogo del IGP (data/raw/igp_sismos_reportados.csv).
  2. Separa cada reporte en registros "evento-localidad" con su intensidad.
  3. Ubica cada localidad con las coordenadas del INEI (centros poblados).
  4. Calcula la distancia epicentral (fórmula de Haversine).
  5. Agrupa la intensidad en 3 clases: I–IV (Baja), V–VI (Moderada), VII–XII (Fuerte).
  6. Guarda data/dataset_sismos_intensidad.csv y data/eventos_igp.csv.

Ejecutar desde la carpeta del proyecto:
    python scripts/01_construir_dataset.py
"""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RAW = BASE / "data" / "raw"
OUT = BASE / "data"

ROMANOS = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7,
           "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}
CLASES = ["I–IV", "V–VI", "VII–XII"]
NOMBRE_CLASE = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte"}

# Nombres que aparecen en los reportes con otra forma en el INEI
ALIAS = {
    "PROVINCIA CONSTITUCIONAL DEL CALLAO": "CALLAO",
    "CERRO DE PASCO": "CHAUPIMARCA",
    "PUERTO MALDONADO": "TAMBOPATA",
    "LA UNION": "LA UNION",
    "SAM MIGUEL": "SAN MIGUEL",
}


def normalizar(texto):
    """Mayúsculas, sin tildes ni espacios dobles."""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto.upper()).strip()


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def grado_a_numero(grado):
    """'III-IV' -> 4 (se toma el valor superior del rango reportado)."""
    partes = [p.strip() for p in grado.split("-")]
    valores = [ROMANOS[p] for p in partes if p in ROMANOS]
    return max(valores) if valores else None


def clase_intensidad(valor):
    if valor <= 4:
        return "I–IV"
    if valor <= 6:
        return "V–VI"
    return "VII–XII"


PATRON_GRADO = re.compile(r"(?<![A-Za-z])([IVX]+(?:\s*-\s*[IVX]+)?)(?![A-Za-z])")


def separar_intensidades(texto):
    """'III-IV Huacho, Huaral III Lima' -> [(4,'Huacho'),(4,'Huaral'),(3,'Lima')]"""
    texto = str(texto).replace("lII", "III").replace("lll", "III")
    texto = re.sub(r"\([^)]*\)", "", texto)           # quita '(Bolivia)'
    marcas = list(PATRON_GRADO.finditer(texto))
    resultado = []
    for i, m in enumerate(marcas):
        valor = grado_a_numero(m.group(1).replace(" ", ""))
        fin = marcas[i + 1].start() if i + 1 < len(marcas) else len(texto)
        lugares = re.split(r"[,;/]| y ", texto[m.end():fin])
        for lugar in lugares:
            lugar = lugar.strip(" .-")
            if valor and len(lugar) >= 3:
                resultado.append((valor, lugar))
    return resultado


def cargar_nomenclator():
    """Diccionario NOMBRE -> lista de (lat, lon, departamento) desde el INEI."""
    ccpp = pd.read_csv(RAW / "inei_centros_poblados.csv", dtype=str)
    dist = pd.read_csv(RAW / "inei_distritos.csv", dtype=str)
    filas = []
    # Centros poblados urbanos (ciudades, pueblos)
    urb = ccpp[ccpp["tipo"] == "Urban"]
    filas += [(r.ccpp, r.latitude, r.longitude, r.departamento, 0) for r in urb.itertuples()]
    # Capitales y nombres de distritos
    filas += [(r.distrito, r.latitude, r.longitude, r.departamento, 1) for r in dist.itertuples()]
    filas += [(r.capital, r.latitude, r.longitude, r.departamento, 1) for r in dist.itertuples()]
    # Centros poblados rurales (último recurso)
    rur = ccpp[ccpp["tipo"] == "Rural"]
    filas += [(r.ccpp, r.latitude, r.longitude, r.departamento, 2) for r in rur.itertuples()]

    nom = {}
    for nombre, lat, lon, dep, prioridad in filas:
        try:
            lat, lon = float(lat), float(lon)
        except (TypeError, ValueError):
            continue
        nom.setdefault(normalizar(nombre), []).append((lat, lon, normalizar(dep), prioridad))
    return nom


def ubicar(lugar, lat_ev, lon_ev, dep_ev, nom):
    clave = ALIAS.get(normalizar(lugar), normalizar(lugar))
    candidatos = nom.get(clave)
    if not candidatos:
        return None
    mejor_prioridad = min(c[3] for c in candidatos)
    candidatos = [c for c in candidatos if c[3] == mejor_prioridad]
    # Si hay homónimos, preferir el del mismo departamento del evento
    mismo_dep = [c for c in candidatos if c[2] == dep_ev]
    if mismo_dep:
        candidatos = mismo_dep
    # y entre ellos, el más cercano al epicentro
    distancias = [haversine_km(lat_ev, lon_ev, c[0], c[1]) for c in candidatos]
    k = int(np.argmin(distancias))
    return candidatos[k][0], candidatos[k][1], candidatos[k][2], distancias[k]


def asignar_departamento(eventos):
    """Departamento del epicentro usando los límites departamentales (GeoJSON)."""
    import json
    from matplotlib.path import Path as Poligono
    geo = json.loads((BASE / "static" / "data_peru_departamentos.geojson").read_text("utf-8"))
    puntos = eventos[["longitud", "latitud"]].to_numpy()
    resultado = np.array([None] * len(eventos), dtype=object)
    for f in geo["features"]:
        nombre = normalizar(f["properties"]["NOMBDEP"])
        g = f["geometry"]
        poligonos = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
        for pol in poligonos:
            dentro = Poligono(np.array(pol[0])).contains_points(puntos)
            resultado[dentro & (resultado == None)] = nombre  # noqa: E711
    # Fuera de los polígonos: mar peruano (al oeste de la costa) o territorio extranjero
    vertices = np.vstack([np.array(pol[0]) for f in geo["features"]
                          for pol in (f["geometry"]["coordinates"]
                                      if f["geometry"]["type"] == "MultiPolygon"
                                      else [f["geometry"]["coordinates"]])])
    franjas = np.round(vertices[:, 1] * 4) / 4                 # franjas de 0.25° de latitud
    costa = pd.Series(vertices[:, 0]).groupby(franjas).min()   # longitud de la costa
    for i in np.where(resultado == None)[0]:  # noqa: E711
        lon, lat = puntos[i]
        franja = round(lat * 4) / 4
        en_mar = franja in costa.index and lon < costa[franja]
        resultado[i] = "MAR PERUANO" if en_mar else "FUERA DEL PERÚ"
    return resultado


def main():
    igp = pd.read_csv(RAW / "igp_sismos_reportados.csv")
    print(f"Eventos en el catálogo IGP: {len(igp):,}")
    # En el catálogo histórico (1960-2019) la profundidad viene en la columna 'profundidad'
    igp["prof"] = igp["prof"].fillna(pd.to_numeric(igp["profundidad"], errors="coerce"))

    # ---- Tabla de eventos (para el dashboard y el mapa) ----
    eventos = pd.DataFrame({
        "evento_id": igp["code"].astype(str),
        "fecha": igp["event_date_et"],
        "hora": igp["event_time_et"],
        "latitud": igp["lat"],
        "longitud": igp["lon"],
        "magnitud": igp["magnitud"],
        "profundidad_km": igp["prof"],
        "departamento": igp["departamento"],
        "referencia": igp["referencia"],
        "intensidad_reportada": igp["intensidad"],
    }).dropna(subset=["latitud", "longitud", "magnitud", "profundidad_km"])
    eventos["departamento"] = asignar_departamento(eventos)
    eventos["codigo_igp"] = eventos["evento_id"]
    sin_codigo = eventos["evento_id"].isin(["nan", "None"])
    eventos.loc[sin_codigo, "evento_id"] = [f"IGP-H{i:05d}" for i in range(sin_codigo.sum())]
    eventos.to_csv(OUT / "eventos_igp.csv", index=False)
    print(f"Eventos válidos guardados: {len(eventos):,} -> data/eventos_igp.csv")

    # ---- Registros evento-localidad con intensidad observada ----
    nom = cargar_nomenclator()
    con_int = eventos[eventos["intensidad_reportada"].fillna("").str.strip() != ""]
    registros, no_ubicados = [], []
    for ev in con_int.itertuples():
        for valor, lugar in separar_intensidades(ev.intensidad_reportada):
            pos = ubicar(lugar, ev.latitud, ev.longitud, ev.departamento, nom)
            if pos is None:
                no_ubicados.append(lugar)
                continue
            lat_l, lon_l, dep_l, dist = pos
            registros.append({
                "evento_id": ev.evento_id, "fecha": ev.fecha,
                "latitud": ev.latitud, "longitud": ev.longitud,
                "localidad": lugar.title(), "departamento_localidad": dep_l,
                "lat_localidad": round(lat_l, 5), "lon_localidad": round(lon_l, 5),
                "magnitud": ev.magnitud, "profundidad_km": ev.profundidad_km,
                "distancia_km": round(float(dist), 1),
                "intensidad_mm": valor,
                "intensidad": clase_intensidad(valor),
            })

    ds = pd.DataFrame(registros)
    # Limpieza: quitar duplicados y ubicaciones incoherentes (> 800 km del epicentro)
    antes = len(ds)
    ds = ds.drop_duplicates(subset=["evento_id", "localidad"])
    ds = ds[ds["distancia_km"] <= 800]
    ds["nivel"] = ds["intensidad"].map(NOMBRE_CLASE)
    ds.to_csv(OUT / "dataset_sismos_intensidad.csv", index=False)

    print(f"\nEventos con intensidad reportada: {len(con_int):,}")
    print(f"Registros evento-localidad: {antes:,} -> tras limpieza: {len(ds):,}")
    print(f"Localidades no ubicadas (descartadas): {len(no_ubicados):,} "
          f"({len(set(no_ubicados))} nombres distintos)")
    print("\nDistribución de clases:")
    print(ds["intensidad"].value_counts().reindex(CLASES).to_string())
    print(f"\nDataset final -> data/dataset_sismos_intensidad.csv ({len(ds):,} filas, "
          f"{ds['evento_id'].nunique():,} eventos)")


if __name__ == "__main__":
    main()
