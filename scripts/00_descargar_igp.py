"""
PASO 0 (opcional) · Actualizar los datos desde la API pública del IGP
=====================================================================
El proyecto ya incluye una copia de los datos del IGP en
data/raw/igp_sismos_reportados.csv (corte: 23/11/2026), obtenida de la copia
pública en GitHub que replica las fuentes oficiales del IGP
(github.com/ifc-chalaco/peru-seismic-dashboard).

Este script descarga directamente los reportes sísmicos del IGP / CENSIS
(https://ultimosismo.igp.gob.pe) para los años indicados y AGREGA los eventos
nuevos al archivo existente. Necesita conexión a internet.

Uso:
    python scripts/00_descargar_igp.py            # año actual
    python scripts/00_descargar_igp.py 2024 2026  # rango de años
"""
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ARCHIVO = BASE / "data" / "raw" / "igp_sismos_reportados.csv"
API = "https://ultimosismo.igp.gob.pe/api/ultimo-sismo/ajaxb/{anio}"


def descargar_anio(anio):
    req = urllib.request.Request(API.format(anio=anio), headers={"User-Agent": "SISMO-AI-PERU/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        datos = json.loads(r.read().decode("utf-8"))
    filas = []
    for d in datos:
        fecha = str(d.get("fecha_local", ""))[:10]
        hora = str(d.get("hora_local", ""))[:8]
        referencia = d.get("referencia") or ""
        departamento = referencia.rsplit(" - ", 1)[-1] if " - " in referencia else None
        filas.append({
            "code": d.get("codigo"),
            "event_date_et": fecha, "event_time_et": hora,
            "lat": d.get("latitud"), "lon": d.get("longitud"),
            "magnitud": d.get("magnitud"),
            "prof": d.get("profundidad"), "profundidad": d.get("profundidad"),
            "intensidad": d.get("intensidad"),
            "departamento": departamento, "referencia": referencia,
        })
    return pd.DataFrame(filas)


def main():
    anios = [int(a) for a in sys.argv[1:]] or [datetime.now().year]
    if len(anios) == 2:
        anios = list(range(anios[0], anios[1] + 1))
    actual = pd.read_csv(ARCHIVO)
    nuevos = []
    for anio in anios:
        try:
            df = descargar_anio(anio)
            print(f"{anio}: {len(df)} reportes descargados del IGP")
            nuevos.append(df)
        except Exception as e:  # sin internet, API caída, etc.
            print(f"{anio}: no se pudo descargar ({e})")
    if not nuevos:
        print("No se descargó nada. Se mantiene la copia incluida en el proyecto.")
        return
    nuevos = pd.concat(nuevos, ignore_index=True)
    nuevos = nuevos[~nuevos["code"].astype(str).isin(actual["code"].astype(str))]
    combinado = pd.concat([nuevos, actual], ignore_index=True)
    combinado.to_csv(ARCHIVO, index=False)
    print(f"Eventos nuevos agregados: {len(nuevos)} · total: {len(combinado):,}")
    print("Ahora vuelve a ejecutar los pasos 1, 2 y 3.")


if __name__ == "__main__":
    main()
