"""
SISMO AI PERÚ · Aplicación web (Flask)
=======================================
Sistema de análisis y estimación de intensidad sísmica con KNN.
Datos: Instituto Geofísico del Perú (IGP) + complemento simulado identificado.

Ejecutar:
    python app.py
y abrir en el navegador:  http://127.0.0.1:5000
"""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELO = BASE / "modelo"

app = Flask(__name__)
app.json.ensure_ascii = False
app.json.sort_keys = False      # respeta el orden de las clases

# ─────────────────────────── Carga de datos y modelo ───────────────────────────
def cargar_modelo():
    """Carga el modelo entrenado. Si falta o fue creado con otra versión de
    pandas/scikit-learn, lo vuelve a entrenar automáticamente (paso 3)."""
    try:
        return joblib.load(MODELO / "knn_sismo.joblib")
    except Exception as error:  # archivo ausente o incompatible
        print(f"  No se pudo cargar el modelo ({type(error).__name__}). Reentrenando...")
        import runpy
        runpy.run_path(str(BASE / "scripts" / "03_entrenar_modelo.py"), run_name="__main__")
        return joblib.load(MODELO / "knn_sismo.joblib")


paquete = cargar_modelo()
MODELOS = paquete["modelos"]                 # {k: Pipeline(StandardScaler + KNN)}
MEJOR_K = paquete["mejor_k"]
VARIABLES = paquete["variables"]
CLASES = paquete["clases"]
TRAIN = pd.DataFrame(paquete["train"])   # registros de entrenamiento (para mostrar vecinos)
METRICAS = json.loads((MODELO / "metricas.json").read_text(encoding="utf-8"))
NOMBRES = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte"}

EVENTOS = pd.read_csv(DATA / "eventos_igp.csv")
EVENTOS["anio"] = EVENTOS["fecha"].str[:4].astype(int)
DATASET = pd.read_csv(DATA / "dataset_final.csv")

DEPS = pd.read_csv(DATA / "raw" / "inei_departamentos.csv")
LOCALIDADES = [
    {"departamento": r.departamento.title(), "capital": r.capital,
     "lat": round(float(r.latitude), 4), "lon": round(float(r.longitude), 4)}
    for r in DEPS.itertuples()
]

LIMITES = {"magnitud": (3.0, 9.5), "profundidad_km": (0.0, 700.0),
           "distancia_km": (0.0, 1000.0)}
RANGO_ENTRENAMIENTO = {v: (float(TRAIN[v].min()), float(TRAIN[v].max())) for v in VARIABLES}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


# ─────────────────────────────────── Páginas ───────────────────────────────────
@app.route("/")
def inicio():
    return render_template("index.html")


# ───────────────────────────────────── API ─────────────────────────────────────
@app.route("/api/resumen")
def api_resumen():
    ev = EVENTOS
    bins = pd.cut(ev["magnitud"], [-1, 3.99, 5.99, 20], labels=["< 4.0", "4.0 – 5.9", "≥ 6.0"])
    tierra = ev[~ev["departamento"].isin(["MAR PERUANO", "FUERA DEL PERÚ"])]
    top = tierra["departamento"].value_counts().head(5)
    por_anio = ev.groupby("anio").size()
    return jsonify({
        "total_eventos": int(len(ev)),
        "magnitud_promedio": round(float(ev["magnitud"].mean()), 2),
        "profundidad_promedio": round(float(ev["profundidad_km"].mean()), 1),
        "regiones": int(tierra["departamento"].nunique()),
        "periodo": [str(ev["fecha"].min()), str(ev["fecha"].max())],
        "dist_magnitud": {str(k): int(v) for k, v in bins.value_counts().sort_index().items()},
        "top_departamentos": {k.title(): int(v) for k, v in top.items()},
        "por_anio": {int(k): int(v) for k, v in por_anio.items()},
        "huanuco": {
            "eventos": int((ev["departamento"] == "HUANUCO").sum()),
            "registros_intensidad": int((DATASET["departamento_localidad"] == "HUANUCO").sum()),
        },
        "ultimo": ev.sort_values("fecha").iloc[-1][
            ["fecha", "hora", "magnitud", "profundidad_km", "referencia"]].fillna("").to_dict(),
        "metricas": METRICAS,
        "dataset": {
            "total": int(len(DATASET)),
            "observados": int((DATASET["origen"] == "IGP observado").sum()),
            "simulados": int((DATASET["origen"] != "IGP observado").sum()),
            "por_clase": DATASET.groupby(["intensidad", "origen"]).size()
                                .unstack(fill_value=0).to_dict(orient="index"),
        },
        "rango_entrenamiento": RANGO_ENTRENAMIENTO,
    })


@app.route("/api/eventos")
def api_eventos():
    ev = EVENTOS
    min_mag = request.args.get("min_mag", type=float, default=0)
    desde = request.args.get("desde", type=int, default=1960)
    hasta = request.args.get("hasta", type=int, default=2100)
    dep = request.args.get("dep", default="")
    filtro = (ev["magnitud"] >= min_mag) & ev["anio"].between(desde, hasta)
    if dep:
        filtro &= ev["departamento"] == dep.upper()
    sel = ev[filtro]
    datos = sel[["latitud", "longitud", "magnitud", "profundidad_km", "fecha",
                 "departamento", "referencia", "intensidad_reportada"]]
    datos = datos.astype(object).where(datos.notna(), None)
    return jsonify({"total": int(len(sel)), "eventos": datos.values.tolist()})


@app.route("/api/muestra")
def api_muestra():
    """Muestra estratificada del dataset para el gráfico de dispersión."""
    muestra = pd.concat([g.sample(min(len(g), 700), random_state=1)
                         for _, g in DATASET.groupby("intensidad")])
    return jsonify(muestra[["magnitud", "distancia_km", "intensidad"]].values.tolist())


@app.route("/api/localidades")
def api_localidades():
    return jsonify(LOCALIDADES)


@app.route("/api/departamentos")
def api_departamentos():
    return jsonify(sorted(EVENTOS["departamento"].unique().tolist()))


@app.route("/api/predecir", methods=["POST"])
def api_predecir():
    datos = request.get_json(silent=True) or {}
    errores, valores = [], {}

    # Si se envían coordenadas de epicentro y localidad, se calcula la distancia
    if all(datos.get(c) not in (None, "") for c in ("lat_epi", "lon_epi", "lat_loc", "lon_loc")):
        try:
            datos["distancia_km"] = round(haversine_km(
                float(datos["lat_epi"]), float(datos["lon_epi"]),
                float(datos["lat_loc"]), float(datos["lon_loc"])), 1)
        except (TypeError, ValueError):
            errores.append("Coordenadas no válidas.")

    # Validación de entradas
    for var, (mn, mx) in LIMITES.items():
        try:
            v = float(datos.get(var))
            if not math.isfinite(v):
                raise ValueError
        except (TypeError, ValueError):
            errores.append(f"'{var}' debe ser un número.")
            continue
        if not mn <= v <= mx:
            errores.append(f"'{var}' debe estar entre {mn:g} y {mx:g}.")
        valores[var] = v
    try:
        k = int(datos.get("k", MEJOR_K))
    except (TypeError, ValueError):
        k = -1
    if k not in MODELOS:
        errores.append(f"K debe ser uno de {sorted(MODELOS)}.")
    if errores:
        return jsonify({"ok": False, "errores": errores}), 400

    # Predicción con el modelo KNN entrenado
    modelo = MODELOS[k]
    X = pd.DataFrame([[valores[v] for v in VARIABLES]], columns=VARIABLES)
    prediccion = modelo.predict(X)[0]
    proba = modelo.predict_proba(X)[0]
    clases_modelo = list(modelo.classes_)

    # Vecinos más cercanos (en el espacio escalado)
    X_esc = modelo.named_steps["escalador"].transform(X)
    dist, idx = modelo.named_steps["knn"].kneighbors(X_esc, n_neighbors=k)
    vecinos = []
    for d, i in zip(dist[0], idx[0]):
        f = TRAIN.iloc[i]
        vecinos.append({
            "localidad": f["localidad"], "fecha": f["fecha"],
            "magnitud": float(f["magnitud"]), "profundidad_km": float(f["profundidad_km"]),
            "distancia_km": float(f["distancia_km"]), "intensidad": f["intensidad"],
            "nivel": NOMBRES[f["intensidad"]], "origen": f["origen"],
            "distancia_escalada": round(float(d), 3),
        })
    votos = {c: sum(1 for v in vecinos if v["intensidad"] == c) for c in CLASES}

    avisos = [f"'{v}' = {valores[v]:g} está fuera del rango de entrenamiento "
              f"({mn:g} – {mx:g}); la estimación es menos confiable."
              for v, (mn, mx) in RANGO_ENTRENAMIENTO.items()
              if not mn <= valores[v] <= mx]

    return jsonify({
        "ok": True, "k": k, "entrada": valores,
        "clase": prediccion, "nivel": NOMBRES[prediccion],
        "probabilidades": {c: round(float(proba[clases_modelo.index(c)]), 3) for c in CLASES},
        "votos": votos, "vecinos": vecinos, "avisos": avisos,
    })


@app.route("/api/dataset")
def api_dataset():
    df = DATASET
    origen = request.args.get("origen", "")
    clase = request.args.get("clase", "")
    buscar = request.args.get("q", "").strip().lower()
    if origen:
        df = df[df["origen"].str.startswith(origen)]
    if clase:
        df = df[df["intensidad"] == clase]
    if buscar:
        df = df[df["localidad"].str.lower().str.contains(buscar, na=False)]
    pagina = max(1, request.args.get("pagina", type=int, default=1))
    por_pagina = 15
    total = len(df)
    sub = df.iloc[(pagina - 1) * por_pagina: pagina * por_pagina]
    cols = ["evento_id", "fecha", "localidad", "departamento_localidad", "magnitud",
            "profundidad_km", "distancia_km", "intensidad_mm", "intensidad", "origen"]
    filas = sub[cols].astype(object).where(sub[cols].notna(), None).values.tolist()
    return jsonify({"total": total, "pagina": pagina,
                    "paginas": max(1, math.ceil(total / por_pagina)),
                    "columnas": cols, "filas": filas})


@app.route("/descargar/<nombre>")
def descargar(nombre):
    archivos = {"dataset_final.csv": DATA / "dataset_final.csv",
                "eventos_igp.csv": DATA / "eventos_igp.csv"}
    if nombre not in archivos:
        return "Archivo no disponible", 404
    return send_file(archivos[nombre], as_attachment=True)


if __name__ == "__main__":
    print("\n  SISMO AI PERÚ  ·  abre en tu navegador:  http://127.0.0.1:5000\n")
    app.run(debug=False, port=5000)
