"""
PASO 3 · Entrenamiento y evaluación del clasificador KNN (SISMO AI PERÚ)
=========================================================================
  * Variables: magnitud, profundidad_km, distancia_km  ->  intensidad (3 clases)
  * División 80 % / 20 % AGRUPADA POR EVENTO (un mismo sismo nunca está en
    entrenamiento y prueba a la vez -> evita fuga de información).
  * Pipeline: StandardScaler + KNeighborsClassifier.
  * Comparación de K = 3, 5, 7, 9 con validación cruzada agrupada (5 pliegues)
    solo sobre el conjunto de entrenamiento.
  * Evaluación final en prueba: exactitud, precisión, recall, F1, matriz de confusión.

Ejecutar:
    python scripts/03_entrenar_modelo.py
"""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold,
                                     cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
MODELO = BASE / "modelo"
MODELO.mkdir(exist_ok=True)

VARIABLES = ["magnitud", "profundidad_km", "distancia_km"]
CLASES = ["I–IV", "V–VI", "VII–XII"]
NOMBRES = {"I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte"}
VALORES_K = [3, 5, 7, 9]


def crear_pipeline(k):
    return Pipeline([
        ("escalador", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=k)),
    ])


def metricas(y_real, y_pred):
    return {
        "exactitud": accuracy_score(y_real, y_pred),
        "precision": precision_score(y_real, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_real, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_real, y_pred, average="macro", zero_division=0),
    }


def main():
    """Entrena, evalúa y guarda el modelo."""
    df = pd.read_csv(DATA / "dataset_final.csv")
    df = df.dropna(subset=VARIABLES + ["intensidad"])
    print(f"Registros: {len(df):,} · Eventos: {df['evento_id'].nunique():,}")

    # 1. División 80/20 agrupada por evento
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_test = next(gss.split(df, df["intensidad"], groups=df["evento_id"]))
    train, test = df.iloc[idx_train], df.iloc[idx_test]
    assert not set(train["evento_id"]) & set(test["evento_id"]), "Fuga de eventos"
    print(f"Entrenamiento: {len(train):,} registros · Prueba: {len(test):,} registros")
    print("Eventos compartidos entre entrenamiento y prueba: 0 ✔")

    X_train, y_train, g_train = train[VARIABLES], train["intensidad"], train["evento_id"]
    X_test, y_test = test[VARIABLES], test["intensidad"]

    # 2. Validación cruzada agrupada para K = 3, 5, 7, 9
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    comparacion = []
    print("\nValidación cruzada agrupada (5 pliegues, solo entrenamiento):")
    for k in VALORES_K:
        r = cross_validate(crear_pipeline(k), X_train, y_train, groups=g_train, cv=cv,
                           scoring=["accuracy", "f1_macro"])
        fila = {"k": k,
                "exactitud_media": float(r["test_accuracy"].mean()),
                "exactitud_std": float(r["test_accuracy"].std()),
                "f1_macro_media": float(r["test_f1_macro"].mean()),
                "f1_macro_std": float(r["test_f1_macro"].std())}
        comparacion.append(fila)
        print(f"  K={k}:  exactitud {fila['exactitud_media']:.3f} ± {fila['exactitud_std']:.3f}"
              f"   F1-macro {fila['f1_macro_media']:.3f} ± {fila['f1_macro_std']:.3f}")
    mejor_k = max(comparacion, key=lambda f: f["f1_macro_media"])["k"]
    print(f"\nK seleccionado (mayor F1-macro): {mejor_k}")

    # 3. Entrenar un pipeline por cada K (el simulador web permite elegir K)
    modelos = {k: crear_pipeline(k).fit(X_train, y_train) for k in VALORES_K}

    # 4. Evaluación final en prueba con el K seleccionado
    y_pred = modelos[mejor_k].predict(X_test)
    m_total = metricas(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=CLASES)
    obs = test["origen"] == "IGP observado"
    m_obs = metricas(y_test[obs], y_pred[obs])
    por_clase = classification_report(y_test, y_pred, labels=CLASES, output_dict=True,
                                      zero_division=0)
    print("\nResultados en PRUEBA (K = %d):" % mejor_k)
    for n, v in m_total.items():
        print(f"  {n:<10} {v:.3f}")
    print("\nMatriz de confusión (filas = real, columnas = predicho):")
    print(pd.DataFrame(cm, index=CLASES, columns=CLASES).to_string())
    print(f"\nSolo registros OBSERVADOS por el IGP en prueba ({obs.sum()}): "
          f"exactitud {m_obs['exactitud']:.3f}")
    print("\nReporte por clase (todo el conjunto de prueba):")
    print(classification_report(y_test, y_pred, labels=CLASES, zero_division=0))

    # 5. Guardar modelo, datos de entrenamiento (para mostrar vecinos) y métricas
    #    (los datos se guardan como diccionario simple para que carguen en pandas 2 y 3)
    joblib.dump({"modelos": modelos, "mejor_k": mejor_k, "variables": VARIABLES,
                 "clases": CLASES, "train": train.reset_index(drop=True).to_dict(orient="list")},
                MODELO / "knn_sismo.joblib")
    resumen = {
        "mejor_k": mejor_k, "comparacion_k": comparacion,
        "prueba": m_total, "prueba_solo_igp": m_obs,
        "n_prueba_igp": int(obs.sum()),
        "matriz_confusion": cm.tolist(), "clases": CLASES,
        "por_clase": {c: por_clase[c] for c in CLASES},
        "n_train": len(train), "n_test": len(test),
        "n_eventos_train": int(train["evento_id"].nunique()),
        "n_eventos_test": int(test["evento_id"].nunique()),
        "n_total": len(df),
        "n_observados": int((df["origen"] == "IGP observado").sum()),
        "n_simulados": int((df["origen"] != "IGP observado").sum()),
    }
    (MODELO / "metricas.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False),
                                          encoding="utf-8")

    # 6. Gráficos de evidencia
    fig, ax = plt.subplots(figsize=(7, 4))
    ks = [f["k"] for f in comparacion]
    ax.errorbar(ks, [f["f1_macro_media"] for f in comparacion],
                yerr=[f["f1_macro_std"] for f in comparacion], marker="o", capsize=4,
                label="F1-macro", color="#2E86AB")
    ax.errorbar(ks, [f["exactitud_media"] for f in comparacion],
                yerr=[f["exactitud_std"] for f in comparacion], marker="s", capsize=4,
                label="Exactitud", color="#E4572E")
    ax.set_xticks(ks)
    ax.set_xlabel("K (número de vecinos)")
    ax.set_title("Validación cruzada agrupada por evento (5 pliegues)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(MODELO / "comparacion_k.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "#1B3A5C")
    etiquetas = [f"{NOMBRES[c]}\n{c}" for c in CLASES]
    ax.set_xticks(range(3), etiquetas)
    ax.set_yticks(range(3), etiquetas)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(f"Matriz de confusión · prueba (K={mejor_k})")
    fig.tight_layout()
    fig.savefig(MODELO / "matriz_confusion.png", dpi=150)
    print("Modelo guardado -> modelo/knn_sismo.joblib · métricas -> modelo/metricas.json")


if __name__ == "__main__":
    main()
