# 🌎 SISMO AI PERÚ

**Machine Learning aplicado a la estimación de intensidad sísmica · SENATI**
Taller de Desarrollo de Aplicaciones con Machine Learning · Reto individual · Huánuco, Perú

Esta aplicación web (Flask) estima la **categoría de intensidad sísmica** (escala de Mercalli modificada) que tendría un sismo en una localidad. Para eso usa un clasificador **K-Nearest Neighbors (KNN)** entrenado con datos del **Instituto Geofísico del Perú (IGP)**.

| Clase | Mercalli | Significado |
|-------|----------|-------------|
| 🟢 Baja | I – IV | Sentido levemente, sin daños |
| 🟡 Moderada | V – VI | Sentido por todos, daños leves |
| 🔴 Fuerte | VII – XII | Daños moderados a severos |

> ⚠️ **Proyecto educativo.** No predice la fecha, la ubicación ni la magnitud de futuros terremotos. No es una alerta ni una herramienta oficial de seguridad. La fuente oficial es el [IGP · CENSIS](https://ultimosismo.igp.gob.pe/).

---

## ⚙️ Instalación y ejecución

```bash
pip install -r requirements.txt
python app.py
```

Luego abre **http://127.0.0.1:5000** en el navegador.

El proyecto ya trae los datos procesados y el modelo entrenado. Para **reproducir todo desde cero**, ejecuta en orden:

```bash
python scripts/00_descargar_igp.py          # (opcional) actualiza los datos desde la API del IGP
python scripts/01_construir_dataset.py      # dataset evento-localidad con datos IGP
python scripts/02_complementar_simulados.py # completa las clases con pocos datos (identificados)
python scripts/03_entrenar_modelo.py        # Pipeline + KNN, validación cruzada agrupada, métricas
python app.py                               # aplicación web
```

---

## 📊 Datos

| Archivo | Contenido |
|---------|-----------|
| `data/raw/igp_sismos_reportados.csv` | Catálogo sísmico del IGP 1960–2026 (27,556 sismos) y reportes CENSIS 2020–2026 con la intensidad observada por localidad |
| `data/raw/inei_*.csv` | Coordenadas de centros poblados, distritos y departamentos (INEI) |
| `data/eventos_igp.csv` | Eventos limpios, con el departamento del epicentro (se usan en el mapa y el dashboard) |
| `data/dataset_sismos_intensidad.csv` | **4,544 registros evento-localidad observados por el IGP** |
| `data/dataset_final.csv` | Dataset de entrenamiento: 4,544 registros observados + 1,721 simulados (**6,265 en total**) |

**Campos:** `evento_id, fecha, latitud, longitud, localidad, magnitud, profundidad_km, distancia_km, intensidad_mm, intensidad, origen`

**Origen de los datos.** Los reportes del IGP indican la intensidad observada en una o varias localidades, por ejemplo *"III-IV Ambo, Huánuco, Cerro de Pasco; II Panao"*. El script 01 separa cada reporte en registros evento-localidad, ubica cada localidad con las coordenadas del INEI y calcula la distancia epicentral con la fórmula de Haversine. Los datos se obtuvieron de la API pública `ultimosismo.igp.gob.pe` a través de la copia pública [ifc-chalaco/peru-seismic-dashboard](https://github.com/ifc-chalaco/peru-seismic-dashboard), con corte al 23/11/2026.

**Datos simulados identificados.** En los datos observados del IGP solo hay 76 registros de la clase V–VI y 3 de la clase VII–XII, que no alcanzan para entrenar el modelo. Siguiendo la guía del reto, esas dos clases se completaron con 1,721 registros **simulados**, marcados en la columna `origen`. Para generarlos se usaron epicentros y profundidades reales del IGP, localidades reales del INEI y la ecuación de intensidad de *Allen, Wald & Worden (2012)*. A esa ecuación se le sumó una corrección regional de +0.55 grados, calculada comparándola con los 4,544 registros del IGP; con esa corrección acierta la clase en el 95 % de los registros reales.

---

## 🧠 Modelo

- **Variables:** magnitud, profundidad (km) y distancia epicentral (km).
- **División 80/20 agrupada por evento** (`GroupShuffleSplit`): un mismo sismo nunca queda a la vez en entrenamiento y en prueba, lo que evita la fuga de información.
- **Pipeline:** `StandardScaler` + `KNeighborsClassifier`.
- **Selección de K:** se compararon K = 3, 5, 7 y 9 con `StratifiedGroupKFold` (5 pliegues) sobre el conjunto de entrenamiento.

| K | Exactitud (CV) | F1-macro (CV) |
|---|----------------|---------------|
| 3 | 0.903 | 0.793 |
| 5 | 0.912 | 0.812 |
| 7 | 0.914 | 0.816 |
| **9** ✔ | **0.915** | **0.818** |

**Resultados en el conjunto de prueba (K = 9, 1,262 registros de 943 sismos):**

| Exactitud | Precisión | Recall | F1 (macro) | Exactitud solo en registros observados por el IGP |
|-----------|-----------|--------|------------|--------------------------------------|
| 91.2 % | 0.81 | 0.81 | 0.81 | 97.3 % |

---

## 🖥️ Aplicación web

| Sección | Qué muestra |
|---------|-------------|
| **Dashboard** | Indicadores, simulador rápido, mapa de epicentros, distribución por magnitud, top 5 de departamentos, matriz de confusión y métricas |
| **Análisis KNN** | Cómo funciona el modelo, comparación de K y gráfico de dispersión de los datos |
| **Simulador** | Clic en el mapa para ubicar el epicentro (la distancia se calcula sola), votación y tabla de los K vecinos |
| **Mapa de eventos** | 27,556 sismos con filtros por magnitud, años y departamento |
| **Dataset** | Tabla paginada con filtros y descarga del CSV |
| **Evaluación** | Matriz de confusión y métricas por clase |
| **Reportes** | 3 escenarios hipotéticos que se ejecutan en vivo |
| **Ayuda** | Respuestas a las preguntas de sustentación, fuentes y limitaciones |

Todas las predicciones las hace el modelo KNN entrenado, a través del endpoint `POST /api/predecir`, que valida las entradas. No hay reglas if/else ni resultados fijos.

---

## 🗂️ Estructura

```
SISMO_AI_PERU/
├── app.py                  ← servidor Flask + API
├── requirements.txt
├── data/                   ← datasets (CSV)
├── modelo/                 ← modelo entrenado, métricas y gráficos
├── scripts/                ← pasos 00 a 03
├── templates/index.html    ← interfaz
├── static/                 ← CSS, JS, Leaflet, Chart.js, GeoJSON del Perú
├── evidencias/             ← capturas de la app y de la consola
└── informe/                ← presentación paso a paso (PPTX y PDF)
```

## 📚 Fuentes

- Instituto Geofísico del Perú (IGP). Reportes sísmicos CENSIS y catálogo sísmico. https://ultimosismo.igp.gob.pe · https://www.gob.pe/igp
- INEI · centros poblados y ubigeos (vía *jmcastagnetto/ubigeo-peru-aumentado*).
- Límites departamentales: *juaneladio/peru-geojson*.
- Allen, T. I., Wald, D. J., & Worden, C. B. (2012). Intensity attenuation for active crustal regions. *Journal of Seismology*, 16, 409–433.

**Autor:** Wilfredo Alcedo Santiago · SENATI
