/* SISMO AI PERÚ · lógica de la interfaz
   Todas las predicciones se piden al modelo KNN entrenado en Python (/api/predecir). */

const COLORES = { "I–IV": "#16a34a", "V–VI": "#eab308", "VII–XII": "#dc2626" };
const NOMBRE = { "I–IV": "Baja", "V–VI": "Moderada", "VII–XII": "Fuerte" };
const $ = (id) => document.getElementById(id);
let RESUMEN = null, LOCALIDADES = [], GEO = null;
const graficos = {};
Chart.defaults.color = "#4b5b5f";
Chart.defaults.borderColor = "#e6ebe4";
Chart.defaults.font.family = "Segoe UI, system-ui, sans-serif";

/* ─────────────── utilidades ─────────────── */
const fmt = (n, d = 0) => Number(n).toLocaleString("es-PE", { minimumFractionDigits: d, maximumFractionDigits: d });
const pct = (v) => (v * 100).toFixed(1) + " %";
function colorMag(m) { return m >= 6 ? "#dc2626" : m >= 4 ? "#f97316" : "#facc15"; }
function grafico(id, config) {
  if (graficos[id]) graficos[id].destroy();
  graficos[id] = new Chart($(id), config);
  return graficos[id];
}
async function api(url, opciones) {
  const r = await fetch(url, opciones);
  return r.json();
}

/* ─────────────── navegación ─────────────── */
document.querySelectorAll(".menu a").forEach((a) => a.addEventListener("click", () => {
  document.querySelectorAll(".menu a").forEach((x) => x.classList.remove("activo"));
  document.querySelectorAll(".pagina").forEach((p) => p.classList.remove("visible"));
  a.classList.add("activo");
  $("pag-" + a.dataset.pag).classList.add("visible");
  iniciarPagina(a.dataset.pag);
}));
const iniciadas = new Set();
function iniciarPagina(p) {
  setTimeout(() => Object.values(mapas).forEach((m) => m.invalidateSize()), 60);
  if (iniciadas.has(p)) return;
  iniciadas.add(p);
  ({ analisis: paginaAnalisis, simulador: paginaSimulador, mapa: paginaMapa,
     dataset: paginaDataset, evaluacion: paginaEvaluacion })[p]?.();
}

/* ─────────────── mapas (Leaflet) ─────────────── */
const mapas = {};
function crearMapa(id, zoom = 5) {
  const m = L.map(id, { preferCanvas: true, zoomControl: true, attributionControl: true, zoomSnap: 0.25 })
    .setView([-9.3, -75.5], zoom);
  const base = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { attribution: "© OpenStreetMap", opacity: 0.6 });
  const deps = L.geoJSON(GEO, {
    style: (f) => ({
      color: "#8a9a7b", weight: 1, fillOpacity: f.properties.NOMBDEP === "HUANUCO" ? 0.75 : 0.9,
      fillColor: f.properties.NOMBDEP === "HUANUCO" ? "#5eead4" : "#eef0e2",
    }),
    onEachFeature: (f, capa) => capa.bindTooltip(f.properties.NOMBDEP, { sticky: true }),
  }).addTo(m);
  m.fitBounds(deps.getBounds(), { padding: [0, 0] });
  L.control.layers({}, { "Departamentos": deps, "Mapa base (OpenStreetMap)": base },
    { position: "topright" }).addTo(m);
  const ley = L.control({ position: "bottomleft" });
  ley.onAdd = () => {
    const d = L.DomUtil.create("div", "leyenda");
    d.innerHTML = "<b>Magnitud</b><br><i style='background:#facc15'></i>&lt; 4.0 (Baja)<br>"
      + "<i style='background:#f97316'></i>4.0 – 5.9 (Moderada)<br><i style='background:#dc2626'></i>≥ 6.0 (Alta)";
    return d;
  };
  ley.addTo(m);
  L.control.scale({ imperial: false }).addTo(m);
  mapas[id] = m;
  return m;
}
function dibujarEventos(mapa, eventos, capaPrevia) {
  if (capaPrevia) mapa.removeLayer(capaPrevia);
  const capa = L.layerGroup();
  eventos.forEach(([lat, lon, mag, prof, fecha, dep, ref, inten]) => {
    L.circleMarker([lat, lon], {
      radius: Math.max(2, (mag - 3.5) * 1.8), color: "#7c2d12", weight: 0.5,
      fillColor: colorMag(mag), fillOpacity: 0.8,
    }).bindPopup(`<b>${dep}</b><br>Fecha: ${fecha}<br>Magnitud: ${mag} · Profundidad: ${prof} km`
      + (ref ? `<br>${ref}` : "") + (inten ? `<br>Intensidad IGP: ${inten}` : ""))
      .addTo(capa);
  });
  return capa.addTo(mapa);
}

/* ─────────────── predicción KNN ─────────────── */
async function predecir(datos) {
  return api("/api/predecir", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(datos),
  });
}
function pintarResultado(caja, r) {
  caja.className = "caja-res " + r.nivel + (caja.classList.contains("grande") ? " grande" : "");
  caja.innerHTML = `<small>Intensidad estimada</small><b>${r.nivel}</b><small>Mercalli ${r.clase} · K = ${r.k}</small>`;
}
function pintarProbs(div, r) {
  div.innerHTML = Object.entries(r.probabilidades).map(([c, p]) =>
    `<div class="fila"><span>${NOMBRE[c]}</span><div class="barra"><i style="width:${p * 100}%;background:${COLORES[c]}"></i></div><b>${Math.round(p * 100)} %</b></div>`).join("");
}
function opcionesK(select) {
  const ks = RESUMEN.metricas.comparacion_k.map((f) => f.k);
  select.innerHTML = ks.map((k) => `<option value="${k}" ${k === RESUMEN.metricas.mejor_k ? "selected" : ""}>K = ${k}${k === RESUMEN.metricas.mejor_k ? " (óptimo)" : ""}</option>`).join("");
}
function opcionesLocalidad(select) {
  select.innerHTML = LOCALIDADES.map((l, i) =>
    `<option value="${i}" ${l.departamento === "Huanuco" ? "selected" : ""}>${l.capital} (${l.departamento})</option>`).join("");
}

/* ─────────────── DASHBOARD ─────────────── */
async function paginaDashboard() {
  const r = RESUMEN;
  $("kTotal").textContent = fmt(r.total_eventos);
  $("kPeriodo").textContent = `${r.periodo[0].slice(0, 4)}–${r.periodo[1].slice(0, 4)}`;
  $("kMag").innerHTML = `${fmt(r.magnitud_promedio, 1)} <small>M</small>`;
  $("kProf").innerHTML = `${fmt(r.profundidad_promedio, 0)} <small>km</small>`;
  $("kReg").textContent = r.regiones;
  $("fechaCorte").textContent = r.periodo[1];

  const mapa = crearMapa("mapaDash", 5);
  const ev = await api("/api/eventos?min_mag=5.5");
  dibujarEventos(mapa, ev.eventos);
  const hco = LOCALIDADES.find((l) => l.departamento === "Huanuco");
  L.marker([hco.lat, hco.lon]).addTo(mapa).bindPopup(
    `<b>Huánuco</b><br>${r.huanuco.eventos} epicentros en el departamento<br>${r.huanuco.registros_intensidad} registros de intensidad`);

  const dm = r.dist_magnitud;
  grafico("gDona", {
    type: "doughnut",
    data: { labels: Object.keys(dm).map((k) => `${k} (${pct(dm[k] / r.total_eventos)})`),
      datasets: [{ data: Object.values(dm), backgroundColor: ["#facc15", "#f97316", "#dc2626"], borderColor: "#fff" }] },
    options: { maintainAspectRatio: false, cutout: "62%", plugins: { legend: { position: "right" } } },
  });
  const top = r.top_departamentos;
  grafico("gTop", {
    type: "bar",
    data: { labels: Object.keys(top), datasets: [{ data: Object.values(top), backgroundColor: "#0f766e", borderRadius: 6 }] },
    options: { indexAxis: "y", maintainAspectRatio: false, plugins: { legend: { display: false } } },
  });
  matriz($("mcDash"), r.metricas);
  const p = r.metricas.prueba;
  $("rendDash").innerHTML = [["Exactitud", p.exactitud], ["Precisión", p.precision], ["Recall", p.recall], ["F1-Score", p.f1]]
    .map(([n, v]) => `<div class="m"><span>${n}</span><b>${v.toFixed(2)}</b></div>`).join("")
    + `<small style="color:#667479">Conjunto de prueba · K = ${r.metricas.mejor_k} · promedio macro</small>`;

  opcionesLocalidad($("dLoc"));
  opcionesK($("dK"));
  const estimar = async () => {
    const res = await predecir({ magnitud: $("dMag").value, profundidad_km: $("dProf").value,
      distancia_km: $("dDist").value, k: $("dK").value });
    if (!res.ok) { $("dErr").textContent = res.errores.join(" "); return; }
    $("dErr").textContent = "";
    pintarResultado($("dRes"), res);
    pintarProbs($("dProbs"), res);
  };
  $("dBtn").onclick = estimar;
  estimar();
}
function matriz(div, m) {
  const max = Math.max(...m.matriz_confusion.flat());
  let h = `<div class="cab"></div>` + m.clases.map((c) => `<div class="cab">${NOMBRE[c]}</div>`).join("");
  m.matriz_confusion.forEach((fila, i) => {
    h += `<div class="cab">${NOMBRE[m.clases[i]]}</div>`;
    fila.forEach((v, j) => {
      const a = 0.15 + 0.85 * v / max;
      const color = i === j ? `rgba(15,118,110,${Math.max(0.18, a)})` : `rgba(217,119,6,${Math.max(0.1, a * 0.7)})`;
      h += `<div class="celda" style="background:${color}">${v}</div>`;
    });
  });
  div.innerHTML = h + `<div></div><div class="cab" style="grid-column:span 3">Predicho → (filas = real)</div>`;
}

/* ─────────────── ANÁLISIS KNN ─────────────── */
async function paginaAnalisis() {
  const m = RESUMEN.metricas;
  $("infoSplit").innerHTML = `<b>División por evento:</b> ${fmt(m.n_train)} registros de entrenamiento (${fmt(m.n_eventos_train)} sismos)
    y ${fmt(m.n_test)} de prueba (${fmt(m.n_eventos_test)} sismos). Ningún sismo aparece en ambos conjuntos.`;
  const c = m.comparacion_k;
  grafico("gK", {
    type: "line",
    data: { labels: c.map((f) => "K=" + f.k), datasets: [
      { label: "F1-macro", data: c.map((f) => f.f1_macro_media), borderColor: "#0f766e", backgroundColor: "#0f766e", tension: .3 },
      { label: "Exactitud", data: c.map((f) => f.exactitud_media), borderColor: "#d97706", backgroundColor: "#d97706", tension: .3 }] },
    options: { maintainAspectRatio: false },
  });
  $("tablaK").innerHTML = `<table class="tabla"><tr><th>K</th><th>Exactitud (CV)</th><th>F1-macro (CV)</th></tr>`
    + c.map((f) => `<tr><td>${f.k}${f.k === m.mejor_k ? " ✔" : ""}</td><td>${f.exactitud_media.toFixed(3)} ± ${f.exactitud_std.toFixed(3)}</td><td>${f.f1_macro_media.toFixed(3)} ± ${f.f1_macro_std.toFixed(3)}</td></tr>`).join("") + "</table>";
  const mu = await api("/api/muestra");
  grafico("gDisp", {
    type: "scatter",
    data: { datasets: Object.keys(COLORES).map((c) => ({
      label: `${NOMBRE[c]} (${c})`, backgroundColor: COLORES[c] + "aa", pointRadius: 2.5,
      data: mu.filter((f) => f[2] === c).map((f) => ({ x: f[1], y: f[0] })) })) },
    options: { maintainAspectRatio: false, scales: {
      x: { title: { display: true, text: "Distancia epicentral (km)" }, max: 400 },
      y: { title: { display: true, text: "Magnitud" } } } },
  });
}

/* ─────────────── SIMULADOR ─────────────── */
let marcaEpi = null, lineaEpi = null, marcaLoc = null;
function paginaSimulador() {
  opcionesLocalidad($("sLoc"));
  opcionesK($("sK"));
  const mapa = crearMapa("sMapa", 5);
  const ubicarLocalidad = () => {
    const l = LOCALIDADES[$("sLoc").value];
    if (marcaLoc) mapa.removeLayer(marcaLoc);
    marcaLoc = L.marker([l.lat, l.lon]).addTo(mapa).bindTooltip(l.capital, { permanent: true });
    mapa.setView([l.lat, l.lon], 6);
    actualizarLinea();
  };
  const actualizarLinea = () => {
    if (!marcaEpi) return;
    const l = LOCALIDADES[$("sLoc").value];
    const e = marcaEpi.getLatLng();
    if (lineaEpi) mapa.removeLayer(lineaEpi);
    lineaEpi = L.polyline([[l.lat, l.lon], e], { color: "#dc2626", dashArray: "6", weight: 2.5 }).addTo(mapa);
    $("sDist").value = (mapa.distance([l.lat, l.lon], e) / 1000).toFixed(1);
  };
  mapa.on("click", (ev) => {
    if (marcaEpi) mapa.removeLayer(marcaEpi);
    marcaEpi = L.circleMarker(ev.latlng, { radius: 10, color: "#fff", fillColor: "#dc2626", fillOpacity: 1 })
      .addTo(mapa).bindTooltip("Epicentro hipotético");
    $("sEpi").value = `${ev.latlng.lat.toFixed(3)}, ${ev.latlng.lng.toFixed(3)}`;
    actualizarLinea();
  });
  $("sLoc").onchange = ubicarLocalidad;
  ubicarLocalidad();
  $("sBtn").onclick = async () => {
    const res = await predecir({ magnitud: $("sMag").value, profundidad_km: $("sProf").value,
      distancia_km: $("sDist").value, k: $("sK").value });
    if (!res.ok) { $("sErr").textContent = res.errores.join(" "); return; }
    $("sErr").textContent = "";
    pintarResultado($("sRes"), res);
    $("sAvisos").innerHTML = res.avisos.map((a) => `<p class="aviso">⚠ ${a}</p>`).join("");
    grafico("gVotos", {
      type: "bar",
      data: { labels: Object.keys(res.votos).map((c) => `${NOMBRE[c]} (${c})`),
        datasets: [{ label: "Votos", data: Object.values(res.votos), backgroundColor: Object.values(COLORES), borderRadius: 6 }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false },
        title: { display: true, text: `Votación de los ${res.k} vecinos más cercanos` } },
        scales: { y: { ticks: { stepSize: 1 }, beginAtZero: true } } },
    });
    $("sVecinos").innerHTML = "<tr><th>#</th><th>Localidad</th><th>Fecha</th><th>M</th><th>Prof.</th><th>Dist.</th><th>Clase</th><th>Origen</th></tr>"
      + res.vecinos.map((v, i) => `<tr><td>${i + 1}</td><td>${v.localidad}</td><td>${v.fecha}</td><td>${v.magnitud}</td>
        <td>${v.profundidad_km} km</td><td>${v.distancia_km} km</td><td><span class="etq ${v.nivel}">${v.nivel}</span></td>
        <td>${v.origen.startsWith("IGP") ? "IGP" : "<span class='sim'>Simulado</span>"}</td></tr>`).join("");
  };
}

/* ─────────────── MAPA DE EVENTOS ─────────────── */
let capaGrande = null;
async function paginaMapa() {
  const mapa = crearMapa("mapaGrande", 5);
  const deps = await api("/api/departamentos");
  $("mDep").innerHTML += deps.map((d) => `<option>${d}</option>`).join("");
  const filtrar = async () => {
    const q = `min_mag=${$("mMin").value}&desde=${$("mDesde").value}&hasta=${$("mHasta").value}&dep=${encodeURIComponent($("mDep").value)}`;
    const r = await api("/api/eventos?" + q);
    capaGrande = dibujarEventos(mapa, r.eventos, capaGrande);
    $("mTotal").textContent = `${fmt(r.total)} eventos`;
    if ($("mDep").value && r.eventos.length) {
      mapa.fitBounds(L.latLngBounds(r.eventos.map((e) => [e[0], e[1]])), { padding: [30, 30], maxZoom: 8 });
    }
    const porAnio = {};
    r.eventos.forEach((e) => { const a = e[4].slice(0, 4); porAnio[a] = (porAnio[a] || 0) + 1; });
    grafico("gAnio", { type: "bar",
      data: { labels: Object.keys(porAnio), datasets: [{ data: Object.values(porAnio), backgroundColor: "#0f766e" }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false } } } });
    const muestra = r.eventos.length > 4000 ? r.eventos.filter((_, i) => i % Math.ceil(r.eventos.length / 4000) === 0) : r.eventos;
    grafico("gMagProf", { type: "scatter",
      data: { datasets: [{ data: muestra.map((e) => ({ x: e[2], y: e[3] })), backgroundColor: muestra.map((e) => colorMag(e[2]) + "99"), pointRadius: 2 }] },
      options: { maintainAspectRatio: false, plugins: { legend: { display: false } },
        scales: { x: { title: { display: true, text: "Magnitud" } }, y: { reverse: true, title: { display: true, text: "Profundidad (km)" } } } } });
  };
  $("mBtn").onclick = filtrar;
  filtrar();
}

/* ─────────────── DATASET ─────────────── */
let paginaActual = 1;
async function paginaDataset() {
  const d = RESUMEN.dataset;
  $("dsResumen").innerHTML = [
    ["Registros totales", fmt(d.total)], ["Observados IGP", fmt(d.observados)],
    ["Simulados (identificados)", fmt(d.simulados)], ["Eventos IGP en catálogo", fmt(RESUMEN.total_eventos)]]
    .map(([t, v], i) => `<div class="kpi k${i + 1}"><div><small>${t}</small><b>${v}</b></div></div>`).join("");
  const cargar = async () => {
    const q = `pagina=${paginaActual}&origen=${$("fOrigen").value}&clase=${encodeURIComponent($("fClase").value)}&q=${encodeURIComponent($("fBuscar").value)}`;
    const r = await api("/api/dataset?" + q);
    $("tablaDs").innerHTML = "<tr>" + r.columnas.map((c) => `<th>${c}</th>`).join("") + "</tr>"
      + r.filas.map((f) => "<tr>" + f.map((v, i) => {
        if (r.columnas[i] === "intensidad") return `<td><span class="etq ${NOMBRE[v]}">${v}</span></td>`;
        if (r.columnas[i] === "origen") return `<td>${v.startsWith("IGP") ? v : `<span class="sim">${v}</span>`}</td>`;
        return `<td>${v ?? ""}</td>`;
      }).join("") + "</tr>").join("");
    $("pInfo").textContent = `Página ${r.pagina} de ${r.paginas} · ${fmt(r.total)} registros`;
    $("pAnt").disabled = r.pagina <= 1;
    $("pSig").disabled = r.pagina >= r.paginas;
  };
  $("fBtn").onclick = () => { paginaActual = 1; cargar(); };
  $("pAnt").onclick = () => { paginaActual--; cargar(); };
  $("pSig").onclick = () => { paginaActual++; cargar(); };
  cargar();
}

/* ─────────────── EVALUACIÓN ─────────────── */
function paginaEvaluacion() {
  const m = RESUMEN.metricas, p = m.prueba;
  $("evKpis").innerHTML = [["Exactitud", p.exactitud], ["Precisión (macro)", p.precision], ["Recall (macro)", p.recall], ["F1 (macro)", p.f1]]
    .map(([t, v], i) => `<div class="kpi k${i + 1}"><span class="ico">✓</span><div><small>${t}</small><b>${pct(v)}</b><em>Prueba · K = ${m.mejor_k}</em></div></div>`).join("");
  matriz($("mcGrande"), m);
  const pc = m.por_clase;
  grafico("gClase", { type: "bar",
    data: { labels: m.clases.map((c) => `${NOMBRE[c]} (${c})`), datasets: ["precision", "recall", "f1-score"].map((k, i) => ({
      label: k, data: m.clases.map((c) => pc[c][k]), backgroundColor: ["#0f766e", "#5eead4", "#d97706"][i] })) },
    options: { maintainAspectRatio: false, scales: { y: { max: 1, beginAtZero: true } } } });
  $("tablaClase").innerHTML = `<table class="tabla"><tr><th>Clase</th><th>Precisión</th><th>Recall</th><th>F1</th><th>Soporte</th></tr>`
    + m.clases.map((c) => `<tr><td>${NOMBRE[c]} (${c})</td><td>${pc[c].precision.toFixed(2)}</td><td>${pc[c].recall.toFixed(2)}</td><td>${pc[c]["f1-score"].toFixed(2)}</td><td>${pc[c].support}</td></tr>`).join("") + "</table>";
  $("evNota").innerHTML = `<b>Interpretación.</b> En los ${fmt(m.n_prueba_igp)} registros de prueba <b>observados por el IGP</b> la exactitud es
    <b>${pct(m.prueba_solo_igp.exactitud)}</b>. La clase Moderada es la más difícil porque está entre las otras dos y se confunde con ambas.
    Las clases Moderada y Fuerte incluyen datos simulados, así que sus métricas muestran qué tan bien el KNN aprende la ley de atenuación,
    no su desempeño frente a sismos fuertes reales.`;
}

/* ─────────────── REPORTES: 3 escenarios ─────────────── */
const ESCENARIOS = [
  { titulo: "1 · Sismo moderado cerca de Huánuco", desc: "Sismo cortical como el del prototipo de la guía.", magnitud: 5.2, profundidad_km: 35, distancia_km: 25 },
  { titulo: "2 · Gran sismo frente a la costa de Lima", desc: "Sismo de subducción hipotético frente al Callao.", magnitud: 8.0, profundidad_km: 40, distancia_km: 60 },
  { titulo: "3 · Sismo profundo bajo Ucayali", desc: "Sismo de foco profundo, típico de la selva.", magnitud: 7.0, profundidad_km: 600, distancia_km: 40 },
];
$("rBtn").onclick = async () => {
  const k = RESUMEN.metricas.mejor_k;
  const res = await Promise.all(ESCENARIOS.map((e) => predecir({ ...e, k })));
  $("escenarios").innerHTML = res.map((r, i) => {
    const e = ESCENARIOS[i];
    return `<div class="esc"><h4>${e.titulo}</h4><p>${e.desc}</p>
      <ul><li>Magnitud: ${e.magnitud} Mw</li><li>Profundidad: ${e.profundidad_km} km</li><li>Distancia: ${e.distancia_km} km</li></ul>
      <div class="caja-res ${r.nivel}"><small>Intensidad estimada</small><b>${r.nivel}</b><small>Mercalli ${r.clase}</small></div>
      <div class="probs">${Object.entries(r.probabilidades).map(([c, p]) =>
        `<div class="fila"><span>${NOMBRE[c]}</span><div class="barra"><i style="width:${p * 100}%;background:${COLORES[c]}"></i></div><b>${Math.round(p * 100)} %</b></div>`).join("")}</div>
      <small>Votos (K=${r.k}): ${Object.entries(r.votos).map(([c, v]) => `${NOMBRE[c]} ${v}`).join(" · ")}</small>
      ${r.avisos.map((a) => `<p class="aviso">⚠ ${a}</p>`).join("")}</div>`;
  }).join("");
};

/* ─────────────── inicio ─────────────── */
(async function () {
  [RESUMEN, LOCALIDADES, GEO] = await Promise.all([
    api("/api/resumen"), api("/api/localidades"), api("/static/data_peru_departamentos.geojson")]);
  iniciadas.add("dashboard");
  paginaDashboard();
})();
