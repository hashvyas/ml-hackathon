const apiInput = document.querySelector("#api-url");
const tripSelect = document.querySelector("#trip-select");
const refreshButton = document.querySelector("#refresh-button");
const predictButton = document.querySelector("#predict-button");
const statusMessage = document.querySelector("#status");
const predictorBadge = document.querySelector("#predictor-badge");
const metricsContainer = document.querySelector("#metrics");
const segmentSummary = document.querySelector("#segment-summary");
const segmentsList = document.querySelector("#segments");

const map = L.map("map", { preferCanvas: true }).setView([49.86, 8.58], 12);
const gpsRenderer = L.canvas();
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors",
}).addTo(map);
const gpsLayer = L.layerGroup().addTo(map);
let routeLayer = null;

apiInput.value = localStorage.getItem("pathsnap-api-url") || apiInput.value;

function apiBase() {
  return apiInput.value.trim().replace(/\/+$/, "");
}

function setStatus(message, kind = "") {
  statusMessage.textContent = message;
  statusMessage.className = `status ${kind}`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(`${apiBase()}${path}`, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || `API returned ${response.status}`);
  }
  return body;
}

async function loadTrips() {
  refreshButton.disabled = true;
  predictButton.disabled = true;
  tripSelect.disabled = true;
  localStorage.setItem("pathsnap-api-url", apiBase());
  setStatus("Connecting to API…");
  try {
    const data = await requestJson("/trips");
    tripSelect.replaceChildren();
    for (const trip of data.trips) {
      const option = document.createElement("option");
      option.value = trip.trip_id;
      option.textContent = trip.trip_id;
      tripSelect.append(option);
    }
    tripSelect.disabled = data.trips.length === 0;
    predictButton.disabled = data.trips.length === 0;
    predictorBadge.textContent = data.predictor || "API connected";
    setStatus(
      data.trips.length
        ? `Connected. ${data.trips.length} trip(s) available.`
        : "API is connected, but no trips were found.",
      data.trips.length ? "success" : "",
    );
  } catch (error) {
    predictorBadge.textContent = "API not connected";
    tripSelect.replaceChildren(new Option("Unable to load trips", ""));
    setStatus(`Could not load trips: ${error.message}`, "error");
  } finally {
    refreshButton.disabled = false;
  }
}

function displayMetrics(metrics) {
  const entries = [
    ["Top-1 accuracy", metrics.top1_acc],
    ["Macro F1", metrics.macro_f1],
    ["Route precision", metrics.route_precision],
    ["Route recall", metrics.route_recall],
    ["Continuity", metrics.continuity],
    ["Observations", metrics.n_obs],
    ["Inference time", metrics.time_ms == null ? null : `${metrics.time_ms} ms`],
  ];
  metricsContainer.replaceChildren();
  for (const [label, rawValue] of entries) {
    const card = document.createElement("div");
    card.className = "metric";
    const name = document.createElement("p");
    name.className = "metric-label";
    name.textContent = label;
    const value = document.createElement("p");
    value.className = "metric-value";
    value.textContent =
      rawValue == null
        ? "—"
        : typeof rawValue === "number" && label !== "Observations"
          ? `${(rawValue * 100).toFixed(1)}%`
          : rawValue;
    card.append(name, value);
    metricsContainer.append(card);
  }
}

function updateMap(result) {
  gpsLayer.clearLayers();
  for (const point of result.gps) {
    L.circleMarker(point, {
      renderer: gpsRenderer,
      radius: 2,
      color: "#287dcc",
      weight: 1,
      fillOpacity: 0.65,
    }).addTo(gpsLayer);
  }
  if (routeLayer) {
    map.removeLayer(routeLayer);
  }
  routeLayer = L.polyline(result.predicted_route, {
    color: "#168a5b",
    weight: 5,
    opacity: 0.9,
  }).addTo(map);

  const bounds = L.latLngBounds(result.gps);
  if (bounds.isValid()) {
    map.fitBounds(bounds.pad(0.08), { maxZoom: 16 });
  }
}

function displaySegments(arcs) {
  const segments = [];
  for (const arcId of arcs) {
    if (arcId >= 0 && segments[segments.length - 1] !== arcId) {
      segments.push(arcId);
    }
  }
  segmentSummary.textContent = segments.length
    ? `${segments.length} consecutive road segment(s)${segments.length > 100 ? " — showing first 100" : ""}.`
    : "No road candidates were decoded for this trip.";
  segmentsList.replaceChildren();
  for (const arcId of segments.slice(0, 100)) {
    const item = document.createElement("li");
    item.textContent = `Arc ${arcId}`;
    segmentsList.append(item);
  }
}

async function predict() {
  if (!tripSelect.value) {
    return;
  }
  predictButton.disabled = true;
  refreshButton.disabled = true;
  setStatus(`Matching trip ${tripSelect.value}…`);
  try {
    const result = await requestJson("/predict", {
      method: "POST",
      body: JSON.stringify({ trip_id: tripSelect.value }),
    });
    updateMap(result);
    displayMetrics(result.metrics);
    displaySegments(result.per_obs_arcs);
    predictorBadge.textContent = result.predictor;
    setStatus(
      `Finished trip ${result.trip_id} with ${result.predictor}.`,
      "success",
    );
  } catch (error) {
    setStatus(`Prediction failed: ${error.message}`, "error");
  } finally {
    predictButton.disabled = !tripSelect.value;
    refreshButton.disabled = false;
  }
}

refreshButton.addEventListener("click", loadTrips);
predictButton.addEventListener("click", predict);
apiInput.addEventListener("change", loadTrips);
loadTrips();
