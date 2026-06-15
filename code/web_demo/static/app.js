const state = {
  options: null,
  graph: null,
  pickMode: "source",
  selectedSource: "",
  selectedTarget: "",
  lastPathEdges: new Set(),
  subgraphRequestId: 0,
};

const els = {
  alert: document.getElementById("alert"),
  graphSelect: document.getElementById("graph-select"),
  modelSelect: document.getElementById("model-select"),
  dimSelect: document.getElementById("dim-select"),
  sourceInput: document.getElementById("source-input"),
  targetInput: document.getElementById("target-input"),
  sourceSuggestions: document.getElementById("source-suggestions"),
  targetSuggestions: document.getElementById("target-suggestions"),
  astarMaxVisits: document.getElementById("astar-max-visits"),
  astarMaxVisitsSource: document.getElementById("astar-max-visits-source"),
  embeddingThreshold: document.getElementById("embedding-threshold"),
  pickSource: document.getElementById("pick-source"),
  pickTarget: document.getElementById("pick-target"),
  controls: document.getElementById("controls"),
  runButton: document.getElementById("run-button"),
  reloadSubgraph: document.getElementById("reload-subgraph"),
  caption: document.getElementById("graph-caption"),
  resultStatus: document.getElementById("result-status"),
  metricHops: document.getElementById("metric-hops"),
  metricVisited: document.getElementById("metric-visited"),
  metricRuntime: document.getElementById("metric-runtime"),
  pathList: document.getElementById("path-list"),
  graphContainer: document.getElementById("graph"),
};

const colors = {
  normal: "#8fb6d9",
  source: "#1b9e77",
  target: "#d95f02",
  path: "#e7298a",
  link: "rgba(190, 205, 220, 0.38)",
  pathLink: "#ffcc33",
};

const graph = ForceGraph3D()(els.graphContainer)
  .backgroundColor("#121417")
  .nodeId("id")
  .nodeLabel((node) => `${node.label}<br>degree: ${node.degree}`)
  .nodeColor(nodeColor)
  .nodeVal((node) => Math.max(2.5, Math.min(12, 2 + Math.log2((node.degree || 1) + 1))))
  .linkColor((link) => link.path ? colors.pathLink : colors.link)
  .linkWidth((link) => link.path ? 4 : 0.7)
  .linkDirectionalArrowLength((link) => link.path ? 5 : 2.5)
  .linkDirectionalArrowRelPos(1)
  .linkDirectionalParticles((link) => link.path ? 4 : 0)
  .linkDirectionalParticleWidth((link) => link.path ? 3 : 0)
  .onNodeClick(handleNodeClick);

graph.cooldownTicks(120);
try {
  graph.d3Force("charge").strength(-110);
  graph.d3Force("link").distance((link) => link.path ? 34 : 48);
} catch (error) {
  console.debug("Force graph tuning unavailable", error);
}

window.addEventListener("resize", () => {
  graph.width(els.graphContainer.clientWidth);
  graph.height(els.graphContainer.clientHeight);
});

document.addEventListener("DOMContentLoaded", init);
els.modelSelect.addEventListener("change", async () => {
  renderDimensionOptions();
  await loadAdvancedDefaults();
});
els.dimSelect.addEventListener("change", loadAdvancedDefaults);
els.graphSelect.addEventListener("change", handleGraphChange);
els.pickSource.addEventListener("click", () => setPickMode("source"));
els.pickTarget.addEventListener("click", () => setPickMode("target"));
els.reloadSubgraph.addEventListener("click", () => loadSubgraph());
els.controls.addEventListener("submit", runAStar);
els.sourceInput.addEventListener("input", debounce(() => updateSuggestions("source"), 180));
els.targetInput.addEventListener("input", debounce(() => updateSuggestions("target"), 180));

async function init() {
  setBusy(true, "Loading options...");
  try {
    state.options = await apiGet("/api/options");
    renderOptions();
    await loadAdvancedDefaults();
    await loadSubgraph();
    setStatus("Ready", "idle");
  } catch (error) {
    showError(error.message);
    setStatus("Could not initialize demo", "error");
  } finally {
    setBusy(false);
  }
}

function renderOptions() {
  els.graphSelect.innerHTML = "";
  for (const graphOption of state.options.graphs) {
    els.graphSelect.append(new Option(graphOption.label, graphOption.id));
  }
  els.graphSelect.value = state.options.defaults.graph;

  els.modelSelect.innerHTML = "";
  for (const model of state.options.models) {
    els.modelSelect.append(new Option(model.label, model.id));
  }
  els.modelSelect.value = state.options.defaults.model;
  renderDimensionOptions();
}

function renderDimensionOptions() {
  const model = selectedModel();
  els.dimSelect.innerHTML = "";
  if (!model) return;

  for (const dim of model.dims) {
    const suffix = dim === model.model_dim ? " (full)" : "";
    els.dimSelect.append(new Option(`${dim}${suffix}`, String(dim)));
  }
}

async function loadSubgraph(center = null) {
  hideError();
  const graphName = els.graphSelect.value;
  const requestId = ++state.subgraphRequestId;
  const params = new URLSearchParams({ graph: graphName, limit: "240" });
  if (center) params.set("center", center);
  if (center) params.set("depth", "2");
  if (els.sourceInput.value.trim()) params.set("source", els.sourceInput.value.trim());
  if (els.targetInput.value.trim()) params.set("target", els.targetInput.value.trim());

  els.caption.textContent = center ? `Loading neighborhood for ${center}...` : "Loading graph sample...";
  try {
    const payload = await apiGet(`/api/subgraph?${params.toString()}`);
    if (requestId !== state.subgraphRequestId) return;
    drawGraph(payload);
    els.caption.textContent = graphCaption(payload, center);
  } catch (error) {
    if (requestId !== state.subgraphRequestId) return;
    showError(error.message);
    els.caption.textContent = "Graph sample unavailable";
  }
}

function drawGraph(payload) {
  graph.graphData(payload);
  graph.width(els.graphContainer.clientWidth);
  graph.height(els.graphContainer.clientHeight);
  setTimeout(() => graph.zoomToFit(650, 72), 350);
  setTimeout(() => graph.zoomToFit(650, 72), 1300);
}

async function handleNodeClick(node) {
  if (state.pickMode === "source") {
    els.sourceInput.value = node.id;
    state.selectedSource = node.id;
    setPickMode("target");
  } else {
    els.targetInput.value = node.id;
    state.selectedTarget = node.id;
    setPickMode("source");
  }

  updateLocalNodeStatuses();
  await loadSubgraph(node.id);
  updateLocalNodeStatuses();
}

async function handleGraphChange() {
  resetGraphSelectionState();
  await loadAdvancedDefaults();
  await loadSubgraph();
}

function resetGraphSelectionState() {
  state.subgraphRequestId += 1;
  state.selectedSource = "";
  state.selectedTarget = "";
  state.lastPathEdges = new Set();
  els.sourceInput.value = "";
  els.targetInput.value = "";
  els.sourceSuggestions.innerHTML = "";
  els.targetSuggestions.innerHTML = "";
  clearPath();
  setStatus("Ready", "idle");
  graph.graphData({ nodes: [], links: [] });
  const selected = selectedGraphOption();
  els.caption.textContent = selected ? `Loading ${selected.label} preview...` : "Loading graph sample...";
}

function updateLocalNodeStatuses() {
  const data = graph.graphData();
  for (const node of data.nodes) {
    const onPath = state.lastPathEdges.has(node.id);
    if (node.id === els.sourceInput.value.trim()) {
      node.status = "source";
    } else if (node.id === els.targetInput.value.trim()) {
      node.status = "target";
    } else if (onPath) {
      node.status = "path";
    } else {
      node.status = "normal";
    }
  }
  graph.nodeColor(nodeColor);
}

function setPickMode(mode) {
  state.pickMode = mode;
  els.pickSource.classList.toggle("active", mode === "source");
  els.pickTarget.classList.toggle("active", mode === "target");
}

async function updateSuggestions(kind) {
  const input = kind === "source" ? els.sourceInput : els.targetInput;
  const list = kind === "source" ? els.sourceSuggestions : els.targetSuggestions;
  const query = input.value.trim();
  if (!query) {
    list.innerHTML = "";
    return;
  }

  const params = new URLSearchParams({
    graph: els.graphSelect.value,
    q: query,
    limit: "24",
  });

  try {
    const payload = await apiGet(`/api/nodes?${params.toString()}`);
    list.innerHTML = "";
    for (const node of payload.nodes) {
      const option = document.createElement("option");
      option.value = node;
      list.append(option);
    }
  } catch (error) {
    showError(error.message);
  }
}

async function runAStar(event) {
  event.preventDefault();
  hideError();
  clearPath();

  const body = {
    graph: els.graphSelect.value,
    model: els.modelSelect.value,
    dim: Number(els.dimSelect.value),
    source: els.sourceInput.value.trim(),
    target: els.targetInput.value.trim(),
    config: {
      astar_max_visits: readIntegerInput(els.astarMaxVisits),
      embedding_index_min_successors: readIntegerInput(els.embeddingThreshold),
    },
  };

  if (!body.source || !body.target) {
    showError("Select both a start and target concept.");
    return;
  }

  setBusy(true, "Running A*...");
  setStatus("Running A*...", "idle");
  try {
    const result = await apiPost("/api/astar", body);
    renderResult(result);
    drawGraph(result.graph);
    els.caption.textContent = result.found
      ? `Path view: ${result.graph.nodes.length} nodes, ${result.graph.links.length} edges`
      : `Selected-node view: ${result.graph.nodes.length} nodes, ${result.graph.links.length} edges`;
  } catch (error) {
    showError(error.message);
    setStatus(statusForApiError(error), "error");
  } finally {
    setBusy(false);
  }
}

async function loadAdvancedDefaults() {
  const model = selectedModel();
  if (!model || !els.dimSelect.value) return;

  const params = new URLSearchParams({
    graph: els.graphSelect.value,
    model: els.modelSelect.value,
    dim: els.dimSelect.value,
  });

  try {
    const payload = await apiGet(`/api/config?${params.toString()}`);
    els.astarMaxVisits.value = String(payload.astar_max_visits);
    els.astarMaxVisitsSource.textContent = payload.astar_max_visits_source;
    els.embeddingThreshold.value = String(payload.embedding_index_min_successors);
  } catch (error) {
    els.astarMaxVisits.value = "-1";
    els.astarMaxVisitsSource.textContent = "uncapped; default lookup failed";
    els.embeddingThreshold.value = "16";
    showError(error.message);
  }
}

function renderResult(result) {
  els.metricHops.textContent = String(result.hops);
  els.metricVisited.textContent = formatNumber(result.visited_nodes);
  els.metricRuntime.textContent = `${formatNumber(result.runtime_ms)} ms`;
  els.pathList.innerHTML = "";
  state.lastPathEdges = new Set(result.path || []);

  if (result.found) {
    setStatus("Path found", "found");
    for (const node of result.path) {
      const item = document.createElement("li");
      item.textContent = node;
      els.pathList.append(item);
    }
  } else {
    setStatus("No path found", "missing");
    const item = document.createElement("li");
    item.textContent = `${result.source} -> ${result.target}`;
    els.pathList.append(item);
  }
}

function clearPath() {
  state.lastPathEdges = new Set();
  els.metricHops.textContent = "-";
  els.metricVisited.textContent = "-";
  els.metricRuntime.textContent = "-";
  els.pathList.innerHTML = "";
}

function setStatus(text, mode) {
  els.resultStatus.textContent = text;
  els.resultStatus.className = `result-status ${mode}`;
}

function setBusy(isBusy, caption = null) {
  els.runButton.disabled = isBusy;
  els.graphSelect.disabled = isBusy;
  els.modelSelect.disabled = isBusy;
  els.dimSelect.disabled = isBusy;
  els.astarMaxVisits.disabled = isBusy;
  els.embeddingThreshold.disabled = isBusy;
  if (caption) els.caption.textContent = caption;
}

function selectedModel() {
  return state.options.models.find((model) => model.id === els.modelSelect.value);
}

function selectedGraphOption() {
  return state.options.graphs.find((graphOption) => graphOption.id === els.graphSelect.value);
}

function graphCaption(payload, center = null) {
  const label = payload.meta?.label || selectedGraphOption()?.label || els.graphSelect.value;
  const view = center ? `neighborhood for ${center}` : "preview";
  const cache = payload.meta?.cache_status ? `, ${payload.meta.cache_status} cache` : "";
  return `${label} ${view}: ${payload.nodes.length} nodes, ${payload.links.length} edges${cache}`;
}

function nodeColor(node) {
  if (node.status === "source") return colors.source;
  if (node.status === "target") return colors.target;
  if (node.status === "path") return colors.path;
  return colors.normal;
}

async function apiGet(url) {
  const response = await fetch(url);
  return readApiResponse(response);
}

async function apiPost(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return readApiResponse(response);
}

async function readApiResponse(response) {
  const payload = await response.json();
  if (!response.ok) {
    const detail = payload.detail || "Backend request failed.";
    const message = typeof detail === "string" ? detail : JSON.stringify(detail);
    const error = new Error(message);
    error.status = response.status;
    error.detail = detail;
    throw error;
  }
  return payload;
}

function statusForApiError(error) {
  const message = error.message || "";
  if (error.status === 400 && message.includes("not found in graph")) {
    return "Missing node";
  }
  return "Inference error";
}

function showError(message) {
  els.alert.textContent = message;
  els.alert.hidden = false;
}

function hideError() {
  els.alert.hidden = true;
  els.alert.textContent = "";
}

function debounce(fn, delay) {
  let handle = null;
  return (...args) => {
    window.clearTimeout(handle);
    handle = window.setTimeout(() => fn(...args), delay);
  };
}

function formatNumber(value) {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
  }).format(value);
}

function readIntegerInput(input) {
  if (!input.value.trim()) return null;
  return Number.parseInt(input.value, 10);
}
