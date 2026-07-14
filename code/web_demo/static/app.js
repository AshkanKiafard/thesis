const state = {
  options: null,
  pickMode: "source",
  selectedSource: "",
  selectedTarget: "",
  lastPathNodes: new Set(),
  subgraphRequestId: 0,
  bfsCapMode: "default_p95",
  bfsDefaults: new Map(),
  suggestionRequestIds: { source: 0, target: 0 },
  busy: false,
};

const els = {
  alert: document.getElementById("alert"),
  graphSelect: document.getElementById("graph-select"),
  graphSize: document.getElementById("graph-size"),
  methodSelect: document.getElementById("method-select"),
  methodValidation: document.getElementById("method-validation"),
  astarDimensionControl: document.getElementById("astar-dimension-control"),
  astarDimensionSelect: document.getElementById("astar-dimension-select"),
  sourceInput: document.getElementById("source-input"),
  targetInput: document.getElementById("target-input"),
  sourceSuggestions: document.getElementById("source-suggestions"),
  targetSuggestions: document.getElementById("target-suggestions"),
  bfsSettings: document.getElementById("bfs-settings"),
  bfsSearchCap: document.getElementById("bfs-search-cap"),
  bfsCapMode: document.getElementById("bfs-cap-mode"),
  bfsCapSource: document.getElementById("bfs-cap-source"),
  rlSettings: document.getElementById("rl-settings"),
  rlDetails: document.getElementById("rl-details"),
  astarSettings: document.getElementById("astar-settings"),
  astarDetails: document.getElementById("astar-details"),
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
  source: "#2f80ed",
  target: "#e53935",
  path: "#8b5cf6",
  link: "rgba(190, 205, 220, 0.38)",
  pathLink: "#c084fc",
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

const debouncedUpdateSuggestions = {
  source: debounce(() => updateSuggestions("source"), 180),
  target: debounce(() => updateSuggestions("target"), 180),
};

document.addEventListener("DOMContentLoaded", init);
els.methodSelect.addEventListener("change", handleMethodChange);
els.astarDimensionSelect.addEventListener("change", handleAstarDimensionChange);
els.graphSelect.addEventListener("change", handleGraphChange);
els.bfsSearchCap.addEventListener("input", handleBfsCapInput);
els.pickSource.addEventListener("click", () => setPickMode("source"));
els.pickTarget.addEventListener("click", () => setPickMode("target"));
els.reloadSubgraph.addEventListener("click", () => loadSubgraph());
els.controls.addEventListener("submit", runInference);
els.sourceInput.addEventListener("input", () => handleEndpointInput("source"));
els.targetInput.addEventListener("input", () => handleEndpointInput("target"));

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
  renderGraphInfo();
  renderMethodOptions(state.options.defaults.method);
  renderAdvancedSettings();
}

function renderMethodOptions(preferredMethod = els.methodSelect.value) {
  const currentGraph = els.graphSelect.value;
  const selectedValue = preferredMethod || state.options.defaults.method;
  const selectedDimension = selectedAstarDimension();
  els.methodSelect.innerHTML = "";

  for (const method of state.options.methods) {
    const option = new Option(method.label, method.id);
    option.disabled = !isMethodSupportedForGraph(method, currentGraph);
    els.methodSelect.append(option);
  }

  if (selectedValue) {
    els.methodSelect.value = selectedValue;
  }
  if (!els.methodSelect.value && state.options.methods.length) {
    els.methodSelect.value = state.options.methods[0].id;
  }
  renderAstarDimensionOptions(selectedDimension);
  validateSelectedMethod();
}

function renderAstarDimensionOptions(preferredDimension = null) {
  const method = selectedMethod();
  const dimensions = method?.algorithm === "astar" ? method.config.dimensions || [] : [];
  els.astarDimensionSelect.innerHTML = "";
  els.astarDimensionControl.hidden = dimensions.length === 0;
  if (!dimensions.length) return;

  for (const dimension of dimensions) {
    els.astarDimensionSelect.append(new Option(`d = ${dimension}`, String(dimension)));
  }

  const currentDimension = Number(preferredDimension ?? els.astarDimensionSelect.value);
  const defaultDimension = method.config.default_dimension;
  const selectedDimension = dimensions.includes(currentDimension)
    ? currentDimension
    : defaultDimension;
  els.astarDimensionSelect.value = String(selectedDimension);
}

function renderGraphInfo() {
  const graphOption = selectedGraphOption();
  els.graphSize.textContent = graphOption ? graphOption.size_label : "";
}

function renderAdvancedSettings() {
  const method = selectedMethod();
  hideAllMethodSettings();
  if (!method) return;

  if (method.algorithm === "bfs") {
    els.bfsSettings.hidden = false;
    renderBfsCapInfo();
  } else if (method.algorithm === "rl") {
    els.rlSettings.hidden = false;
    renderRlDetails(method.config);
  } else if (method.algorithm === "astar") {
    els.astarSettings.hidden = false;
    renderAstarDetails({
      ...method.config,
      dimension: selectedAstarDimension(),
    });
  }
}

function hideAllMethodSettings() {
  els.bfsSettings.hidden = true;
  els.rlSettings.hidden = true;
  els.astarSettings.hidden = true;
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

  clearPath();
  updateLocalNodeStatuses();
  await loadSubgraph(node.id);
  updateLocalNodeStatuses();
}

async function handleGraphChange() {
  const previousBfsMode = state.bfsCapMode;
  resetGraphSelectionState();
  renderGraphInfo();
  renderMethodOptions();
  if (selectedMethod()?.algorithm === "bfs" && previousBfsMode === "default_p95") {
    state.bfsCapMode = "default_p95";
  }
  renderAdvancedSettings();
  await loadAdvancedDefaults();
  await loadSubgraph();
}

async function handleMethodChange() {
  clearPath();
  setStatus("Ready", "idle");
  renderAstarDimensionOptions();
  validateSelectedMethod();
  renderAdvancedSettings();
  await loadAdvancedDefaults();
}

async function handleAstarDimensionChange() {
  clearPath();
  setStatus("Ready", "idle");
  renderAdvancedSettings();
  await loadAdvancedDefaults();
}

function resetGraphSelectionState() {
  state.subgraphRequestId += 1;
  state.selectedSource = "";
  state.selectedTarget = "";
  state.lastPathNodes = new Set();
  state.suggestionRequestIds.source += 1;
  state.suggestionRequestIds.target += 1;
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
    const onPath = state.lastPathNodes.has(node.id);
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
  for (const link of data.links) {
    link.path = false;
  }
  graph.nodeColor(nodeColor);
  graph.linkColor((link) => link.path ? colors.pathLink : colors.link);
  graph.linkWidth((link) => link.path ? 4 : 0.7);
  graph.linkDirectionalArrowLength((link) => link.path ? 5 : 2.5);
  graph.linkDirectionalParticles((link) => link.path ? 4 : 0);
  graph.linkDirectionalParticleWidth((link) => link.path ? 3 : 0);
}

function setPickMode(mode) {
  state.pickMode = mode;
  els.pickSource.classList.toggle("active", mode === "source");
  els.pickTarget.classList.toggle("active", mode === "target");
}

function handleEndpointInput(kind) {
  const input = kind === "source" ? els.sourceInput : els.targetInput;
  if (kind === "source") {
    state.selectedSource = input.value.trim();
  } else {
    state.selectedTarget = input.value.trim();
  }

  clearPath();
  setStatus("Ready", "idle");
  updateLocalNodeStatuses();
  debouncedUpdateSuggestions[kind]();
}

function handleBfsCapInput() {
  const cap = readIntegerInput(els.bfsSearchCap);
  if (cap === null) {
    state.bfsCapMode = "default_p95";
  } else if (cap === -1) {
    state.bfsCapMode = "uncapped";
  } else {
    state.bfsCapMode = "custom";
  }
  renderBfsCapInfo();
}

async function updateSuggestions(kind) {
  const input = kind === "source" ? els.sourceInput : els.targetInput;
  const list = kind === "source" ? els.sourceSuggestions : els.targetSuggestions;
  const query = input.value.trim();
  const requestId = ++state.suggestionRequestIds[kind];
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
    if (requestId !== state.suggestionRequestIds[kind]) return;
    list.innerHTML = "";
    for (const node of payload.nodes) {
      const option = document.createElement("option");
      option.value = node;
      list.append(option);
    }
  } catch (error) {
    if (requestId !== state.suggestionRequestIds[kind]) return;
    showError(error.message);
  }
}

async function runInference(event) {
  event.preventDefault();
  hideError();
  clearPath();

  const method = selectedMethod();
  if (!method) {
    showError("Select a search method.");
    return;
  }
  if (!isMethodSupportedForGraph(method, els.graphSelect.value)) {
    showError(unsupportedMethodMessage(method));
    return;
  }

  const body = {
    algorithm: method.algorithm,
    graph_id: els.graphSelect.value,
    source: els.sourceInput.value.trim(),
    target: els.targetInput.value.trim(),
    config: configForSelectedMethod(method),
  };

  if (!body.source || !body.target) {
    showError("Select both a cause and effect concept.");
    return;
  }

  const runLabel = method.algorithm === "astar"
    ? `${method.label} (d=${selectedAstarDimension()})`
    : resultMethodLabel(method);
  setBusy(true, `Running ${runLabel}...`);
  setStatus(`Running ${runLabel}...`, "idle");
  try {
    const result = await apiPost("/api/infer", body);
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

function configForSelectedMethod(method) {
  if (method.algorithm === "bfs") {
    return { cap: readIntegerInput(els.bfsSearchCap) };
  }
  if (method.algorithm === "rl") {
    return { policy_config_id: method.config.policy_config_id };
  }
  return {
    model_config_id: method.id,
    dimension: selectedAstarDimension(),
    astar_max_visits: readIntegerInput(els.astarMaxVisits),
    embedding_index_min_successors: readIntegerInput(els.embeddingThreshold),
  };
}

async function loadAdvancedDefaults() {
  const method = selectedMethod();
  if (!method || !isMethodSupportedForGraph(method, els.graphSelect.value)) {
    renderAdvancedSettings();
    return;
  }

  const params = new URLSearchParams({
    graph: els.graphSelect.value,
    algorithm: method.algorithm,
  });
  if (method.algorithm === "astar") {
    params.set("method", method.id);
    params.set("dim", String(selectedAstarDimension()));
  }

  try {
    const payload = await apiGet(`/api/config?${params.toString()}`);
    if (payload.algorithm === "bfs") {
      state.bfsDefaults.set(els.graphSelect.value, payload);
      if (state.bfsCapMode === "default_p95" || !els.bfsSearchCap.value.trim()) {
        els.bfsSearchCap.value = String(payload.bfs_cap);
        state.bfsCapMode = "default_p95";
      }
      renderBfsCapInfo(payload);
    } else if (payload.algorithm === "rl") {
      renderRlDetails(payload.policy);
    } else {
      els.astarMaxVisits.value = String(payload.astar_max_visits);
      els.astarMaxVisitsSource.textContent = payload.astar_max_visits_source;
      els.embeddingThreshold.value = String(payload.embedding_index_min_successors);
      renderAstarDetails({
        ...method.config,
        dimension: payload.model?.selected_dim ?? selectedAstarDimension(),
      });
    }
  } catch (error) {
    if (method.algorithm === "bfs") {
      els.bfsSearchCap.value = "-1";
      state.bfsCapMode = "uncapped";
      renderBfsCapInfo();
    } else if (method.algorithm === "astar") {
      els.astarMaxVisits.value = "-1";
      els.astarMaxVisitsSource.textContent = "uncapped; default lookup failed";
      els.embeddingThreshold.value = "16";
    }
    showError(error.message);
  }
}

function renderResult(result) {
  els.metricHops.textContent = String(result.hops);
  els.metricVisited.textContent = formatNumber(result.visited_nodes);
  els.metricRuntime.textContent = `${formatNumber(result.runtime_ms)} ms`;
  els.pathList.innerHTML = "";
  els.pathList.className = `path-list ${result.found ? "path-found" : "path-missing"}`;
  state.lastPathNodes = new Set(result.path || []);

  if (result.found) {
    setResultStatus(result.config_label, "Path found", "found");
    for (const node of result.path) {
      const item = document.createElement("li");
      item.textContent = node;
      els.pathList.append(item);
    }
  } else {
    setResultStatus(
      result.config_label,
      formatTermination(result.termination_reason),
      "missing",
    );
    const item = document.createElement("li");
    item.textContent = "No path found";
    els.pathList.append(item);
  }
}

function clearPath() {
  state.lastPathNodes = new Set();
  els.metricHops.textContent = "-";
  els.metricVisited.textContent = "-";
  els.metricRuntime.textContent = "-";
  els.pathList.innerHTML = "";
  els.pathList.className = "path-list";
}

function renderBfsCapInfo(payload = null) {
  const graphId = els.graphSelect.value;
  const defaults = payload || state.bfsDefaults.get(graphId);
  const graphOption = selectedGraphOption();
  const defaultCap = defaults?.bfs_cap ?? graphOption?.bfs_p95_cap ?? -1;
  const defaultSource = defaults?.bfs_cap_source || graphOption?.bfs_p95_cap_source || "";
  const currentCap = readIntegerInput(els.bfsSearchCap);

  if (state.bfsCapMode === "default_p95" && !els.bfsSearchCap.value.trim()) {
    els.bfsSearchCap.value = String(defaultCap);
  }

  if (state.bfsCapMode === "uncapped" || currentCap === -1) {
    els.bfsCapMode.textContent = "BFS (uncapped)";
    els.bfsCapSource.textContent = "-1 disables the visited-node cap.";
  } else if (state.bfsCapMode === "default_p95") {
    els.bfsCapMode.textContent = "BFS (p95 cap)";
    els.bfsCapSource.textContent = defaultSource;
  } else {
    els.bfsCapMode.textContent = "BFS (custom cap)";
    els.bfsCapSource.textContent = `Current p95 default for this graph: ${formatNumber(defaultCap)}.`;
  }
}

function renderRlDetails(config) {
  renderDetails(els.rlDetails, [
    ["Policy", config.policy_label || "RL"],
    ["Description", config.description],
    ["Checkpoint", config.checkpoint],
    ["Parameters", config.parameters],
    ["Beam width", config.rl_beam_width],
    ["Max path length", config.rl_max_path_len],
    ["Max actions", config.rl_max_actions],
    ["Max visits", formatCap(config.rl_max_visits)],
  ]);
}

function renderAstarDetails(config) {
  renderDetails(els.astarDetails, [
    ["Embedding", config.model_label],
    ["Identifier", config.model_identifier],
    ["Parameters", config.parameters],
    ["Dimension", config.dimension],
    ["Activation", labelValue(config.activation)],
    ["Distance", labelValue(config.distance)],
    ["Variant", labelValue(config.variant)],
  ]);
}

function renderDetails(container, rows) {
  container.innerHTML = "";
  for (const [label, value] of rows) {
    if (value === null || value === undefined || value === "") continue;
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = String(value);
    container.append(term, description);
  }
}

function setStatus(text, mode) {
  els.resultStatus.textContent = text;
  els.resultStatus.className = `result-status ${mode}`;
}

function setResultStatus(modelLabel, outcome, mode) {
  const copy = document.createElement("span");
  copy.className = "result-status-copy";

  const model = document.createElement("span");
  model.className = "result-status-model";
  model.textContent = modelLabel;
  model.title = modelLabel;

  const result = document.createElement("span");
  result.className = "result-status-outcome";
  result.textContent = outcome;

  copy.append(model, result);
  els.resultStatus.replaceChildren(copy);
  els.resultStatus.className = `result-status has-summary ${mode}`;
}

function setBusy(isBusy, caption = null) {
  state.busy = isBusy;
  els.graphSelect.disabled = isBusy;
  els.methodSelect.disabled = isBusy;
  els.astarDimensionSelect.disabled = isBusy;
  els.bfsSearchCap.disabled = isBusy;
  els.astarMaxVisits.disabled = isBusy;
  els.embeddingThreshold.disabled = isBusy;
  if (caption) els.caption.textContent = caption;
  validateSelectedMethod();
}

function validateSelectedMethod() {
  const method = selectedMethod();
  if (!method) {
    els.methodValidation.textContent = "";
    els.runButton.disabled = true;
    return false;
  }
  const supported = isMethodSupportedForGraph(method, els.graphSelect.value);
  els.methodValidation.textContent = supported ? "" : unsupportedMethodMessage(method);
  els.runButton.disabled = state.busy || !supported;
  return supported;
}

function selectedMethod() {
  return state.options?.methods.find((method) => method.id === els.methodSelect.value);
}

function selectedAstarDimension() {
  const dimension = Number.parseInt(els.astarDimensionSelect.value, 10);
  return Number.isInteger(dimension) ? dimension : null;
}

function selectedGraphOption() {
  return state.options?.graphs.find((graphOption) => graphOption.id === els.graphSelect.value);
}

function isMethodSupportedForGraph(method, graphId) {
  return method?.supported_graphs?.includes(graphId);
}

function unsupportedMethodMessage(method) {
  const graphOption = selectedGraphOption();
  return `${method.label} is not supported for ${graphOption?.label || els.graphSelect.value}.`;
}

function resultMethodLabel(method) {
  if (method.algorithm === "bfs") {
    if (state.bfsCapMode === "uncapped") return "BFS (uncapped)";
    if (state.bfsCapMode === "custom") return "BFS (custom cap)";
    return "BFS (p95 cap)";
  }
  return method.label;
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

function formatCap(value) {
  if (value === null || value === undefined) return "-";
  if (Number(value) === -1) return "uncapped";
  return formatNumber(value);
}

function formatTermination(reason) {
  const labels = {
    path_found: "path found",
    target_unreachable: "target unreachable",
    cap_reached: "cap reached",
    frontier_exhausted: "frontier exhausted",
    rl_policy_terminated: "RL policy terminated",
    invalid_source_or_target: "invalid source or target",
    error: "error",
  };
  return labels[reason] || reason || "no path found";
}

function labelValue(value) {
  if (!value) return value;
  const labels = {
    relu: "ReLU",
    gelu: "GELU",
    cosine: "Cosine",
    euclid: "Euclidean",
    euclidean: "Euclidean",
    base: "Base",
    finetuned: "Fine-tuned",
    ablation: "Ablation",
  };
  return labels[value] || value;
}

function readIntegerInput(input) {
  if (!input.value.trim()) return null;
  return Number.parseInt(input.value, 10);
}
