# Causal Graph Inference Web Demo

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

Choose the startup model scope in `app.py`:

```python
load_all = False  # CauseNet Precision + Granite FT d=32 + A* only
load_all = True   # every graph, A* model/dimension, BFS, and RL
```

The limited setting constrains Granite to `d=32` before its embedding index is
allocated, exposes only CauseNet Precision, and skips BFS and RL. This is
appropriate for the local 12 GB GPU. Change it to `True` on a server with
sufficient GPU memory to expose and preload everything.

The demo serves a Webis-template-based page and API endpoints from
`web_demo.server`. Limited mode exposes only CauseNet Precision. Full mode
exposes all three causal graphs:

```text
CauseNet Precision   80,214 nodes      197,376 edges
CauseNet Full    12,185,920 nodes   11,606,975 edges
CEG Filtered         77,264 nodes   21,507,177 edges
```

Graph metadata is centralized in `core.graph_config`. Each entry stores the
stable graph ID, display name, backend path, node/edge counts, supported
algorithms, BFS p95 cap, cache suffix, and node-universe setting.

Supported search methods:

```text
BFS
RL
A* embedding configurations
```

BFS and A* call the existing repository traversal path through
`core.utils.traverse_graph(...)`. RL uses the existing LSTM policy baseline from
the evaluation code: `DEFAULT_RL_MODEL_PATH`, beam width 50, max path length 2,
max actions 5000, and `rl_max_visits=-1`.

Request schema:

```json
{
  "algorithm": "bfs",
  "graph_id": "causenet",
  "source": "...",
  "target": "...",
  "config": { "cap": -1 }
}
```

Use `algorithm: "rl"` with `config.policy_config_id`, or `algorithm: "astar"`
with `config.model_config_id` and `config.dimension`.

BFS cap behavior:

```text
-1  uncapped BFS
0+  stop when the configured visited-node cap is reached
```

The default BFS cap is resolved from the evaluation
`visited_nodes_analysis.json` p95 value when available, then from the central
graph registry fallback. In this checkout the central evaluation workflow uses
`causenet/msmarco_train/v3` as the p95 source, so the exposed graphs currently
resolve to cap `12,170`.

Model labels and method identities are centralized in `core.model_registry`.
Visible labels use the canonical short names:

```text
MPNet Base
BGE Base
Granite FT ReLU+Euclidean
Granite AB ReLU+Cosine
```

The search-method dropdown has one entry for each model configuration, such as
MPNet Base or MPNet FT. It does not repeat models for every Matryoshka
dimension. Selecting an A* model reveals a separate dimension dropdown directly
below it. Granite's normal fine-tuned model and all available activation/distance
ablation models are included.

The result area shows only Hops, Visited, and Runtime metrics.

Graph loading cache:

```text
Before the server accepts requests, it loads every enabled graph, builds and
retains each initial preview in memory, preloads the A* runtimes and indexes, and
preloads the RL runtime and graph views. The selected graph ID is still part of
every request and result. Parsed graphs are also serialized under
data/cache/web_demo_graphs/. If a selected runtime cannot be prepared, startup
fails rather than serving a page that would incur a delayed first request.
```

The disk cache is invalidated automatically when the source graph file path,
size, or modification time changes.

```powershell
$env:WEB_DEMO_DISABLE_GRAPH_DISK_CACHE="1"
```

Initial graph previews use a separate graph-specific disk cache under
`data/cache/web_demo_graph_previews/` and an in-memory cache during the running
server process, so opening or switching back to a graph does not repeat preview
construction.

A* model warmup:

```text
Startup preloads every A* model exposed by the UI and prepares one runtime
embedding table plus indexed graph adjacencies for each compatible graph node
universe. CauseNet Full therefore uses its own full-node embedding cache rather
than the smaller merged CauseNet/CEG cache. Each table is built at the
model's full dimension, so lower Matryoshka dimensions reuse it by slicing.
Startup also preloads A* p95 defaults and dynamically selects a two-hop query
from each graph's topology for every exposed model and dimension. The warmup
uses the same visit cap as normal inference and contains no fixed concept names.

Cause and effect inputs retain their graph-backed autocomplete lists. Clicking
a graph node fills the active cause/effect input and requests that node's
two-hop neighborhood.
```

Optional environment variables:

```powershell
$env:WEB_DEMO_EMBEDDING_DEVICE="cpu"
$env:WEB_DEMO_ASTAR_MAX_VISITS="50000"
$env:WEB_DEMO_SUBGRAPH_LIMIT="240"
$env:WEB_DEMO_PRELOAD_MODELS="none"
$env:WEB_DEMO_MODEL_WARMUP_BATCH_SIZE="128"
```
