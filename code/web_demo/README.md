# Causal Graph A* Web Demo

Run from the repository root:

```powershell
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:8000
```

The demo serves a Webis-template-based page and API endpoints from
`web_demo.server`. The web demo currently exposes only `causenet` and
`causalbank`; the full graph variants remain available to CLI
training/evaluation code, but are not accepted by the demo API. A* inference
calls the existing repository path:
`core.utils.traverse_graph(..., traverse_strategies.astar_traverse, ...)`.

Graph loading cache:

```text
Startup preloads the enabled web-demo graphs before the UI is served.
This keeps A* runtime measurements from including graph load/deserialization.
The loaded NetworkX graphs are reused for all later requests in that server run.
The parsed graphs are also serialized under data/cache/web_demo_graphs/ so later
server starts can restore them without reparsing the raw JSONL/TXT graph.
```

Model warmup:

```text
Startup also preloads every model exposed by the UI and prepares one runtime
embedding table plus indexed graph adjacencies for the enabled demo graphs.
This moves the first-request model/index setup cost to server startup for all
UI model choices. Each table is built at the model's full dimension, so lower
matryoshka dimensions for that same model reuse it by slicing instead of
loading a second model. Startup also preloads the A* config defaults and runs a
tiny real one-hop A* traversal for each exposed graph/model/dimension choice so
the first UI inference does not parse the large p95 config file or initialize
the traversal path lazily.
```

The disk cache is invalidated automatically when the source graph file path,
size, or modification time changes. To choose a subset of the enabled demo
graphs, set `WEB_DEMO_GRAPHS` to `causenet`, `causalbank`, or
`causenet,causalbank`. Full graph variants are ignored by the web demo. To
bypass the parsed graph disk cache for debugging:

```powershell
$env:WEB_DEMO_DISABLE_GRAPH_DISK_CACHE="1"
```

Initial graph previews use a separate graph-specific cache under
`data/cache/web_demo_graph_previews/`. The older
`data/cache/web_demo_visuals/` cache is ignored and can be deleted:

```powershell
Remove-Item -Recurse -Force data\cache\web_demo_visuals
```

The advanced settings panel is collapsed by default. It exposes only A* config
keys already used by evaluation:

```text
astar_max_visits
embedding_index_min_successors
```

`embedding_index_min_successors` defaults to 16. Benchmarks showed that
`embed_many` helps medium and dense successor batches, but tiny batches are
still better served by single-row lookups.

`astar_max_visits` defaults to the matching `msmarco_train` p95 successful A*
cap when a local `visited_nodes_analysis.json` entry exists, otherwise `-1`
for uncapped traversal.

Optional environment variables:

```powershell
$env:WEB_DEMO_EMBEDDING_DEVICE="cpu"
$env:WEB_DEMO_ASTAR_MAX_VISITS="50000"
$env:WEB_DEMO_SUBGRAPH_LIMIT="240"
$env:WEB_DEMO_PRELOAD_MODELS="none"
$env:WEB_DEMO_MODEL_WARMUP_BATCH_SIZE="64"
```
