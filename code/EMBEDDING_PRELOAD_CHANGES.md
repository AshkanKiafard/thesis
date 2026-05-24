# Embedding and Preload Changes

This note summarizes the embedding-cache and graph-preload changes made for the
CauseNet full / Qwen finetuned / Matryoshka dim 128 evaluation runs.

## Problem

The old cache format used `np.save(...)` on a Python dictionary. That makes NumPy
store the cache through pickle. For a huge graph cache, this had two practical
problems:

- A partially written cache could fail later with `EOFError: Ran out of input`.
- Loading a dictionary with millions of embeddings is memory-heavy and cannot be
  memory-mapped cleanly.

For the explicit test runs, we also only need the selected best model dimension
(`--best-model-dim 128`). Preloading full embeddings first wastes RAM/GPU memory.

## Files Changed

- `core/embeddings.py`
- `core/pre_embed.py`
- `core/indexed_graph.py`
- `evaluation/evaluation.py`
- `traverse_strategies/astar.py`
- `traverse_strategies/dijkstra.py`

## Cache Format

The new cache format separates text keys and vectors:

- `*_embeddings_texts.jsonl`
- `*_embeddings_vectors.npy`

The `.npy` vector file is saved with `allow_pickle=False`, so it can be loaded
with NumPy memory mapping. The text file stores one graph node string per line in
the same order as the vector rows.

Example for the full CauseNet cache:

```text
data/embeddings/Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_best_v2_finetuned_causenet_full_embeddings_texts.jsonl
data/embeddings/Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_best_v2_finetuned_causenet_full_embeddings_vectors.npy
```

Example for the 128-dim sliced CauseNet cache:

```text
data/embeddings/Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_best_v2_finetuned_causenet_full_dim128_embeddings_texts.jsonl
data/embeddings/Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_best_v2_finetuned_causenet_full_dim128_embeddings_vectors.npy
```

## Legacy Cache Compatibility

Old pickle `.npy` caches are still readable as a fallback.

Load order:

1. Try the new non-pickle `texts.jsonl` + `vectors.npy` pair.
2. If that does not exist, try the old legacy pickle `.npy` cache.
3. If a cache is corrupt, move it aside as `*.corrupt.<timestamp>` and continue.

New saves use the non-pickle format.

## Corrupt Cache Handling

If NumPy raises errors such as `EOFError: Ran out of input`, `ValueError`, or
file read errors while loading a legacy cache, the code no longer crashes the
whole run immediately. It renames the broken file to a `.corrupt.<timestamp>`
name and rebuilds what it needs.

This does not restore a corrupt pickle cache. It only prevents the corrupt file
from being reused.

## Memory Mapping

The non-pickle vector `.npy` files are loaded with:

```python
np.load(..., allow_pickle=False, mmap_mode="r")
```

That means the full vector file does not need to be loaded into RAM as one giant
Python object. NumPy can read rows from disk as needed.

## Best-Model Dim Before Preload

For explicit test runs with:

```text
--best-model-path ...
--best-model-dim 128
```

`evaluation/evaluation.py` now sets the embedder Matryoshka dimension to `128`
before graph preload starts.

That means CauseNet full preload uses the active 128 dimensions instead of first
building full model-dimensional tensors.

## Sliced Preload Cache

When the active dimension is smaller than the model dimension, preload now uses a
sliced index path:

1. Build one big torch embedding table with shape:

   ```text
   number_of_graph_nodes x active_dim
   ```

2. For dim 128, save that same sliced table as a reusable dim-specific cache.
3. On the next eval run with the same model, graph, and dim, load this dim cache
   directly instead of encoding all graph nodes again.

This matters for running both `msmarco_test` and `sem_test`: the first run can
create the 128-dim cache, and the second run can reuse it.

## Atomic Cache Writes

Cache saves write to temporary files first:

```text
*.tmp.<pid>
```

Only after the vector file and text file are fully written does the code replace
the final cache files.

This lowers the chance of leaving a final-looking cache file that is only half
written.

## Integer Graph Indices

`core/indexed_graph.py` adds an indexed graph wrapper.

During preload:

1. The graph nodes are assigned integer row ids.
2. The embedding table stores the vector for each node at that row id.
3. A*/Dijkstra can fetch successor embeddings by integer ids instead of looking
   up strings and stacking many tensors repeatedly.

The traversal still returns normal text paths. The integer ids are only a runtime
speed/memory optimization.

For the same graph file and same graph loading order, the node order is stable.
The important part is that the graph index and embedding table are built together
in the same preload step, so row `i` always refers to the same node for that run.

## A*/Dijkstra Fast Path

`traverse_strategies/astar.py` and `traverse_strategies/dijkstra.py` now use the
indexed graph when `evaluation.py` passes `_indexed_graph` in the runtime config.

If no indexed graph exists, the old string-node path is still used.

## What Did Not Change

Evaluation outputs are not rewritten just because this code changed.

Existing result JSON/CSV files stay untouched unless you rerun with force flags
or missing algorithms/dimensions need to be filled.

The expected F1/precision/recall logic did not change. These changes target
cache format, preload memory behavior, and traversal embedding lookup speed.

You do not need to rerun old evaluation results only because of this change,
unless you are comparing runtime/efficiency numbers.

## Commands for the Two CauseNet Full Test Runs

First run, which can create the dim-128 cache:

```bash
python -m evaluation.evaluation data/datasets/filtered/msmarco_test_filtered.json --run-suffix best_v2 --graph causenet_full --config-source-dataset msmarco_train --config-source-graph causenet --best-model-path data/models/lightning/Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_best_v2_finetuned --best-model-dim 128
```

Second run, which should reuse the dim-128 cache:

```bash
python -m evaluation.evaluation data/datasets/filtered/sem_test_filtered.json --run-suffix best_v2 --graph causenet_full --config-source-dataset msmarco_train --config-source-graph causenet --best-model-path data/models/lightning/Qwen3-Embedding-0.6B_relu_euclid_nonorm_matryoshka_best_v2_finetuned --best-model-dim 128
```

Do not pass `--no-save-embedding-cache` if you want the first run to create the
cache for the second run.

## TLDR

The code now avoids pickle caches, memory-maps embedding vectors, handles corrupt
old caches, sets dim 128 before preloading best-model test runs, saves a reusable
dim-128 CauseNet full cache, and lets A*/Dijkstra use integer graph indices for
faster embedding lookup.

## ELI5

Before, the code kept embeddings in one huge fragile box. If the box broke, the
whole run failed, and opening it used too much memory.

Now the code keeps a list of node names and a table of numbers. For dim 128 it
stores only the 128 numbers we actually need. The first eval fills that table;
the second eval opens the table and starts much faster.
