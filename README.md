# Efficient Causal Question Answering with Learned A* Heuristics

A Python/PyTorch project for binary causal question answering. Given a question such as **“Does A cause B?”**, it searches for a directed path from `A` to `B` in a causal knowledge graph. Fine-tuned Sentence Transformer embeddings guide bounded A* search on graphs as large as CauseNet Full, which contains 12.2 million nodes.

**Core stack:** Python · PyTorch · PyTorch Lightning · Sentence Transformers · NetworkX · Optuna · FastAPI · 3d-force-graph

> Instead of learning a traversal policy, this project learns the heuristic used by A*.

On the 12.2-million-node CauseNet Full graph, the selected A* configuration averages **7.5 visited nodes and 7.2 ms per MS MARCO query**. Uncapped BFS averages 30,099.3 visited nodes and 59.5 ms, with F1 scores of 82.9 and 89.9 respectively.

## Problem

A graph-based answer can include the path that connects a cause to an effect. The difficult part is finding that path without exploring a large part of the graph. Uncapped breadth-first search (BFS) is exhaustive, but a single query can visit thousands or even tens of thousands of nodes. The evaluated RL baseline explores less of the graph, but repeatedly runs a neural policy during beam search.

This project keeps traversal explicit with A* and learns an embedding space that steers the search toward the target. Graph-node embeddings are computed ahead of time, so a query mainly requires indexed lookups, distance calculations, and priority-queue operations. A visit budget limits the worst-case work on large or dense graphs.

## How it works

Training and evaluation are split into five stages:

```mermaid
flowchart TB
  subgraph Data["1. Build path-derived training data"]
    PAIRS["Positive MS MARCO train/validation pairs"] --> COVER["Keep pairs covered by CauseNet Precision"]
    COVER --> BASE["Run uncapped A* with each base embedding model"]
    BASE --> PATHS["Keep successful paths"]
    PATHS --> QUADS["Create ranking examples:<br/>current · target · preferred successor · alternative successor"]
  end

  subgraph Tuning["2. Tune each backbone"]
    QUADS --> OPTUNA["30–50 budget-counting Optuna trials per backbone:<br/>learning rate · activation · distance"]
    OPTUNA --> TRIALVAL["Choose the trial with the lowest<br/>validation average visited nodes"]
  end

  subgraph Training["3. Train all models and precompute embeddings"]
    TRIALVAL -->|Best hyperparameters per backbone| TRAIN["Final Matryoshka training<br/>for all five backbones"]
    TRAIN --> CHECKPOINT["Keep each backbone's checkpoint with the<br/>lowest validation average visited nodes"]
    CHECKPOINT --> CACHE["Precompute graph-node embeddings<br/>for every evaluated prefix dimension"]
  end

  subgraph Selection["4. Evaluate and select on validation"]
    CACHE --> BUDGET["Set one A* budget per model/dimension<br/>from successful training searches"]
    BUDGET --> MODELEVAL["Evaluate every fine-tuned model × dimension<br/>on MS MARCO validation"]
    MODELEVAL --> FRONTIER["Discard effectiveness–efficiency<br/>Pareto-dominated candidates"]
    FRONTIER --> KNEE["Select the F1–log-visited-nodes<br/>Pareto knee"]
    KNEE --> SELECTED["Granite R2 · d = 64<br/>ReLU · Euclidean · τ = 23"]
  end

  subgraph Use["5. Final evaluation and inference"]
    SELECTED --> TEST["Compare with BFS and RL<br/>on the held-out test sets"]
    SELECTED --> QUESTION["Question: Does A cause B?"]
    QUESTION --> MAP["Resolve A and B to graph nodes"]
    MAP --> ASTAR["Run bounded A* with cached 64D embeddings"]
    ASTAR --> REACHED{"Target reached?"}
    REACHED -->|Yes| YES["Yes + causal path"]
    REACHED -->|No| NO["No within the search budget"]
  end
```

1. **Build the training data.** For each base model, positive MS MARCO pairs covered by CauseNet Precision are searched with uncapped A*. A successful path produces `(current node, target, preferred successor, alternative successor)` examples. Cosine and Euclidean searches are generated separately because they can produce different paths.

2. **Tune each embedding model.** Optuna first runs at least 30 budget-counting terminal trials per backbone over the learning rate, activation (`ReLU` or `GELU`), and distance metric (`cosine` or `Euclidean`). Only `COMPLETE` and `PRUNED` trials count. Thereafter, trials launch one at a time until five consecutive counting trials fail to improve the best completed visited-node objective, with a hard maximum of 50; a new completed best resets the counter, while failed or interrupted trials consume neither budget nor counter. This step chooses training hyperparameters for each backbone; it does not choose the final backbone or embedding dimension.

3. **Train and cache every candidate.** All five backbones are trained with their own best hyperparameters for up to 50 epochs. For each backbone, the checkpoint with the lowest validation average visited-node count is kept. The Matryoshka objective sums the task/ranking loss over all configured prefixes without averaging or self-normalization and applies the existing embedding regularizer once globally. The resulting graph-node embeddings are stored in indexed, memory-mapped NumPy arrays for each evaluated dimension.

4. **Select one configuration on validation data.** Each fine-tuned model and Matryoshka dimension receives its own A* budget, calculated as the 95th percentile of visited nodes on successful MS MARCO training searches. The candidates are then evaluated on MS MARCO validation with CauseNet Precision. Pareto-dominated configurations are discarded using F1 and average visited nodes; the normalized F1–log-visited-nodes knee on the remaining frontier identifies the point of diminishing returns without an arbitrary effectiveness cutoff. Granite Embedding R2 at 64 dimensions is selected. Its tuned setup uses ReLU and Euclidean distance, and its search budget is `τ = 23`.

5. **Evaluate the selected system.** The Granite configuration is frozen before the final comparison with BFS and RL on the MS MARCO and SemEval test sets across three graphs. Test data are not used for model selection. Binary evaluation uses reachability-only A*/BFS: it returns as soon as the target is discovered and avoids parent maps and path reconstruction. The interactive frontend uses path mode, preserving the explicit causal path for interpretation.

For a candidate node `n`, A* prioritizes the smallest estimated total cost:

```text
f(n) = g(n) + h(n)

g(n): accumulated embedding distance from the source to n
h(n): Euclidean embedding distance from n to the target
```

In the selected configuration, adjacent concepts are connected by their Euclidean embedding distance, and the Euclidean distance from `n` to the target provides the heuristic. Fine-tuning reshapes the embedding space so that useful successors tend to receive better A* priorities than competing neighbors.

The same objective is applied to nested Matryoshka prefixes. This makes the first 64 dimensions useful on their own, reducing stored vector width and distance-computation work by **12×** relative to Granite’s native 768-dimensional representation.

### Final selected configuration

| Component | Selection |
|---|---|
| Embedding backbone | `ibm-granite/granite-embedding-english-r2` |
| Representation | 64-dimensional Matryoshka prefix |
| Training activation | ReLU |
| Search distance | Euclidean |
| A* visit budget | `τ = 23` |
| Selection data | MS MARCO validation with CauseNet Precision |
| Validation result | 83.5 F1 · 9.1 visited nodes · 1.5 ms/query |
| Selection rule | F1–log-visited-nodes Pareto knee after removing dominated candidates |

## Example: “Does sleep deprivation cause cancer?”

The interactive demo can return the directed path:

```text
sleep deprivation → stress → cancer
```

The answer is **Yes** because the graph contains a directed path from the source to the target. The path explains the graph lookup; it is not independent proof that the causal claim is true in the real world.

## Results

The final system was evaluated on graph-covered MS MARCO and SemEval questions across CauseNet Precision, CauseNet Full, and CEG Filtered. The table reports the **MS MARCO test** results.

| Graph | Method | F1 | Avg. visited nodes | Time/query |
|---|---|---:|---:|---:|
| CauseNet Precision | Selected A* | 86.7 | **6.5** | 1.1 ms |
|  | BFS, uncapped | **90.6** | 1,157.3 | 1.4 ms |
|  | BFS, capped | 90.2 | 134.9 | **0.5 ms** |
|  | RL baseline | 70.3 | 32.0 | 67.7 ms |
| CauseNet Full | Selected A* | 82.9 | **7.5** | 7.2 ms |
|  | BFS, uncapped | **89.9** | 30,099.3 | 59.5 ms |
|  | BFS, capped | 87.2 | 135.1 | **9.4 ms** |
|  | RL baseline | 69.9 | 44.6 | 658.1 ms |
| CEG Filtered | Selected A* | **92.0** | **2.3** | **0.8 ms** |
|  | BFS, uncapped | 90.9 | 492.1 | 44.3 ms |
|  | BFS, capped | **92.0** | 30.2 | 5.8 ms |
|  | RL baseline | 84.6 | 24.3 | 5,129.0 ms |

The clearest efficiency gain appears on CauseNet Full. Relative to uncapped BFS, A* visits about 4,011× fewer nodes and runs 8.2× faster on MS MARCO; on SemEval it visits about 11,308× fewer nodes and runs 18.9× faster. The corresponding F1 differences are -7.1 and +9.5 points. Compared with capped BFS, A* visits 18.0× to 31.7× fewer nodes; its F1 is 4.3 points lower on MS MARCO and 0.6 points lower on SemEval.

On CEG Filtered / MS MARCO, A* ties capped BFS for the highest F1 and visits about 213× fewer nodes than uncapped BFS. That F1 difference is not statistically significant after correction. Across all six graph/dataset test settings, A* has higher F1 and lower runtime than the evaluated LSTM-based RL baseline.

Runtime was measured on Webis SLURM jobs with eight CPU cores, 64 GB RAM, and one Hopper GPU; graph loading and embedding-cache loading were excluded. Visited nodes are therefore the more hardware-independent efficiency measure. CEG Filtered covers 48 MS MARCO test examples, so its F1 result should be interpreted with that sample size in mind. RL visited-node accounting follows the baseline implementation and differs from BFS/A*, so direct node-count comparisons with RL are less meaningful.

### Why 64 dimensions?

The validation sweep compares F1, the p95 search budget, visited nodes, and runtime across Matryoshka dimensions. Granite at 64 dimensions reached 83.5 F1 with 9.1 visited nodes and 1.5 ms per query. Smaller prefixes can reduce search effort further, while larger prefixes can gain effectiveness, but the normalized Pareto-knee criterion selects 64 dimensions as the empirical point of diminishing returns between these objectives.

![Validation performance across Matryoshka embedding dimensions](assets/readme/matryoshka-validation.png)

*Validation on MS MARCO with CauseNet Precision. Cost-related axes use logarithmic scales. Figure adapted from the thesis.*

Exact final-test values are in [`thesis/tables/test_res.tex`](thesis/tables/test_res.tex). Evaluation runs write their machine-readable results to `code/data/evaluation/`.

## Technical highlights

- Successful A* paths are converted into ranking examples for fine-tuning the embedding heuristic.
- A Matryoshka training objective makes several prefix dimensions usable from the same checkpoint.
- A*/BFS share path and reachability-only modes, avoiding path-specific allocation during binary evaluation.
- Precomputed NumPy memmaps and indexed adjacency structures avoid encoder inference during graph traversal.
- The evaluation covers fine-tuned and pretrained embeddings, capped and uncapped BFS, Dijkstra, and an LSTM-based RL baseline.
- The repository includes preprocessing, Optuna tuning, PyTorch Lightning training, model selection, statistical tests, plots, and an interactive FastAPI demo.

## Interactive demonstration

The FastAPI demo uses 3d-force-graph to display the discovered path and its surrounding graph neighborhood. It also reports the path length, visited nodes, and runtime for the selected search configuration.

Hosted demo: [causalpathfinding.web.webis.de](https://causalpathfinding.web.webis.de/)

![Interactive causal path visualization](assets/readme/causal-path-demo.png)

### Run locally

The default local launcher uses the selected Granite model at `d=64` on CauseNet Precision:

```powershell
cd code
python app.py
```

Open `http://127.0.0.1:9000`. The demo requires the CauseNet Precision graph, selected exported model directory, and matching embedding cache under `code/data/`. In [`code/app.py`](code/app.py), `load_all = False` keeps startup to the selected A* configuration; setting it to `True` preloads all supported graphs, A* variants, BFS, and RL and requires substantially more memory.

The frontend loads 3d-force-graph and supporting UI libraries from CDNs, so the browser also needs internet access when the demo page starts.

## Repository structure

```text
.
├── assets/readme/         # Images embedded by this README
├── code/
│   ├── core/                 # Graph/model registries, embedding caches, indexed inference
│   ├── traverse_strategies/  # A*, BFS, Dijkstra, and RL traversal
│   ├── finetune/             # Training data, Optuna search, final training, ablations
│   ├── evaluation/           # Evaluation, model selection, statistics, and visualization
│   ├── preprocessing/        # Dataset normalization and CEG filtering
│   ├── web_demo/             # FastAPI API and 3d-force-graph frontend
│   ├── tests/                # Cache, graph, registry, and reporting tests
│   ├── data/                 # Local reproducibility artifacts; ignored by Git except placeholders
│   │   ├── cache/            # Generated web-demo/report caches; contents are not released
│   │   ├── checkpoints/      # Raw Lightning checkpoints; optional and not released
│   │   ├── datasets/         # Raw and normalized evaluation datasets
│   │   ├── docker/           # Runtime/deployment data workspace
│   │   ├── docs/             # SHA-256 checksum allowlist for the release archive
│   │   ├── embeddings/       # Node indices and memory-mapped embedding matrices
│   │   ├── evaluation/       # JSON/CSV results and p95 analyses
│   │   ├── graphs/           # Provider-downloaded graph files; not included in the artifact
│   │   ├── lightning_logs/   # Generated training logs; not released
│   │   ├── models/           # Exported v4 models and RL checkpoint
│   │   ├── optuna_studies/   # Final v4 hyperparameter-search SQLite studies
│   │   ├── plots/            # Generated evaluation figures
│   │   └── reports/          # Generated tables and summaries
│   ├── app.py                # Local demo launcher
│   └── requirements.txt
└── thesis/
    ├── chapters/             # Thesis chapters
    ├── tables/               # Exact experimental tables
    └── figures/              # Thesis figures
```

## Installation

From the repository root:

```powershell
python -m venv .venv
# Activate .venv in the current shell, then:
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
python -m nltk.downloader punkt punkt_tab stopwords
cd code
```

A CUDA-capable GPU is recommended for training, embedding precomputation, and the complete evaluation. CPU execution is supported by the relevant `--embedding-device cpu` options, but full-graph runs are resource intensive.

## Reproducibility artifacts and graphs

Download the v4 reproducibility artifact as [`datasets.zip` from Google Drive](https://drive.google.com/file/d/108g4Sz_kOrKyDriroRmI8R6oqoDdgNzl/view) (approximately 30.6 GiB). Extract the archive into `code/data/`. A correctly extracted copy starts with this layout:

```text
code/data/
├── datasets/
├── docs/
├── embeddings/
├── evaluation/
├── models/
├── optuna_studies/
├── plots/
└── reports/
```

The Google Drive artifact contains:

| Archive path | Contents and purpose |
|---|---|
| `datasets/` | Raw MS MARCO/SemEval inputs and the normalized files under `filtered/` used by evaluation |
| `embeddings/` | Shared and CauseNet Full node-index JSONL files, retained base-model caches, v4 fine-tuned caches, and `glove.6B/glove.6B.300d.txt` for RL |
| `models/lightning/` | Five primary v4 SentenceTransformer exports and three Granite v4 ablation exports |
| `models/rl/msmarco_no_inverse_state_dict.pt` | Released RL baseline checkpoint |
| `evaluation/` | Final v4 JSON/CSV results, p95 visited-node analysis, significance tests, and budget-tradeoff results |
| `plots/` | Regenerated v4, ablation, tradeoff, and thesis plots |
| `reports/` | Machine-readable and LaTeX dataset, graph, hyperparameter, and p95 reports |
| `optuna_studies/hparam_search/` | The five final v4 Optuna SQLite studies used by `finetune_best` |
| `docs/v4-sha256sums.txt` | SHA-256 checksums for every payload file in the artifact |

The complete `plots/` tree and the five final Optuna studies under `optuna_studies/hparam_search/` are included. External causal graph files are distributed by their original providers and are downloaded separately as described below. Runtime caches, raw Lightning checkpoints, and training logs are not part of the reproducibility artifact.

Retraining and embedding precomputation are not required to inspect the measurements or regenerate the plots. Re-running an evaluation additionally requires the corresponding graph, exported model, node index, and embedding matrix. RL evaluation also uses the included checkpoint and 300-dimensional GloVe vectors.

### Verify the downloaded archive

`code/data/docs/v4-sha256sums.txt` uses the standard GNU `sha256sum` format and lists every payload file except the checksum file itself. From `code/data/`, verify the extracted artifact with:

```bash
sha256sum --check docs/v4-sha256sums.txt
```

This command is available directly on Linux and through WSL or Git Bash on Windows. Every line should report `OK`.

The required dataset files are:

```text
code/data/datasets/msmarco_train.json
code/data/datasets/msmarco_valid.json
code/data/datasets/msmarco_test.json
code/data/datasets/sem_test.csv
code/data/datasets/filtered/msmarco_train_filtered.json
code/data/datasets/filtered/msmarco_valid_filtered.json
code/data/datasets/filtered/msmarco_test_filtered.json
code/data/datasets/filtered/sem_test_filtered.json
```

The final selected model and its directly required caches are:

```text
code/data/models/lightning/granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v4_finetuned/
code/data/embeddings/merged_causenet_ceg_nodes.jsonl
code/data/embeddings/causenet_full_nodes.jsonl
code/data/embeddings/granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v4_finetuned_dim64_embeddings_vectors.npy
code/data/embeddings/granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v4_finetuned_causenet_full_dim64_embeddings_vectors.npy
code/data/evaluation/causenet/msmarco_train/v4/visited_nodes_analysis.json
```

All final test result files follow this layout:

```text
code/data/evaluation/<causenet|causenet_full|ceg>/<msmarco_test|sem_test>/v4/evaluation_results.{json,csv}
```

The validation/model-selection source is `code/data/evaluation/causenet/msmarco_valid/v4/evaluation_results.json`. All test graphs reuse the per-model/dimension p95 caps from `code/data/evaluation/causenet/msmarco_train/v4/visited_nodes_analysis.json`; this file gives Granite-64 `τ=23` and capped BFS `τ=1316`.

### Download the graphs

The original graph files are downloaded separately:

| Graph | Official source | Repository path |
|---|---|---|
| CauseNet Precision | [CauseNet downloads](https://causenet.org/) | `code/data/graphs/causenet-precision.jsonl` |
| CauseNet Full | [CauseNet downloads](https://causenet.org/) | `code/data/graphs/causenet-full.jsonl` |
| Cause Effect Graph (CEG) | [CausalBank / CEG](https://github.com/eecrazy/CausalBank) | `code/data/graphs/Lexical_Cause_Effect_Graph.txt` |

CauseNet downloads are distributed as `.jsonl.bz2`; decompress them and keep the filenames shown above. Download both CauseNet Precision and CauseNet Full to rerun all reported graph settings. Use the CausalBank **Cause Effect Graph**, not the sentence corpus. Place the raw `Lexical_Cause_Effect_Graph.txt` as shown, then create the filtered CEG from `code/`:

```powershell
python -m preprocessing.filter_ceg_graph
```

This writes `data/graphs/Lexical_Cause_Effect_Graph.filtered.txt`. The code’s current graph identifiers are `causenet`, `causenet_full`, and `ceg`; `causalbank` remains only as a legacy alias. The binary causal dataset splits originate from the [RL baseline repository](https://github.com/ds-jrg/causal-qa-rl).

## Reproduce the experiments

All commands in this section run from `code/` after the required graphs and research artifacts are in place.

### Inspect released results and regenerate plots

Machine-readable final results are already included in the artifact. Regenerate the standard, ablation, thesis, and budget-tradeoff figures without loading a graph or model:

```powershell
python -m evaluation.evaluation_viz --all --run-suffix v4
python -m evaluation.evaluation_viz --all --ablation --run-suffix v4 --dim 64
python -m evaluation.evaluation_viz --thesis --run-suffix v4
python -m evaluation.evaluation_viz --tradeoff
```

These commands write under `data/plots/`, preserving the graph/dataset/v4 hierarchy where applicable.

### Check the complete evaluation plan

Print the evaluation commands without running them:

```powershell
python -m evaluation.run_all_evaluations --run-suffix v4 --select-best-from-validation --skip-dijkstra --skip-ablation --dry-run
```

The dry run reads the released validation results and should select the local Granite v4 export at dimension 64. It also prints the six final test commands for the two datasets across `causenet`, `causenet_full`, and `ceg`.

### Rerun only the frozen final configuration

This skips validation/model selection, uses the fixed Granite-64 default, and reruns A*, capped/uncapped BFS, and RL on all six test settings:

```powershell
python -m evaluation.run_all_evaluations --run-suffix v4 --skip-validation --skip-dijkstra --skip-ablation
```

The final model needs the shared Granite-64 cache for CauseNet Precision and CEG, plus the graph-specific Granite-64 cache for CauseNet Full. The RL rows additionally need the released RL checkpoint and GloVe 300d.

### Rerun validation selection and final tests

The full primary v4 package contains all five exported fine-tuned models, their shared-universe Matryoshka caches, and the matching base-model caches used for the pretrained comparison rows. Rerun validation, apply the recorded Pareto-knee rule, and use the selected winner on the test settings:

```powershell
python -m evaluation.run_all_evaluations --run-suffix v4 --select-best-from-validation --skip-dijkstra --skip-ablation
```

Evaluation is expensive and the orchestration command refreshes existing baseline and model rows by default. Add `--no-force` to retain completed rows. Remove `--skip-dijkstra` to include Dijkstra. Remove `--skip-ablation` to rerun the four Granite activation/distance configurations at `d=64`; the package includes the required ablation exports and shared/full-graph caches.

To regenerate the training-split visit budgets before evaluation:

```powershell
python -m evaluation.visited_nodes_analysis data/datasets/filtered/msmarco_train_filtered.json --run-suffix v4 --graph causenet --embedding-device cuda
```

The released p95 file is the source of truth for the thesis results; regenerating it rebuilds that analysis from the available v4 models and caches.

### Optional: normalize datasets

The artifact already contains normalized files. When starting from the raw splits, run:

```powershell
python -m preprocessing.normalize_datasets msmarco_train
python -m preprocessing.normalize_datasets msmarco_valid
python -m preprocessing.normalize_datasets msmarco_test
python -m preprocessing.normalize_datasets sem_test
```

<details>
<summary><strong>Optional: reproduce the full v4 training pipeline</strong></summary>

The released exports make retraining unnecessary. For an exact from-scratch pipeline, run hyperparameter search and final training for each of the five backbones in `code/core/config.py`, precompute their shared caches, regenerate p95 budgets, run validation/model selection, then run the frozen test evaluation. Training data derived from A* paths are generated and cached under `data/datasets/` automatically when absent.

Granite’s final v4 study used a 50-trial maximum, 30-trial minimum, study patience 5, per-trial early-stopping patience 3, 10 epochs per trial, no embedding normalization, and Matryoshka training. The released Optuna study selected ReLU, Euclidean distance, and learning rate `2.7088928792005436e-05`; final training used up to 50 epochs with patience 10 and effective batch size 128.

```powershell
python -m finetune.hparam_search --model ibm-granite/granite-embedding-english-r2 --run-suffix v4 --trials 50 --min-trials 30 --study-patience 5 --epochs 10 --patience 3
python -m finetune.finetune_best --model ibm-granite/granite-embedding-english-r2 --run-suffix v4 --epochs 50 --patience 10
```

`finetune_best` uses the latest matching v4 Optuna SQLite study. To rebuild every primary shared cache and all configured Matryoshka prefixes after training:

```powershell
python -m core.pre_embed --run-suffix v4 --all-dims --embedding-device cuda
```

For only the selected shared and CauseNet Full caches:

```powershell
python -m core.pre_embed --model data/models/lightning/granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v4_finetuned --dim 64 --run-suffix v4
python -m core.pre_embed --model data/models/lightning/granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v4_finetuned --dim 64 --run-suffix v4 --graph causenet_full
```

CauseNet Precision and filtered CEG share `merged_causenet_ceg_nodes.jsonl`, so CEG does not need a separate pre-embedding run. CauseNet Full uses `causenet_full_nodes.jsonl` and a `_causenet_full_...` cache name.

</details>

## Technologies

| Area | Technologies |
|---|---|
| ML and optimization | PyTorch, PyTorch Lightning, Sentence Transformers, Hugging Face Datasets, Optuna |
| Graph search | NetworkX, A*, BFS, Dijkstra, LSTM-based RL traversal |
| Efficient inference | NumPy memmaps, precomputed embeddings, indexed adjacency, Matryoshka representations |
| Evaluation | scikit-learn, pandas, Matplotlib, paired stratified bootstrap tests |
| Demo | FastAPI, Uvicorn, JavaScript, 3d-force-graph |

## Scope and limitations

This is a system for **causal path discovery**, not general causal inference. Automatically extracted knowledge graphs can contain missing, noisy, ambiguous, or overly general relations, and causal edges are not necessarily transitive. A found path means that the graph encodes a connection. A negative answer can also mean that the graph lacks the relation or that bounded A* exhausted `τ=23` before reaching it.

The system does not estimate effect sizes, validate causal claims from raw evidence, or perform interventional or counterfactual reasoning.

## Thesis and citations

The LaTeX source for the bachelor thesis **“Learning Heuristics for Efficient Causal Question Answering Using A\* Search”** is included in [`thesis/`](thesis/).

The datasets, causal graphs, embedding models, search methods, and baseline systems used in this project are cited in the thesis. Their BibTeX entries are collected in [`thesis/literature.bib`](thesis/literature.bib).
