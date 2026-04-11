# Causal Graph Search with Learned Heuristics

## Overview

This repository contains the code for a Bachelor thesis on causal
question answering using graph search over CauseNet. The main focus is
improving A\* search using learned embedding-based heuristics.

The project is structured so that experiments can be reproduced
end-to-end and extended with analysis tools such as histogram-based
diagnostics.

------------------------------------------------------------------------

## Project Structure

All code is located inside the `code/` directory.

    code/
    │
    ├── data/
    │   ├── checkpoints/         # Model checkpoints during training
    │   ├── datasets/            # MSMARCO-based causal QA datasets
    │   ├── docker/              # Optional docker setup
    │   ├── docs/                # Additional documentation
    │   ├── embeddings/          # Cached node embeddings (.npy)
    │   ├── evaluation/          # Evaluation outputs (JSON, CSV, histogram exports)
    │   ├── graphs/              # CauseNet graph file
    │   ├── lightning_logs/      # PyTorch Lightning logs
    │   ├── models/              # Saved trained models
    │   ├── optuna_studies/      # Hyperparameter search database
    │   ├── plots/               # Generated plots (including histograms)
    │   ├── tb_logs/             # TensorBoard logs
    │   └── .gitkeep
    │
    ├── evaluation/
    │   ├── evaluation.py            # Main evaluation loop (metrics, JSON/CSV output)
    │   ├── evaluation_viz.py        # Standard metric plots (accuracy, nodes, etc.)
    │   ├── export_histogram_data.py # Per-example A* export (hop count, nodes visited)
    │   ├── plot_astar_histograms.py # Histogram plotting (hop count, visited nodes)
    │   ├── peak_analysis.py         # Distribution / tail analysis (e.g. p95)
    │
    ├── traverse_strategies/     # Graph search algorithms
    │   ├── astar.py
    │   ├── bfs.py
    │   ├── dijkstra.py
    │   └── rl.py
    │
    ├── embeddings.py            # Embedding wrappers (SBERT, GloVe)
    ├── finetune.py              # Training pipeline (Lightning + Optuna)
    ├── pre_embed.py             # Precompute embeddings
    ├── rl_model.py              # RL agent architecture
    ├── utils.py                 # Graph loading and helper functions
    ├── app.py                   # Gradio demo interface
    ├── main.py                  # Entry point / experiments (if used)
    ├── requirements.txt
    └── .gitignore

------------------------------------------------------------------------

## External Resources

### CauseNet Graph

Download from: https://causenet.org/

Place in:

    code/data/graphs/causenet-precision.jsonl

------------------------------------------------------------------------

### MSMARCO Causal QA Dataset

Required files: - msmarco_train.json\
- msmarco_valid.json\
- msmarco_test.json\
- msmarco_train_valid.json

Download from: https://github.com/ds-jrg/causal-qa-rl (datasets folder)

Place in:

    code/data/datasets/

------------------------------------------------------------------------

## Setup

Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------------------

## Reproducing Experiments

### 1. (Optional) Train model

    python finetune.py

------------------------------------------------------------------------

### 2. Precompute embeddings

    python pre_embed.py

------------------------------------------------------------------------

### 3. Run evaluation

    python evaluation/evaluation.py

Outputs:

    code/data/evaluation/evaluation_results_valid.json
    code/data/evaluation/evaluation_results_valid.csv

------------------------------------------------------------------------

### 4. Generate standard plots

    python evaluation/evaluation_viz.py

------------------------------------------------------------------------

### 5. Peak / distribution analysis

    python evaluation/peak_analysis.py

------------------------------------------------------------------------

## Histogram-Based Analysis (New)

### 6. Export per-example A\* statistics

This generates CSV files containing, for each cause--effect pair:

-   hop count\
-   nodes visited\
-   path found

```{=html}
<!-- -->
```
    python evaluation/export_histogram_data.py

Outputs:

    code/data/evaluation/histogram_exports/*.csv

------------------------------------------------------------------------

### 7. Plot histograms

Creates histograms for:

-   hop count distribution\
-   nodes visited distribution

```{=html}
<!-- -->
```
    python evaluation/plot_astar_histograms.py

Outputs:

    code/data/plots/histograms_astar/

------------------------------------------------------------------------

## Metrics

-   Accuracy / F1 score\
-   Average nodes visited\
-   Average path length\
-   Average path cost\
-   Cost per hop

------------------------------------------------------------------------

## Demo

Run interactive demo:

    python app.py

------------------------------------------------------------------------

## Reproducibility

To reproduce results:

1.  Install dependencies\
2.  Download graph + datasets\
3.  Run:

```bash
python pre_embed.py
python evaluation/evaluation.py
python evaluation/evaluation_viz.py
```

Optional analysis:

```bash
python evaluation/export_histogram_data.py
python evaluation/plot_astar_histograms.py
```

------------------------------------------------------------------------

## Notes

-   Large graphs require significant memory\
-   Precomputing embeddings is strongly recommended\
-   RL baseline requires GloVe embeddings (300d)\
-   Histogram exports help diagnose search behavior (e.g., visit limits,
    path depth) beyond aggregate metrics
