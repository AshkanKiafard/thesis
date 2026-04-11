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
    │   ├── checkpoints/
    │   ├── datasets/
    │   ├── docker/
    │   ├── docs/
    │   ├── embeddings/
    │   ├── evaluation/
    │   ├── graphs/
    │   ├── lightning_logs/
    │   ├── models/
    │   ├── optuna_studies/
    │   ├── plots/
    │   ├── tb_logs/
    │   └── .gitkeep
    │
    ├── evaluation/
    │   ├── evaluation.py
    │   ├── evaluation_viz.py
    │   ├── export_histogram_data.py
    │   ├── plot_astar_histograms.py
    │   ├── peak_analysis.py
    │
    ├── traverse_strategies/
    │   ├── astar.py
    │   ├── bfs.py
    │   ├── dijkstra.py
    │   └── rl.py
    │
    ├── embeddings.py
    ├── finetune.py
    ├── pre_embed.py
    ├── rl_model.py
    ├── utils.py
    ├── app.py
    ├── main.py
    ├── requirements.txt
    └── .gitignore

------------------------------------------------------------------------

## Setup

    pip install -r requirements.txt

------------------------------------------------------------------------

## Reproducing Experiments

### 1. Train (optional)

    python finetune.py

### 2. Precompute embeddings

    python pre_embed.py

### 3. Run evaluation

    python evaluation/evaluation.py

### 4. Generate plots

    python evaluation/evaluation_viz.py

### 5. Peak analysis

    python evaluation/peak_analysis.py

------------------------------------------------------------------------

## Histogram Analysis

### Export data

    python evaluation/export_histogram_data.py

### Plot histograms

    python evaluation/plot_astar_histograms.py

------------------------------------------------------------------------

## Notes

-   Large graphs require significant memory
-   Precomputing embeddings is recommended
-   Histogram analysis helps diagnose search behavior
