# Causal Graph Search with Learned Heuristics

## Overview

This repository contains the code for a Bachelor thesis on causal question answering using graph search over CauseNet.  
The goal is to improve A* search efficiency using learned embedding-based heuristics.

The project supports end-to-end experiment reproduction and provides additional tools for analyzing search behavior.

---

## Project Structure

All code is located inside the `code/` directory.

```
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
│   └── peak_analysis.py         # Distribution / tail analysis (e.g. p95)
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
├── main.py                  # Entry point / experiments
├── requirements.txt
└── .gitignore
```

---

## External Resources

### CauseNet Graph
Download: https://causenet.org/  
Place in:
```
code/data/graphs/causenet-precision.jsonl
```

### MSMARCO Causal QA Dataset
Download: https://github.com/ds-jrg/causal-qa-rl  

Required files:
- msmarco_train.json  
- msmarco_valid.json  
- msmarco_test.json  
- msmarco_train_valid.json  

Place in:
```
code/data/datasets/
```

---

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

---

## Reproducing the Main Results

### 1. (Optional) Train model
```
python finetune.py
```

### 2. Precompute embeddings
```
python pre_embed.py
```

### 3. Run evaluation
```
python evaluation/evaluation.py
```

Outputs:
```
code/data/evaluation/evaluation_results_valid.json
code/data/evaluation/evaluation_results_valid.csv
```

### 4. Generate plots
```
python evaluation/evaluation_viz.py
```

---

## Additional Analysis

### Peak / distribution analysis
```
python evaluation/peak_analysis.py
```

### Export per-example A* statistics
```
python evaluation/export_histogram_data.py
```

Output:
```
code/data/evaluation/histogram_exports/*.csv
```

### Plot histograms
```
python evaluation/plot_astar_histograms.py
```

Output:
```
code/data/plots/histograms_astar/
```

---

## Metrics

- Accuracy / F1 score  
- Recall / Precision  
- Average nodes visited  
- Average path length  
- Average path cost  
- Cost per hop  

---

## Demo

```
python app.py
```

---

## Notes

- Large graphs require significant memory  
- Precomputing embeddings is strongly recommended  
- RL baseline uses GloVe embeddings (300d)  
- Histogram exports help diagnose search behavior beyond aggregate metrics  
