# Causal Graph Search with Learned Heuristics

## Overview
This repository contains the code for a Bachelor thesis on causal question answering using graph search over CauseNet. The main focus is improving A* search using learned embedding-based heuristics.

The project is structured so that experiments can be reproduced end-to-end.

---

## Project Structure

All code is located inside the `code/` directory.

```
code/
│
├── data/
│   ├── checkpoints/        # Model checkpoints during training
│   ├── datasets/           # MSMARCO-based causal QA datasets
│   ├── docker/             # Optional docker setup
│   ├── docs/               # Additional documentation
│   ├── embeddings/         # Cached node embeddings (.npy)
│   ├── evaluation/         # Evaluation outputs (JSON + CSV)
│   ├── graphs/             # CauseNet graph file
│   ├── lightning_logs/     # PyTorch Lightning logs
│   ├── models/             # Saved trained models
│   ├── optuna_studies/     # Hyperparameter search database
│   ├── plots/              # Generated plots
│   ├── tb_logs/            # TensorBoard logs
│   └── .gitkeep
│
├── traverse_strategies/    # Graph search algorithms
│   ├── astar.py
│   ├── bfs.py
│   ├── dijkstra.py
│   └── rl.py
│
├── embeddings.py           # Embedding wrappers (SBERT, GloVe)
├── finetune.py             # Training pipeline (Lightning + Optuna)
├── evaluation.py           # Evaluation and metrics computation
├── pre_embed.py            # Precompute embeddings
├── peak_analysis.py        # Distribution analysis of visited nodes
├── viz.py                  # Plotting evaluation results
├── rl_model.py             # RL agent architecture
├── utils.py                # Graph loading and helper functions
├── app.py                  # Gradio demo interface
└── requirements.txt
```

---

## Data Folder Details

All experiment data is stored under `code/data/`:

- `checkpoints/` – intermediate model checkpoints
- `datasets/` – MSMARCO-based causal QA datasets
- `embeddings/` – cached embeddings for graph nodes
- `evaluation/` – evaluation results (JSON and CSV)
- `graphs/` – CauseNet graph file
- `models/` – final trained models
- `optuna_studies/` – hyperparameter search database
- `plots/` – generated figures
- `lightning_logs/`, `tb_logs/` – training logs

---

## External Resources

### CauseNet Graph
Download from:
https://causenet.org/

Place in:
```
code/data/graphs/causenet-precision.jsonl
```

---

### MSMARCO Causal QA Dataset

Required files:
- msmarco_train.json
- msmarco_valid.json
- msmarco_test.json
- msmarco_train_valid.json

Download from:
https://github.com/ds-jrg/causal-qa-rl (datasets folder)

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

## Reproducing Experiments

### 1. Precompute embeddings
```
python pre_embed.py
```

---

### 2. (Optional) Train model
```
python finetune.py
```

---

### 3. Run evaluation
```
python evaluation.py
```

Outputs:
```
code/data/evaluation/evaluation_results_valid.json
code/data/evaluation/evaluation_results_valid.csv
```

---

### 4. Generate plots
```
python viz.py
```

---

### 5. Peak analysis (search complexity)
```
python peak_analysis.py
```

---

## Metrics

- Accuracy / F1 score
- Average nodes visited
- Average path cost
- Cost per hop

---

## Demo

Run interactive demo:

```
python app.py
```

---

## Reproducibility

To reproduce results:

1. Install dependencies  
2. Download graph + datasets  
3. Run:
```
pre_embed.py → evaluation.py → viz.py
```

Training is optional unless reproducing fine-tuned models.

---

## Notes

- Large graph may require significant memory
- Precomputing embeddings is strongly recommended
- RL baseline requires GloVe embeddings (300 dimensions)
