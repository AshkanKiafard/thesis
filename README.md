# Causal Graph Search with Learned Heuristics

## Overview
This repository contains the code for a Bachelor thesis on causal question answering using graph search over CauseNet, focusing on improving A* with learned embedding-based heuristics.

---

## Project Structure

### Root
- traverse_strategies/: search algorithms (A*, BFS, Dijkstra, RL)
- embeddings.py: embedding wrappers
- finetune.py: training pipeline
- evaluation.py: evaluation + metrics
- pre_embed.py: embedding caching
- peak_analysis.py: search complexity analysis
- viz.py: plotting
- rl_model.py: RL agent
- utils.py: graph + helper functions
- app.py: demo UI

---

## Data Folder Explanation (data/)

- checkpoints/: saved training checkpoints
- datasets/: MSMARCO-based causal QA datasets
- docker/: docker-related files (if used)
- docs/: documentation or auxiliary files
- embeddings/: cached node embeddings (numpy)
- evaluation/: JSON + CSV experiment results
- graphs/: CauseNet graph file
- lightning_logs/: PyTorch Lightning logs
- models/: trained models
- optuna_studies/: hyperparameter search database
- plots/: generated plots
- tb_logs/: TensorBoard logs

---

## External Resources

### CauseNet Graph
Download from:
https://causenet.org/

Place at:
data/graphs/causenet-precision.jsonl

### MSMARCO Causal QA Datasets
Required files:
- msmarco_train.json
- msmarco_valid.json
- msmarco_test.json
- msmarco_train_valid.json

Download from:
https://github.com/ds-jrg/causal-qa-rl (datasets folder)

Place in:
data/datasets/

---

## Setup

pip install -r requirements.txt

---

## Reproducing Experiments

1. Precompute embeddings:
python pre_embed.py

2. (Optional) Train model:
python finetune.py

3. Run evaluation:
python evaluation.py

Outputs:
- data/evaluation/evaluation_results_valid.json
- data/evaluation/evaluation_results_valid.csv

4. Generate plots:
python viz.py

5. Peak analysis:
python peak_analysis.py

---

## Metrics

- Accuracy / F1
- Avg nodes visited
- Avg path cost
- Cost per hop

---

## Demo

python app.py

---

## Reproducibility

Run:
pre_embed.py → evaluation.py → viz.py
