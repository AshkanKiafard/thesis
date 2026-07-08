# Thesis reports

The modules in this package perform read-only analyses and write one canonical
JSON/CSV/LaTeX triplet per report to data/reports/<report_name>/.

Run them from the repository's code directory:

    python -m reports.dataset_statistics
    python -m reports.graph_statistics
    python -m reports.glove_coverage
    python -m reports.bfs_path_lengths
    python -m reports.training_hyperparameters
    python -m reports.p95_visited_nodes

The training-hyperparameter report reads final Lightning logs, exported
training_metadata.json files, retained checkpoints, and Optuna SQLite studies.
Epochs use Lightning's zero-based convention. The reported epoch is the final
strict improvement in val/astar_cost; it is not the stopping epoch.

The p95 visited-node report reads
data/evaluation/causenet/msmarco_train/v3/visited_nodes_analysis.json by
default and renders the integer A* traversal caps used by evaluation, i.e.
ceil(p95_visited_successful_only), in a compact thesis table.

The generators accept --output-dir to select a different report root. Each
generator still creates its own <report_name>/ subdirectory below that root.
