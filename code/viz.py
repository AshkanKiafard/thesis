import os
import json
import pandas as pd
import matplotlib.pyplot as plt


def ensure_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_bfs_baseline(json_data):
    for entry in json_data:
        if entry.get("model") == "BFS_Baseline" and "evaluation" in entry:
            bfs_data = entry["evaluation"].get("BFS", {})
            return {
                "avg_nodes_visited": bfs_data.get("avg_nodes_visited"),
                "avg_time_sec": bfs_data.get("avg_time_sec"),
                "avg_path_length": bfs_data.get("avg_path_length"),
                "accuracy": bfs_data.get("accuracy")  # Added accuracy
            }
    return None


def get_rl_baseline(json_data):
    for entry in json_data:
        if entry.get("model") == "RL_Baseline" and "evaluation" in entry:
            rl_data = entry["evaluation"].get("RL", {})
            return {
                "avg_nodes_visited": rl_data.get("avg_nodes_visited"),
                "avg_time_sec": rl_data.get("avg_time_sec"),
                "avg_path_length": rl_data.get("avg_path_length"),
                "accuracy": rl_data.get("accuracy")  # Added accuracy
            }
    return None


def extract_astar_data(json_data):
    rows = []
    for entry in json_data:
        if entry.get("model") in ["BFS_Baseline", "RL_Baseline"]:
            continue

        if 'dimension' in entry and 'evaluation' in entry:
            eval_data = entry['evaluation']
            if 'A*' in eval_data:
                rows.append({
                    'model': entry['model'],
                    'dimension': int(entry['dimension']),
                    'astar_nodes': eval_data['A*']['avg_nodes_visited'],
                    'astar_path_len': eval_data['A*']['avg_path_length'],
                    'astar_time': eval_data['A*']['avg_time_sec'],
                    'astar_accuracy': eval_data['A*']['accuracy']  # Added accuracy
                })
    return pd.DataFrame(rows)


def plot_nodes_visited_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df['model'].unique()
    for model in models:
        subset = df[df['model'] == model].sort_values(by='dimension')
        plt.plot(subset['dimension'], subset['astar_nodes'], marker='o', label=model)

    if bfs_baseline:
        plt.axhline(y=bfs_baseline['avg_nodes_visited'], color='black', linestyle='--', label='BFS Baseline')

    if rl_baseline:
        plt.axhline(y=rl_baseline['avg_nodes_visited'], color='red', linestyle='-.', label='RL Baseline')

    plt.title('Average Nodes Visited (A*) vs Embedding Size')
    plt.xlabel('Embedding Size')
    plt.ylabel('Avg Nodes Visited')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df['dimension'].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_execution_time_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df['model'].unique()
    for model in models:
        subset = df[df['model'] == model].sort_values(by='dimension')
        plt.plot(subset['dimension'], subset['astar_time'], marker='o', label=model)

    if bfs_baseline:
        plt.axhline(y=bfs_baseline['avg_time_sec'], color='black', linestyle='--', label='BFS Baseline')

    if rl_baseline:
        plt.axhline(y=rl_baseline['avg_time_sec'], color='red', linestyle='-.', label='RL Baseline')

    plt.title('Average Execution Time (A*) vs Embedding Size')
    plt.xlabel('Embedding Size')
    plt.ylabel('Avg Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df['dimension'].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_path_length_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df['model'].unique()
    for model in models:
        subset = df[df['model'] == model].sort_values(by='dimension')
        plt.plot(subset['dimension'], subset['astar_path_len'], marker='o', label=model)

    if bfs_baseline:
        plt.axhline(y=bfs_baseline['avg_path_length'], color='black', linestyle='--', label='BFS Baseline')

    if rl_baseline:
        plt.axhline(y=rl_baseline['avg_path_length'], color='red', linestyle='-.', label='RL Baseline')

    plt.title('Average Path Length (A*) vs Embedding Size')
    plt.xlabel('Embedding Size')
    plt.ylabel('Avg Path Length')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0)

    if not df.empty:
        plt.xticks(sorted(df['dimension'].unique()))

    plt.savefig(output_path)
    plt.close()


def plot_accuracy_vs_dimension(df, bfs_baseline, rl_baseline, output_path):
    plt.figure(figsize=(10, 6))

    models = df['model'].unique()
    for model in models:
        subset = df[df['model'] == model].sort_values(by='dimension')
        plt.plot(subset['dimension'], subset['astar_accuracy'], marker='o', label=model)

    if bfs_baseline:
        plt.axhline(y=bfs_baseline['accuracy'], color='black', linestyle='--', label='BFS Baseline')

    if rl_baseline:
        plt.axhline(y=rl_baseline['accuracy'], color='red', linestyle='-.', label='RL Baseline')

    plt.title('Accuracy (A*) vs Embedding Size')
    plt.xlabel('Embedding Size')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0, top=1.05)  # Accuracy is 0-1, slightly over 1 for clarity

    if not df.empty:
        plt.xticks(sorted(df['dimension'].unique()))

    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    with open("data/evaluation/evaluation_results_valid.json") as f:
        data = json.load(f)

    output_dir = "data/plots"
    ensure_directory(output_dir)

    df_astar = extract_astar_data(data)
    bfs_data = get_bfs_baseline(data)
    rl_data = get_rl_baseline(data)

    plot_nodes_visited_vs_dimension(df_astar, bfs_data, rl_data, os.path.join(output_dir, "metric_nodes_visited.png"))
    plot_execution_time_vs_dimension(df_astar, bfs_data, rl_data, os.path.join(output_dir, "metric_time.png"))
    plot_path_length_vs_dimension(df_astar, bfs_data, rl_data, os.path.join(output_dir, "metric_path_length.png"))
    plot_accuracy_vs_dimension(df_astar, bfs_data, rl_data, os.path.join(output_dir, "metric_accuracy.png"))