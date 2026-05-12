import argparse
import csv
import json
import re
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export per-example evaluation outputs to CSV."
    )
    parser.add_argument(
        "dataset",
        help="e.g. msmarco_valid or data/datasets/filtered/msmarco_valid_filtered.json",
    )
    return parser.parse_args()


def dataset_name_from_arg(dataset_arg):
    path = Path(dataset_arg)

    if path.suffix == ".json":
        return path.stem.replace("_filtered", "")

    return dataset_arg.replace("_filtered", "")


def safe_filename(value):
    value = str(value)
    value = value.replace("*", "astar")
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = re.sub(r'[:<>|"?]', "_", value)
    return value


if __name__ == "__main__":
    args = parse_args()

    dataset_name = dataset_name_from_arg(args.dataset)

    eval_path = (
        Path("data/evaluation")
        / dataset_name
        / "evaluation_results.json"
    )

    output_dir = (
        Path("data/evaluation")
        / dataset_name
        / "per_example_exports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    for entry in eval_data:
        model = entry["model"]
        dimension = entry.get("dimension", "baseline")

        for algorithm, result in entry["evaluation"].items():
            per_example = result.get("per_example", [])

            if not per_example:
                continue

            filename = safe_filename(
                f"{model}_dim{dimension}_{algorithm}.csv"
            )

            output_file = output_dir / filename

            rows = []

            for row in per_example:
                rows.append(
                    {
                        "id": row.get("id"),
                        "cause": row.get("cause"),
                        "effect": row.get("effect"),
                        "true": row.get("true"),
                        "pred": row.get("pred"),
                        "correct": row.get("correct"),
                        "nodes_visited": row.get("nodes_visited"),
                        "path_length": row.get("path_length"),
                        "time_sec": row.get("time_sec"),
                        "path_cost": row.get("path_cost"),
                    }
                )

            with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile,
                    fieldnames=rows[0].keys(),
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"Saved: {output_file}")

    print("Done.")