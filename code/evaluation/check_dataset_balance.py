import json
from pathlib import Path

DATASET_DIR = Path("../data/datasets/filtered")


def load_dataset(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_labels(data):
    positive = 0
    negative = 0

    for item in data:
        label = item.get("answer")

        if label is True:
            positive += 1
        elif label is False:
            negative += 1
        else:
            print(
                f"Warning: missing/invalid label in example "
                f"{item.get('id', 'unknown')}"
            )

    total = positive + negative

    positive_ratio = (positive / total * 100) if total > 0 else 0
    negative_ratio = (negative / total * 100) if total > 0 else 0

    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "positive_ratio": positive_ratio,
        "negative_ratio": negative_ratio,
    }


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATASET_DIR}"
        )

    dataset_files = sorted(DATASET_DIR.glob("*.json"))

    if not dataset_files:
        print("No dataset files found.")
        return

    print("=" * 90)
    print(
        f"{'Dataset':40}"
        f"{'Total':>10}"
        f"{'Yes':>10}"
        f"{'No':>10}"
        f"{'Yes %':>10}"
        f"{'No %':>10}"
    )
    print("=" * 90)

    for dataset_file in dataset_files:
        data = load_dataset(dataset_file)
        stats = count_labels(data)

        print(
            f"{dataset_file.name:40}"
            f"{stats['total']:>10}"
            f"{stats['positive']:>10}"
            f"{stats['negative']:>10}"
            f"{stats['positive_ratio']:>9.2f}%"
            f"{stats['negative_ratio']:>9.2f}%"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()