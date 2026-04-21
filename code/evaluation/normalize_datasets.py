import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import get_concept

INPUT_DIR = Path("data/datasets")
INPUT_FILES = [
    "msmarco_train.json",
    "msmarco_valid.json",
    "msmarco_test.json",
    "msmarco_train_valid.json",
]
OUTPUT_DIR = Path("data/datasets/filtered")


def extract_answer(example: Dict[str, Any]) -> Optional[bool]:
    values_to_check: List[Any] = []

    for key in ["answer:Extracted", "answer", "answers"]:
        if key in example:
            values_to_check.append(example[key])

    for value in values_to_check:
        text = flatten_to_text(value).strip().lower()
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False

    return None


def flatten_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(flatten_to_text(v) for v in value)
    if isinstance(value, dict):
        return " ".join(flatten_to_text(v) for v in value.values())
    return str(value)


def normalize_record(example: Dict[str, Any], idx: int, dataset_name: str) -> Optional[Dict[str, Any]]:
    try:
        cause = get_concept(example, 0)
        effect = get_concept(example, 1)
    except Exception:
        return None

    answer = extract_answer(example)
    if not cause or not effect or answer is None:
        return None

    question_text = example.get("question")
    if not question_text:
        question_text = " ".join(token[0] for token in example.get("question:POS", []))

    source_id = example.get("id", idx)

    return {
        "id": f"{dataset_name}_{source_id}",
        "question": str(question_text).strip(),
        "cause": cause,
        "effect": effect,
        "answer": answer,
    }


def load_json(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a list in {path}, got {type(data).__name__}")

    return data


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        if not input_path.exists():
            print(f"[SKIP] Missing file: {input_path}")
            continue

        dataset_name = input_path.stem
        print(f"[READ] {input_path}")
        data = load_json(input_path)

        normalized: List[Dict[str, Any]] = []
        skipped = 0

        for idx, example in enumerate(data):
            record = normalize_record(example, idx, dataset_name)
            if record is None:
                skipped += 1
                continue
            normalized.append(record)

        output_path = OUTPUT_DIR / f"{dataset_name}_filtered.json"
        save_json(output_path, normalized)
        print(f"[SAVE] {output_path} | kept {len(normalized)} / {len(data)} | skipped {skipped}")


if __name__ == "__main__":
    main()
