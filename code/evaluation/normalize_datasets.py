import argparse
import csv
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from nltk.tokenize import word_tokenize as nltk_word_tokenize
except ImportError:
    nltk_word_tokenize = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from core.utils import get_concept

INPUT_DIR = REPO_ROOT / "data" / "datasets"
OUTPUT_DIR = INPUT_DIR / "filtered"


def is_eval_split(dataset_name: str) -> bool:
    """
    Match the original causal-qa-rl behavior.

    In the original repo:
    - train loading uses valid=False -> only Yes examples are kept
    - valid/test loading uses valid=True -> Yes and No examples are kept

    msmarco_train_valid.json is still used as a training file in their setup,
    so we treat it as train, not validation.
    """
    return dataset_name in {"msmarco_valid", "msmarco_test"}


def extract_msmarco_answer(example: Dict[str, Any], include_negative: bool) -> Optional[bool]:
    """
    Original-compatible MS MARCO answer extraction.

    Important:
    "No Answer Present." must NOT be interpreted as False.
    Only the literal answer:Extracted[0] == "No" is a negative label.
    """
    extracted = example.get("answer:Extracted")

    if not isinstance(extracted, list) or len(extracted) == 0:
        return None

    answer = str(extracted[0]).strip()

    if answer == "Yes":
        return True

    if answer == "No" and include_negative:
        return False

    return None


def normalize_causal_concept(text: str) -> str:
    """
    Match original causal-qa-rl concept normalization:
    - tokenize
    - lowercase
    - Unicode NFKC normalization
    """
    if nltk_word_tokenize is None:
        tokens = str(text).split()
    else:
        try:
            tokens = nltk_word_tokenize(text)
        except LookupError:
            tokens = str(text).split()

    tokens = [
        unicodedata.normalize("NFKC", token.lower())
        for token in tokens
    ]
    return " ".join(tokens)


def normalize_question_text(example: Dict[str, Any]) -> str:
    question_text = example.get("question")

    if not question_text:
        question_text = " ".join(token[0] for token in example.get("question:POS", []))

    question_text = str(question_text).strip()

    # Original repo embeds question['question'] + '?'
    if not question_text.endswith("?"):
        question_text += "?"

    return question_text


def normalize_msmarco_record(
    example: Dict[str, Any],
    idx: int,
    dataset_name: str,
    include_negative: bool,
) -> Optional[Dict[str, Any]]:
    answer = extract_msmarco_answer(example, include_negative=include_negative)
    if answer is None:
        return None

    try:
        cause = normalize_causal_concept(get_concept(example, 0))
        effect = normalize_causal_concept(get_concept(example, 1))
    except Exception:
        return None

    if not cause or not effect:
        return None

    question_text = normalize_question_text(example)
    source_id = example.get("id", idx)

    return {
        "id": f"{dataset_name}_{source_id}",
        "question": question_text,
        "cause": cause,
        "effect": effect,
        "answer": answer,
    }


def extract_semeval_answer(example: Dict[str, Any]) -> Optional[bool]:
    label = str(example.get("causal", "")).strip().lower()

    if label in {"causal", "true", "yes", "1"}:
        return True

    if label in {"non_causal", "non-causal", "noncausal", "false", "no", "0"}:
        return False

    return None


def normalize_semeval_record(
    example: Dict[str, Any],
    idx: int,
    dataset_name: str,
) -> Optional[Dict[str, Any]]:
    answer = extract_semeval_answer(example)
    if answer is None:
        return None

    raw_cause = example.get("cause")
    raw_effect = example.get("effect")

    if raw_cause is None or raw_effect is None:
        return None

    cause = normalize_causal_concept(str(raw_cause).strip())
    effect = normalize_causal_concept(str(raw_effect).strip())

    if not cause or not effect:
        return None

    source_id = example.get("id", idx)

    return {
        "id": f"{dataset_name}_{source_id}",
        "question": f"can {cause} cause {effect}?",
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


def load_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_label_stats(data: List[Dict[str, Any]]) -> None:
    positives = sum(1 for item in data if item["answer"] is True)
    negatives = sum(1 for item in data if item["answer"] is False)

    print(f"       positives: {positives}")
    print(f"       negatives: {negatives}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize one causal dataset into filtered JSON format."
    )
    parser.add_argument(
        "dataset_name",
        help=(
            "Dataset name without extension, e.g. msmarco_valid or sem_train. "
            "A .json or .csv filename/path is also accepted."
        ),
    )

    return parser.parse_args()


def resolve_input_path(dataset_name: str) -> Path:
    requested_path = Path(dataset_name)

    if requested_path.suffix:
        candidates = [
            requested_path if requested_path.is_absolute() else REPO_ROOT / requested_path,
            INPUT_DIR / requested_path.name,
        ]
    else:
        candidates = [
            INPUT_DIR / f"{dataset_name}.json",
            INPUT_DIR / f"{dataset_name}.csv",
        ]

    for path in candidates:
        if path.exists():
            return path

    expected_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find dataset '{dataset_name}'. Tried: {expected_paths}"
    )


def normalize_dataset_file(input_path: Path) -> None:
    if not input_path.exists():
        print(f"[SKIP] Missing file: {input_path}")
        return

    dataset_name = input_path.stem

    print(f"[READ] {input_path}")

    if input_path.suffix.lower() == ".json":
        include_negative = is_eval_split(dataset_name)
        print(f"       format: msmarco-json")
        print(f"       include_negative: {include_negative}")
        data = load_json(input_path)

        def normalize_example(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            return normalize_msmarco_record(
                example=example,
                idx=idx,
                dataset_name=dataset_name,
                include_negative=include_negative,
            )

    elif input_path.suffix.lower() == ".csv":
        print(f"       format: semeval-csv")
        data = load_csv(input_path)

        def normalize_example(example: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
            return normalize_semeval_record(
                example=example,
                idx=idx,
                dataset_name=dataset_name,
            )

    else:
        print(f"[SKIP] Unsupported file type: {input_path}")
        return

    normalized: List[Dict[str, Any]] = []
    skipped = 0

    for idx, example in enumerate(data):
        record = normalize_example(example, idx)

        if record is None:
            skipped += 1
            continue

        normalized.append(record)

    output_path = OUTPUT_DIR / f"{dataset_name}_filtered.json"
    save_json(output_path, normalized)

    print(f"[SAVE] {output_path} | kept {len(normalized)} / {len(data)} | skipped {skipped}")
    print_label_stats(normalized)


def main() -> None:
    args = parse_args()
    normalize_dataset_file(resolve_input_path(args.dataset_name))


if __name__ == "__main__":
    main()
