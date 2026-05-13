import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from nltk.tokenize import word_tokenize

from core.utils import get_concept

INPUT_DIR = Path("../data/datasets")
INPUT_FILES = [
    "msmarco_train.json",
    "msmarco_valid.json",
    "msmarco_test.json",
    "msmarco_train_valid.json",
]
OUTPUT_DIR = Path("../data/datasets/filtered")


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
    tokens = word_tokenize(text)
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


def normalize_record(
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


def print_label_stats(data: List[Dict[str, Any]]) -> None:
    positives = sum(1 for item in data if item["answer"] is True)
    negatives = sum(1 for item in data if item["answer"] is False)

    print(f"       positives: {positives}")
    print(f"       negatives: {negatives}")


def main() -> None:
    for filename in INPUT_FILES:
        input_path = INPUT_DIR / filename
        if not input_path.exists():
            print(f"[SKIP] Missing file: {input_path}")
            continue

        dataset_name = input_path.stem
        include_negative = is_eval_split(dataset_name)

        print(f"[READ] {input_path}")
        print(f"       include_negative: {include_negative}")

        data = load_json(input_path)

        normalized: List[Dict[str, Any]] = []
        skipped = 0

        for idx, example in enumerate(data):
            record = normalize_record(
                example=example,
                idx=idx,
                dataset_name=dataset_name,
                include_negative=include_negative,
            )

            if record is None:
                skipped += 1
                continue

            normalized.append(record)

        output_path = OUTPUT_DIR / f"{dataset_name}_filtered.json"
        save_json(output_path, normalized)

        print(f"[SAVE] {output_path} | kept {len(normalized)} / {len(data)} | skipped {skipped}")
        print_label_stats(normalized)


if __name__ == "__main__":
    main()