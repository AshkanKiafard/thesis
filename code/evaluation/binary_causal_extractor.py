import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

# General causal rules adapted from the notebook.
PATTERN1 = re.compile(
    r"\Awhy|(?=\Aif)(?=.*why )|(?=\Awhen)(?=.*why )| and why | why is \w+ | is \w+ why ",
    re.IGNORECASE,
)
PATTERN2 = re.compile(r"\Acause.{0,1} | cause.{0,1} |because of what", re.IGNORECASE)
PATTERN3 = re.compile(r"\s*how come |\s*how did ", re.IGNORECASE)
PATTERN4 = re.compile(r"^(?!.*dopplar).*effect.{0,1} .*$| affect{0,1} ", re.IGNORECASE)
PATTERN5 = re.compile(r" lead to", re.IGNORECASE)
PATTERN6 = re.compile(
    r"(?=.*what happens)(?=.*if)|(?=.*what will happen)(?=.*if)|(?=.*what might happen)(?=.*if)|"
    r"(?=.*what happens)(?=.*when)|(?=.*what will happen)(?=.*when)|(?=.*what might happen)(?=.*when)",
    re.IGNORECASE,
)
PATTERN7 = re.compile(
    r"\Awhat to do if |\Awhat to do when |\Awhat to do to |"
    r"\Awhat should be done if |\Awhat should be done when |\Awhat should be done to ",
    re.IGNORECASE,
)
GENERAL_CAUSAL_PATTERNS = [PATTERN1, PATTERN2, PATTERN3, PATTERN4, PATTERN5, PATTERN6, PATTERN7]

CAUSAL_CUES = [
    "cause", "causes", "caused", "causing",
    "induce", "induces", "induced",
    "give rise to", "gives rise to", "gave rise to",
    "produce", "produces", "produced",
    "generate", "generates", "generated",
    "effect", "affect", "affects", "affected",
    "bring about", "brings about", "brought about",
    "provoke", "provokes", "provoked",
    "arouse", "arouses", "aroused",
    "elicit", "elicits", "elicited",
    "lead to", "leads to", "led to",
    "derive from", "derives from", "derived from",
    "associate with", "associates with", "associated with",
    "relate to", "relates to", "related to",
    "link to", "links to", "linked to",
    "stem from", "stems from", "stemmed from",
    "originate", "originates", "originated",
    "originate from", "originates from", "originated from",
    "bring forth", "brings forth", "brought forth",
    "lead up", "leads up",
    "trigger off", "triggers off",
    "bring on", "brings on",
    "result from", "results from",
    "result in", "results in",
    "trigger", "triggers", "triggered",
]

QUESTION_AUX = r"(?:can|could|may|might|will|would|do|does|did|is|are|was|were|has|have|had)"
SUBJECT = r"[a-z0-9][a-z0-9\-'/]*(?:\s+[a-z0-9][a-z0-9\-'/]*){0,12}"

BINARY_CAUSAL_PATTERNS = [
    re.compile(
        rf"^\s*{QUESTION_AUX}\s+{SUBJECT}\s+(?:directly\s+|indirectly\s+)?"
        rf"(?:{'|'.join(re.escape(c) for c in CAUSAL_CUES)})\s+{SUBJECT}\s*\??$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*{QUESTION_AUX}\s+{SUBJECT}\s+(?:be\s+)?(?:a\s+|an\s+|the\s+)?cause\s+of\s+{SUBJECT}\s*\??$",
        re.IGNORECASE,
    ),
    re.compile(
        rf"^\s*(?:is|are|was|were)\s+{SUBJECT}\s+(?:caused|triggered|induced|produced|generated)\s+by\s+{SUBJECT}\s*\??$",
        re.IGNORECASE,
    ),
]

NON_BINARY_PREFIX = re.compile(r"^\s*(why|how|what|which|who|whom|whose|where)\b", re.IGNORECASE)

TEXT_COLUMN_CANDIDATES = [
    "question",
    "query",
    "question_text",
    "title",
    "Question",
    "query_text",
]


def strip_punct(text: str) -> str:
    text = re.sub(r"[^А-Яа-яЁёЙйA-Za-z0-9]", " ", str(text).lower())
    return " ".join(text.split())


def normalize_question(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def looks_causal(text: str) -> Tuple[bool, str]:
    q = strip_punct(text)
    for idx, pattern in enumerate(GENERAL_CAUSAL_PATTERNS, start=1):
        if pattern.search(q):
            return True, f"general_rule_{idx}"

    for cue in CAUSAL_CUES:
        if re.search(rf"\b{re.escape(cue)}\b", q, re.IGNORECASE):
            return True, f"cue:{cue}"

    return False, ""


def looks_binary_causal(text: str) -> Tuple[bool, str]:
    q = normalize_question(text)
    if not q:
        return False, "empty"
    if NON_BINARY_PREFIX.search(q):
        return False, "non_binary_wh"

    causal, causal_reason = looks_causal(q)
    if not causal:
        return False, "not_causal"

    for idx, pattern in enumerate(BINARY_CAUSAL_PATTERNS, start=1):
        if pattern.search(q):
            return True, f"binary_rule_{idx}|{causal_reason}"

    return False, f"causal_but_not_binary|{causal_reason}"


def extract_cause_effect(question: str) -> Tuple[Optional[str], Optional[str]]:
    q = normalize_question(question).rstrip("?")

    active = re.compile(
        r"^(?:can|could|may|might|will|would|do|does|did)\s+(.+?)\s+"
        r"(cause|causes|lead to|leads to|result in|results in|trigger|triggers|produce|produces|induce|induces|generate|generates|bring about|brings about)\s+(.+)$",
        re.IGNORECASE,
    )
    passive = re.compile(
        r"^(?:is|are|was|were)\s+(.+?)\s+(caused|triggered|produced|induced|generated)\s+by\s+(.+)$",
        re.IGNORECASE,
    )
    cause_of = re.compile(
        r"^(?:can|could|may|might|will|would|do|does|did|is|are|was|were|has|have|had)\s+(.+?)\s+"
        r"(?:be\s+)?(?:a\s+|an\s+|the\s+)?cause\s+of\s+(.+)$",
        re.IGNORECASE,
    )

    match = active.match(q)
    if match:
        cause, _, effect = match.groups()
        return cause.strip(), effect.strip()

    match = passive.match(q)
    if match:
        effect, _, cause = match.groups()
        return cause.strip(), effect.strip()

    match = cause_of.match(q)
    if match:
        cause, effect = match.groups()
        return cause.strip(), effect.strip()

    return None, None


def extract_binary_label(row: Dict[str, Any]) -> Optional[bool]:
    for key in ["answer_processed", "answer", "label", "target"]:
        if key not in row:
            continue
        value = row[key]
        if pd.isna(value):
            continue
        text = str(value).strip().lower()
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False
    return None


def detect_text_column(columns: Sequence[str]) -> Optional[str]:
    lowered = {c.lower(): c for c in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def iter_json(path: Path) -> Iterator[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            for item in data["data"]:
                if isinstance(item, dict):
                    yield item
        else:
            yield data


def load_to_dataframe(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path, low_memory=False)
        if path.name.endswith(".jsonl") or path.name.endswith(".jsonl.gz"):
            return pd.DataFrame(iter_jsonl(path))
        if path.name.endswith(".json") or path.name.endswith(".json.gz"):
            return pd.DataFrame(iter_json(path))
    except Exception as exc:
        print(f"[WARN] Failed to read {path}: {exc}")
        return None
    return None


def iter_supported_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith((".csv", ".json", ".jsonl", ".json.gz", ".jsonl.gz")):
            yield path


def build_output_record(
    row: Dict[str, Any],
    question: str,
    dataset_name: str,
    source_path: Path,
    row_idx: int,
    text_col: str,
    reason: str,
) -> Dict[str, Any]:
    cause, effect = extract_cause_effect(question)
    binary_answer = extract_binary_label(row)

    return {
        "id": f"{dataset_name}_{row_idx}",
        "dataset": dataset_name,
        "source_file": str(source_path),
        "row_index": int(row_idx),
        "question": question,
        "question_field": text_col,
        "cause": cause,
        "effect": effect,
        "answer": None,
        "binary_answer": binary_answer,
        "binary_causal_reason": reason,
        "raw": row,
    }


def build_answer_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": record["id"],
        "question": record["question"],
        "cause": record["cause"],
        "effect": record["effect"],
        "answer": record["binary_answer"],
    }


def main() -> None:
    input_dir = Path("../data/datasets/webis")
    output_all_path = Path("../data/datasets/webis_binary_causal_all.json")
    output_answered_path = Path("../data/datasets/filtered/webis_binary_causal_answered.json")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    all_results: List[Dict[str, Any]] = []
    answered_results: List[Dict[str, Any]] = []

    for path in iter_supported_files(input_dir):
        print(f"[READ] {path}")
        df = load_to_dataframe(path)
        if df is None or df.empty:
            continue

        text_col = detect_text_column(df.columns)
        if text_col is None:
            print(f"[SKIP] No question-like column found in {path}")
            continue

        dataset_name = path.stem
        dataset_name = dataset_name.replace("_train_original_split", "")
        dataset_name = dataset_name.replace("_valid_original_split", "")

        for row_idx, row in df.iterrows():
            question = str(row.get(text_col, "") or "").strip()
            keep, reason = looks_binary_causal(question)
            if not keep:
                continue

            record = build_output_record(
                row=row.to_dict(),
                question=question,
                dataset_name=dataset_name,
                source_path=path,
                row_idx=row_idx,
                text_col=text_col,
                reason=reason,
            )
            all_results.append(record)

            if record["binary_answer"] is not None and record["cause"] is not None and record["effect"] is not None:
                answered_results.append(build_answer_record(record))

    output_all_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_all_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    output_answered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_answered_path, "w", encoding="utf-8") as f:
        json.dump(answered_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_results)} records to {output_all_path}")
    print(f"Saved {len(answered_results)} answered records to {output_answered_path}")


if __name__ == "__main__":
    main()
