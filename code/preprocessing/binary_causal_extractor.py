import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = REPO_ROOT / "data" / "datasets" / "webis"
OUTPUT_ALL_PATH = REPO_ROOT / "data" / "datasets" / "webis_binary_causal_all.json"
OUTPUT_ANSWERED_PATH = (
    REPO_ROOT / "data" / "datasets" / "filtered" / "webis_binary_causal_answered.json"
)

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
        rf"^\s*(?:is|are|was|were|can|could|may|might|will|would)\s+{SUBJECT}\s+(?:be\s+)?"
        rf"(?:caused|triggered|induced|produced|generated)\s+by\s+{SUBJECT}\s*\??$",
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
    text = re.sub(r"[^A-Za-z0-9]", " ", str(text).lower())
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
    q = normalize_question(question).rstrip("?").strip()

    def clean(text: str) -> Optional[str]:
        text = re.sub(r"\s+", " ", text).strip(" \t\n\r.,;:!?\"'")
        text = re.sub(r"^(that|whether|if)\s+", "", text, flags=re.IGNORECASE)
        return text or None

    def bad_pair(cause: Optional[str], effect: Optional[str]) -> bool:
        if not cause or not effect:
            return True

        c = cause.lower().strip()
        e = effect.lower().strip()

        # Broken noun-phrase cases:
        # "does chaucer have a cause for ..."
        if c.endswith((" have a", " has a", " had a", " have the", " has the", " had the")):
            return True

        # Existential questions:
        # "is there a genetic cause of ..."
        if c.startswith(("there ", "there a ", "there is ", "there are ")):
            return True

        # Broken passive extraction:
        # "can nausea be caused by hunger" must not become cause="nausea be", effect="by hunger".
        if c.endswith(" be") or e.startswith("by "):
            return True

        # Usually not a clean causal effect phrase.
        if e.startswith(("for ", "of ")):
            return True

        # Too vague for graph traversal.
        if c in {"there", "it", "this", "that"} or e in {"a difference", "difference"}:
            return True

        return False

    causal_verbs = [
        "bring about", "brings about", "brought about",
        "give rise to", "gives rise to", "gave rise to",
        "lead to", "leads to", "led to",
        "result in", "results in", "resulted in",
        "trigger", "triggers", "triggered",
        "produce", "produces", "produced",
        "induce", "induces", "induced",
        "generate", "generates", "generated",
        "cause", "causes", "caused",
    ]
    verb_alt = "|".join(re.escape(v) for v in sorted(causal_verbs, key=len, reverse=True))

    aux = r"(?:can|could|may|might|will|would|do|does|did|is|are|was|were|has|have|had)"
    modifiers = r"(?:(?:directly|indirectly|normally|usually|possibly|probably|also|really|actually)\s+)*"
    modal_modifiers = rf"(?:(?:can|could|may|might|will|would|do|does|did)\s+)?{modifiers}"

    # Example:
    # "Do aviation experts say that weather alone would normally cause a crash"
    reported_active = re.compile(
        rf"^{aux}?\s*.*?\b"
        rf"(?:say|says|said|claim|claims|claimed|suggest|suggests|suggested|report|reports|reported)"
        rf"\s+that\s+(.+?)\s+{modal_modifiers}(?:{verb_alt})\s+(.+)$",
        re.IGNORECASE,
    )

    # Example:
    # "is cancer caused by smoking"
    # "can nausea be caused by hunger"
    passive = re.compile(
        r"^(?:(?:is|are|was|were)\s+|(?:can|could|may|might|will|would|do|does|did)\s+)"
        r"(.+?)\s+"
        r"(?:be\s+)?"
        r"(caused|triggered|produced|induced|generated|brought about)\s+by\s+"
        r"(.+)$",
        re.IGNORECASE,
    )

    # Example:
    # "can alcohol cause dehydration"
    active = re.compile(
        rf"^(?:can|could|may|might|will|would|do|does|did)\s+(.+?)\s+"
        rf"{modifiers}(?:{verb_alt})\s+(.+)$",
        re.IGNORECASE,
    )

    # Example:
    # "is smoking a cause of cancer"
    # But reject: "is there a genetic cause of ..."
    cause_of = re.compile(
        rf"^(?:is|are|was|were|can|could|may|might|will|would)\s+(.+?)\s+"
        rf"(?:be\s+)?(?:a\s+|an\s+|the\s+)?cause\s+of\s+(.+)$",
        re.IGNORECASE,
    )

    # Passive must run before active, otherwise:
    # "can nausea be caused by hunger" becomes cause="nausea be", effect="by hunger".
    for pattern, mode in [
        (reported_active, "active"),
        (passive, "passive"),
        (active, "active"),
        (cause_of, "cause_of"),
    ]:
        match = pattern.match(q)
        if not match:
            continue

        if mode == "passive":
            effect, _, cause = match.groups()
        else:
            cause, effect = match.groups()

        cause = clean(cause)
        effect = clean(effect)

        if bad_pair(cause, effect):
            return None, None

        return cause, effect

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
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input folder does not exist: {INPUT_DIR}")

    all_results: List[Dict[str, Any]] = []
    answered_results: List[Dict[str, Any]] = []

    for path in iter_supported_files(INPUT_DIR):
        if "msmarco" in path.name.lower():
            print(f"[SKIP] MS MARCO file: {path}")
            continue

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

    OUTPUT_ALL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ALL_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    OUTPUT_ANSWERED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_ANSWERED_PATH, "w", encoding="utf-8") as f:
        json.dump(answered_results, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(all_results)} records to {OUTPUT_ALL_PATH}")
    print(f"Saved {len(answered_results)} answered records to {OUTPUT_ANSWERED_PATH}")


if __name__ == "__main__":
    main()
