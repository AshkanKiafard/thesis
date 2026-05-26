import argparse
import csv
import json
from pathlib import Path

import numpy as np

from core.graph_config import DEFAULT_GRAPH_NAME, graph_choices


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_NAME = "msmarco_test"
DEFAULT_RUN_SUFFIX = "best_v2"
DEFAULT_REFERENCE_CONTAINS = [
    "Qwen3-Embedding-0.6B",
    "finetuned",
]
DEFAULT_REFERENCE_DIMENSION = 128
DEFAULT_REFERENCE_ALGORITHM = "A*"
DEFAULT_COMPARISONS = [
    "model=BFS_Uncapped_Baseline,algorithm=BFS",
    "model=BFS_Baseline,algorithm=BFS",
    "model=RL_Baseline,algorithm=RL",
]


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_evaluation_results_path(dataset_name_or_path, run_suffix, graph_name):
    requested_path = Path(dataset_name_or_path)

    if requested_path.suffix == ".json":
        candidates = [
            requested_path,
            REPO_ROOT / requested_path,
        ]
    elif len(requested_path.parts) > 1:
        candidates = [
            requested_path / "evaluation_results.json",
            REPO_ROOT / requested_path / "evaluation_results.json",
        ]
    else:
        candidates = [
            REPO_ROOT
            / "data"
            / "evaluation"
            / graph_name
            / dataset_name_or_path
            / run_suffix
            / "evaluation_results.json"
        ]

    for path in candidates:
        if path.exists():
            return path

    expected_paths = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find evaluation results for '{dataset_name_or_path}'. "
        f"Tried: {expected_paths}"
    )


def parse_optional_dimension(value):
    if value is None:
        return None

    if isinstance(value, str):
        value = value.strip()

        if value == "" or value.lower() in {"none", "null", "baseline"}:
            return None

    return int(value)


def normalize_id(value, fallback):
    if value is None:
        return f"__row_{fallback}"

    return str(value)


def f1_from_arrays(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    denominator = (2 * tp) + fp + fn

    if denominator == 0:
        return 0.0

    return float((2 * tp) / denominator)


def f1_many(y_true_samples, y_pred_samples):
    tp = np.sum(y_true_samples & y_pred_samples, axis=1)
    fp = np.sum(~y_true_samples & y_pred_samples, axis=1)
    fn = np.sum(y_true_samples & ~y_pred_samples, axis=1)

    denominator = (2 * tp) + fp + fn

    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(tp, dtype=float),
        where=denominator != 0,
    )


def flatten_runs(evaluation_data):
    runs = []

    for entry in evaluation_data:
        model = entry.get("model")
        dimension = entry.get("dimension")
        split = entry.get("split")
        run_suffix = entry.get("run_suffix")

        for algorithm, result in entry.get("evaluation", {}).items():
            per_example = result.get("per_example", [])

            if not per_example:
                continue

            runs.append(
                {
                    "model": model,
                    "model_path": entry.get("model_path"),
                    "dimension": dimension,
                    "algorithm": algorithm,
                    "split": split,
                    "run_suffix": run_suffix,
                    "metrics": result.get("metrics", {}),
                    "per_example": per_example,
                }
            )

    return runs


def run_label(run):
    dimension = run.get("dimension")

    if dimension is None:
        dimension_text = "baseline"
    else:
        dimension_text = f"dim {dimension}"

    return (
        f"{run.get('algorithm')} | "
        f"{run.get('model')} | "
        f"{dimension_text}"
    )


def print_available_runs(runs):
    print("Available runs:")

    for index, run in enumerate(runs, start=1):
        metrics = run.get("metrics", {})
        f1_score = metrics.get("f1_score")

        if f1_score is None:
            f1_text = "n/a"
        else:
            f1_text = f"{float(f1_score):.6f}"

        print(f"{index:02d}. {run_label(run)} | F1={f1_text}")


def parse_selector(text):
    """
    Parse a comparison selector.

    Accepted forms:
    - "BFS_Uncapped_Baseline"
    - "BFS_Baseline"
    - "RL_Baseline"
    - "model=BFS_Uncapped_Baseline,algorithm=BFS"
    - "model=BFS_Baseline,algorithm=BFS"
    - "model=RL_Baseline,algorithm=RL"
    - "model_contains=Qwen,dimension=128,algorithm=A*"
    """
    selector = {}
    text = text.strip()

    if "=" not in text:
        selector["model_contains"] = [text]
        return selector

    for part in text.split(","):
        key, value = part.split("=", 1)
        key = key.strip().lower().replace("-", "_")
        value = value.strip()

        if key in {"model", "model_contains", "contains"}:
            selector.setdefault("model_contains", []).append(value)
        elif key in {"dimension", "dim"}:
            selector["dimension"] = parse_optional_dimension(value)
        elif key in {"algorithm", "algo"}:
            selector["algorithm"] = value
        else:
            raise ValueError(f"Unknown selector key: {key}")

    return selector


def run_matches(run, selector):
    model_text = f"{run.get('model')} {run.get('model_path')}".lower()

    for needle in selector.get("model_contains", []):
        if needle.lower() not in model_text:
            return False

    if "dimension" in selector:
        if parse_optional_dimension(run.get("dimension")) != selector["dimension"]:
            return False

    if "algorithm" in selector:
        if str(run.get("algorithm")) != selector["algorithm"]:
            return False

    return True


def select_single_run(runs, selector, description):
    matches = [run for run in runs if run_matches(run, selector)]

    if not matches:
        raise ValueError(f"No run matched {description}: {selector}")

    if len(matches) > 1:
        print_available_runs(matches)
        raise ValueError(
            f"{description} selector is ambiguous. "
            "Add model_contains, dimension, or algorithm filters."
        )

    return matches[0]


def examples_by_id(run):
    by_id = {}

    for index, row in enumerate(run.get("per_example", [])):
        example_id = normalize_id(row.get("id"), index)

        if example_id in by_id:
            raise ValueError(
                f"Duplicate example id {example_id!r} in {run_label(run)}"
            )

        by_id[example_id] = {
            "true": bool(row.get("true")),
            "pred": bool(row.get("pred")),
        }

    return by_id


def align_examples(reference_run, comparison_run):
    reference_examples = examples_by_id(reference_run)
    comparison_examples = examples_by_id(comparison_run)

    common_ids = [
        normalize_id(row.get("id"), index)
        for index, row in enumerate(reference_run.get("per_example", []))
        if normalize_id(row.get("id"), index) in comparison_examples
    ]

    if not common_ids:
        raise ValueError(
            f"No shared example ids between {run_label(reference_run)} "
            f"and {run_label(comparison_run)}"
        )

    y_true = []
    reference_pred = []
    comparison_pred = []
    mismatched_truth = []

    for example_id in common_ids:
        reference_row = reference_examples[example_id]
        comparison_row = comparison_examples[example_id]

        if reference_row["true"] != comparison_row["true"]:
            mismatched_truth.append(example_id)
            continue

        y_true.append(reference_row["true"])
        reference_pred.append(reference_row["pred"])
        comparison_pred.append(comparison_row["pred"])

    if mismatched_truth:
        preview = ", ".join(mismatched_truth[:5])
        raise ValueError(
            f"Ground-truth labels differ for {len(mismatched_truth)} shared "
            f"examples. First mismatches: {preview}"
        )

    return (
        np.asarray(y_true, dtype=bool),
        np.asarray(reference_pred, dtype=bool),
        np.asarray(comparison_pred, dtype=bool),
        common_ids,
    )


def make_bootstrap_indices(y_true, rng, size, stratified):
    n_examples = len(y_true)

    if not stratified:
        return rng.integers(0, n_examples, size=(size, n_examples))

    positive_idx = np.flatnonzero(y_true)
    negative_idx = np.flatnonzero(~y_true)

    if len(positive_idx) == 0 or len(negative_idx) == 0:
        return rng.integers(0, n_examples, size=(size, n_examples))

    positive_samples = rng.choice(
        positive_idx,
        size=(size, len(positive_idx)),
        replace=True,
    )
    negative_samples = rng.choice(
        negative_idx,
        size=(size, len(negative_idx)),
        replace=True,
    )

    return np.concatenate([positive_samples, negative_samples], axis=1)


def paired_bootstrap_f1_difference(
        y_true,
        reference_pred,
        comparison_pred,
        n_bootstrap,
        seed,
        alpha,
        stratified,
        chunk_size,
):
    rng = np.random.default_rng(seed)

    reference_f1 = f1_from_arrays(y_true, reference_pred)
    comparison_f1 = f1_from_arrays(y_true, comparison_pred)
    observed_delta = reference_f1 - comparison_f1

    bootstrap_deltas = np.empty(n_bootstrap, dtype=float)

    for start in range(0, n_bootstrap, chunk_size):
        end = min(start + chunk_size, n_bootstrap)
        current_size = end - start

        indices = make_bootstrap_indices(
            y_true=y_true,
            rng=rng,
            size=current_size,
            stratified=stratified,
        )

        true_samples = y_true[indices]
        reference_samples = reference_pred[indices]
        comparison_samples = comparison_pred[indices]

        reference_scores = f1_many(true_samples, reference_samples)
        comparison_scores = f1_many(true_samples, comparison_samples)

        bootstrap_deltas[start:end] = reference_scores - comparison_scores

    lower_q = 100 * (alpha / 2)
    upper_q = 100 * (1 - alpha / 2)
    ci_low, ci_high = np.percentile(bootstrap_deltas, [lower_q, upper_q])

    centered_deltas = bootstrap_deltas - observed_delta
    p_value = (
        np.sum(np.abs(centered_deltas) >= abs(observed_delta)) + 1
    ) / (n_bootstrap + 1)

    return {
        "reference_f1": reference_f1,
        "comparison_f1": comparison_f1,
        "f1_delta_reference_minus_comparison": observed_delta,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_value": float(min(max(p_value, 0.0), 1.0)),
    }


def build_default_output_path(evaluation_results_path, suffix):
    path = Path(evaluation_results_path)
    return path.with_name(f"f1_significance_tests.{suffix}")


def write_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


def write_csv(path, results):
    fieldnames = [
        "reference_algorithm",
        "reference_model",
        "reference_dimension",
        "comparison_algorithm",
        "comparison_model",
        "comparison_dimension",
        "num_examples",
        "num_positive",
        "num_negative",
        "reference_f1",
        "comparison_f1",
        "f1_delta_reference_minus_comparison",
        "ci_low",
        "ci_high",
        "p_value",
        "p_value_bonferroni",
        "alpha",
        "alpha_bonferroni",
        "significant_after_bonferroni",
        "n_bootstrap",
        "stratified",
    ]

    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    "reference_algorithm": result["reference"]["algorithm"],
                    "reference_model": result["reference"]["model"],
                    "reference_dimension": result["reference"]["dimension"],
                    "comparison_algorithm": result["comparison"]["algorithm"],
                    "comparison_model": result["comparison"]["model"],
                    "comparison_dimension": result["comparison"]["dimension"],
                    "num_examples": result["num_examples"],
                    "num_positive": result["num_positive"],
                    "num_negative": result["num_negative"],
                    "reference_f1": result["reference_f1"],
                    "comparison_f1": result["comparison_f1"],
                    "f1_delta_reference_minus_comparison": (
                        result["f1_delta_reference_minus_comparison"]
                    ),
                    "ci_low": result["ci_low"],
                    "ci_high": result["ci_high"],
                    "p_value": result["p_value"],
                    "p_value_bonferroni": result["p_value_bonferroni"],
                    "alpha": result["alpha"],
                    "alpha_bonferroni": result["alpha_bonferroni"],
                    "significant_after_bonferroni": (
                        result["significant_after_bonferroni"]
                    ),
                    "n_bootstrap": result["n_bootstrap"],
                    "stratified": result["stratified"],
                }
            )


def run_summary(run):
    return {
        "algorithm": run.get("algorithm"),
        "model": run.get("model"),
        "model_path": run.get("model_path"),
        "dimension": run.get("dimension"),
        "split": run.get("split"),
        "run_suffix": run.get("run_suffix"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paired bootstrap significance test for F1 differences between "
            "completed evaluation runs."
        )
    )
    parser.add_argument(
        "dataset_name",
        nargs="?",
        default=DEFAULT_DATASET_NAME,
        help=(
            "Evaluation dataset name, e.g. msmarco_test or sem_test. "
            "A path to evaluation_results.json or its parent directory is also accepted."
        ),
    )
    parser.add_argument(
        "--run-suffix",
        default=DEFAULT_RUN_SUFFIX,
        help=(
            "Run suffix used when dataset_name is not a direct path, "
            "e.g. best_v2."
        ),
    )
    parser.add_argument(
        "--graph",
        choices=graph_choices(),
        default=DEFAULT_GRAPH_NAME,
        help="Graph results to test. Defaults to CauseNet.",
    )
    parser.add_argument(
        "--reference-model-contains",
        action="append",
        default=None,
        help=(
            "Substring that must appear in the reference model or model path. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--reference-dimension",
        type=parse_optional_dimension,
        default=DEFAULT_REFERENCE_DIMENSION,
        help="Reference dimension. Use 'none' for a baseline run.",
    )
    parser.add_argument(
        "--reference-algorithm",
        default=DEFAULT_REFERENCE_ALGORITHM,
        help="Reference algorithm name.",
    )
    parser.add_argument(
        "--comparison",
        action="append",
        default=None,
        help=(
            "Comparison selector. Examples: 'BFS_Uncapped_Baseline', "
            "'BFS_Baseline', 'RL_Baseline', "
            "'model=BFS_Uncapped_Baseline,algorithm=BFS', "
            "'model=BFS_Baseline,algorithm=BFS', "
            "'model=RL_Baseline,algorithm=RL', "
            "'model_contains=Qwen,dimension=64,algorithm=A*'. "
            "Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--all-against-reference",
        action="store_true",
        help="Compare every other run against the selected reference run.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=10000,
        help="Number of paired bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap resampling.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Family-wise significance level.",
    )
    parser.add_argument(
        "--no-stratified",
        action="store_true",
        help="Disable label-stratified bootstrap resampling.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Bootstrap samples processed per vectorized chunk.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Output JSON path. Defaults next to evaluation_results.json.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path. Defaults next to evaluation_results.json.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="Print available runs and exit.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.n_bootstrap < 1:
        raise ValueError("--n-bootstrap must be >= 1")

    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be >= 1")

    evaluation_results_path = resolve_evaluation_results_path(
        args.dataset_name,
        args.run_suffix,
        args.graph,
    )
    evaluation_data = load_json(evaluation_results_path)
    runs = flatten_runs(evaluation_data)

    if not runs:
        raise ValueError("No runs with per_example predictions were found.")

    if args.list_runs:
        print_available_runs(runs)
        raise SystemExit(0)

    reference_contains = (
        args.reference_model_contains
        if args.reference_model_contains is not None
        else DEFAULT_REFERENCE_CONTAINS
    )

    reference_selector = {
        "model_contains": reference_contains,
        "dimension": args.reference_dimension,
        "algorithm": args.reference_algorithm,
    }
    reference_run = select_single_run(
        runs,
        reference_selector,
        "reference",
    )

    if args.all_against_reference:
        comparison_runs = [run for run in runs if run is not reference_run]
    else:
        comparison_texts = args.comparison or DEFAULT_COMPARISONS
        comparison_runs = [
            select_single_run(runs, parse_selector(text), f"comparison {text!r}")
            for text in comparison_texts
        ]

    n_comparisons = len(comparison_runs)

    if n_comparisons == 0:
        raise ValueError("No comparison runs selected.")

    alpha_bonferroni = args.alpha / n_comparisons
    stratified = not args.no_stratified

    print("F1 significance testing")
    print("=" * 60)
    print(f"Dataset:     {args.dataset_name}")
    print(f"Run suffix:  {args.run_suffix}")
    print(f"Input:       {evaluation_results_path}")
    print(f"Reference:   {run_label(reference_run)}")
    print(f"Comparisons: {n_comparisons}")
    print(f"Bootstrap:   n={args.n_bootstrap}, stratified={stratified}")
    print(
        f"Bonferroni:  alpha={args.alpha:.4f}, "
        f"corrected alpha={alpha_bonferroni:.4f}"
    )

    results = []

    for comparison_run in comparison_runs:
        y_true, reference_pred, comparison_pred, common_ids = align_examples(
            reference_run,
            comparison_run,
        )

        stats = paired_bootstrap_f1_difference(
            y_true=y_true,
            reference_pred=reference_pred,
            comparison_pred=comparison_pred,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
            alpha=args.alpha,
            stratified=stratified,
            chunk_size=args.chunk_size,
        )

        p_value_bonferroni = min(stats["p_value"] * n_comparisons, 1.0)
        significant = p_value_bonferroni < args.alpha

        result = {
            "reference": run_summary(reference_run),
            "comparison": run_summary(comparison_run),
            "num_examples": int(len(y_true)),
            "num_positive": int(np.sum(y_true)),
            "num_negative": int(np.sum(~y_true)),
            "common_ids": common_ids,
            "alpha": float(args.alpha),
            "alpha_bonferroni": float(alpha_bonferroni),
            "n_comparisons": int(n_comparisons),
            "n_bootstrap": int(args.n_bootstrap),
            "seed": int(args.seed),
            "stratified": bool(stratified),
            **stats,
            "p_value_bonferroni": float(p_value_bonferroni),
            "significant_after_bonferroni": bool(significant),
        }
        results.append(result)

        print("\nComparison")
        print("-" * 60)
        print(f"Against:     {run_label(comparison_run)}")
        print(f"Examples:    {len(y_true)}")
        print(f"F1 ref:      {stats['reference_f1']:.6f}")
        print(f"F1 comp:     {stats['comparison_f1']:.6f}")
        print(
            "Delta:       "
            f"{stats['f1_delta_reference_minus_comparison']:.6f} "
            "(reference - comparison)"
        )
        print(
            f"{100 * (1 - args.alpha):.1f}% CI:     "
            f"[{stats['ci_low']:.6f}, {stats['ci_high']:.6f}]"
        )
        print(f"p:           {stats['p_value']:.6f}")
        print(f"p Bonf.:     {p_value_bonferroni:.6f}")
        print(f"Significant: {significant}")

    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else build_default_output_path(evaluation_results_path, "json")
    )
    output_csv = (
        Path(args.output_csv)
        if args.output_csv is not None
        else build_default_output_path(evaluation_results_path, "csv")
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset_name": args.dataset_name,
        "run_suffix": args.run_suffix,
        "evaluation_results_path": str(evaluation_results_path),
        "method": (
            "paired stratified bootstrap for F1 difference"
            if stratified
            else "paired bootstrap for F1 difference"
        ),
        "p_value_method": "two-sided centered bootstrap",
        "bonferroni_correction": {
            "n_comparisons": n_comparisons,
            "alpha": args.alpha,
            "alpha_bonferroni": alpha_bonferroni,
            "applied": n_comparisons > 1,
        },
        "results": results,
    }

    write_json(output_json, payload)
    write_csv(output_csv, results)

    print("\nSaved:")
    print(f"- {output_json}")
    print(f"- {output_csv}")
