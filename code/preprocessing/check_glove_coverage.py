import argparse
import json
from pathlib import Path

from filter_causalbank_graph import FILTERED_CAUSALBANK_GRAPH_PATH

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GLOVE_PATH = (
    REPO_ROOT / "data" / "embeddings" / "glove.6B" / "glove.6B.300d.txt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "reports" / "coverage"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "datasets" / "filtered"

# These are the datasets where the graph-membership report should mirror evaluation.
# For every row we check whether cause and effect both exist in the graph. If not,
# evaluation would skip the example, so the report marks it as not eval-usable.
DEFAULT_EVAL_DATASETS = (
    "msmarco_valid",
    "msmarco_test",
    "semeval_valid",
    "semeval_test",
)

DATASET_FILE_ALIASES = {
    "msmarco_valid": ("msmarco_valid_filtered.json", "msmarco_valid.json"),
    "msmarco_test": ("msmarco_test_filtered.json", "msmarco_test.json"),
    "semeval_valid": (
        "semeval_valid_filtered.json",
        "semeval_validation_filtered.json",
        "sem_valid_filtered.json",
        "sem_eval_valid_filtered.json",
        "semeval_valid.json",
        "sem_valid.json",
    ),
    "semeval_test": (
        "semeval_test_filtered.json",
        "sem_test_filtered.json",
        "sem_eval_test_filtered.json",
        "semeval_test.json",
        "sem_test.json",
    ),
}
CAUSENET_GRAPH_PATH = REPO_ROOT / "data" / "graphs" / "causenet-precision.jsonl"
CAUSALBANK_GRAPH_PATH = FILTERED_CAUSALBANK_GRAPH_PATH


def normalize_causenet_concept(value):
    return value.replace("_", " ").strip()


def iter_causenet_nodes(file_path, progress_every=500_000):
    with open(file_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if progress_every and line_number % progress_every == 0:
                print(f"Read {line_number:,} CauseNet lines...")

            if not line.strip():
                continue

            item = json.loads(line)
            relation = item["causal_relation"]
            cause = normalize_causenet_concept(relation["cause"]["concept"])
            effect = normalize_causenet_concept(relation["effect"]["concept"])

            if cause and effect and cause != effect:
                yield cause
                yield effect


def iter_causalbank_nodes(file_path, progress_every=1_000_000):
    with open(file_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if progress_every and line_number % progress_every == 0:
                print(f"Read {line_number:,} CausalBank filtered lines...")

            parts = line.rstrip("\n").split("\t")

            if not parts or "->" not in parts[0]:
                continue

            cause, effect = parts[0].split("->", 1)

            if cause and effect and cause != effect:
                yield cause
                yield effect


def load_nodes(graph_name, graph_path):
    if graph_name == "causenet":
        return set(iter_causenet_nodes(graph_path))

    if graph_name == "causalbank":
        return set(iter_causalbank_nodes(graph_path))

    raise ValueError(f"Unknown graph name: {graph_name}")


def load_glove_vocab(glove_path, progress_every=100_000):
    vocab = set()

    with open(glove_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if progress_every and line_number % progress_every == 0:
                print(f"Read {line_number:,} GloVe rows...")

            if not line:
                continue

            word = line.split(" ", 1)[0]

            if word:
                vocab.add(word)

    return vocab


def split_entity_tokens(node):
    # Matches GloveEmbeder.embed_entity(), which uses text.split(" ").
    return [part for part in node.split(" ") if part]


def get_node_glove_details(node, glove_vocab, lowercase=False):
    lookup_node = node.lower() if lowercase else node
    tokens = split_entity_tokens(lookup_node)
    known_tokens = [token for token in tokens if token in glove_vocab]
    missing_tokens = [token for token in tokens if token not in glove_vocab]

    return {
        "node": node,
        "exact_in_glove": lookup_node in glove_vocab,
        "tokens": tokens,
        "known_tokens": known_tokens,
        "missing_tokens": missing_tokens,
        "any_token_in_glove": bool(known_tokens),
        "all_tokens_in_glove": bool(tokens) and not missing_tokens,
    }


def analyze_node_coverage(nodes, glove_vocab, lowercase=False):
    exact_nodes = []
    missing_exact_nodes = []
    any_token_nodes = []
    all_token_nodes = []
    no_token_nodes = []
    partial_token_nodes = []
    missing_tokens = set()

    for node in sorted(nodes):
        details = get_node_glove_details(
            node=node,
            glove_vocab=glove_vocab,
            lowercase=lowercase,
        )
        known_tokens = details["known_tokens"]
        unknown_tokens = details["missing_tokens"]

        if details["exact_in_glove"]:
            exact_nodes.append(node)
        else:
            missing_exact_nodes.append(node)

        if details["any_token_in_glove"]:
            any_token_nodes.append(node)
        else:
            no_token_nodes.append(node)

        if details["all_tokens_in_glove"]:
            all_token_nodes.append(node)
        elif known_tokens and unknown_tokens:
            partial_token_nodes.append(
                {
                    "node": node,
                    "known_tokens": known_tokens,
                    "missing_tokens": unknown_tokens,
                }
            )

        missing_tokens.update(unknown_tokens)

    total_nodes = len(nodes)

    return {
        "total_nodes": total_nodes,
        "exact_nodes": exact_nodes,
        "missing_exact_nodes": missing_exact_nodes,
        "any_token_nodes": any_token_nodes,
        "all_token_nodes": all_token_nodes,
        "no_token_nodes": no_token_nodes,
        "partial_token_nodes": partial_token_nodes,
        "missing_tokens": sorted(missing_tokens),
        "summary": {
            "total_nodes": total_nodes,
            "exact_nodes_in_glove": len(exact_nodes),
            "exact_nodes_missing_from_glove": len(missing_exact_nodes),
            "nodes_with_any_glove_token": len(any_token_nodes),
            "nodes_with_all_tokens_in_glove": len(all_token_nodes),
            "nodes_with_no_glove_tokens": len(no_token_nodes),
            "nodes_with_partial_token_coverage": len(partial_token_nodes),
            "unique_missing_tokens": len(missing_tokens),
        },
    }


def percent(part, total):
    if total == 0:
        return 0.0

    return part / total * 100.0


def dataset_family(dataset_name):
    name = dataset_name.lower()

    if name.startswith("msmarco"):
        return "msmarco"

    if name.startswith("sem"):
        return "semeval"

    return "other"


def load_dataset(file_path):
    with open(file_path, encoding="utf-8") as file:
        return json.load(file)


def analyze_dataset_nodes(dataset_file, glove_vocab, lowercase=False):
    data = load_dataset(dataset_file)
    unique_nodes = set()
    missing_example_rows = []
    node_rows_by_node = {}

    endpoint_total = 0
    endpoint_exact = 0
    endpoint_any_token = 0
    endpoint_all_tokens = 0
    endpoint_no_token = 0
    examples_with_both_any_token = 0
    examples_with_any_missing_endpoint = 0
    examples_with_missing_cause = 0
    examples_with_missing_effect = 0
    examples_with_both_missing = 0

    for item in data:
        cause = item.get("cause", "")
        effect = item.get("effect", "")
        unique_nodes.update([cause, effect])

        cause_details = get_node_glove_details(
            node=cause,
            glove_vocab=glove_vocab,
            lowercase=lowercase,
        )
        effect_details = get_node_glove_details(
            node=effect,
            glove_vocab=glove_vocab,
            lowercase=lowercase,
        )

        endpoint_details = [
            ("cause", cause_details),
            ("effect", effect_details),
        ]

        for role, details in endpoint_details:
            endpoint_total += 1

            if details["exact_in_glove"]:
                endpoint_exact += 1

            if details["any_token_in_glove"]:
                endpoint_any_token += 1
            else:
                endpoint_no_token += 1

            if details["all_tokens_in_glove"]:
                endpoint_all_tokens += 1

            if not details["any_token_in_glove"]:
                node_row = node_rows_by_node.setdefault(
                    details["node"],
                    {
                        "dataset": dataset_file.stem.replace("_filtered", ""),
                        "family": dataset_family(dataset_file.stem),
                        "node": details["node"],
                        "roles": set(),
                        "example_ids": set(),
                        "missing_tokens": set(details["missing_tokens"]),
                    },
                )
                node_row["roles"].add(role)
                node_row["example_ids"].add(str(item.get("id", "")))
                node_row["missing_tokens"].update(details["missing_tokens"])

        cause_missing = not cause_details["any_token_in_glove"]
        effect_missing = not effect_details["any_token_in_glove"]

        if cause_details["any_token_in_glove"] and effect_details["any_token_in_glove"]:
            examples_with_both_any_token += 1

        if cause_missing or effect_missing:
            examples_with_any_missing_endpoint += 1

            if cause_missing:
                examples_with_missing_cause += 1

            if effect_missing:
                examples_with_missing_effect += 1

            if cause_missing and effect_missing:
                examples_with_both_missing += 1

            missing_example_rows.append(
                {
                    "dataset": dataset_file.stem.replace("_filtered", ""),
                    "family": dataset_family(dataset_file.stem),
                    "id": item.get("id", ""),
                    "answer": item.get("answer", ""),
                    "cause": cause,
                    "cause_missing": cause_missing,
                    "cause_missing_tokens": cause_details["missing_tokens"],
                    "effect": effect,
                    "effect_missing": effect_missing,
                    "effect_missing_tokens": effect_details["missing_tokens"],
                }
            )

    unique_coverage = analyze_node_coverage(
        nodes=unique_nodes,
        glove_vocab=glove_vocab,
        lowercase=lowercase,
    )
    missing_node_rows = []

    for row in sorted(node_rows_by_node.values(), key=lambda value: value["node"]):
        missing_node_rows.append(
            {
                "dataset": row["dataset"],
                "family": row["family"],
                "node": row["node"],
                "roles": sorted(row["roles"]),
                "example_ids": sorted(row["example_ids"]),
                "missing_tokens": sorted(row["missing_tokens"]),
            }
        )

    total_examples = len(data)

    return {
        "summary": {
            "total_examples": total_examples,
            "endpoint_occurrences": endpoint_total,
            "endpoint_occurrences_exact_in_glove": endpoint_exact,
            "endpoint_occurrences_with_any_glove_token": endpoint_any_token,
            "endpoint_occurrences_with_all_tokens_in_glove": endpoint_all_tokens,
            "endpoint_occurrences_with_no_glove_tokens": endpoint_no_token,
            "examples_with_both_endpoints_covered": examples_with_both_any_token,
            "examples_with_any_missing_endpoint": examples_with_any_missing_endpoint,
            "examples_with_missing_cause": examples_with_missing_cause,
            "examples_with_missing_effect": examples_with_missing_effect,
            "examples_with_both_missing": examples_with_both_missing,
            "unique_nodes": unique_coverage["summary"]["total_nodes"],
            "unique_nodes_exact_in_glove": unique_coverage["summary"][
                "exact_nodes_in_glove"
            ],
            "unique_nodes_with_any_glove_token": unique_coverage["summary"][
                "nodes_with_any_glove_token"
            ],
            "unique_nodes_with_no_glove_tokens": unique_coverage["summary"][
                "nodes_with_no_glove_tokens"
            ],
        },
        "unique_nodes": sorted(unique_nodes),
        "missing_examples": missing_example_rows,
        "missing_nodes": missing_node_rows,
    }


def print_dataset_summary(dataset_name, summary):
    total_examples = summary["total_examples"]
    endpoint_total = summary["endpoint_occurrences"]

    print(f"\nDataset: {dataset_name}")
    print("=" * (len(dataset_name) + 9))
    print(f"Examples:                            {total_examples:,}")
    print(f"Unique cause/effect nodes:           {summary['unique_nodes']:,}")
    print(
        "Unique nodes with any GloVe token:   "
        f"{summary['unique_nodes_with_any_glove_token']:,} "
        f"({percent(summary['unique_nodes_with_any_glove_token'], summary['unique_nodes']):.2f}%)"
    )
    print(
        "Unique nodes with no GloVe tokens:   "
        f"{summary['unique_nodes_with_no_glove_tokens']:,} "
        f"({percent(summary['unique_nodes_with_no_glove_tokens'], summary['unique_nodes']):.2f}%)"
    )
    print(
        "Cause/effect node occurrences covered:        "
        f"{summary['endpoint_occurrences_with_any_glove_token']:,} "
        f"({percent(summary['endpoint_occurrences_with_any_glove_token'], endpoint_total):.2f}%)"
    )
    print(
        "Cause/effect node occurrences missing:        "
        f"{summary['endpoint_occurrences_with_no_glove_tokens']:,} "
        f"({percent(summary['endpoint_occurrences_with_no_glove_tokens'], endpoint_total):.2f}%)"
    )
    print(
        "Examples with both cause/effect nodes covered:"
        f" {summary['examples_with_both_endpoints_covered']:,} "
        f"({percent(summary['examples_with_both_endpoints_covered'], total_examples):.2f}%)"
    )
    print(
        "Examples with any missing endpoint:  "
        f"{summary['examples_with_any_missing_endpoint']:,} "
        f"({percent(summary['examples_with_any_missing_endpoint'], total_examples):.2f}%)"
    )


def print_graph_summary(graph_label, coverage):
    summary = coverage["summary"]
    total = summary["total_nodes"]

    print(f"\n{graph_label}")
    print("=" * len(graph_label))
    print(f"Total nodes:                         {total:,}")
    print(
        "Exact node strings in GloVe:         "
        f"{summary['exact_nodes_in_glove']:,} "
        f"({percent(summary['exact_nodes_in_glove'], total):.2f}%)"
    )
    print(
        "Exact node strings missing:          "
        f"{summary['exact_nodes_missing_from_glove']:,} "
        f"({percent(summary['exact_nodes_missing_from_glove'], total):.2f}%)"
    )
    print(
        "Nodes with any GloVe token:          "
        f"{summary['nodes_with_any_glove_token']:,} "
        f"({percent(summary['nodes_with_any_glove_token'], total):.2f}%)"
    )
    print(
        "Nodes with all tokens in GloVe:      "
        f"{summary['nodes_with_all_tokens_in_glove']:,} "
        f"({percent(summary['nodes_with_all_tokens_in_glove'], total):.2f}%)"
    )
    print(
        "Nodes with no GloVe tokens:          "
        f"{summary['nodes_with_no_glove_tokens']:,} "
        f"({percent(summary['nodes_with_no_glove_tokens'], total):.2f}%)"
    )
    print(
        "Nodes with partial token coverage:   "
        f"{summary['nodes_with_partial_token_coverage']:,} "
        f"({percent(summary['nodes_with_partial_token_coverage'], total):.2f}%)"
    )
    print(f"Unique missing tokens:               {summary['unique_missing_tokens']:,}")


def write_lines(path, values):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        for value in values:
            file.write(f"{value}\n")


def write_partial_nodes(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("node\tknown_tokens\tmissing_tokens\n")

        for row in rows:
            file.write(
                f"{row['node']}\t"
                f"{' '.join(row['known_tokens'])}\t"
                f"{' '.join(row['missing_tokens'])}\n"
            )


def write_graph_report(output_dir, graph_name, coverage):
    output_dir.mkdir(parents=True, exist_ok=True)

    write_lines(
        output_dir / f"{graph_name}_missing_exact_nodes.txt",
        coverage["missing_exact_nodes"],
    )
    write_lines(
        output_dir / f"{graph_name}_missing_entity_nodes.txt",
        coverage["no_token_nodes"],
    )
    write_partial_nodes(
        output_dir / f"{graph_name}_partial_entity_nodes.tsv",
        coverage["partial_token_nodes"],
    )
    write_lines(
        output_dir / f"{graph_name}_missing_tokens.txt",
        coverage["missing_tokens"],
    )


def write_missing_dataset_examples(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(
            "dataset\tfamily\tid\tanswer\tcause\tcause_missing\t"
            "cause_missing_tokens\teffect\teffect_missing\teffect_missing_tokens\n"
        )

        for row in rows:
            file.write(
                f"{row['dataset']}\t"
                f"{row['family']}\t"
                f"{row['id']}\t"
                f"{row['answer']}\t"
                f"{row['cause']}\t"
                f"{row['cause_missing']}\t"
                f"{' '.join(row['cause_missing_tokens'])}\t"
                f"{row['effect']}\t"
                f"{row['effect_missing']}\t"
                f"{' '.join(row['effect_missing_tokens'])}\n"
            )


def write_missing_dataset_nodes(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("dataset\tfamily\tnode\troles\texample_ids\tmissing_tokens\n")

        for row in rows:
            file.write(
                f"{row['dataset']}\t"
                f"{row['family']}\t"
                f"{row['node']}\t"
                f"{' '.join(row['roles'])}\t"
                f"{' '.join(row['example_ids'])}\t"
                f"{' '.join(row['missing_tokens'])}\n"
            )


def summarize_dataset_family(dataset_results):
    family_nodes = {}
    family_missing_nodes = {}
    family_summary = {}

    for dataset_name, result in dataset_results.items():
        family = dataset_family(dataset_name)
        family_summary.setdefault(
            family,
            {
                "total_examples": 0,
                "endpoint_occurrences": 0,
                "endpoint_occurrences_with_any_glove_token": 0,
                "endpoint_occurrences_with_no_glove_tokens": 0,
                "examples_with_both_endpoints_covered": 0,
                "examples_with_any_missing_endpoint": 0,
            },
        )
        summary = result["summary"]

        for key in family_summary[family]:
            family_summary[family][key] += summary[key]

        family_nodes.setdefault(family, set())
        family_missing_nodes.setdefault(family, set())

        for row in result["missing_nodes"]:
            family_missing_nodes[family].add(row["node"])

        family_nodes[family].update(result["unique_nodes"])

    for family, summary in family_summary.items():
        summary["unique_nodes"] = len(family_nodes[family])
        summary["unique_nodes_with_no_glove_tokens"] = len(family_missing_nodes[family])
        summary["unique_nodes_with_any_glove_token"] = (
            summary["unique_nodes"] - summary["unique_nodes_with_no_glove_tokens"]
        )

    return family_summary


def round_percent(part, total):
    return round(percent(part, total), 2)


def get_example_id(item, index):
    value = item.get("id", "")
    if value == "":
        return str(index)

    return str(value)


def resolve_eval_dataset_files(dataset_dir, requested_datasets):
    dataset_files = {}

    for dataset_name in requested_datasets:
        aliases = DATASET_FILE_ALIASES.get(
            dataset_name,
            (f"{dataset_name}_filtered.json", f"{dataset_name}.json"),
        )

        found_path = None

        for filename in aliases:
            candidate = dataset_dir / filename

            if candidate.exists():
                found_path = candidate
                break

        if found_path is None:
            print(
                f"WARNING: Could not find dataset '{dataset_name}' in "
                f"{dataset_dir}. Tried: {', '.join(aliases)}"
            )
            continue

        dataset_files[dataset_name] = found_path

    return dataset_files


def analyze_dataset_graph_membership(
    dataset_file, dataset_name, graph_name, graph_nodes
):
    data = load_dataset(dataset_file)

    kept_examples = []
    skipped_examples = []
    missing_node_rows_by_node = {}
    unique_cause_effect_nodes = set()
    unique_cause_effect_nodes_in_graph = set()
    unique_cause_effect_nodes_missing_from_graph = set()

    cause_effect_node_occurrences_total = 0
    cause_effect_node_occurrences_in_graph = 0
    cause_effect_node_occurrences_missing_from_graph = 0

    examples_eval_usable = 0
    examples_skipped_by_eval = 0
    examples_missing_cause_in_graph = 0
    examples_missing_effect_in_graph = 0
    examples_missing_both_in_graph = 0

    for index, item in enumerate(data):
        cause = item.get("cause", "")
        effect = item.get("effect", "")
        example_id = get_example_id(item, index)

        cause_in_graph = cause in graph_nodes
        effect_in_graph = effect in graph_nodes
        both_cause_and_effect_in_graph = cause_in_graph and effect_in_graph

        unique_cause_effect_nodes.update([cause, effect])
        cause_effect_node_occurrences_total += 2

        for role, node, node_in_graph in (
            ("cause", cause, cause_in_graph),
            ("effect", effect, effect_in_graph),
        ):
            if node_in_graph:
                cause_effect_node_occurrences_in_graph += 1
                unique_cause_effect_nodes_in_graph.add(node)
            else:
                cause_effect_node_occurrences_missing_from_graph += 1
                unique_cause_effect_nodes_missing_from_graph.add(node)

                row = missing_node_rows_by_node.setdefault(
                    node,
                    {
                        "graph": graph_name,
                        "dataset": dataset_name,
                        "family": dataset_family(dataset_name),
                        "node": node,
                        "roles": set(),
                        "example_ids": set(),
                    },
                )
                row["roles"].add(role)
                row["example_ids"].add(example_id)

        base_row = {
            "graph": graph_name,
            "dataset": dataset_name,
            "family": dataset_family(dataset_name),
            "id": example_id,
            "answer": item.get("answer", ""),
            "cause": cause,
            "cause_in_graph": cause_in_graph,
            "effect": effect,
            "effect_in_graph": effect_in_graph,
            "both_cause_and_effect_in_graph": both_cause_and_effect_in_graph,
        }

        if both_cause_and_effect_in_graph:
            examples_eval_usable += 1
            kept_examples.append(base_row)
        else:
            examples_skipped_by_eval += 1

            if not cause_in_graph:
                examples_missing_cause_in_graph += 1

            if not effect_in_graph:
                examples_missing_effect_in_graph += 1

            if not cause_in_graph and not effect_in_graph:
                examples_missing_both_in_graph += 1

            if not cause_in_graph and not effect_in_graph:
                skip_reason = "cause_and_effect_missing_from_graph"
            elif not cause_in_graph:
                skip_reason = "cause_missing_from_graph"
            else:
                skip_reason = "effect_missing_from_graph"

            skipped_examples.append({**base_row, "skip_reason": skip_reason})

    total_examples = len(data)
    unique_cause_effect_nodes_total = len(unique_cause_effect_nodes)
    unique_cause_effect_nodes_in_graph_count = len(unique_cause_effect_nodes_in_graph)
    unique_cause_effect_nodes_missing_count = len(
        unique_cause_effect_nodes_missing_from_graph
    )

    missing_nodes = [
        {
            "graph": row["graph"],
            "dataset": row["dataset"],
            "family": row["family"],
            "node": row["node"],
            "roles": sorted(row["roles"]),
            "example_ids": sorted(row["example_ids"]),
        }
        for row in sorted(
            missing_node_rows_by_node.values(),
            key=lambda value: (value["dataset"], value["node"]),
        )
    ]

    summary = {
        "n_examples_total": total_examples,
        "n_examples_eval_usable_both_cause_and_effect_in_graph": examples_eval_usable,
        "n_examples_skipped_by_eval_missing_cause_or_effect_in_graph": examples_skipped_by_eval,
        "n_examples_missing_cause_in_graph": examples_missing_cause_in_graph,
        "n_examples_missing_effect_in_graph": examples_missing_effect_in_graph,
        "n_examples_missing_both_cause_and_effect_in_graph": examples_missing_both_in_graph,
        "pct_examples_eval_usable": round_percent(examples_eval_usable, total_examples),
        "pct_examples_skipped_by_eval": round_percent(
            examples_skipped_by_eval, total_examples
        ),
        "n_cause_effect_node_occurrences_total": cause_effect_node_occurrences_total,
        "n_cause_effect_node_occurrences_in_graph": cause_effect_node_occurrences_in_graph,
        "n_cause_effect_node_occurrences_missing_from_graph": cause_effect_node_occurrences_missing_from_graph,
        "pct_cause_effect_node_occurrences_in_graph": round_percent(
            cause_effect_node_occurrences_in_graph,
            cause_effect_node_occurrences_total,
        ),
        "pct_cause_effect_node_occurrences_missing_from_graph": round_percent(
            cause_effect_node_occurrences_missing_from_graph,
            cause_effect_node_occurrences_total,
        ),
        "n_unique_dataset_cause_effect_nodes_total": unique_cause_effect_nodes_total,
        "n_unique_dataset_cause_effect_nodes_in_graph": unique_cause_effect_nodes_in_graph_count,
        "n_unique_dataset_cause_effect_nodes_missing_from_graph": unique_cause_effect_nodes_missing_count,
        "pct_unique_dataset_cause_effect_nodes_in_graph": round_percent(
            unique_cause_effect_nodes_in_graph_count,
            unique_cause_effect_nodes_total,
        ),
        "pct_unique_dataset_cause_effect_nodes_missing_from_graph": round_percent(
            unique_cause_effect_nodes_missing_count,
            unique_cause_effect_nodes_total,
        ),
    }

    return {
        "summary": summary,
        "kept_examples": kept_examples,
        "skipped_examples": skipped_examples,
        "missing_nodes": missing_nodes,
        "unique_cause_effect_nodes": sorted(unique_cause_effect_nodes),
        "unique_cause_effect_nodes_missing_from_graph": sorted(
            unique_cause_effect_nodes_missing_from_graph
        ),
    }


def summarize_graph_membership_family(dataset_graph_results):
    family_nodes = {}
    family_missing_nodes = {}
    family_summary = {}

    for dataset_name, result in dataset_graph_results.items():
        family = dataset_family(dataset_name)
        family_summary.setdefault(
            family,
            {
                "n_examples_total": 0,
                "n_examples_eval_usable_both_cause_and_effect_in_graph": 0,
                "n_examples_skipped_by_eval_missing_cause_or_effect_in_graph": 0,
                "n_examples_missing_cause_in_graph": 0,
                "n_examples_missing_effect_in_graph": 0,
                "n_examples_missing_both_cause_and_effect_in_graph": 0,
                "n_cause_effect_node_occurrences_total": 0,
                "n_cause_effect_node_occurrences_in_graph": 0,
                "n_cause_effect_node_occurrences_missing_from_graph": 0,
            },
        )

        summary = result["summary"]

        for key in family_summary[family]:
            family_summary[family][key] += summary[key]

        family_nodes.setdefault(family, set()).update(
            result["unique_cause_effect_nodes"]
        )
        family_missing_nodes.setdefault(family, set()).update(
            result["unique_cause_effect_nodes_missing_from_graph"]
        )

    for family, summary in family_summary.items():
        total_examples = summary["n_examples_total"]
        endpoint_total = summary["n_cause_effect_node_occurrences_total"]
        unique_total = len(family_nodes[family])
        unique_missing = len(family_missing_nodes[family])
        unique_in_graph = unique_total - unique_missing

        summary["pct_examples_eval_usable"] = round_percent(
            summary["n_examples_eval_usable_both_cause_and_effect_in_graph"],
            total_examples,
        )
        summary["pct_examples_skipped_by_eval"] = round_percent(
            summary["n_examples_skipped_by_eval_missing_cause_or_effect_in_graph"],
            total_examples,
        )
        summary["pct_cause_effect_node_occurrences_in_graph"] = round_percent(
            summary["n_cause_effect_node_occurrences_in_graph"],
            endpoint_total,
        )
        summary["pct_cause_effect_node_occurrences_missing_from_graph"] = round_percent(
            summary["n_cause_effect_node_occurrences_missing_from_graph"],
            endpoint_total,
        )
        summary["n_unique_dataset_cause_effect_nodes_total"] = unique_total
        summary["n_unique_dataset_cause_effect_nodes_in_graph"] = unique_in_graph
        summary["n_unique_dataset_cause_effect_nodes_missing_from_graph"] = (
            unique_missing
        )
        summary["pct_unique_dataset_cause_effect_nodes_in_graph"] = round_percent(
            unique_in_graph,
            unique_total,
        )
        summary["pct_unique_dataset_cause_effect_nodes_missing_from_graph"] = (
            round_percent(
                unique_missing,
                unique_total,
            )
        )

    return family_summary


def print_graph_membership_summary(graph_label, dataset_name, summary):
    total_examples = summary["n_examples_total"]
    endpoint_total = summary["n_cause_effect_node_occurrences_total"]

    print(f"\nEval graph coverage: {graph_label} / {dataset_name}")
    print("=" * (len(graph_label) + len(dataset_name) + 23))
    print(f"Examples total:                      {total_examples:,}")
    print(
        "Examples kept by evaluation:         "
        f"{summary['n_examples_eval_usable_both_cause_and_effect_in_graph']:,} "
        f"({summary['pct_examples_eval_usable']:.2f}%)"
    )
    print(
        "Examples skipped by evaluation:      "
        f"{summary['n_examples_skipped_by_eval_missing_cause_or_effect_in_graph']:,} "
        f"({summary['pct_examples_skipped_by_eval']:.2f}%)"
    )
    print(
        "Missing cause endpoint:              "
        f"{summary['n_examples_missing_cause_in_graph']:,}"
    )
    print(
        "Missing effect endpoint:             "
        f"{summary['n_examples_missing_effect_in_graph']:,}"
    )
    print(
        "Missing both cause/effect nodes:              "
        f"{summary['n_examples_missing_both_cause_and_effect_in_graph']:,}"
    )
    print(
        "Cause/effect node occurrences in graph:       "
        f"{summary['n_cause_effect_node_occurrences_in_graph']:,}/{endpoint_total:,} "
        f"({summary['pct_cause_effect_node_occurrences_in_graph']:.2f}%)"
    )
    print(
        "Unique dataset cause/effect nodes in graph:   "
        f"{summary['n_unique_dataset_cause_effect_nodes_in_graph']:,}/"
        f"{summary['n_unique_dataset_cause_effect_nodes_total']:,} "
        f"({summary['pct_unique_dataset_cause_effect_nodes_in_graph']:.2f}%)"
    )


def write_eval_example_rows(path, rows, include_skip_reason):
    columns = [
        "graph",
        "dataset",
        "family",
        "id",
        "answer",
        "cause",
        "cause_in_graph",
        "effect",
        "effect_in_graph",
        "both_cause_and_effect_in_graph",
    ]

    if include_skip_reason:
        columns.append("skip_reason")

    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("\t".join(columns) + "\n")

        for row in rows:
            file.write("\t".join(str(row.get(column, "")) for column in columns) + "\n")


def write_eval_missing_node_rows(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write("graph\tdataset\tfamily\tnode\troles\texample_ids\n")

        for row in rows:
            file.write(
                f"{row['graph']}\t"
                f"{row['dataset']}\t"
                f"{row['family']}\t"
                f"{row['node']}\t"
                f"{' '.join(row['roles'])}\t"
                f"{' '.join(row['example_ids'])}\n"
            )


def write_graph_membership_reports(output_dir, graph_name, dataset_graph_results):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_kept_examples = []
    all_skipped_examples = []
    all_missing_nodes = []

    for result in dataset_graph_results.values():
        all_kept_examples.extend(result["kept_examples"])
        all_skipped_examples.extend(result["skipped_examples"])
        all_missing_nodes.extend(result["missing_nodes"])

    write_eval_example_rows(
        output_dir / f"{graph_name}_eval_usable_examples.tsv",
        all_kept_examples,
        include_skip_reason=False,
    )
    write_eval_example_rows(
        output_dir / f"{graph_name}_eval_skipped_examples.tsv",
        all_skipped_examples,
        include_skip_reason=True,
    )
    write_eval_missing_node_rows(
        output_dir / f"{graph_name}_eval_missing_cause_effect_nodes.tsv",
        all_missing_nodes,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Check CauseNet and filtered CausalBank node coverage in "
            "glove.6B.300d embeddings."
        )
    )
    parser.add_argument(
        "--glove-path",
        type=Path,
        default=DEFAULT_GLOVE_PATH,
        help="Path to glove.6B.300d.txt.",
    )
    parser.add_argument(
        "--causenet-graph",
        type=Path,
        default=CAUSENET_GRAPH_PATH,
        help="Path to CauseNet JSONL graph.",
    )
    parser.add_argument(
        "--causalbank-graph",
        type=Path,
        default=CAUSALBANK_GRAPH_PATH,
        help="Path to filtered CausalBank graph.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for missing-node and missing-token reports.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Directory with normalized *_filtered.json datasets.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help=(
            "Lowercase nodes before GloVe lookup. Leave disabled to match "
            "GloveEmbeder.embed_entity() exactly."
        ),
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Only print coverage summaries; do not write report files.",
    )
    parser.add_argument(
        "--skip-datasets",
        action="store_true",
        help="Skip MSMARCO/SemEval dataset GloVe-token coverage checks.",
    )
    parser.add_argument(
        "--skip-graph-eval-coverage",
        action="store_true",
        help=("Skip evaluation-style graph membership checks for valid/test datasets."),
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=list(DEFAULT_EVAL_DATASETS),
        help=(
            "Dataset names for evaluation-style graph membership reports. "
            "Default: msmarco_valid msmarco_test semeval_valid semeval_test."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    for path in (args.glove_path, args.causenet_graph, args.causalbank_graph):
        if not path.exists():
            raise FileNotFoundError(path)

    if (
        not args.skip_datasets or not args.skip_graph_eval_coverage
    ) and not args.dataset_dir.exists():
        raise FileNotFoundError(args.dataset_dir)

    print(f"Loading GloVe vocabulary from: {args.glove_path}")
    glove_vocab = load_glove_vocab(args.glove_path)
    print(f"Loaded {len(glove_vocab):,} GloVe tokens.")

    graph_paths = {
        "causenet": args.causenet_graph,
        "causalbank": args.causalbank_graph,
    }
    graph_labels = {
        "causenet": "CauseNet",
        "causalbank": "CausalBank filtered",
    }

    full_summary = {
        "inputs": {
            "glove_path": str(args.glove_path),
            "causenet_graph": str(args.causenet_graph),
            "causalbank_graph": str(args.causalbank_graph),
            "dataset_dir": str(args.dataset_dir),
            "lowercase_glove_lookup": args.lowercase,
            "eval_datasets": args.eval_datasets,
        },
        "glove_coverage": {
            "graphs": {},
            "datasets": {},
            "dataset_families": {},
        },
        "evaluation_graph_membership": {},
    }

    graph_nodes_by_name = {}

    for graph_name, graph_path in graph_paths.items():
        graph_label = graph_labels[graph_name]
        print(f"\nLoading {graph_label} nodes from: {graph_path}")
        nodes = load_nodes(graph_name, graph_path)
        graph_nodes_by_name[graph_name] = nodes
        print(f"Loaded {len(nodes):,} unique {graph_label} nodes.")

        coverage = analyze_node_coverage(
            nodes=nodes,
            glove_vocab=glove_vocab,
            lowercase=args.lowercase,
        )
        print_graph_summary(graph_label, coverage)
        full_summary["glove_coverage"]["graphs"][graph_name] = coverage["summary"]

        if not args.no_write:
            write_graph_report(args.output_dir, graph_name, coverage)

    if not args.skip_datasets:
        dataset_files = sorted(args.dataset_dir.glob("*.json"))
        dataset_results = {}
        all_missing_examples = []
        all_missing_nodes = []

        print(f"\nChecking dataset GloVe-token coverage in: {args.dataset_dir}")

        for dataset_file in dataset_files:
            dataset_name = dataset_file.stem.replace("_filtered", "")
            result = analyze_dataset_nodes(
                dataset_file=dataset_file,
                glove_vocab=glove_vocab,
                lowercase=args.lowercase,
            )
            dataset_results[dataset_name] = result
            full_summary["glove_coverage"]["datasets"][dataset_name] = result["summary"]
            all_missing_examples.extend(result["missing_examples"])
            all_missing_nodes.extend(result["missing_nodes"])
            print_dataset_summary(dataset_name, result["summary"])

        family_summary = summarize_dataset_family(dataset_results)
        full_summary["glove_coverage"]["dataset_families"] = family_summary

        for family, summary in family_summary.items():
            print_dataset_summary(f"{family} aggregate", summary)

        if not args.no_write:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            write_missing_dataset_examples(
                args.output_dir / "glove_dataset_missing_endpoint_examples.tsv",
                all_missing_examples,
            )
            write_missing_dataset_nodes(
                args.output_dir / "glove_dataset_missing_entity_nodes.tsv",
                all_missing_nodes,
            )

    if not args.skip_graph_eval_coverage:
        eval_dataset_files = resolve_eval_dataset_files(
            dataset_dir=args.dataset_dir,
            requested_datasets=args.eval_datasets,
        )

        if not eval_dataset_files:
            raise FileNotFoundError(
                "No evaluation datasets were found. Check --dataset-dir or --eval-datasets."
            )

        print(
            "\nChecking evaluation-style graph membership coverage "
            f"in: {args.dataset_dir}"
        )

        for graph_name, graph_nodes in graph_nodes_by_name.items():
            graph_label = graph_labels[graph_name]
            dataset_graph_results = {}
            full_summary["evaluation_graph_membership"][graph_name] = {
                "datasets": {},
                "dataset_families": {},
            }

            for dataset_name, dataset_file in eval_dataset_files.items():
                result = analyze_dataset_graph_membership(
                    dataset_file=dataset_file,
                    dataset_name=dataset_name,
                    graph_name=graph_name,
                    graph_nodes=graph_nodes,
                )
                dataset_graph_results[dataset_name] = result
                full_summary["evaluation_graph_membership"][graph_name]["datasets"][
                    dataset_name
                ] = result["summary"]
                print_graph_membership_summary(
                    graph_label=graph_label,
                    dataset_name=dataset_name,
                    summary=result["summary"],
                )

            family_summary = summarize_graph_membership_family(dataset_graph_results)
            full_summary["evaluation_graph_membership"][graph_name][
                "dataset_families"
            ] = family_summary

            for family, summary in family_summary.items():
                print_graph_membership_summary(
                    graph_label=graph_label,
                    dataset_name=f"{family} aggregate",
                    summary=summary,
                )

            if not args.no_write:
                write_graph_membership_reports(
                    output_dir=args.output_dir,
                    graph_name=graph_name,
                    dataset_graph_results=dataset_graph_results,
                )

    if not args.no_write:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = args.output_dir / "summary.json"

        with open(summary_path, "w", encoding="utf-8", newline="\n") as file:
            json.dump(full_summary, file, indent=2)

        print(f"\nWrote reports to: {args.output_dir}")


if __name__ == "__main__":
    main()
