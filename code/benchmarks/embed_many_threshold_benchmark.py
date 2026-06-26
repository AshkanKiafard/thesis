import argparse
import gc
import json
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import torch

# Make code/ importable when this benchmark is executed by file path.
CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from traverse_strategies.astar import astar_traverse
from traverse_strategies.dijkstra import dijkstra_traverse
from core.config import EMBEDDING_INDEX_MIN_SUCCESSORS


CURRENT_THRESHOLD = 128
SELECTED_THRESHOLD = EMBEDDING_INDEX_MIN_SUCCESSORS


@dataclass(frozen=True)
class Workload:
    name: str
    nodes: int
    fanout: int


WORKLOADS = (
    Workload("small_sparse_low_successors", 96, 3),
    Workload("small_dense_below_128", 96, 48),
    Workload("large_sparse_low_successors", 1024, 4),
    Workload("large_medium_below_128", 1024, 32),
    Workload("large_dense_high_successors", 768, 192),
)

TRAVERSALS = (
    ("astar", astar_traverse),
    ("dijkstra", dijkstra_traverse),
)


class TimedEmbeddingTable:
    """
    Minimal embedder that mirrors the traversal runtime cache.

    `embed()` returns one row from an in-memory embedding table. `embed_many()`
    gathers a batch with one indexed table select. This isolates the runtime
    embedding lookup choice from transformer inference.
    """

    def __init__(self, num_nodes, dim, device, seed=1234):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        table = torch.randn(
            num_nodes,
            dim,
            generator=generator,
            dtype=torch.float32,
        )
        table = torch.nn.functional.normalize(table, p=2, dim=1)

        self.table = table.to(device)
        self.indexed_text_to_idx = {
            f"n{index}": index
            for index in range(num_nodes)
        }
        self.device = device
        self.reset_stats()

    def clone_fresh(self):
        clone = object.__new__(TimedEmbeddingTable)
        clone.table = self.table
        clone.indexed_text_to_idx = self.indexed_text_to_idx
        clone.device = self.device
        clone.reset_stats()
        return clone

    def reset_stats(self):
        self.single_calls = 0
        self.many_calls = 0
        self.single_lookup_seconds = 0.0
        self.many_lookup_seconds = 0.0
        self.batch_sizes = []

    def has_embedding_index(self):
        return True

    def has_normalized_runtime_embeddings(self):
        return True

    def get_model_dim(self):
        return self.table.shape[1]

    def embed(self, text):
        start = time.perf_counter()
        self.single_calls += 1
        embedding = self.table[self.indexed_text_to_idx[text]].flatten()
        self.single_lookup_seconds += time.perf_counter() - start
        return embedding

    def embed_many(self, texts):
        texts = list(texts)
        start = time.perf_counter()
        self.many_calls += 1
        self.batch_sizes.append(len(texts))

        if texts:
            indices = torch.tensor(
                [self.indexed_text_to_idx[text] for text in texts],
                device=self.device,
                dtype=torch.long,
            )
            embeddings = self.table.index_select(0, indices)
        else:
            embeddings = torch.empty(
                (0, self.table.shape[1]),
                device=self.device,
                dtype=torch.float32,
            )

        self.many_lookup_seconds += time.perf_counter() - start
        return embeddings

    def get_distances(self, source, embeddings, assume_normalized=False):
        if isinstance(embeddings, torch.Tensor):
            if embeddings.numel() == 0:
                return []
            matrix = embeddings.reshape(embeddings.shape[0], -1)
        else:
            embeddings = list(embeddings)
            if not embeddings:
                return []
            matrix = torch.stack(embeddings)

        distances = 1.0 - torch.mv(matrix, source.flatten())
        return distances.detach().cpu().tolist()


def parse_thresholds(value):
    thresholds = [int(part) for part in value.split(",") if part.strip()]

    if not thresholds:
        raise argparse.ArgumentTypeError("at least one threshold is required")
    if any(threshold < 1 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be positive integers")

    return tuple(dict.fromkeys(thresholds))


def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return value


def synchronize(device):
    if device == "cuda":
        torch.cuda.synchronize()


def build_dag(num_nodes, fanout):
    graph = nx.DiGraph()
    nodes = [f"n{index}" for index in range(num_nodes)]
    graph.add_nodes_from(nodes)

    for source in range(num_nodes - 1):
        target_end = min(num_nodes, source + fanout + 1)
        graph.add_edges_from(
            (nodes[source], nodes[target])
            for target in range(source + 1, target_end)
        )

    return graph, nodes[0], nodes[-1]


def run_once(traverse, graph, start_node, end_node, base_embedder, threshold, device):
    embedder = base_embedder.clone_fresh()
    config = {"embedding_index_min_successors": threshold}

    synchronize(device)
    start = time.perf_counter()
    path, visits = traverse(graph, start_node, end_node, embedder, config)
    synchronize(device)

    elapsed = time.perf_counter() - start
    lookup_seconds = (
        embedder.single_lookup_seconds
        + embedder.many_lookup_seconds
    )

    return {
        "elapsed_seconds": elapsed,
        "path_length": len(path),
        "visits": visits,
        "single_calls": embedder.single_calls,
        "many_calls": embedder.many_calls,
        "lookup_seconds": lookup_seconds,
        "single_lookup_seconds": embedder.single_lookup_seconds,
        "many_lookup_seconds": embedder.many_lookup_seconds,
        "avg_many_batch": (
            statistics.mean(embedder.batch_sizes)
            if embedder.batch_sizes
            else 0.0
        ),
        "small_many_batches": sum(
            1
            for batch_size in embedder.batch_sizes
            if batch_size < CURRENT_THRESHOLD
        ),
    }


def summarize_samples(samples):
    representative = samples[0]
    return {
        "median_seconds": statistics.median(
            sample["elapsed_seconds"]
            for sample in samples
        ),
        "median_lookup_seconds": statistics.median(
            sample["lookup_seconds"]
            for sample in samples
        ),
        "visits": representative["visits"],
        "path_length": representative["path_length"],
        "single_calls": representative["single_calls"],
        "many_calls": representative["many_calls"],
        "avg_many_batch": representative["avg_many_batch"],
        "small_many_batches": representative["small_many_batches"],
    }


def benchmark_threshold(
    traverse,
    graph,
    start_node,
    end_node,
    base_embedder,
    threshold,
    repeats,
    warmups,
    device,
):
    for _ in range(warmups):
        run_once(
            traverse,
            graph,
            start_node,
            end_node,
            base_embedder,
            threshold,
            device,
        )

    samples = [
        run_once(
            traverse,
            graph,
            start_node,
            end_node,
            base_embedder,
            threshold,
            device,
        )
        for _ in range(repeats)
    ]
    return summarize_samples(samples)


def benchmark_lookup_batch(base_embedder, batch_size, iterations, device):
    nodes = [f"n{index}" for index in range(batch_size)]
    loop_samples = []
    many_samples = []

    for _ in range(iterations):
        embedder = base_embedder.clone_fresh()
        synchronize(device)
        start = time.perf_counter()
        matrix = torch.stack([embedder.embed(node) for node in nodes])
        _ = matrix.shape
        synchronize(device)
        loop_samples.append(time.perf_counter() - start)

        embedder = base_embedder.clone_fresh()
        synchronize(device)
        start = time.perf_counter()
        matrix = embedder.embed_many(nodes)
        _ = matrix.shape
        synchronize(device)
        many_samples.append(time.perf_counter() - start)

    loop_median = statistics.median(loop_samples)
    many_median = statistics.median(many_samples)
    return {
        "batch_size": batch_size,
        "loop_stack_seconds": loop_median,
        "embed_many_seconds": many_median,
        "embed_many_delta_percent": (many_median / loop_median - 1.0) * 100.0,
    }


def run_benchmark(args):
    device = resolve_device(args.device)
    thresholds = parse_thresholds(args.thresholds)
    results = {
        "device": device,
        "dim": args.dim,
        "repeats": args.repeats,
        "warmups": args.warmups,
        "thresholds": thresholds,
        "selected_threshold": SELECTED_THRESHOLD,
        "workloads": [],
        "microbench": [],
    }

    print(
        "embed_many threshold benchmark "
        f"device={device} dim={args.dim} repeats={args.repeats} "
        f"warmups={args.warmups}"
    )

    for workload in WORKLOADS:
        graph, start_node, end_node = build_dag(workload.nodes, workload.fanout)
        base_embedder = TimedEmbeddingTable(
            workload.nodes,
            args.dim,
            device,
            seed=args.seed,
        )
        workload_result = {
            "name": workload.name,
            "nodes": workload.nodes,
            "fanout": workload.fanout,
            "edges": graph.number_of_edges(),
            "traversals": [],
        }

        print(
            f"\n{workload.name}: nodes={workload.nodes} "
            f"fanout={workload.fanout} edges={graph.number_of_edges()}"
        )

        for traversal_name, traverse in TRAVERSALS:
            threshold_results = []

            for threshold in thresholds:
                summary = benchmark_threshold(
                    traverse,
                    graph,
                    start_node,
                    end_node,
                    base_embedder,
                    threshold,
                    args.repeats,
                    args.warmups,
                    device,
                )
                summary["threshold"] = threshold
                threshold_results.append(summary)

            best_seconds = min(
                summary["median_seconds"]
                for summary in threshold_results
            )
            print(f"  {traversal_name}")
            for summary in threshold_results:
                marker = (
                    "*"
                    if summary["median_seconds"] == best_seconds
                    else " "
                )
                print(
                    "   "
                    f"{marker} threshold={summary['threshold']:3d} "
                    f"median={summary['median_seconds'] * 1000:8.2f}ms "
                    f"lookup={summary['median_lookup_seconds'] * 1000:7.2f}ms "
                    f"vs_best={(summary['median_seconds'] / best_seconds - 1.0) * 100.0:6.1f}% "
                    f"visits={summary['visits']:4d} "
                    f"path={summary['path_length']:3d} "
                    f"single={summary['single_calls']:6d} "
                    f"many={summary['many_calls']:5d} "
                    f"avg_many={summary['avg_many_batch']:5.1f}"
                )

            workload_result["traversals"].append(
                {
                    "name": traversal_name,
                    "thresholds": threshold_results,
                }
            )

        results["workloads"].append(workload_result)
        del base_embedder
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    if not args.skip_microbench:
        print("\nsuccessor lookup microbenchmark")
        base_embedder = TimedEmbeddingTable(
            max(args.micro_batches),
            args.dim,
            device,
            seed=args.seed,
        )

        for batch_size in args.micro_batches:
            summary = benchmark_lookup_batch(
                base_embedder,
                batch_size,
                args.micro_iterations,
                device,
            )
            results["microbench"].append(summary)
            print(
                f"  batch={batch_size:3d} "
                f"loop_stack={summary['loop_stack_seconds'] * 1_000_000:8.2f}us "
                f"embed_many={summary['embed_many_seconds'] * 1_000_000:8.2f}us "
                f"delta={summary['embed_many_delta_percent']:+6.1f}%"
            )

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the traversal successor embedding threshold for "
            "embed_many versus single-row embedding lookups."
        )
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device to benchmark. Default: auto.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=128,
        help="Synthetic embedding dimension.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="1,8,16,32,64,128",
        help="Comma-separated successor thresholds to compare.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Measured repeats per workload/traversal/threshold.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Warmup runs per workload/traversal/threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for deterministic synthetic embeddings.",
    )
    parser.add_argument(
        "--micro-iterations",
        type=int,
        default=200,
        help="Iterations per successor lookup microbenchmark batch size.",
    )
    parser.add_argument(
        "--micro-batches",
        type=parse_thresholds,
        default=parse_thresholds("1,2,4,8,16,32,64,96,128,192,256,512"),
        help="Comma-separated microbenchmark batch sizes.",
    )
    parser.add_argument(
        "--skip-microbench",
        action="store_true",
        help="Skip the isolated successor lookup microbenchmark.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write structured benchmark results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dim <= 0:
        raise ValueError("--dim must be greater than 0")
    if args.repeats <= 0:
        raise ValueError("--repeats must be greater than 0")
    if args.warmups < 0:
        raise ValueError("--warmups must not be negative")

    results = run_benchmark(args)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_output, "w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
            file.write("\n")
        print(f"\nWrote benchmark results to {args.json_output}")


if __name__ == "__main__":
    main()
