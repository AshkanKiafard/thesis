import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import networkx as nx
from fastapi import HTTPException

from core.graph_config import GRAPH_CONFIGS, SUPPORTED_INFERENCE_GRAPHS
from core.inference_registry import get_rl_policy_config, graph_supports_rl
from core.utils import RLGraph
from core.model_registry import (
    activation_label,
    distance_label,
    format_model_display_label,
    stable_config_identity,
)
from web_demo.server import (
    BFSConfig,
    GraphBundle,
    InferenceAStarConfig,
    ModelRuntime,
    RLConfig,
    build_inference_response,
    config_defaults,
    discover_search_methods,
    get_default_bfs_cap,
    get_demo_models,
    get_graph_warmup_query,
    graph_option_payload,
    group_graphs_by_embedding_universe,
    is_ignorable_graph_node,
    parse_algorithm_config,
    remove_ignorable_graph_nodes,
    remove_ignorable_rl_nodes,
    resolve_astar_model_option,
    search_nodes,
    subgraph,
    validate_search_cap,
    warm_runtime_traversals,
)


def fake_model(**overrides):
    model = {
        "id": "sentence-transformers/all-mpnet-base-v2",
        "label": "MPNet Base",
        "base_label": "MPNet",
        "config_label": "MPNet Base",
        "model_dim": 768,
        "dims": [32],
        "distance": "cosine",
        "distance_label": "Cosine",
        "is_finetuned": False,
        "cache_name": "all-mpnet-base-v2",
        "variant": "base",
        "activation": None,
        "model_key": "mpnet",
        "normalize": None,
        "metadata": {},
    }
    model.update(overrides)
    return model


class InferenceRegistryTests(unittest.TestCase):
    def test_label_generation_normalizes_canonical_names_and_terms(self):
        self.assertEqual(
            format_model_display_label("sentence-transformers/all-mpnet-base-v2"),
            "MPNet Base",
        )
        self.assertEqual(
            format_model_display_label("BAAI/bge-large-en-v1.5"),
            "BGE Base",
        )
        self.assertEqual(
            format_model_display_label("ibm-granite/granite-embedding-english-r2"),
            "Granite Base",
        )
        self.assertEqual(
            format_model_display_label("mixedbread-ai/mxbai-embed-large-v1"),
            "mxbai Base",
        )
        self.assertEqual(
            format_model_display_label("Qwen/Qwen3-Embedding-0.6B"),
            "Qwen3-0.6B Base",
        )
        self.assertEqual(activation_label("relu"), "ReLU")
        self.assertEqual(activation_label("gelu"), "GELU")
        self.assertEqual(distance_label("cos"), "Cosine")
        self.assertEqual(distance_label("cosine_distance"), "Cosine")
        self.assertEqual(distance_label("euclid"), "Euclidean")
        self.assertEqual(distance_label("l2"), "Euclidean")

    def test_finetuned_ablation_and_conditional_dimension_labels(self):
        self.assertEqual(
            format_model_display_label(
                "data/models/lightning/all-mpnet-base-v2_relu_cosine_nonorm_matryoshka_v3_finetuned",
                is_finetuned=True,
            ),
            "MPNet FT ReLU+Cosine",
        )
        self.assertEqual(
            format_model_display_label(
                "granite-embedding-english-r2_gelu_euclid_nonorm_matryoshka_v3_ablation_finetuned",
                is_finetuned=True,
            ),
            "Granite AB GELU+Euclidean",
        )
        self.assertEqual(
            format_model_display_label(
                "granite-embedding-english-r2_relu_euclid_nonorm_matryoshka_v3_finetuned",
                is_finetuned=True,
                dimension=32,
                include_dimension=True,
            ),
            "Granite FT ReLU+Euclidean (d=32)",
        )
        self.assertEqual(
            format_model_display_label(
                "Qwen3-Embedding-0.6B_gelu_cosine_nonorm_matryoshka_v3_finetuned",
                is_finetuned=True,
                dimension=64,
                include_dimension=True,
            ),
            "Qwen3-0.6B FT GELU+Cosine (d=64)",
        )

    def test_method_identity_deduplicates_only_equivalent_configs(self):
        methods = discover_search_methods(
            (
                fake_model(dims=[32, 64]),
                fake_model(
                    id="data/models/lightning/all-mpnet-base-v2_relu_cosine_nonorm_matryoshka_v3_finetuned",
                    is_finetuned=True,
                    variant="finetuned",
                    activation="relu",
                    cache_name="all-mpnet-base-v2_relu_cosine_nonorm_matryoshka_v3_finetuned",
                ),
            )
        )
        astar_methods = [
            method for method in methods if method["algorithm"] == "astar"
        ]
        identities = [method["id"] for method in astar_methods]

        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(len(astar_methods), 2)
        self.assertTrue(
            any(
                method["config"]["dimensions"] == [32, 64]
                for method in astar_methods
            )
        )
        self.assertTrue(
            all("dimension" not in method["config"] for method in astar_methods)
        )
        self.assertTrue(
            any(
                method["config"]["variant"] == "finetuned"
                for method in astar_methods
            )
        )

    def test_method_order_is_deterministic(self):
        methods = discover_search_methods(
            (
                fake_model(
                    id="BAAI/bge-large-en-v1.5",
                    model_key="bge",
                    cache_name="bge-large-en-v1.5",
                    model_dim=1024,
                ),
                fake_model(),
            )
        )
        labels = [method["label"] for method in methods[:4]]

        self.assertEqual(labels[0], "BFS")
        self.assertEqual(labels[1], "RL")
        self.assertEqual(labels[2], "MPNet Base")
        self.assertEqual(labels[3], "BGE Base")

    def test_limited_demo_mode_exposes_only_granite_ft_at_dimension_32(self):
        granite = fake_model(
            id=(
                "data/models/lightning/"
                "granite-embedding-english-r2_relu_euclid_nonorm_"
                "matryoshka_v3_finetuned"
            ),
            label="Granite FT ReLU+Euclidean",
            model_key="granite",
            model_dim=768,
            dims=[768, 32, 16],
            distance="euclid",
            is_finetuned=True,
            cache_name=(
                "granite-embedding-english-r2_relu_euclid_nonorm_"
                "matryoshka_v3_finetuned"
            ),
            variant="finetuned",
            activation="relu",
            normalize=False,
        )
        models = get_demo_models((fake_model(), granite), load_all=False)
        methods = discover_search_methods(models)
        astar_methods = [
            method
            for method in methods
            if method["algorithm"] == "astar"
        ]

        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["label"], "Granite FT ReLU+Euclidean")
        self.assertEqual(models[0]["dims"], [32])
        self.assertEqual(len(astar_methods), 1)
        self.assertEqual(astar_methods[0]["config"]["dimensions"], [32])
        self.assertEqual(
            get_demo_models((fake_model(), granite), load_all=True),
            (fake_model(), granite),
        )

    def test_astar_dimension_is_selected_separately_from_model_choice(self):
        model = fake_model(dims=[32, 64])
        method = next(
            entry
            for entry in discover_search_methods((model,))
            if entry["algorithm"] == "astar"
        )

        with (
            patch("web_demo.server.get_search_method", return_value=method),
            patch("web_demo.server.get_model_option", return_value=model),
        ):
            selected = resolve_astar_model_option(method["id"], None, 64)

        self.assertEqual(method["config"]["dimensions"], [32, 64])
        self.assertEqual(selected["selected_dim"], 64)
        self.assertIn("dimension=64", selected["config_id"])
        self.assertEqual(selected["selected_label"], "MPNet Base (d=64)")

    def test_stable_identity_keeps_bfs_cap_out_of_identity(self):
        self.assertEqual(stable_config_identity(algorithm="bfs"), "algorithm=bfs")

    def test_exposed_graphs_and_metadata_are_exact(self):
        graphs = [
            graph_option_payload(graph_id)
            for graph_id in SUPPORTED_INFERENCE_GRAPHS
        ]

        self.assertEqual(
            [graph["label"] for graph in graphs],
            ["CauseNet Precision", "CauseNet Full", "CEG Filtered"],
        )
        self.assertEqual(
            [(graph["nodes"], graph["edges"]) for graph in graphs],
            [
                (80_214, 197_376),
                (12_185_920, 11_606_975),
                (77_264, 21_507_177),
            ],
        )
        self.assertNotIn("ceg_full", SUPPORTED_INFERENCE_GRAPHS)
        self.assertEqual(GRAPH_CONFIGS["causenet"]["label"], "CauseNet Precision")

    def test_full_graph_uses_its_own_embedding_universe_during_warmup(self):
        groups = group_graphs_by_embedding_universe(
            ("causenet", "causenet_full", "ceg")
        )
        self.assertEqual(
            groups,
            (
                (None, "merged_causenet_ceg", ("causenet", "ceg")),
                ("causenet_full", "causenet_full", ("causenet_full",)),
            ),
        )

    def test_startup_warmup_selects_query_from_graph_topology(self):
        graph = nx.DiGraph()
        successors = [f"successor-{index}" for index in range(16)]
        graph.add_edges_from(("source", successor) for successor in successors)
        graph.add_edge(successors[0], "two-hop-target")
        nodes = sorted(graph.nodes)
        bundle = GraphBundle(
            name="dynamic-warmup-test",
            label="Dynamic Warmup Test",
            path=Path("graph.jsonl"),
            graph=graph,
            nodes=nodes,
            lowercase_to_node={node.lower(): node for node in nodes},
        )

        class FakeEmbedder:
            device = "cpu"

            def set_matryoshka_dim(self, dim):
                self.dim = dim

        runtime = ModelRuntime(
            model_path="dynamic-model",
            cache_suffix=None,
            node_universe="dynamic-universe",
            embedder=FakeEmbedder(),
            indexed_graphs={},
        )

        self.assertEqual(
            get_graph_warmup_query(bundle),
            ("source", "two-hop-target"),
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "traverse_strategies": SimpleNamespace(
                        astar_traverse=object(),
                    )
                },
            ),
            patch(
                "web_demo.server.get_loaded_graph_bundle",
                return_value=bundle,
            ),
            patch(
                "web_demo.server.get_default_astar_max_visits",
                return_value={"value": 27, "source": "test"},
            ),
            patch(
                "web_demo.server.traverse_graph",
                return_value=([], 27),
            ) as traverse,
        ):
            warm_runtime_traversals(runtime, (bundle.name,), [32])

        self.assertEqual(traverse.call_count, 1)
        self.assertEqual(traverse.call_args.args[1:3], ("source", "two-hop-target"))
        self.assertEqual(traverse.call_args.args[5]["astar_max_visits"], 27)

    def test_bfs_default_cap_and_validation(self):
        self.assertEqual(get_default_bfs_cap("causenet")["value"], 12_170)
        self.assertEqual(
            config_defaults(graph="causenet", algorithm="bfs")["bfs_cap"],
            12_170,
        )
        self.assertEqual(
            config_defaults(graph="causenet_full", algorithm="bfs")["bfs_cap"],
            12_170,
        )
        self.assertEqual(BFSConfig.model_validate({"cap": -1}).cap, -1)
        self.assertEqual(BFSConfig.model_validate({"cap": 0}).cap, 0)
        self.assertEqual(BFSConfig.model_validate({"cap": 25}).cap, 25)
        with self.assertRaises(HTTPException):
            validate_search_cap(-2, "BFS search cap")

    def test_rl_policy_config_and_graph_support(self):
        policy = get_rl_policy_config()

        self.assertEqual(policy.label, "RL")
        self.assertEqual(policy.parameters, "12M")
        self.assertEqual(
            policy.runtime_config(),
            {
                "rl_model_path": str(policy.checkpoint_path),
                "rl_beam_width": 50,
                "rl_max_path_len": 2,
                "rl_max_actions": 5000,
                "rl_max_visits": -1,
            },
        )
        self.assertEqual(set(policy.supported_graphs), set(SUPPORTED_INFERENCE_GRAPHS))
        self.assertTrue(graph_supports_rl("causenet", policy.id))
        self.assertFalse(graph_supports_rl("ceg_full", policy.id))
        self.assertNotIn("dimension", config_defaults(graph="causenet", algorithm="rl"))

    def test_backend_rejects_algorithm_specific_field_leakage(self):
        with self.assertRaises(HTTPException):
            parse_algorithm_config(BFSConfig, {"model_config_id": "mpnet"})

        with self.assertRaises(HTTPException):
            parse_algorithm_config(RLConfig, {"dimension": 32})

        with self.assertRaises(HTTPException):
            parse_algorithm_config(InferenceAStarConfig, {"policy_config_id": "rl"})

        with self.assertRaises(HTTPException):
            resolve_astar_model_option(None, None, None)

    def test_inference_response_shape_is_frontend_compatible(self):
        graph = nx.DiGraph()
        graph.add_edge("rain", "flood")
        bundle = GraphBundle(
            name="causenet",
            label="CauseNet Precision",
            path=Path("graph.jsonl"),
            graph=graph,
            nodes=["flood", "rain"],
            lowercase_to_node={"flood": "flood", "rain": "rain"},
        )

        response = build_inference_response(
            bundle=bundle,
            algorithm="bfs",
            source="rain",
            target="flood",
            path=["rain", "flood"],
            visited_nodes=1,
            runtime_ms=1.25,
            config_id="bfs",
            config_label="BFS (p95 cap)",
            used_config={"bfs_max_visits": 12_170},
            applied_cap=12_170,
            termination_reason="path_found",
        )

        self.assertEqual(response["algorithm"], "bfs")
        self.assertEqual(response["graph_id"], "causenet")
        self.assertTrue(response["found"])
        self.assertEqual(
            response["path_edges"],
            [{"source": "rain", "target": "flood"}],
        )
        self.assertEqual(response["visited_nodes"], 1)
        self.assertEqual(response["termination_reason"], "path_found")
        self.assertTrue(response["graph"]["nodes"])

    def test_autocomplete_and_clicked_node_neighborhood_remain_available(self):
        graph = nx.DiGraph()
        graph.add_edge("ai", "intermediate")
        graph.add_edge("intermediate", "target concept")
        graph.add_edge("ai", "ai safety")
        nodes = sorted(graph.nodes)
        bundle = GraphBundle(
            name="causenet",
            label="CauseNet Precision",
            path=Path("graph.jsonl"),
            graph=graph,
            nodes=nodes,
            lowercase_to_node={node.lower(): node for node in nodes},
        )

        self.assertEqual(search_nodes(bundle, "ai", 24), ["ai", "ai safety"])

        with patch(
            "web_demo.server.get_loaded_graph_bundle",
            return_value=bundle,
        ):
            payload = subgraph(
                graph="causenet",
                center="ai",
                source=None,
                target=None,
                depth=2,
                limit=20,
            )

        self.assertEqual(
            {node["id"] for node in payload["nodes"]},
            {"ai", "ai safety", "intermediate", "target concept"},
        )

    def test_frontend_keeps_autocomplete_and_node_click_handlers(self):
        static_dir = Path(__file__).resolve().parents[1] / "web_demo" / "static"
        app_js = (static_dir / "app.js").read_text(encoding="utf-8")
        index_html = (static_dir / "index.html").read_text(encoding="utf-8")
        styles_css = (static_dir / "styles.css").read_text(encoding="utf-8")

        self.assertIn(".onNodeClick(handleNodeClick)", app_js)
        self.assertIn("await loadSubgraph(node.id)", app_js)
        self.assertIn('item.textContent = "No path found"', app_js)
        self.assertNotIn('item.textContent = `${result.source} -> ${result.target}`', app_js)
        self.assertIn('result.found ? "path-found" : "path-missing"', app_js)
        self.assertIn('setResultStatus(result.config_label, "Path found", "found")', app_js)
        self.assertIn('result-status-outcome', app_js)
        self.assertIn("@keyframes result-success-pulse", styles_css)
        self.assertIn("@keyframes result-missing-wobble", styles_css)
        self.assertIn("prefers-reduced-motion: reduce", styles_css)
        self.assertIn('list="source-suggestions"', index_html)
        self.assertIn('list="target-suggestions"', index_html)

    def test_ignorable_graph_nodes_are_removed_before_indexing(self):
        graph = nx.DiGraph()
        graph.add_edge("", "effect")
        graph.add_edge("# #", "effect")
        graph.add_edge("???", "effect")
        graph.add_edge("C#", "effect")
        graph.add_edge("cause", "effect")

        removed = remove_ignorable_graph_nodes(graph, "causenet")

        self.assertEqual(removed, 3)
        self.assertNotIn("", graph.nodes)
        self.assertNotIn("# #", graph.nodes)
        self.assertNotIn("???", graph.nodes)
        self.assertIn("C#", graph.nodes)
        self.assertIn("cause", graph.nodes)
        self.assertIn("effect", graph.nodes)
        self.assertTrue(is_ignorable_graph_node("# #"))
        self.assertFalse(is_ignorable_graph_node("C#"))

    def test_ignorable_nodes_are_removed_from_rl_graph(self):
        graph = RLGraph(
            adjacency={"cause": ["effect", "# #"], "# #": ["effect"]},
            edge_sources={
                ("cause", "effect"): "valid",
                ("cause", "# #"): "invalid",
                ("# #", "effect"): "invalid",
            },
            nodes={"cause", "effect", "# #", "stop stop action"},
        )

        removed = remove_ignorable_rl_nodes(graph, "causenet")

        self.assertEqual(removed, 1)
        self.assertEqual(graph.successors("cause"), ["effect"])
        self.assertNotIn("# #", graph.nodes)
        self.assertNotIn(("cause", "# #"), graph.edge_sources)


if __name__ == "__main__":
    unittest.main()
