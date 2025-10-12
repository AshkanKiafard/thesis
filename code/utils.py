import json

import networkx as nx


def load_graph(file_path):
    with open(file_path) as f:
        return nx.DiGraph([
            (
                c,
                e,
                {
                    "support": d.get("support", 0),
                    "sentence": d.get("sources", [{}])[0].get("payload", {}).get("sentence", "")
                }
            )
            for d in map(json.loads, f)
            if (c := d["causal_relation"]["cause"]["concept"]) != (e := d["causal_relation"]["effect"]["concept"])
        ])
