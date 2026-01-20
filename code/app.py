import os
import time

import gradio as gr

import traverse_strategies as ts
from embeddings import Embeder, DistanceMetric
from utils import load_graph, traverse_graph

causal_graph = load_graph("data/graphs/causenet-precision.jsonl")

with gr.Blocks() as demo:
    with gr.Group():
        # TODO instead of algorithm return all results as table
        algorithm_dropdown = gr.Dropdown(["BFS", "A*", "Dijkstra"], value="A*", label="Algorithm")
        # TODO add base models
        model_dropdown = gr.Dropdown(
            [f"data/models/lightning/{model}" for model in os.listdir("data/models/lightning") if os.path.isdir(f"data/models/lightning/{model}")],
            label="Model")
        with gr.Row():
            cause_dropdown = gr.Dropdown([node for node in causal_graph.nodes], value=None, label="Cause")
            effect_dropdown = gr.Dropdown([node for node in causal_graph.nodes], value=None, label="Effect")
    inference_button = gr.Button("INFER")
    result_textbox = gr.Textbox(label="Result", lines=3)


    def infer(algorithm, model, cause, effect):
        embeder = Embeder(model, DistanceMetric.COSINE)

        start_time = time.time()
        match algorithm:
            case "BFS":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, ts.bfs_traverse)
            case "A*":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, ts.astar_traverse)
            case "Dijkstra":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, embeder, ts.dijkstra_traverse)
            case _:
                return "Invalid algorithm"
        end_time = time.time()
        inference_time = end_time - start_time

        return f"Path found: {path}\nVisited nodes: {visited_nodes}\nInference time: {inference_time:.4f}s"


    inference_button.click(infer, [algorithm_dropdown, model_dropdown, cause_dropdown, effect_dropdown], [result_textbox])

demo.queue().launch(debug=True)
