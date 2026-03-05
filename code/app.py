import os
import time
import gradio as gr
import traverse_strategies as ts
from embeddings import STEmbeder, DistanceMetric, GloveEmbeder
from utils import load_graph, traverse_graph

# ==========================================
# 1. WEBIS TEMPLATE CONFIGURATION
# ==========================================

head_config = """
<link rel="stylesheet" href="https://assets.webis.de/css/style.css?1760090551">
<script src="https://assets.webis.de/js/thirdparty/uikit/uikit.min.js"></script>
<script src="https://assets.webis.de/js/thirdparty/uikit/uikit-icons.min.js"></script>
<style>
    footer.svelte-1rjryqp { display: none !important; } 
    .gradio-container { font-family: "Lato", sans-serif !important; }
    .uk-icon > svg { width: 20px; height: 20px; }
</style>
"""

webis_header_html = """
<div class="uk-background-secondary global-nav" data-uk-sticky>
    <nav class="uk-navbar-container uk-navbar-transparent uk-container uk-light" data-uk-navbar="mode: click">
        <div class="uk-navbar-left">
            <ul class="uk-navbar-nav">
                <li>
                    <a href="https://webis.de/"><img src="https://assets.webis.de/img/webis-logo.png" alt="Webis Logo" class="uk-logo"> Webis.de</a>
                </li>
            </ul>
        </div>
    </nav>
</div>

<nav class="uk-container uk-margin-top">
    <ul class="uk-breadcrumb">
        <li><a href="https://webis.de">Webis.de</a></li>
        <li><a href="https://webis.de/research.html">Research</a></li>
        <li><a href="#">Ashkan's Project</a></li> 
        <li class="uk-disabled"><a href="#">Demo</a></li>
    </ul>
</nav>

<div class="uk-container uk-margin-small-bottom">
    <h1>Causal Graph Traversal</h1>
</div>
"""

webis_footer_html = """
<footer class="uk-section uk-section-muted footer-section uk-margin-large-top">
    <div class="uk-container">
        <div class="uk-grid uk-grid-small uk-margin-top">
            <div class="uk-width-1-5@s"></div>
            <div class="uk-width-expand uk-visible@s"></div>
            <div>
              &copy; 2025 <a href="https://webis.de/">Webis Group</a> <span class="uk-padding-small">&bullet;</span>

              <a href="https://github.com/webis-de" class="uk-icon uk-margin-small-right" style="color: #666; display:inline-block;">
                <svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M10,1 C5.03,1 1,5.03 1,10 C1,13.98 3.58,17.35 7.16,18.54 C7.61,18.62 7.77,18.34 7.77,18.11 C7.77,17.9 7.76,17.33 7.76,16.58 C5.26,17.12 4.73,15.37 4.73,15.37 C4.32,14.33 3.73,14.05 3.73,14.05 C2.91,13.49 3.79,13.5 3.79,13.5 C4.69,13.56 5.17,14.43 5.17,14.43 C5.97,15.8 7.28,15.41 7.79,15.18 C7.87,14.6 8.1,14.2 8.36,13.98 C6.36,13.75 4.26,12.98 4.26,9.53 C4.26,8.55 4.61,7.74 5.19,7.11 C5.1,6.88 4.79,5.97 5.28,4.73 C5.28,4.73 6.04,4.49 7.75,5.65 C8.47,5.45 9.24,5.35 10,5.35 C10.76,5.35 11.53,5.45 12.25,5.65 C13.97,4.48 14.72,4.73 14.72,4.73 C15.21,5.97 14.9,6.88 14.81,7.11 C15.39,7.74 15.73,8.54 15.73,9.53 C15.73,12.99 13.63,13.75 11.62,13.97 C11.94,14.25 12.23,14.8 12.23,15.64 C12.23,16.84 12.22,17.81 12.22,18.11 C12.22,18.35 12.38,18.63 12.84,18.54 C16.42,17.35 19,13.98 19,10 C19,5.03 14.97,1 10,1 L10,1 Z"></path></svg>
              </a> 

              <a href="https://x.com/webis_de" class="uk-icon uk-margin-small-right" style="color: #666; display:inline-block;">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M18.901 1.153h3.68l-8.04 9.19L24 22.846h-7.406l-5.8-7.584-6.638 7.584H.474l8.6-9.83L0 1.154h7.594l5.243 6.932ZM17.61 20.644h2.039L6.486 3.24H4.298Z"/></svg>
              </a> 

              <a href="https://www.youtube.com/webis" class="uk-icon uk-margin-small-right" style="color: #666; display:inline-block;">
                 <svg viewBox="0 0 20 20" xmlns="http://www.w3.org/2000/svg" fill="currentColor"><path d="M15,4.1 C17.033,4.1 18.5,5.567 18.5,7.6 L18.5,12.4 C18.5,14.433 17.033,15.9 15,15.9 L5,15.9 C2.967,15.9 1.5,14.433 1.5,12.4 L1.5,7.6 C1.5,5.567 2.967,4.1 5,4.1 L15,4.1 Z M8,7.5 L8,12.5 L13,10 L8,7.5 Z"></path></svg>
               </a>

              <span class="uk-padding-small">&bullet;</span>
              <a href="https://webis.de/people.html">Contact</a> <span class="uk-padding-small">&bullet;</span>
              <a href="https://webis.de/legal.html">Impressum / Terms / Privacy</a>
            </div>
        </div>
    </div>
</footer>
"""

# ==========================================
# 2. LOGIC & APP
# ==========================================

causal_graph = load_graph("data/graphs/causenet-precision.jsonl")


def infer(model, cause, effect):
    if not cause or not effect:
        return [["Error", "Please select both Cause and Effect", 0, 0.0]]

    st_embeder = STEmbeder(model, DistanceMetric.COSINE)
    glove_embeder = GloveEmbeder('data/embeddings/glove.6B/glove.6B.300d.txt', DistanceMetric.COSINE)

    algorithms = ["BFS", "A*", "Dijkstra", "RL"]
    results = []

    for algo in algorithms:
        start_time = time.time()

        path = []
        visited_nodes = 0
        try:
            if algo == "BFS":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.bfs_traverse)
            elif algo == "A*":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.astar_traverse)
            elif algo == "Dijkstra":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, st_embeder, ts.dijkstra_traverse)
            elif algo == "RL":
                path, visited_nodes = traverse_graph(causal_graph, cause, effect, glove_embeder, ts.rl_traverse)
        except Exception as e:
            path = [f"Error: {str(e)}"]

        end_time = time.time()
        inference_time = end_time - start_time

        path_str = " -> ".join(path) if isinstance(path, list) else str(path)

        results.append([algo, path_str, visited_nodes, round(inference_time, 4)])

    return results


with gr.Blocks(head=head_config, title="Webis CausalNet Demo", theme=gr.themes.Default(spacing_size="sm")) as demo:
    gr.HTML(webis_header_html)

    with gr.Column(elem_classes=["uk-container"]):
        with gr.Group():
            model_path = "data/models/lightning"
            models = ["all-mpnet-base-v2"]
            if os.path.exists(model_path):
                models += [f"{model_path}/{m}" for m in os.listdir(model_path) if os.path.isdir(f"{model_path}/{m}")]
            model_dropdown = gr.Dropdown(models, label="Model", value=models[0] if models else None)

            with gr.Row():
                cause_dropdown = gr.Dropdown([node for node in causal_graph.nodes], value=None, label="Cause")
                effect_dropdown = gr.Dropdown([node for node in causal_graph.nodes], value=None, label="Effect")

        inference_button = gr.Button("INFER", variant="primary")

        result_dataframe = gr.Dataframe(
            headers=["Algorithm", "Path Found", "Visited Nodes", "Time (s)"],
            datatype=["str", "str", "number", "number"],
            label="Inference Results",
            interactive=False,
            wrap=True
        )

    gr.HTML(webis_footer_html)

    inference_button.click(
        infer,
        inputs=[model_dropdown, cause_dropdown, effect_dropdown],
        outputs=[result_dataframe]
    )

demo.queue().launch(debug=True)
