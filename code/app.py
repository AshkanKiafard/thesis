"""Local launcher for the interactive causal-graph inference web demo."""

import os

# False: CauseNet Precision with Granite FT at d=32 and A* only.
# True: every graph, A* model/dimension, BFS, and RL.
load_all = False
os.environ["WEB_DEMO_LOAD_ALL"] = str(load_all).lower()


def main():
    import uvicorn

    uvicorn.run(
        "web_demo.server:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
    )


if __name__ == "__main__":
    main()
