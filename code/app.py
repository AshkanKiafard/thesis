"""Local launcher for the interactive causal-graph inference web demo."""

import os

# False: expose and preload BFS, RL, and Granite FT at d=32 only.
# True: expose and preload every available A* model and Matryoshka dimension.
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
