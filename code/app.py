"""Local launcher for the interactive causal-graph A* web demo."""

from web_demo.server import app


def main():
    import uvicorn

    uvicorn.run(
        "web_demo.server:app",
        host="127.0.0.1",
        port=8001,
        reload=False,
    )


if __name__ == "__main__":
    main()
