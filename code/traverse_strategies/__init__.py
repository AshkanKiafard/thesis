from .astar import astar_traverse
from .bfs import bfs_traverse
from .dijkstra import dijkstra_traverse

__all__ = ["astar_traverse", "bfs_traverse", "dijkstra_traverse"]

# TODO add metadata to nodes: if visited + embedding