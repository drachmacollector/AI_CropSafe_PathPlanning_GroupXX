"""Helpers shared by every search algorithm."""

import time
from dataclasses import dataclass, field


@dataclass
class SearchResult:
    algorithm: str
    path: list = None               # list of cells from start to goal, or None
    nodes_expanded: int = 0         # how many states were taken off the frontier
    time_ms: float = 0.0
    path_cost: float = None         # cost under the problem's step costs
    explored: list = field(default_factory=list)  # expansion order (for plots)

    @property
    def reached(self):
        return self.path is not None

    @property
    def path_length(self):
        """Number of moves in the path."""
        return len(self.path) - 1 if self.path else None


def build_path(parent, goal):
    """Walk back through the parent links from the goal to the start."""
    path = [goal]
    while parent[path[-1]] is not None:
        path.append(parent[path[-1]])
    path.reverse()
    return path


def timed(search_fn):
    """Decorator: measure run time and fill in the path cost."""

    def wrapper(problem, *args, **kwargs):
        t0 = time.perf_counter()
        result = search_fn(problem, *args, **kwargs)
        result.time_ms = (time.perf_counter() - t0) * 1000
        if result.path:
            result.path_cost = problem.path_cost(result.path)
        return result

    wrapper.__name__ = search_fn.__name__
    wrapper.__doc__ = search_fn.__doc__
    return wrapper
