"""
Task 6 - Performance evaluation.

Offline comparison: every algorithm plans once on the robot's known map.
  * path length      - number of moves
  * crop-safe cost   - moves + penalties for driving near protected regions
                       (the same yardstick for every algorithm)
  * near-crop cells  - how many cells of the path lie in the "too close" zone
  * nodes expanded   - states taken off the frontier
  * time             - median over several runs (single runs are noisy)
  * reached          - whether a path to the destination was found

Online comparison: the replanning robot versus an A* plan made with the
full true map (a "perfect knowledge" reference), plus the number of replans.
"""

import statistics

from .informed import a_star_search, greedy_best_first_search
from .problem import PathProblem
from .replanning import navigate_with_replanning
from .uninformed import breadth_first_search, depth_first_search

REPEATS = 25


def _median_time(fn, problem):
    times = [fn(problem).time_ms for _ in range(REPEATS)]
    return statistics.median(times)


def run_offline_comparison(field):
    safe = PathProblem(field, crop_safe=True)
    distance_only = PathProblem(field, crop_safe=False)

    runs = [
        ("BFS", breadth_first_search, safe),
        ("DFS", depth_first_search, safe),
        ("Greedy Best-First", greedy_best_first_search, safe),
        ("A* (distance only)", a_star_search, distance_only),
        ("A* (crop-safe)", a_star_search, safe),
    ]

    results = []
    for name, fn, problem in runs:
        res = fn(problem)
        res.algorithm = name
        res.time_ms = _median_time(fn, problem)
        if res.reached:
            res.safety_cost = safe.path_cost(res.path)
            res.near_crop_cells = sum(1 for cell in res.path if safe.penalty[cell] > 0)
        results.append(res)
    return results


def run_online_comparison(field):
    online = navigate_with_replanning(field)

    # Reference: A* that knows every obstacle from the start.
    oracle = a_star_search(PathProblem(field, grid=field.grid, crop_safe=True))
    oracle.algorithm = "A* with full map (reference)"
    return online, oracle


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
OFFLINE_HEADERS = ["Algorithm", "Reached", "Path length", "Crop-safe cost",
                   "Near-crop cells", "Nodes expanded", "Time (ms)"]

ONLINE_HEADERS = ["Strategy", "Reached", "Path length", "Crop-safe cost",
                  "Nodes expanded", "Time (ms)", "Obstacles detected", "Replans"]


def offline_rows(results):
    rows = []
    for r in results:
        if r.reached:
            rows.append([r.algorithm, "Yes", str(r.path_length), f"{r.safety_cost:g}",
                         str(r.near_crop_cells), str(r.nodes_expanded), f"{r.time_ms:.3f}"])
        else:
            rows.append([r.algorithm, "No", "-", "-", "-", str(r.nodes_expanded),
                         f"{r.time_ms:.3f}"])
    return rows


def online_rows(online, oracle):
    return [
        ["A* + online replanning", "Yes" if online.reached else "No",
         str(online.path_length), f"{online.path_cost:g}", str(online.nodes_expanded),
         f"{online.time_ms:.3f}", str(len(online.detections)), str(online.num_replans)],
        [oracle.algorithm, "Yes" if oracle.reached else "No",
         str(oracle.path_length), f"{oracle.path_cost:g}", str(oracle.nodes_expanded),
         f"{oracle.time_ms:.3f}", "-", "0"],
    ]


def format_text_table(headers, rows):
    """Plain text table for the console."""
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    out = [line,
           "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
           line]
    for row in rows:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")
    out.append(line)
    return "\n".join(out)


def format_markdown_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(out)
