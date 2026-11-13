"""
Task 3 (part 2) - Informed search: Greedy Best-First Search and A*.

Heuristic
---------
h(n) = Manhattan distance from cell n to the goal = |r1 - r2| + |c1 - c2|.

* The robot moves only Up/Down/Left/Right, so it needs at least that many
  moves to reach the goal, even on an empty field.
* Every move costs at least 1 (crop-safe penalties only ADD cost), so h(n)
  never overestimates the true remaining cost -> h is ADMISSIBLE.
* One move changes h by at most 1 while costing at least 1, so
  h(n) <= cost(n, n') + h(n') -> h is also CONSISTENT.
Because of this, A* with h returns the cheapest path.
"""

import heapq
from itertools import count

from .search_result import SearchResult, build_path, timed


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


@timed
def greedy_best_first_search(problem, heuristic=manhattan):
    """
    Always expands the frontier cell that LOOKS closest to the goal,
    f(n) = h(n). Very fast when the way is open, but it ignores the cost
    already paid, so it can be lured into dead ends and is not optimal.
    """
    result = SearchResult("Greedy Best-First")
    tie = count()                   # breaks ties in insertion order
    start = problem.initial
    frontier = [(heuristic(start, problem.goal), next(tie), start)]
    parent = {start: None}
    expanded = set()

    while frontier:
        _h, _, state = heapq.heappop(frontier)
        if state in expanded:
            continue
        expanded.add(state)
        result.nodes_expanded += 1
        result.explored.append(state)

        if problem.goal_test(state):
            result.path = build_path(parent, state)
            return result

        for nxt, _cost in problem.neighbours(state):
            if nxt not in parent:
                parent[nxt] = state
                heapq.heappush(frontier, (heuristic(nxt, problem.goal), next(tie), nxt))

    return result


@timed
def a_star_search(problem, heuristic=manhattan, name="A*"):
    """
    Expands the cell with the smallest f(n) = g(n) + h(n), where g(n) is the
    cost paid so far. With an admissible, consistent heuristic the first time
    the goal is popped its path is the cheapest one.
    """
    result = SearchResult(name)
    tie = count()
    start = problem.initial
    g = {start: 0}                  # best known cost to reach each cell
    parent = {start: None}
    frontier = [(heuristic(start, problem.goal), next(tie), start)]
    expanded = set()

    while frontier:
        _f, _, state = heapq.heappop(frontier)
        if state in expanded:       # an older, more expensive copy
            continue
        expanded.add(state)
        result.nodes_expanded += 1
        result.explored.append(state)

        if problem.goal_test(state):
            result.path = build_path(parent, state)
            return result

        for nxt, cost in problem.neighbours(state):
            new_g = g[state] + cost
            if nxt not in g or new_g < g[nxt]:
                g[nxt] = new_g
                parent[nxt] = state
                f = new_g + heuristic(nxt, problem.goal)
                heapq.heappush(frontier, (f, next(tie), nxt))

    return result
