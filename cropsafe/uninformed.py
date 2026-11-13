"""
Task 3 (part 1) - Uninformed search: Breadth-First Search and Depth-First Search.

Both are graph searches: a cell is never added to the frontier twice.
Neither looks at step costs or at where the goal is.
"""

from collections import deque

from .search_result import SearchResult, build_path, timed


@timed
def breadth_first_search(problem):
    """
    BFS explores the field in "rings" around the start: first all cells one
    move away, then two moves away, and so on (FIFO queue).
    It always finds a path with the fewest moves, but it ignores step costs,
    so it does not care how close that path is to protected crops.
    """
    result = SearchResult("BFS")
    start = problem.initial
    if problem.goal_test(start):
        result.path = [start]
        return result

    frontier = deque([start])
    parent = {start: None}          # also works as the "reached" set

    while frontier:
        state = frontier.popleft()
        result.nodes_expanded += 1
        result.explored.append(state)

        for nxt, _cost in problem.neighbours(state):
            if nxt in parent:
                continue
            parent[nxt] = state
            if problem.goal_test(nxt):     # early goal test is safe for BFS
                result.path = build_path(parent, nxt)
                return result
            frontier.append(nxt)

    return result  # no path


@timed
def depth_first_search(problem):
    """
    DFS always continues from the most recently discovered cell (LIFO stack),
    so it runs as deep as possible along one lane before backing up.
    It uses little memory but the path it returns is usually long and
    winding - DFS is neither optimal nor cost-aware.
    """
    result = SearchResult("DFS")
    frontier = [problem.initial]
    parent = {problem.initial: None}
    expanded = set()

    while frontier:
        state = frontier.pop()
        if state in expanded:
            continue
        expanded.add(state)
        result.nodes_expanded += 1
        result.explored.append(state)

        if problem.goal_test(state):
            result.path = build_path(parent, state)
            return result

        # Push in reverse so the first action (Up) is tried first.
        for nxt, _cost in reversed(list(problem.neighbours(state))):
            if nxt not in expanded:
                parent[nxt] = state
                frontier.append(nxt)

    return result  # no path
