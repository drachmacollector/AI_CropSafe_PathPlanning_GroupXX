"""
Task 2 - Problem formulation.

  State          : the robot's cell, a (row, col) tuple.
  Initial state  : the start cell 'S' of the field.
  Actions        : Up, Down, Left, Right (4-connected moves, no diagonals).
                   An action is only available if the target cell is inside
                   the field and is not a potato row, crop-protection cell
                   or a known obstacle.
  Transition     : result((r, c), action) = (r + dr, c + dc).
  Goal test      : state == destination cell 'G'.
  Path cost      : sum of step costs. Each move costs 1.
"""

from .field import Field

# action name -> (row change, col change)
ACTIONS = {
    "Up": (-1, 0),
    "Down": (1, 0),
    "Left": (0, -1),
    "Right": (0, 1),
}


class PathProblem:
    """Search problem: drive from `start` to the field's goal on a given map."""

    def __init__(self, field: Field, grid=None, start=None):
        self.field = field
        # The map the robot believes in. By default it is the robot's known map
        # (hidden obstacles are invisible). Replanning passes an updated map.
        self.grid = field.known_grid() if grid is None else grid
        self.initial = field.start if start is None else start
        self.goal = field.goal

    def actions(self, state):
        """Moves that keep the robot inside the field and on safe soil."""
        valid = []
        for name, (dr, dc) in ACTIONS.items():
            nxt = (state[0] + dr, state[1] + dc)
            if self.field.is_free(nxt, self.grid):
                valid.append(name)
        return valid

    def result(self, state, action):
        dr, dc = ACTIONS[action]
        return (state[0] + dr, state[1] + dc)

    def goal_test(self, state):
        return state == self.goal

    def step_cost(self, state, action, next_state):
        return 1

    def neighbours(self, state):
        """(next_state, step_cost) pairs - a shortcut used by the searches."""
        for action in self.actions(state):
            nxt = self.result(state, action)
            yield nxt, self.step_cost(state, action, nxt)

    def path_cost(self, path):
        """Total cost of a path given as a list of cells."""
        return sum(self.step_cost(a, None, b) for a, b in zip(path, path[1:]))
