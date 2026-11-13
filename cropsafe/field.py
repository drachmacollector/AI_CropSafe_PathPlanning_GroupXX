"""
Task 1 - Field environment.

The potato field is modelled as a 2D grid. Every cell holds one cell type
(free soil, potato row, crop-protection region or obstacle). The map is
written as a list of strings so that anyone can read and edit it by hand.

Positions are (row, col) tuples. Row 0 is the top of the field.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Cell types stored in the grid
# ---------------------------------------------------------------------------
FREE = 0        # safe soil the robot may drive on
POTATO = 1      # potato plants (driving here damages the crop)
PROTECTED = 2   # crop-protection region, e.g. a diseased / seedling patch
OBSTACLE = 3    # stones, tools or other objects

# The robot is never allowed to enter these cells.
BLOCKED_TYPES = (POTATO, PROTECTED, OBSTACLE)

# ---------------------------------------------------------------------------
# Default field map
#   .  free soil (lanes between rows and the headlands at the edges)
#   P  potato row
#   X  crop-protection region
#   #  obstacle the robot already knows about
#   ?  obstacle the robot does NOT know about (found only while driving)
#   S  start position
#   G  destination
# ---------------------------------------------------------------------------
DEFAULT_LAYOUT = [
    "............?............",
    ".........................",
    "..P.P.P.P.P.P.P.P.P.P.P..",
    "..P.P.P.PXXXXXP.P.P.P.P..",
    "..P.P.P.PXXXXXP.P.P.P#P..",
    "..P.P.P.PXXXXXP.P.P.P.P.?",
    "..P.P.P.PXXXXXP.P.P.P.P..",
    "S.......................G",
    "..P.P.P.PXXXXXP.P.P.P.P..",
    "..P#P.P.PXXXXXP.P.P.P.P..",
    "..P.P.P.PXXXXXP.P.P.P.P..",
    "..P.P.P.PXXXXXP.P.P.P.P..",
    "..P.P.P.P.P.P.P.P.P.P.P..",
    ".................#.......",
    ".........................",
]

_CHAR_TO_CELL = {".": FREE, "S": FREE, "G": FREE,
                 "P": POTATO, "X": PROTECTED, "#": OBSTACLE, "?": OBSTACLE}


class Field:
    """A potato field: the true grid, start, goal and hidden obstacles."""

    def __init__(self, layout=DEFAULT_LAYOUT):
        self.height = len(layout)
        self.width = len(layout[0])
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.hidden_obstacles = []   # obstacles unknown to the robot at first
        self.start = None
        self.goal = None

        for r, line in enumerate(layout):
            if len(line) != self.width:
                raise ValueError(f"Layout row {r} has the wrong length")
            for c, ch in enumerate(line):
                if ch not in _CHAR_TO_CELL:
                    raise ValueError(f"Unknown map symbol {ch!r} at {(r, c)}")
                self.grid[r, c] = _CHAR_TO_CELL[ch]
                if ch == "S":
                    self.start = (r, c)
                elif ch == "G":
                    self.goal = (r, c)
                elif ch == "?":
                    self.hidden_obstacles.append((r, c))

        if self.start is None or self.goal is None:
            raise ValueError("Layout needs one 'S' and one 'G'")

    # ----- basic queries --------------------------------------------------
    def in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, pos, grid=None):
        """True if the robot may stand on this cell (using `grid` if given)."""
        grid = self.grid if grid is None else grid
        return self.in_bounds(pos) and grid[pos] not in BLOCKED_TYPES

    def known_grid(self):
        """The map the robot starts with: hidden obstacles look like free soil."""
        grid = self.grid.copy()
        for pos in self.hidden_obstacles:
            grid[pos] = FREE
        return grid

    def cells_of(self, cell_type):
        """All positions holding a given cell type."""
        return [tuple(p) for p in np.argwhere(self.grid == cell_type)]
