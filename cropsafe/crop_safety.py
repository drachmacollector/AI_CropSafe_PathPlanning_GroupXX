"""
Task 4 - Crop-safe planning.

Hard rule  : the robot may never enter potato, crop-protection or obstacle
             cells (handled in PathProblem.actions).
Soft rule  : driving CLOSE to a crop-protection region is allowed but costs
             extra, so a planner that minimises cost (A*) prefers routes that
             keep a safe distance whenever such a route exists.

The penalty depends on the distance to the nearest protected cell, measured
in "rings" around it (Chebyshev distance, so diagonal neighbours count too):

    distance 1 (touching the region)   -> +3 per move
    distance 2 (one cell gap)          -> +1 per move
    distance 3 or more                 -> +0 per move

The step cost of moving INTO a cell is 1 + penalty(cell).
"""

import numpy as np

from .field import PROTECTED

# distance to protected region -> extra cost
PROXIMITY_PENALTY = {1: 3.0, 2: 1.0}


def proximity_penalty_map(grid):
    """Return an array the size of the grid holding the extra cost of every cell."""
    height, width = grid.shape
    penalty = np.zeros((height, width))
    protected = np.argwhere(grid == PROTECTED)
    max_dist = max(PROXIMITY_PENALTY)

    for r, c in protected:
        # Look at the square of cells around this protected cell.
        for rr in range(max(0, r - max_dist), min(height, r + max_dist + 1)):
            for cc in range(max(0, c - max_dist), min(width, c + max_dist + 1)):
                dist = max(abs(rr - r), abs(cc - c))
                if dist in PROXIMITY_PENALTY:
                    # keep the largest penalty if several protected cells are near
                    penalty[rr, cc] = max(penalty[rr, cc], PROXIMITY_PENALTY[dist])

    penalty[grid == PROTECTED] = 0   # those cells cannot be entered anyway
    return penalty
