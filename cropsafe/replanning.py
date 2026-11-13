"""
Task 5 - Obstacle detection and replanning (online navigation).

The robot has two maps:
  * the TRUE field  (field.grid)       - includes the hidden obstacles
  * its KNOWN map   (known)            - hidden obstacles look like free soil

Loop, one move at a time:
  1. SENSE  - look at every cell within SENSOR_RANGE of the robot in the true
              field. A real obstacle the robot did not know about is
              "detected" and written into the known map.
  2. CHECK  - if a newly detected obstacle lies on the remaining route, the
              route is blocked.
  3. REPLAN - run A* again from the current cell on the updated known map.
  4. MOVE   - take the next step of the (possibly new) route.

Detection is simulated in Python; no hardware is needed.
"""

import time
from dataclasses import dataclass, field

from .field import OBSTACLE, Field
from .informed import a_star_search
from .problem import PathProblem

SENSOR_RANGE = 1    # the robot sees its 8 neighbouring cells


@dataclass
class ReplanEvent:
    step: int               # moves made before the replan
    position: tuple         # where the robot was
    obstacles: list         # newly detected obstacles that blocked the route
    old_route: list         # remaining route that became blocked
    new_route: list         # new A* route (None if the goal became unreachable)


@dataclass
class OnlineResult:
    initial_route: list
    travelled: list                                   # cells actually visited
    detections: list = field(default_factory=list)   # (step, obstacle cell)
    replans: list = field(default_factory=list)       # ReplanEvent objects
    frames: list = field(default_factory=list)        # snapshots for animation
    nodes_expanded: int = 0     # total over the first plan and all replans
    time_ms: float = 0.0
    reached: bool = False
    path_cost: float = 0.0      # crop-safe cost of the route actually driven

    @property
    def num_replans(self):
        return len(self.replans)

    @property
    def path_length(self):
        return len(self.travelled) - 1


def sense(field: Field, known, position, sensor_range=SENSOR_RANGE):
    """Return hidden obstacles near `position` that are not yet in `known`."""
    r0, c0 = position
    found = []
    for r in range(r0 - sensor_range, r0 + sensor_range + 1):
        for c in range(c0 - sensor_range, c0 + sensor_range + 1):
            cell = (r, c)
            if (field.in_bounds(cell) and field.grid[cell] == OBSTACLE
                    and known[cell] != OBSTACLE):
                found.append(cell)
    return found


def plan(field, known, start):
    """Run crop-safe A* on the robot's current knowledge."""
    return a_star_search(PathProblem(field, grid=known, start=start, crop_safe=True))


def navigate_with_replanning(field: Field, sensor_range=SENSOR_RANGE, max_steps=1000):
    """Drive the robot from start to goal, replanning when the route gets blocked."""
    t0 = time.perf_counter()
    known = field.known_grid()
    position = field.start

    first = plan(field, known, position)
    route = first.path                     # remaining route, starts at `position`
    result = OnlineResult(initial_route=route, travelled=[position],
                          nodes_expanded=first.nodes_expanded)

    for step in range(max_steps):
        # 1. SENSE: update the known map with anything new around the robot
        new_obstacles = sense(field, known, position, sensor_range)
        for cell in new_obstacles:
            known[cell] = OBSTACLE
            result.detections.append((step, cell))

        # 2. CHECK + 3. REPLAN if the remaining route now hits an obstacle
        blocking = [cell for cell in new_obstacles if route and cell in route]
        if blocking:
            replan = plan(field, known, position)
            result.nodes_expanded += replan.nodes_expanded
            result.replans.append(ReplanEvent(step, position, blocking, route, replan.path))
            route = replan.path

        result.frames.append({
            "position": position,
            "route": list(route) if route else [],
            "known_obstacles": [c for _, c in result.detections],
            "replanned": bool(blocking),
        })

        if route is None:                  # goal cannot be reached any more
            break
        if position == field.goal:
            result.reached = True
            break

        # 4. MOVE one step along the route
        position = route[1]
        route = route[1:]
        result.travelled.append(position)

    # cost of the route actually driven, measured on the true field
    true_problem = PathProblem(field, grid=field.grid, crop_safe=True)
    result.path_cost = true_problem.path_cost(result.travelled)
    result.time_ms = (time.perf_counter() - t0) * 1000
    return result
