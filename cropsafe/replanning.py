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


class OnlineRobot:
    """The robot's state while driving: its known map, position and route."""

    def __init__(self, field: Field, sensor_range=SENSOR_RANGE):
        self.field = field
        self.sensor_range = sensor_range
        self.known = field.known_grid()
        self.position = field.start
        self.steps = 0

        first = plan(field, self.known, self.position)
        self.route = first.path            # remaining route, starts at `position`
        self.initial_route = self.route
        self.travelled = [self.position]
        self.detections = []               # (step, obstacle cell)
        self.replans = []                  # ReplanEvent objects
        self.nodes_expanded = first.nodes_expanded

    @property
    def reached(self):
        return self.position == self.field.goal

    @property
    def stuck(self):
        """True if no route to the goal exists on the known map."""
        return self.route is None

    def sense_and_replan(self):
        """Steps 1-3. Returns (newly detected obstacles, replan event or None)."""
        # 1. SENSE: update the known map with anything new around the robot
        new_obstacles = sense(self.field, self.known, self.position, self.sensor_range)
        for cell in new_obstacles:
            self.known[cell] = OBSTACLE
            self.detections.append((self.steps, cell))

        # 2. CHECK: does a new obstacle lie on the remaining route?
        blocking = [cell for cell in new_obstacles if self.route and cell in self.route]
        if not blocking:
            return new_obstacles, None

        # 3. REPLAN with A* from the current cell on the updated map
        replan = plan(self.field, self.known, self.position)
        self.nodes_expanded += replan.nodes_expanded
        event = ReplanEvent(self.steps, self.position, blocking, self.route, replan.path)
        self.replans.append(event)
        self.route = replan.path
        return new_obstacles, event

    def move(self):
        """Step 4: take the next step of the route."""
        self.position = self.route[1]
        self.route = self.route[1:]
        self.travelled.append(self.position)
        self.steps += 1


def navigate_with_replanning(field: Field, sensor_range=SENSOR_RANGE, max_steps=1000):
    """Drive the robot from start to goal, replanning when the route gets blocked."""
    t0 = time.perf_counter()
    robot = OnlineRobot(field, sensor_range)
    frames = []

    for _ in range(max_steps):
        _new, event = robot.sense_and_replan()
        frames.append({
            "position": robot.position,
            "route": list(robot.route) if robot.route else [],
            "known_obstacles": [c for _, c in robot.detections],
            "replanned": event is not None,
        })
        if robot.stuck or robot.reached:
            break
        robot.move()

    # cost of the route actually driven, measured on the true field
    true_problem = PathProblem(field, grid=field.grid, crop_safe=True)
    return OnlineResult(
        initial_route=robot.initial_route,
        travelled=robot.travelled,
        detections=robot.detections,
        replans=robot.replans,
        frames=frames,
        nodes_expanded=robot.nodes_expanded,
        time_ms=(time.perf_counter() - t0) * 1000,
        reached=robot.reached,
        path_cost=true_problem.path_cost(robot.travelled),
    )
