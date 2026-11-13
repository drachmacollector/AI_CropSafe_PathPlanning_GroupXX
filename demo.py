"""
Live, step-by-step demo of the project, for presenting in class.

    python demo.py              start from the beginning
    python demo.py --stage 5    jump straight to a stage (1-6)

Stages
    1  The potato field                    (Task 1)
    2  Problem formulation                 (Task 2)
    3  Search race: BFS, DFS, Greedy, A*   (Task 3)
    4  Crop-safe A* vs shortest A*         (Task 4)
    5  Live drive: detect and replan       (Task 5)  - click cells to drop obstacles!
    6  Results table                       (Task 6)

Controls
    Right arrow / n  next stage        Left arrow / b  previous stage
    r                replay stage      p               pause / resume animation
    + / -            faster / slower   q               quit
    Mouse click      (stage 5) drop a surprise obstacle the robot does not know about
"""

import argparse

import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.widgets import Button

from cropsafe import visualize as viz
from cropsafe.crop_safety import PROXIMITY_PENALTY
from cropsafe.evaluation import (OFFLINE_HEADERS, ONLINE_HEADERS, offline_rows,
                                 online_rows, run_offline_comparison, run_online_comparison)
from cropsafe.field import BLOCKED_TYPES, OBSTACLE, POTATO, PROTECTED, Field
from cropsafe.problem import ACTIONS, PathProblem
from cropsafe.replanning import SENSOR_RANGE, OnlineRobot

plt = viz.plt

STAGES = [
    ("1", "The potato field", "Task 1 - field environment"),
    ("2", "Problem formulation", "Task 2 - how the robot is allowed to move"),
    ("3", "Search race", "Task 3 - BFS, DFS, Greedy Best-First and A* explore the field"),
    ("4", "Crop-safe planning", "Task 4 - penalty for driving close to protected crops"),
    ("5", "Live drive with obstacle detection", "Task 5 - detect, update map, replan with A*"),
    ("6", "Performance comparison", "Task 6 - results table"),
]

MONO = {"family": "monospace", "fontsize": 10.5, "color": viz.TEXT}


class Demo:
    def __init__(self, start_stage=1):
        self.fig = plt.figure(figsize=(15, 8.6), facecolor="white")
        self.fig.canvas.manager.set_window_title("Crop-Safe Path Planning - live demo")
        self.stage = start_stage - 1
        self.speed = 1.0
        self.paused = False
        self.timer = None
        self.artists = []          # everything owned by the current stage
        self.on_tick = None
        self.on_click = None

        # search results are computed once and reused by several stages
        self.field = Field()
        self.results = {r.algorithm: r for r in run_offline_comparison(self.field)}
        self.live_robot = None
        self.live_field = None

        self._build_controls()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.fig.canvas.mpl_connect("button_press_event", self._on_mouse)
        self.show_stage()

    # ------------------------------------------------------------------
    # Navigation and shared layout
    # ------------------------------------------------------------------
    def _build_controls(self):
        specs = [("<  Back", 0.62, self.prev_stage), ("Replay", 0.72, self.replay),
                 ("Pause", 0.80, self.toggle_pause), ("Next  >", 0.88, self.next_stage)]
        self.buttons = []
        for label, x, callback in specs:
            ax = self.fig.add_axes([x, 0.015, 0.075, 0.045])
            button = Button(ax, label, color="#f1eee8", hovercolor="#e2ddd2")
            button.label.set_fontsize(10)
            button.on_clicked(lambda _event, cb=callback: cb())
            self.buttons.append(button)
        self.fig.text(0.01, 0.03, "keys:  <- ->  navigate   r replay   p pause   "
                      "+/- speed   q quit", fontsize=9, color="#77736b")

    def show_stage(self):
        self._clear()
        number, title, subtitle = STAGES[self.stage]
        self._add(self.fig.text(0.01, 0.955, f"{number}/6   {title}", fontsize=19,
                                fontweight="bold", color=viz.TEXT))
        self._add(self.fig.text(0.01, 0.92, subtitle, fontsize=11.5, color="#5c5a55"))
        [self.stage_field, self.stage_formulation, self.stage_search_race,
         self.stage_crop_safe, self.stage_live_drive, self.stage_results][self.stage]()
        self.fig.canvas.draw_idle()

    def next_stage(self):
        if self.stage < len(STAGES) - 1:
            self.stage += 1
            self.show_stage()

    def prev_stage(self):
        if self.stage > 0:
            self.stage -= 1
            self.show_stage()

    def replay(self):
        self.show_stage()

    def toggle_pause(self):
        self.paused = not self.paused
        self.buttons[2].label.set_text("Resume" if self.paused else "Pause")
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        actions = {"right": self.next_stage, "n": self.next_stage,
                   "left": self.prev_stage, "b": self.prev_stage,
                   "r": self.replay, "p": self.toggle_pause, " ": self.toggle_pause,
                   "q": lambda: plt.close(self.fig)}
        if event.key in actions:
            actions[event.key]()
        elif event.key in ("+", "="):
            self.speed = min(self.speed * 1.5, 8)
        elif event.key == "-":
            self.speed = max(self.speed / 1.5, 0.25)

    def _on_mouse(self, event):
        if self.on_click and event.inaxes is not None and event.xdata is not None:
            self.on_click(event)

    def _clear(self):
        if self.timer is not None:
            self.timer.stop()
            self.timer = None
        self.on_tick = self.on_click = None
        for artist in self.artists:
            artist.remove()
        self.artists = []
        if self.paused:
            self.toggle_pause()

    def _add(self, artist):
        self.artists.append(artist)
        return artist

    def _axes(self, rect):
        return self._add(self.fig.add_axes(rect))

    def _start_animation(self, tick, interval_ms):
        """Call `tick()` every interval (scaled by the speed) until it returns False."""
        self.on_tick = tick

        def step():
            if self.paused or self.on_tick is None:
                return
            # run several ticks per timer event when sped up
            for _ in range(max(1, round(self.speed))):
                if self.on_tick() is False:
                    self.timer.stop()
                    break
            self.fig.canvas.draw_idle()

        self.timer = self.fig.canvas.new_timer(interval=interval_ms / min(self.speed, 1))
        self.timer.add_callback(step)
        self.timer.start()

    def _text_panel(self, rect, text, **style):
        ax = self._axes(rect)
        ax.axis("off")
        props = dict(MONO)
        props.update(style)
        ax.text(0, 1, text, va="top", ha="left", transform=ax.transAxes, linespacing=1.55,
                **props)
        return ax

    # ------------------------------------------------------------------
    # Stage 1 - field
    # ------------------------------------------------------------------
    def stage_field(self):
        f = self.field
        ax = self._axes([0.02, 0.1, 0.62, 0.78])
        viz.draw_field(ax, f, grid=f.known_grid(), unknown=f.hidden_obstacles)
        counts = {name: int((f.grid == t).sum()) for name, t in
                  [("potato", POTATO), ("protected", PROTECTED)]}
        known_obstacles = int((f.known_grid() == OBSTACLE).sum())
        free = int((~np.isin(f.grid, BLOCKED_TYPES)).sum())
        text = (
            f"Grid            {f.height} rows x {f.width} cols\n"
            f"Start  S        {f.start}\n"
            f"Goal   *        {f.goal}\n\n"
            f"Safe soil       {free} cells\n"
            f"Potato plants   {counts['potato']} cells\n"
            f"Protected zone  {counts['protected']} cells\n"
            f"Known obstacles {known_obstacles}\n"
            f"Hidden (?)      {len(f.hidden_obstacles)}  <- robot does\n"
            f"                    NOT know these\n\n"
            "The robot drives in the lanes\n"
            "between potato rows and on the\n"
            "headlands at the top and bottom.\n"
            "Row 7 is a cross path between\n"
            "two crop-protection regions."
        )
        self._text_panel([0.67, 0.42, 0.31, 0.46], text)
        legend_ax = self._axes([0.67, 0.1, 0.31, 0.3])
        legend_ax.axis("off")
        legend_ax.legend(handles=viz.field_legend_handles(include_hidden=True),
                         loc="upper left", frameon=False, fontsize=10.5, ncol=2)

    # ------------------------------------------------------------------
    # Stage 2 - problem formulation
    # ------------------------------------------------------------------
    def stage_formulation(self):
        f = self.field
        problem = PathProblem(f)
        ax = self._axes([0.02, 0.1, 0.58, 0.78])
        viz.draw_field(ax, f, grid=f.known_grid())

        # show the available actions from two example states
        for state, label in [((4, 3), "example state"), ((1, 10), "example state")]:
            allowed = problem.actions(state)
            for name, (dr, dc) in ACTIONS.items():
                end = (state[0] + dr * 0.85, state[1] + dc * 0.85)
                ok = name in allowed
                colour = viz.START_COLOUR if ok else viz.GOAL_COLOUR
                ax.add_patch(FancyArrowPatch((state[1], state[0]), (end[1], end[0]),
                                             arrowstyle="-|>", mutation_scale=14,
                                             color=colour, linewidth=2.2, zorder=9))
                if not ok:
                    ax.text(end[1], end[0], "x", color=viz.GOAL_COLOUR, fontsize=13,
                            fontweight="bold", ha="center", va="center", zorder=10)
            ax.scatter(state[1], state[0], s=120, color=viz.TRAVELLED, edgecolor="white",
                       marker="s", zorder=10)
            ax.annotate(f"{label} {state}\nactions: {', '.join(allowed)}",
                        (state[1], state[0]), xytext=(state[1] + 1.2, state[0] + 1.6),
                        fontsize=9, color=viz.TEXT, zorder=11,
                        bbox=dict(boxstyle="round", fc="white", ec="#c9c2b2"))

        text = (
            "STATE        robot cell (row, col)\n\n"
            f"INITIAL      S = {f.start}\n\n"
            "ACTIONS      Up, Down, Left, Right\n"
            "             (no diagonal moves)\n"
            "             only onto safe soil:\n"
            "             never potato, protected\n"
            "             or obstacle cells\n\n"
            "TRANSITION   (r, c) + Up    = (r-1, c)\n"
            "             (r, c) + Down  = (r+1, c)\n"
            "             (r, c) + Left  = (r, c-1)\n"
            "             (r, c) + Right = (r, c+1)\n\n"
            f"GOAL TEST    state == G {f.goal}\n\n"
            "PATH COST    sum of step costs\n"
            "             step = 1 + penalty(cell)\n"
            "             (penalty: see stage 4)"
        )
        self._text_panel([0.63, 0.1, 0.36, 0.78], text)
        ax.legend(handles=[Line2D([], [], color=viz.START_COLOUR, lw=2.2, label="allowed move"),
                           Line2D([], [], color=viz.GOAL_COLOUR, lw=2.2, label="blocked move")],
                  loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False,
                  fontsize=10)

    # ------------------------------------------------------------------
    # Stage 3 - search race
    # ------------------------------------------------------------------
    def stage_search_race(self):
        f = self.field
        names = list(self.results)
        boxes = [[0.02, 0.5, 0.31, 0.38], [0.345, 0.5, 0.31, 0.38], [0.67, 0.5, 0.31, 0.38],
                 [0.02, 0.09, 0.31, 0.38], [0.345, 0.09, 0.31, 0.38]]
        panels = []
        for name, rect in zip(names, boxes):
            ax = self._axes(rect)
            viz.draw_field(ax, f, grid=f.known_grid(), show_penalty=(name != "A* (distance only)"))
            overlay = np.zeros(f.grid.shape + (4,))
            image = ax.imshow(overlay, interpolation="nearest", zorder=1.6)
            line, = ax.plot([], [], color=viz.ALGO_COLOURS[name], linewidth=2.8, zorder=5,
                            solid_capstyle="round")
            title = ax.set_title(name, fontsize=11, fontweight="bold", loc="left",
                                 color=viz.TEXT)
            panels.append(dict(name=name, res=self.results[name], overlay=overlay,
                               image=image, line=line, title=title, done=False))

        board_ax = self._axes([0.68, 0.09, 0.3, 0.38])
        board_ax.axis("off")
        board = board_ax.text(0, 1, "", va="top", ha="left", transform=board_ax.transAxes,
                              linespacing=1.6, **MONO)
        explored_rgba = to_rgba(viz.EXPLORED_BLUE, 0.6)
        state = {"n": 0}

        def scoreboard():
            lines = ["NODES EXPANDED (live)", ""]
            for p in panels:
                res = p["res"]
                shown = min(state["n"], res.nodes_expanded)
                bar = "#" * int(18 * shown / 200)
                lines.append(f"{p['name']:<19}{shown:>4} {bar}")
            if not all(p["done"] for p in panels):
                lines += ["", "light blue = cells expanded",
                          "same speed for every algorithm:",
                          "one expansion per tick"]
            else:
                lines += ["", "PATH  moves / crop-safe cost"]
                lines += [f"{p['name']:<19}{p['res'].path_length:>3} / {p['res'].safety_cost:g}"
                          for p in panels]
            board.set_text("\n".join(lines))

        def tick():
            state["n"] += 1
            for p in panels:
                res = p["res"]
                if state["n"] <= len(res.explored):
                    p["overlay"][res.explored[state["n"] - 1]] = explored_rgba
                    p["image"].set_data(p["overlay"])
                    p["title"].set_text(f"{p['name']}   expanded {state['n']}")
                elif not p["done"]:
                    p["done"] = True
                    p["line"].set_data([c for _, c in res.path], [r for r, _ in res.path])
                    p["title"].set_text(f"{p['name']}   expanded {res.nodes_expanded}  "
                                        f"|  path {res.path_length} moves")
            scoreboard()
            return not all(p["done"] for p in panels)

        scoreboard()
        self._start_animation(tick, 25)

    # ------------------------------------------------------------------
    # Stage 4 - crop-safe planning
    # ------------------------------------------------------------------
    def stage_crop_safe(self):
        f = self.field
        safe_problem = PathProblem(f, crop_safe=True)
        compare = [("A* (distance only)", "Shortest route - every move costs 1"),
                   ("A* (crop-safe)", "Crop-safe route - penalty near protected crops")]
        rects = [[0.02, 0.24, 0.47, 0.6], [0.51, 0.24, 0.47, 0.6]]
        lines = []
        for (name, title), rect in zip(compare, rects):
            res = self.results[name]
            ax = self._axes(rect)
            viz.draw_field(ax, f, grid=f.known_grid())
            # write the extra cost into every penalised cell
            for (r, c), value in np.ndenumerate(safe_problem.penalty):
                if value > 0 and f.known_grid()[r, c] not in BLOCKED_TYPES:
                    ax.text(c, r, f"+{value:g}", ha="center", va="center", fontsize=6.5,
                            color="#9a3a12", zorder=4)
            line, = ax.plot([], [], color=viz.ALGO_COLOURS[name], linewidth=3.2, zorder=5,
                            solid_capstyle="round")
            near = sum(1 for cell in res.path if safe_problem.penalty[cell] > 0)
            ax.set_title(f"{title}\n{res.path_length} moves  |  cost {res.safety_cost:g}  |  "
                         f"{near} cells next to protected crops", fontsize=11, loc="left",
                         color=viz.TEXT)
            lines.append((line, res.path))

        penalty_text = ", ".join(f"distance {d}: +{p:g}" for d, p in PROXIMITY_PENALTY.items())
        self._text_panel(
            [0.02, 0.03, 0.58, 0.17],
            "Hard rule: potato, protected and obstacle cells are never allowed.\n"
            f"Soft rule: cells near a protected region cost extra ({penalty_text}).\n"
            "A* minimises total cost, so it takes a slightly longer route that keeps\n"
            "its distance from the protected crops (cost 38 instead of 47).",
            fontsize=10.5)

        state = {"i": 0}

        def tick():
            state["i"] += 1
            for line, path in lines:
                part = path[: state["i"]]
                line.set_data([c for _, c in part], [r for r, _ in part])
            return state["i"] < max(len(p) for _, p in lines)

        self._start_animation(tick, 60)

    # ------------------------------------------------------------------
    # Stage 5 - live drive with detection and replanning
    # ------------------------------------------------------------------
    def stage_live_drive(self):
        world = Field()                     # fresh true world we are allowed to change
        robot = OnlineRobot(world)
        self.live_field, self.live_robot = world, robot

        ax = self._axes([0.02, 0.1, 0.63, 0.78])
        viz.draw_field(ax, world, grid=robot.known.copy())
        markers = {}

        def mark_unknown(cell):
            r, c = cell
            box = ax.add_patch(Rectangle((c - 0.42, r - 0.42), 0.84, 0.84, fill=False,
                                         linestyle="--", edgecolor=viz.OBSTACLE_GREY,
                                         linewidth=1.4, zorder=3))
            q = ax.text(c, r, "?", ha="center", va="center", fontsize=9, zorder=3,
                        color=viz.OBSTACLE_GREY, fontweight="bold")
            markers[cell] = [box, q]

        def mark_detected(cell):
            for artist in markers.pop(cell, []):
                artist.remove()
            r, c = cell
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=viz.OBSTACLE_GREY,
                                   edgecolor=viz.GOAL_COLOUR, linewidth=2.2, zorder=3))
            ax.text(c, r, "!", ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold", zorder=4)

        for cell in world.hidden_obstacles:
            mark_unknown(cell)

        old_line, = ax.plot([], [], "--", color=viz.OLD_ROUTE, linewidth=2, zorder=4.5)
        route_line, = ax.plot([], [], "--", color=viz.PATH_BLUE, linewidth=2.4, zorder=5)
        driven_line, = ax.plot([], [], color=viz.TRAVELLED, linewidth=3.2, zorder=5.5,
                               solid_capstyle="round")
        size = 2 * SENSOR_RANGE + 1
        sensor = ax.add_patch(Rectangle((0, 0), size, size, fill=False, linestyle=":",
                                        edgecolor=viz.TRAVELLED, linewidth=1.5, zorder=7))
        robot_dot = ax.scatter([], [], s=200, marker="s", color=viz.TRAVELLED,
                               edgecolor="white", linewidth=1.5, zorder=8)
        banner = ax.text(0.5, 1.015, "", transform=ax.transAxes, ha="center", va="bottom",
                         fontsize=12, fontweight="bold", color=viz.GOAL_COLOUR)
        ax.legend(handles=[
            Patch(facecolor="none", edgecolor=viz.OBSTACLE_GREY, linestyle="--",
                  label="hidden obstacle (robot can't see)"),
            Patch(facecolor=viz.OBSTACLE_GREY, edgecolor=viz.GOAL_COLOUR, linewidth=2,
                  label="detected"),
            Line2D([], [], color=viz.PATH_BLUE, ls="--", lw=2.4, label="current A* plan"),
            Line2D([], [], color=viz.OLD_ROUTE, ls="--", lw=2, label="blocked plan"),
            Line2D([], [], color=viz.TRAVELLED, lw=3.2, label="driven"),
            Patch(facecolor="none", edgecolor=viz.TRAVELLED, linestyle=":", label="sensor range"),
        ], loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, frameon=False, fontsize=9.5)

        info_ax = self._axes([0.67, 0.1, 0.32, 0.78])
        info_ax.axis("off")
        info = info_ax.text(0, 1, "", va="top", ha="left", transform=info_ax.transAxes,
                            linespacing=1.5, **MONO)
        log = []
        state = {"hold": 0}

        def path_xy(path):
            path = path or []
            return [c for _, c in path], [r for r, _ in path]

        def redraw():
            r, c = robot.position
            robot_dot.set_offsets([[c, r]])
            sensor.set_xy((c - SENSOR_RANGE - 0.5, r - SENSOR_RANGE - 0.5))
            route_line.set_data(*path_xy(robot.route))
            driven_line.set_data(*path_xy(robot.travelled))
            if robot.reached:
                status = "DESTINATION REACHED"
            elif robot.stuck:
                status = "NO ROUTE LEFT - goal is blocked"
            elif self.paused:
                status = "paused"
            else:
                status = "driving"
            info.set_text("\n".join([
                "LOOP EVERY MOVE",
                " 1. sense 8 neighbouring cells",
                " 2. new obstacle? -> update map",
                " 3. on my route?  -> A* replan",
                " 4. move one step",
                "",
                f"status      {status}",
                f"moves       {robot.steps}",
                f"detected    {len(robot.detections)}",
                f"replans     {len(robot.replans)}",
                f"A* expanded {robot.nodes_expanded}",
                "",
                "EVENTS",
                *log[-9:],
                "",
                "CLICK a free cell to drop a",
                "surprise obstacle on the field!",
            ]))

        def tick():
            if state["hold"] > 0:          # pause briefly so the audience sees the replan
                state["hold"] -= 1
                if state["hold"] == 0:
                    banner.set_text("")
                return True
            new, event = robot.sense_and_replan()
            for cell in new:
                mark_detected(cell)
                if event is None or cell not in event.obstacles:
                    log.append(f"m{robot.steps:<3} saw {cell}: not on route")
            if event is not None:
                obs = ", ".join(map(str, event.obstacles))
                log.append(f"m{robot.steps:<3} {obs} BLOCKS route")
                old_line.set_data(*path_xy(event.old_route))
                if event.new_route:
                    log.append(f"     -> A* new route, {len(event.new_route) - 1} moves")
                banner.set_text(f"Obstacle detected at {obs}  ->  route blocked  ->  "
                                "A* replanning")
                state["hold"] = 22
                redraw()
                return True
            if robot.stuck or robot.reached:
                if robot.reached:
                    log.append(f"m{robot.steps:<3} goal reached!")
                    banner.set_color(viz.START_COLOUR)
                    banner.set_text(f"Destination reached in {robot.steps} moves with "
                                    f"{len(robot.replans)} replans")
                else:
                    banner.set_text("No route to the goal exists any more")
                redraw()
                return False
            robot.move()
            redraw()
            return True

        def click(event):
            if event.inaxes is not ax or robot.reached:
                return
            cell = (int(round(event.ydata)), int(round(event.xdata)))
            if (not world.is_free(cell) or cell in (robot.position, world.goal, world.start)):
                return
            world.grid[cell] = OBSTACLE         # real obstacle, unknown to the robot
            world.hidden_obstacles.append(cell)
            mark_unknown(cell)
            log.append(f"m{robot.steps:<3} you dropped {cell}")
            redraw()
            if self.timer is not None and not robot.stuck:
                self.timer.start()               # resume if it had finished
            self.fig.canvas.draw_idle()

        self.on_click = click
        redraw()
        self._start_animation(tick, 160)

    # ------------------------------------------------------------------
    # Stage 6 - results
    # ------------------------------------------------------------------
    def stage_results(self):
        results = list(self.results.values())
        online, oracle = run_online_comparison(Field())
        on_rows = online_rows(online, oracle)
        robot = self.live_robot
        if robot is not None and (robot.reached or robot.stuck):
            cost = PathProblem(self.live_field, grid=self.live_field.grid).path_cost(
                robot.travelled)
            on_rows.append(["Live demo run (stage 5)", "Yes" if robot.reached else "No",
                            str(robot.steps), f"{cost:g}", str(robot.nodes_expanded), "-",
                            str(len(robot.detections)), str(len(robot.replans))])

        best = [r.algorithm for r in results].index("A* (crop-safe)")
        viz.draw_table(self._axes([0.02, 0.6, 0.96, 0.25]), OFFLINE_HEADERS,
                       offline_rows(results), "Search algorithms on the known map",
                       fontsize=11, highlight_row=best)
        viz.draw_table(self._axes([0.02, 0.33, 0.96, 0.05 * (len(on_rows) + 1)]),
                       ONLINE_HEADERS, on_rows, "Online obstacle experiment", fontsize=11)
        self._text_panel(
            [0.02, 0.08, 0.96, 0.18],
            "* BFS: fewest moves but expands most of the field; ignores crop safety.\n"
            "* DFS: reaches the goal but with a very long, winding path (86 moves).\n"
            "* Greedy: fastest (follows the heuristic straight to the goal) but no "
            "optimality guarantee and ignores crop safety.\n"
            "* A* crop-safe: the only route with 0 cells next to protected crops and the "
            "lowest crop-safe cost.\n"
            "* Online: with hidden obstacles the robot still reaches the goal, replanning "
            "only when its route is blocked.",
            fontsize=10.5, family="sans-serif")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", type=int, default=1, choices=range(1, 7),
                        help="stage to start from (1-6)")
    args = parser.parse_args()

    plt.switch_backend("TkAgg")
    for key in ("keymap.back", "keymap.forward", "keymap.pan", "keymap.quit",
                "keymap.home", "keymap.save"):
        plt.rcParams[key] = []              # free the keys used by the demo
    Demo(args.stage)
    plt.show()


if __name__ == "__main__":
    main()
