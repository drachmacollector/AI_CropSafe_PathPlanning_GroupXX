"""
Visualisation of the field, the search results and the replanning run.

All drawing goes through `draw_field`, so every figure uses the same colours:

    light soil      free cell the robot can use
    green + dots    potato row
    red + hatching  crop-protection region
    orange tint     "too close" zone around protected regions (extra cost)
    dark grey       obstacle
    dashed '?'      obstacle the robot does not know about yet
"""

import matplotlib

matplotlib.use("Agg")   # draw to files; main.py switches backend for --show

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from .crop_safety import proximity_penalty_map
from .field import FREE, OBSTACLE, POTATO, PROTECTED

# ----- colours ------------------------------------------------------------
SOIL = "#f4ecdc"
POTATO_GREEN = "#7cb36a"
PLANT_DOT = "#3f7a33"
PROTECTED_RED = "#e6a19a"
PROTECTED_EDGE = "#b8463c"
OBSTACLE_GREY = "#3c3c3c"
PENALTY_ORANGE = "#eb6834"
GRID_LINE = "#d8ccb4"
TEXT = "#2b2b2b"
START_COLOUR = "#1baf7a"
GOAL_COLOUR = "#e34948"
PATH_BLUE = "#2a78d6"
EXPLORED_BLUE = "#9cc3ee"
OLD_ROUTE = "#8a8a8a"
TRAVELLED = "#6250d6"

ALGO_COLOURS = {
    "BFS": "#2a78d6",
    "DFS": "#eb6834",
    "Greedy Best-First": "#c98500",
    "A* (distance only)": "#e87ba4",
    "A* (crop-safe)": "#6250d6",
}

_CELL_CMAP = ListedColormap([SOIL, POTATO_GREEN, PROTECTED_RED, OBSTACLE_GREY])


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def draw_field(ax, field, grid=None, show_penalty=True, unknown=(), detected=()):
    """
    Draw a field map on `ax`.
    grid      : which map to draw (true field by default, or the robot's map)
    unknown   : hidden obstacles to show as dashed '?' boxes
    detected  : obstacles found by the robot, drawn with a red outline
    """
    grid = field.grid if grid is None else grid
    ax.imshow(grid, cmap=_CELL_CMAP, vmin=FREE, vmax=OBSTACLE, interpolation="nearest")

    # orange "too close" tint, stronger nearer the protected region
    if show_penalty:
        penalty = proximity_penalty_map(grid)
        overlay = np.zeros(grid.shape + (4,))
        overlay[..., :3] = to_rgba(PENALTY_ORANGE)[:3]
        overlay[..., 3] = np.where(grid == FREE, penalty / 3.0 * 0.38, 0)
        ax.imshow(overlay, interpolation="nearest")

    # small plants on potato cells
    rows, cols = np.nonzero(grid == POTATO)
    ax.scatter(cols, rows, s=14, color=PLANT_DOT, zorder=2, linewidths=0)

    # hatching on protected cells
    for r, c in np.argwhere(grid == PROTECTED):
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="////",
                               edgecolor=PROTECTED_EDGE, linewidth=0, zorder=2))

    for r, c in unknown:
        ax.add_patch(Rectangle((c - 0.42, r - 0.42), 0.84, 0.84, fill=False,
                               linestyle="--", edgecolor=OBSTACLE_GREY, linewidth=1.2, zorder=3))
        ax.text(c, r, "?", ha="center", va="center", fontsize=8,
                color=OBSTACLE_GREY, fontweight="bold", zorder=3)

    for r, c in detected:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=OBSTACLE_GREY,
                               edgecolor=GOAL_COLOUR, linewidth=2, zorder=3))
        ax.text(c, r, "!", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)

    # thin grid lines between cells
    h, w = grid.shape
    ax.set_xticks(np.arange(-0.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, h, 1), minor=True)
    ax.grid(which="minor", color=GRID_LINE, linewidth=0.5)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    for spine in ax.spines.values():
        spine.set_edgecolor("#b9ad95")

    draw_start_goal(ax, field.start, field.goal)


def draw_start_goal(ax, start, goal):
    ax.scatter(start[1], start[0], s=150, marker="o", color=START_COLOUR,
               edgecolor="white", linewidth=1.5, zorder=6)
    ax.text(start[1], start[0], "S", ha="center", va="center", fontsize=7,
            color="white", fontweight="bold", zorder=7)
    ax.scatter(goal[1], goal[0], s=240, marker="*", color=GOAL_COLOUR,
               edgecolor="white", linewidth=1.2, zorder=6)


def draw_path(ax, path, colour=PATH_BLUE, style="-", width=2.6, alpha=1.0, zorder=5, label=None):
    if not path:
        return
    rows = [p[0] for p in path]
    cols = [p[1] for p in path]
    ax.plot(cols, rows, style, color=colour, linewidth=width, alpha=alpha,
            solid_capstyle="round", solid_joinstyle="round", zorder=zorder, label=label)


def shade_cells(ax, cells, colour, alpha=0.45):
    for r, c in cells:
        ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=colour,
                               edgecolor="none", alpha=alpha, zorder=1.5))


def field_legend_handles(include_hidden=False):
    handles = [
        Patch(facecolor=SOIL, edgecolor="#b9ad95", label="Safe soil"),
        Patch(facecolor=POTATO_GREEN, edgecolor=PLANT_DOT, label="Potato row"),
        Patch(facecolor=PROTECTED_RED, edgecolor=PROTECTED_EDGE, hatch="////",
              label="Crop-protection region"),
        Patch(facecolor=to_rgba(PENALTY_ORANGE, 0.35), edgecolor="none",
              label="Too close (extra cost)"),
        Patch(facecolor=OBSTACLE_GREY, label="Obstacle"),
        Line2D([], [], marker="o", color="w", markerfacecolor=START_COLOUR,
               markersize=10, label="Start"),
        Line2D([], [], marker="*", color="w", markerfacecolor=GOAL_COLOUR,
               markersize=14, label="Destination"),
    ]
    if include_hidden:
        handles.append(Patch(facecolor="none", edgecolor=OBSTACLE_GREY, linestyle="--",
                             label="Unknown obstacle"))
    return handles


def _style_title(ax, title, subtitle=None):
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, loc="left",
                 pad=18 if subtitle else 6)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=8.5,
                color="#5c5a55", va="bottom")


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_field_layout(field, out_path):
    """Task 1: the field with every cell type, start, goal and hidden obstacles."""
    fig, ax = plt.subplots(figsize=(11, 7.2))
    draw_field(ax, field, grid=field.known_grid(), unknown=field.hidden_obstacles)
    _style_title(ax, "Potato field environment",
                 f"{field.height} x {field.width} grid  |  dashed '?' = obstacles the robot "
                 "does not know about at the start")
    ax.legend(handles=field_legend_handles(include_hidden=True), loc="upper center",
              ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
    return _save(fig, out_path)


def plot_algorithm_paths(field, results, out_path):
    """Task 3/4: one panel per algorithm with explored cells and the final path."""
    n = len(results)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.2, rows * 4.6))
    axes = np.atleast_1d(axes).ravel()

    for ax, res in zip(axes, results):
        colour = ALGO_COLOURS.get(res.algorithm, PATH_BLUE)
        draw_field(ax, field, grid=field.known_grid())
        shade_cells(ax, res.explored, EXPLORED_BLUE, alpha=0.35)
        draw_path(ax, res.path, colour=colour)
        if res.reached:
            sub = (f"moves {res.path_length}  |  crop-safe cost {res.safety_cost:g}  |  "
                   f"expanded {res.nodes_expanded}  |  {res.time_ms:.2f} ms")
        else:
            sub = "no path found"
        _style_title(ax, res.algorithm, sub)

    for ax in axes[n:]:
        ax.axis("off")

    handles = field_legend_handles() + [
        Patch(facecolor=EXPLORED_BLUE, alpha=0.6, label="Expanded cells"),
        Line2D([], [], color=PATH_BLUE, linewidth=2.6, label="Planned path"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Paths found by each search algorithm (robot's known map)",
                 fontsize=14, fontweight="bold", color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=3)
    return _save(fig, out_path)


def plot_replanning(field, online, out_path):
    """Task 5: initial plan, every replan, and the route actually driven."""
    panels = 2 + online.num_replans
    cols = min(panels, 2)
    rows = int(np.ceil(panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6.6, rows * 4.9))
    axes = np.atleast_1d(axes).ravel()

    # panel 1: the plan made with the initial (incomplete) knowledge
    ax = axes[0]
    draw_field(ax, field, grid=field.known_grid(), unknown=field.hidden_obstacles)
    draw_path(ax, online.initial_route, colour=PATH_BLUE)
    _style_title(ax, "1. Initial A* plan",
                 f"{len(field.hidden_obstacles)} obstacles are unknown to the robot")

    # one panel per replanning event
    for i, event in enumerate(online.replans, start=2):
        ax = axes[i - 1]
        found_so_far = [c for step, c in online.detections if step <= event.step]
        grid = field.known_grid()
        for c in found_so_far:
            grid[c] = OBSTACLE
        still_unknown = [c for c in field.hidden_obstacles if c not in found_so_far]
        draw_field(ax, field, grid=grid, unknown=still_unknown, detected=found_so_far)
        driven = online.travelled[: event.step + 1]
        draw_path(ax, driven, colour=TRAVELLED, width=3)
        draw_path(ax, event.old_route, colour=OLD_ROUTE, style="--", width=1.8)
        draw_path(ax, event.new_route, colour=PATH_BLUE)
        r, c = event.position
        ax.scatter(c, r, s=170, marker="s", color=TRAVELLED, edgecolor="white", zorder=8)
        obs = ", ".join(str(o) for o in event.obstacles)
        _style_title(ax, f"{i}. Obstacle detected at {obs} - replanning",
                     f"after {event.step} moves, robot at {event.position}")

    # last panel: what the robot actually drove
    ax = axes[panels - 1]
    all_found = [c for _, c in online.detections]
    draw_field(ax, field, grid=field.grid, detected=all_found)
    draw_path(ax, online.initial_route, colour=OLD_ROUTE, style="--", width=1.8)
    draw_path(ax, online.travelled, colour=TRAVELLED, width=3)
    status = "destination reached" if online.reached else "destination NOT reached"
    _style_title(ax, f"{panels}. Route actually driven",
                 f"{online.path_length} moves  |  {online.num_replans} replans  |  {status}")

    for ax in axes[panels:]:
        ax.axis("off")

    handles = field_legend_handles(include_hidden=True) + [
        Patch(facecolor=OBSTACLE_GREY, edgecolor=GOAL_COLOUR, linewidth=2,
              label="Detected obstacle"),
        Line2D([], [], color=PATH_BLUE, linewidth=2.6, label="Current plan"),
        Line2D([], [], color=OLD_ROUTE, linestyle="--", linewidth=1.8, label="Blocked / old plan"),
        Line2D([], [], color=TRAVELLED, linewidth=3, label="Driven so far"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Online obstacle detection and A* replanning",
                 fontsize=14, fontweight="bold", color=TEXT, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=3)
    return _save(fig, out_path)


def animate_replanning(field, online, out_path, fps=4):
    """Task 5: a GIF of the robot driving, sensing and replanning step by step."""
    fig, ax = plt.subplots(figsize=(9, 6))

    def update(i):
        ax.clear()
        frame = online.frames[i]
        found = frame["known_obstacles"]
        grid = field.known_grid()
        for c in found:
            grid[c] = OBSTACLE
        unknown = [c for c in field.hidden_obstacles if c not in found]
        draw_field(ax, field, grid=grid, unknown=unknown, detected=found)
        draw_path(ax, online.travelled[: i + 1], colour=TRAVELLED, width=3)
        draw_path(ax, frame["route"], colour=PATH_BLUE, style="--", width=2)
        r, c = frame["position"]
        # sensor range around the robot
        ax.add_patch(Rectangle((c - 1.5, r - 1.5), 3, 3, fill=False,
                               edgecolor=TRAVELLED, linewidth=1.2, linestyle=":", zorder=7))
        ax.scatter(c, r, s=170, marker="s", color=TRAVELLED, edgecolor="white", zorder=8)
        replans = sum(1 for e in online.replans if e.step <= i)
        note = "  -  OBSTACLE DETECTED, REPLANNING" if frame["replanned"] else ""
        _style_title(ax, "Robot driving with obstacle detection",
                     f"move {i}  |  replans so far: {replans}{note}")

    # hold the last frame for a moment
    frames = list(range(len(online.frames))) + [len(online.frames) - 1] * fps * 2
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps)
    anim.save(out_path, writer=PillowWriter(fps=fps), dpi=80)
    plt.close(fig)


def _column_widths(headers, rows):
    """Column width follows the longest text in that column."""
    return [max(len(h), *(len(row[i]) for row in rows)) + 3 for i, h in enumerate(headers)]


def draw_table(ax, headers, rows, title=None, fontsize=10, highlight_row=None):
    """Draw a styled table that fills `ax`."""
    widths = _column_widths(headers, rows)
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=headers, cellLoc="center", bbox=[0, 0, 1, 1],
                     colWidths=[w / sum(widths) for w in widths])
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    for (r, _c), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d6cf")
        if r == 0:
            cell.set_facecolor("#2b2b2b")
            cell.set_text_props(color="white", fontweight="bold")
        elif highlight_row is not None and r == highlight_row + 1:
            cell.set_facecolor("#e4e0f7")
            cell.set_text_props(fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f6f4ef")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", color=TEXT, loc="left")
    return table


def plot_comparison_table(headers, rows, out_path, title):
    """Task 6: render the comparison table as an image."""
    widths = _column_widths(headers, rows)
    fig, ax = plt.subplots(figsize=(0.11 * sum(widths), 0.4 * (len(rows) + 1)))
    draw_table(ax, headers, rows, title)
    return _save(fig, out_path)
