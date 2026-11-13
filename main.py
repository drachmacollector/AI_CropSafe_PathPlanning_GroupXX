"""
AI-Based Crop-Safe Path Planning for an Autonomous Potato-Field Robot.

Run:  python main.py            (saves all figures and tables to outputs/)
      python main.py --show     (also opens the figures in windows)
      python main.py --no-gif   (skip the animation, a little faster)
"""

import argparse
from pathlib import Path

from cropsafe import visualize as viz
from cropsafe.evaluation import (OFFLINE_HEADERS, ONLINE_HEADERS, format_markdown_table,
                                 format_text_table, offline_rows, online_rows,
                                 run_offline_comparison, run_online_comparison)
from cropsafe.field import Field

OUT = Path(__file__).parent / "outputs"


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--show", action="store_true", help="open the figures in windows")
    parser.add_argument("--no-gif", action="store_true", help="skip the replanning animation")
    args = parser.parse_args()

    if args.show:
        viz.plt.switch_backend("TkAgg")
    OUT.mkdir(exist_ok=True)

    # ---- Task 1: field --------------------------------------------------
    field = Field()
    section("Task 1 - Field environment")
    print(f"Grid size        : {field.height} rows x {field.width} columns")
    print(f"Start            : {field.start}")
    print(f"Destination      : {field.goal}")
    print(f"Hidden obstacles : {field.hidden_obstacles}")
    viz.plot_field_layout(field, OUT / "1_field_layout.png")

    # ---- Tasks 2-4: offline search --------------------------------------
    section("Tasks 3 & 4 - Search algorithms on the robot's known map")
    results = run_offline_comparison(field)
    for r in results:
        print(f"{r.algorithm:<20} path: {r.path}" if r.reached else f"{r.algorithm}: no path")
    viz.plot_algorithm_paths(field, results, OUT / "2_algorithm_paths.png")

    # ---- Task 5: online obstacle detection and replanning ---------------
    section("Task 5 - Obstacle detection and replanning")
    online, oracle = run_online_comparison(field)
    for step, cell in online.detections:
        blocked = any(cell in e.obstacles for e in online.replans)
        action = "route blocked -> A* replans" if blocked else "not on route, keep going"
        print(f"move {step:>3}: detected obstacle at {cell}  ({action})")
    print(f"Destination reached: {online.reached} after {online.path_length} moves")
    viz.plot_replanning(field, online, OUT / "3_replanning.png")
    if not args.no_gif:
        viz.animate_replanning(field, online, OUT / "4_replanning.gif")

    # ---- Task 6: performance tables -------------------------------------
    section("Task 6 - Performance comparison")
    off_rows = offline_rows(results)
    on_rows = online_rows(online, oracle)
    print("Offline search (known map, crop-safe cost used to score every path):")
    print(format_text_table(OFFLINE_HEADERS, off_rows))
    print("\nOnline experiment (hidden obstacles):")
    print(format_text_table(ONLINE_HEADERS, on_rows))

    viz.plot_comparison_table(OFFLINE_HEADERS, off_rows, OUT / "5_offline_table.png",
                              "Search algorithm comparison")
    viz.plot_comparison_table(ONLINE_HEADERS, on_rows, OUT / "6_online_table.png",
                              "Online obstacle experiment")
    (OUT / "results.md").write_text(
        "## Search algorithm comparison\n\n"
        + format_markdown_table(OFFLINE_HEADERS, off_rows)
        + "\n\n## Online obstacle experiment\n\n"
        + format_markdown_table(ONLINE_HEADERS, on_rows) + "\n",
        encoding="utf-8")

    print(f"\nFigures and tables saved in: {OUT}")
    if args.show:
        viz.plt.show()


if __name__ == "__main__":
    main()
