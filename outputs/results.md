## Search algorithm comparison

| Algorithm | Reached | Path length | Crop-safe cost | Near-crop cells | Nodes expanded | Time (ms) |
|---|---|---|---|---|---|---|
| BFS | Yes | 24 | 47 | 9 | 180 | 1.142 |
| DFS | Yes | 86 | 95 | 9 | 112 | 0.708 |
| Greedy Best-First | Yes | 24 | 47 | 9 | 25 | 0.153 |
| A* (distance only) | Yes | 24 | 47 | 9 | 25 | 0.149 |
| A* (crop-safe) | Yes | 38 | 38 | 0 | 196 | 1.374 |

## Online obstacle experiment

| Strategy | Reached | Path length | Crop-safe cost | Nodes expanded | Time (ms) | Obstacles detected | Replans |
|---|---|---|---|---|---|---|---|
| A* + online replanning | Yes | 50 | 55 | 326 | 7.582 | 4 | 2 |
| A* with full map (reference) | Yes | 38 | 38 | 149 | 1.047 | - | 0 |
