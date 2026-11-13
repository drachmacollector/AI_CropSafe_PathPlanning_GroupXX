"""Simple checks for the planner. Run with:  python -m unittest discover tests"""

import unittest

from cropsafe.field import BLOCKED_TYPES, Field
from cropsafe.informed import a_star_search, greedy_best_first_search, manhattan
from cropsafe.problem import PathProblem
from cropsafe.replanning import navigate_with_replanning
from cropsafe.uninformed import breadth_first_search, depth_first_search

ALL_SEARCHES = [breadth_first_search, depth_first_search,
                greedy_best_first_search, a_star_search]


def is_valid_path(field, grid, path):
    """Starts at S, ends at G, single 4-connected steps, never on a blocked cell."""
    if path[0] != field.start or path[-1] != field.goal:
        return False
    for a, b in zip(path, path[1:]):
        if manhattan(a, b) != 1:
            return False
    return all(grid[cell] not in BLOCKED_TYPES for cell in path)


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.field = Field()

    def test_every_algorithm_finds_a_valid_path(self):
        problem = PathProblem(self.field)
        for search in ALL_SEARCHES:
            result = search(problem)
            self.assertTrue(result.reached, search.__name__)
            self.assertTrue(is_valid_path(self.field, problem.grid, result.path),
                            search.__name__)

    def test_bfs_and_astar_agree_on_shortest_length(self):
        problem = PathProblem(self.field, crop_safe=False)
        self.assertEqual(breadth_first_search(problem).path_length,
                         a_star_search(problem).path_length)

    def test_crop_safe_astar_is_cheapest(self):
        problem = PathProblem(self.field, crop_safe=True)
        best = a_star_search(problem).path_cost
        for search in ALL_SEARCHES:
            self.assertGreaterEqual(problem.path_cost(search(problem).path), best)

    def test_crop_safe_route_avoids_penalty_zone(self):
        problem = PathProblem(self.field, crop_safe=True)
        path = a_star_search(problem).path
        self.assertTrue(all(problem.penalty[cell] == 0 for cell in path))

    def test_no_path_when_goal_is_walled_in(self):
        layout = ["S.#..",
                  "..#.G",
                  "..#.."]
        result = a_star_search(PathProblem(Field(layout)))
        self.assertFalse(result.reached)


class TestReplanning(unittest.TestCase):
    def test_robot_detects_hidden_obstacles_and_reaches_goal(self):
        field = Field()
        online = navigate_with_replanning(field)
        self.assertTrue(online.reached)
        self.assertGreaterEqual(online.num_replans, 1)
        # the driven route must be valid on the TRUE map
        self.assertTrue(is_valid_path(field, field.grid, online.travelled))

    def test_no_replan_without_hidden_obstacles(self):
        layout = ["S...G",
                  ".....",
                  "....."]
        online = navigate_with_replanning(Field(layout))
        self.assertTrue(online.reached)
        self.assertEqual(online.num_replans, 0)
        self.assertEqual(online.path_length, 4)


if __name__ == "__main__":
    unittest.main()
