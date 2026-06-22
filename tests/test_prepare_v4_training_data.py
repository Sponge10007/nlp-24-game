import unittest

from data.prepare_data import puzzle_key
from data.prepare_v4_training_data import (
    build_solvable_sft_rows,
    build_unsolvable_sft_rows,
    find_solutions,
)
from src.game24 import CORRECT, judge_answer


class PrepareV4TrainingDataTest(unittest.TestCase):
    def test_find_solutions_returns_distinct_valid_expressions(self):
        solutions = find_solutions([3, 3, 8, 8], max_solutions=3)

        self.assertGreaterEqual(len(solutions), 1)
        self.assertEqual(len(solutions), len(set(solutions)))
        for answer in solutions:
            self.assertEqual(judge_answer(answer, [3, 3, 8, 8]).code, CORRECT)

    def test_solvable_rows_use_short_arithmetic_completions(self):
        rows, missing = build_solvable_sft_rows(
            [{"target_nums": [1, 1, 1, 8], "target_value": 24, "solvable": True}],
            max_solutions=3,
        )

        self.assertEqual(missing, 0)
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("<think>", rows[0]["completion"][0]["content"])
        self.assertIn("<answer>", rows[0]["completion"][0]["content"])
        self.assertNotIn("尝试", rows[0]["completion"][0]["content"])

    def test_unsolvable_rows_use_exact_unsolvable_answer(self):
        rows = build_unsolvable_sft_rows(
            [{"target_nums": [1, 1, 1, 1], "target_value": 24, "solvable": False}]
        )

        self.assertEqual(rows[0]["answer"], "UNSOLVABLE")
        self.assertIn("<answer>UNSOLVABLE</answer>", rows[0]["completion"][0]["content"])
        self.assertFalse(rows[0]["solvable"])

    def test_puzzle_key_is_order_independent_for_split_checks(self):
        self.assertEqual(puzzle_key([1, 2, 3, 4]), puzzle_key([4, 3, 2, 1]))


if __name__ == "__main__":
    unittest.main()
