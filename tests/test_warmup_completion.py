import unittest

from src.game24 import CORRECT, UNSOLVABLE_CLAIM, extract_answer, judge_answer
from src.solver24 import find_expression
from src.warmup_completion import build_warmup_completion


class Solver24Test(unittest.TestCase):
    def test_find_expression_returns_correct_solution(self):
        solution = find_expression([3, 3, 8, 8])

        self.assertIsNotNone(solution)
        judgment = judge_answer(solution or "", [3, 3, 8, 8])
        self.assertTrue(judgment.ok)
        self.assertEqual(judgment.code, CORRECT)

    def test_find_expression_returns_none_for_unsolvable(self):
        self.assertIsNone(find_expression([1, 1, 1, 1]))


class WarmupCompletionTest(unittest.TestCase):
    def test_build_solvable_completion(self):
        warmup = build_warmup_completion([3, 3, 8, 8])

        self.assertTrue(warmup.solvable)
        self.assertIsNotNone(warmup.solution)
        self.assertEqual(extract_answer(warmup.completion), warmup.solution)
        judgment = judge_answer(warmup.solution or "", [3, 3, 8, 8])
        self.assertTrue(judgment.ok)
        self.assertEqual(judgment.code, CORRECT)

    def test_build_unsolvable_completion(self):
        warmup = build_warmup_completion([1, 1, 1, 1], solvable=False)

        self.assertFalse(warmup.solvable)
        self.assertIsNone(warmup.solution)
        self.assertEqual(extract_answer(warmup.completion), "UNSOLVABLE")
        judgment = judge_answer(extract_answer(warmup.completion), [1, 1, 1, 1], solvable=False)
        self.assertTrue(judgment.ok)
        self.assertEqual(judgment.code, UNSOLVABLE_CLAIM)

    def test_solvable_marked_sample_must_have_solution(self):
        with self.assertRaises(ValueError):
            build_warmup_completion([1, 1, 1, 1], solvable=True)

    def test_unsolvable_marked_sample_must_not_have_solution(self):
        with self.assertRaises(ValueError):
            build_warmup_completion([3, 3, 8, 8], solvable=False)


if __name__ == "__main__":
    unittest.main()
