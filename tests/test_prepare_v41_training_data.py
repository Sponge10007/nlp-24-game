import unittest

from data.prepare_data import puzzle_key
from data.prepare_v41_training_data import (
    build_solvable_sft_rows,
    choose_canonical_solution,
    teacher_answers_by_key,
)


class PrepareV41TrainingDataTest(unittest.TestCase):
    def test_canonical_solution_prefers_fewer_parentheses(self):
        answer = choose_canonical_solution(
            ["((4*5)+(10-6))", "(4*5)-(6-10)", "10-(6-(4*5))"]
        )

        self.assertEqual(answer, "(4*5)-(6-10)")

    def test_teacher_answers_are_filtered_to_train_and_validated(self):
        train_keys = {puzzle_key([1, 1, 1, 8])}
        rows = [
            {"target_nums": [1, 1, 1, 8], "answer": "8*(1+1+1)"},
            {"target_nums": [3, 3, 8, 8], "answer": "8/(3-8/3)"},
            {"target_nums": [1, 1, 1, 8], "answer": "1+1+1+8"},
        ]

        answers = teacher_answers_by_key(rows, train_keys)

        self.assertEqual(answers, {puzzle_key([1, 1, 1, 8]): "8*(1+1+1)"})

    def test_one_sft_row_per_solvable_prompt(self):
        train_rows = [
            {"target_nums": [1, 1, 1, 8], "target_value": 24},
            {"target_nums": [4, 5, 6, 10], "target_value": 24},
        ]
        teacher_answers = {puzzle_key([1, 1, 1, 8]): "8*(1+1+1)"}

        rows, stats = build_solvable_sft_rows(
            train_rows,
            teacher_answers,
            search_candidates=8,
        )

        self.assertEqual(len(rows), 2)
        self.assertEqual(
            len({puzzle_key(row["target_nums"]) for row in rows}),
            2,
        )
        self.assertEqual(stats["teacher_solution_rows"], 1)
        self.assertEqual(stats["deterministic_solution_rows"], 1)
        self.assertEqual(stats["missing_solution_rows"], 0)


if __name__ == "__main__":
    unittest.main()
