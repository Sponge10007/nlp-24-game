import unittest

from src.game24 import (
    CORRECT,
    DIVISION_BY_ZERO,
    FABRICATED_UNSOLVABLE,
    FALSE_UNSOLVABLE_CLAIM,
    ILLEGAL_CHARACTER,
    MISSING_ANSWER,
    NUMBER_MISMATCH,
    SYNTAX_ERROR,
    UNSOLVABLE_CLAIM,
    WRONG_VALUE,
    extract_answer,
    has_r1_format,
    judge_answer,
)


class Game24JudgeTest(unittest.TestCase):
    def test_extract_answer_and_format(self):
        text = "<think>try</think>\n<answer>(8/(3-8/3))</answer>"
        self.assertEqual(extract_answer(text), "(8/(3-8/3))")
        self.assertTrue(has_r1_format(text))

    def test_correct_fraction_expression(self):
        judgment = judge_answer("8/(3-8/3)", [3, 3, 8, 8])
        self.assertTrue(judgment.ok)
        self.assertEqual(judgment.code, CORRECT)

    def test_missing_answer(self):
        self.assertEqual(judge_answer("", [1, 2, 3, 4]).code, MISSING_ANSWER)

    def test_number_mismatch(self):
        judgment = judge_answer("(1+2+3)*4", [1, 2, 3, 5])
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, NUMBER_MISMATCH)

    def test_illegal_character(self):
        judgment = judge_answer("(1+2+3)*4=24", [1, 2, 3, 4])
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, ILLEGAL_CHARACTER)

    def test_syntax_error(self):
        judgment = judge_answer("(1+2+3*", [1, 2, 3])
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, SYNTAX_ERROR)

    def test_division_by_zero(self):
        judgment = judge_answer("1/(2-2)+3+4", [1, 2, 2, 3])
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, NUMBER_MISMATCH)

        judgment = judge_answer("1/(2-2)+3", [1, 2, 2, 3])
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, DIVISION_BY_ZERO)

    def test_wrong_value(self):
        judgment = judge_answer("1+2+3+4", [1, 2, 3, 4])
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, WRONG_VALUE)

    def test_unsolvable_claim(self):
        self.assertEqual(judge_answer("UNSOLVABLE", [1, 1, 1, 1], solvable=False).code, UNSOLVABLE_CLAIM)
        self.assertEqual(judge_answer("UNSOLVABLE", [1, 2, 3, 4], solvable=True).code, FALSE_UNSOLVABLE_CLAIM)

    def test_fabricated_unsolvable(self):
        judgment = judge_answer("1+1+1+1", [1, 1, 1, 1], solvable=False)
        self.assertFalse(judgment.ok)
        self.assertEqual(judgment.code, FABRICATED_UNSOLVABLE)


if __name__ == "__main__":
    unittest.main()
