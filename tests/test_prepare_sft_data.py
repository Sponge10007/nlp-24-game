import os
import unittest

from data.prepare_data import puzzle_key
from data.prepare_sft_data import build_sft_samples, load_jsonl, normalize_solution
from src.game24 import CORRECT, judge_answer


class PrepareSftDataTest(unittest.TestCase):
    def test_normalize_solution(self):
        self.assertEqual(normalize_solution("(1+1+1)×8"), "(1+1+1)*8")
        self.assertEqual(normalize_solution("（8÷2）−1"), "(8/2)-1")

    def test_build_sft_samples_filters_to_train_keys_and_validates_answers(self):
        rows = [
            {"numbers": [1, 1, 1, 8], "solutions": ["(1+1+1)×8", "not a solution"]},
            {"numbers": [9, 9, 9, 9], "solutions": ["9+9+9+9"]},
        ]
        train_keys = {puzzle_key([1, 1, 1, 8])}

        samples, stats = build_sft_samples(rows, train_keys)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["answer"], "(1+1+1)*8")
        self.assertNotIn("×", samples[0]["answer"])
        self.assertEqual(judge_answer(samples[0]["answer"], samples[0]["target_nums"]).code, CORRECT)
        self.assertEqual(stats["train_rows_seen"], 1)
        self.assertEqual(stats["invalid_solutions"], 1)

    @unittest.skipUnless(os.path.exists("data/sft_train.jsonl"), "data/sft_train.jsonl has not been generated")
    def test_generated_sft_data_is_strict_and_non_overlapping(self):
        samples = load_jsonl("data/sft_train.jsonl")
        train_keys = {puzzle_key(row["target_nums"]) for row in load_jsonl("data/train.jsonl")}
        test_keys = set()
        for path in ("data/test_hard_900_1000.jsonl", "data/test_low_solved_rate.jsonl", "data/test_all_nonoverlap.jsonl"):
            test_keys.update(puzzle_key(row["target_nums"]) for row in load_jsonl(path))

        self.assertTrue(samples)
        for sample in samples:
            key = puzzle_key(sample["target_nums"])
            self.assertIn(key, train_keys)
            self.assertNotIn(key, test_keys)
            self.assertNotIn("×", sample["answer"])
            self.assertNotIn("×", sample["completion"])
            self.assertEqual(judge_answer(sample["answer"], sample["target_nums"]).code, CORRECT)


if __name__ == "__main__":
    unittest.main()
