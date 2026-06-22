import json
import os
import tempfile
import unittest

from data.prepare_v5_grpo_data import prepare_v5_data


class PrepareV5GrpoDataTest(unittest.TestCase):
    def test_builds_mixed_data_without_test_overlap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = os.path.join(temp_dir, "train.jsonl")
            solvable_test_path = os.path.join(temp_dir, "solvable_test.jsonl")
            unsolvable_test_path = os.path.join(temp_dir, "unsolvable_test.jsonl")
            output_dir = os.path.join(temp_dir, "out")

            with open(train_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"target_nums": [1, 1, 1, 8], "solvable": True}) + "\n")
            with open(solvable_test_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"target_nums": [3, 3, 8, 8], "solvable": True}) + "\n")
            with open(unsolvable_test_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"target_nums": [1, 1, 1, 1], "solvable": False}) + "\n")

            summary = prepare_v5_data(
                train_path=train_path,
                solvable_test_path=solvable_test_path,
                unsolvable_test_path=unsolvable_test_path,
                output_dir=output_dir,
                unsolvable_size=2,
                seed=1,
            )

            self.assertEqual(summary["solvable_train_puzzles"], 1)
            self.assertEqual(summary["unsolvable_train_puzzles"], 2)
            self.assertEqual(summary["grpo_rows"], 3)
            self.assertEqual(summary["solvable_test_overlap"], 0)
            self.assertEqual(summary["unsolvable_test_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
