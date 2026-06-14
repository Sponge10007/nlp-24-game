import unittest

from src.rejection_sampling import build_sft_row, select_rejection_sample, validate_rejection_candidate


class RejectionSamplingTest(unittest.TestCase):
    def test_accepts_correct_r1_completion(self):
        completion = "<think>try division</think>\n<answer>8/(3-(8/3))</answer>"

        sample = validate_rejection_candidate(completion, [3, 3, 8, 8])

        self.assertTrue(sample.accepted)
        self.assertEqual(sample.answer, "8/(3-(8/3))")
        self.assertEqual(sample.code, "correct")

    def test_rejects_correct_answer_without_required_r1_format(self):
        completion = "<answer>8/(3-(8/3))</answer>"

        sample = validate_rejection_candidate(completion, [3, 3, 8, 8], require_r1_format=True)

        self.assertFalse(sample.accepted)
        self.assertEqual(sample.code, "correct")

    def test_selects_first_accepted_attempt(self):
        completions = [
            "<think>bad</think>\n<answer>1+2+3+4</answer>",
            "<think>ok</think>\n<answer>8/(3-(8/3))</answer>",
        ]

        sample = select_rejection_sample(completions, [3, 3, 8, 8])

        self.assertIsNotNone(sample)
        self.assertEqual(sample.attempt_index, 2)

    def test_build_sft_row_records_rejection_metadata(self):
        completion = "<think>ok</think>\n<answer>8/(3-(8/3))</answer>"
        sample = validate_rejection_candidate(completion, [3, 3, 8, 8])
        row = build_sft_row({"target_nums": [3, 3, 8, 8], "solvable": True}, sample)

        self.assertEqual(row["completion"], completion)
        self.assertEqual(row["solution"], "8/(3-(8/3))")
        self.assertEqual(row["rejection_source"], "model_rejection")


if __name__ == "__main__":
    unittest.main()
