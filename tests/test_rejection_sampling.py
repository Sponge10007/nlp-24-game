import unittest
from argparse import Namespace

from generate_rejection_sft import build_rejection_sft_rows
from src.rejection_sampling import (
    build_completion_from_deepseek_response,
    build_sft_row,
    select_rejection_sample,
    validate_rejection_candidate,
)


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
        self.assertEqual(row["rejection_source"], "deepseek_api")

    def test_build_completion_from_deepseek_reasoning_and_content(self):
        completion = build_completion_from_deepseek_response(
            "try division",
            "8/(3-(8/3))",
        )

        self.assertEqual(completion, "<think>try division</think>\n<answer>8/(3-(8/3))</answer>")
        sample = validate_rejection_candidate(completion, [3, 3, 8, 8])
        self.assertTrue(sample.accepted)

    def test_build_completion_keeps_tagged_content(self):
        completion = build_completion_from_deepseek_response(
            "hidden reasoning",
            "<think>visible</think>\n<answer>8/(3-(8/3))</answer>",
        )

        self.assertEqual(completion, "<think>visible</think>\n<answer>8/(3-(8/3))</answer>")

    def test_deepseek_wrong_expression_is_rejected(self):
        completion = build_completion_from_deepseek_response(
            "bad arithmetic",
            "1+2+3+4",
        )

        sample = validate_rejection_candidate(completion, [3, 3, 8, 8])
        self.assertFalse(sample.accepted)

    def test_deepseek_number_mismatch_is_rejected(self):
        completion = build_completion_from_deepseek_response(
            "uses extra number",
            "8/(3-(8/4))",
        )

        sample = validate_rejection_candidate(completion, [3, 3, 8, 8])
        self.assertFalse(sample.accepted)
        self.assertEqual(sample.code, "number_mismatch")

    def test_parallel_row_builder_preserves_input_order(self):
        class FakeClient:
            pass

        cases = [
            {"target_nums": [1, 2, 3, 4], "solvable": True},
            {"target_nums": [3, 3, 8, 8], "solvable": True},
        ]
        args = Namespace(
            attempts_per_case=1,
            require_r1_format=True,
            sleep_seconds=0,
            max_workers=2,
        )

        import generate_rejection_sft

        original_call = generate_rejection_sft.call_deepseek
        try:
            def fake_call(_client, target_nums, _target_value, _args):
                if target_nums == [1, 2, 3, 4]:
                    return "<think>sum</think>\n<answer>1*2*3*4</answer>"
                return "<think>division</think>\n<answer>8/(3-(8/3))</answer>"

            generate_rejection_sft.call_deepseek = fake_call
            rows, counts = build_rejection_sft_rows(FakeClient(), cases, args)
        finally:
            generate_rejection_sft.call_deepseek = original_call

        self.assertEqual([row["target_nums"] for row in rows], [[1, 2, 3, 4], [3, 3, 8, 8]])
        self.assertEqual(counts["accepted"], 2)


if __name__ == "__main__":
    unittest.main()
