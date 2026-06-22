import unittest

from data.prepare_compact_sft_data import arithmetic_trace, build_compact_row


class PrepareCompactSftDataTest(unittest.TestCase):
    def test_arithmetic_trace_contains_only_arithmetic_steps(self):
        trace = arithmetic_trace("8*(1+1+1)")

        self.assertEqual(trace, "1+1=2; 2+1=3; 8*3=24")

    def test_fraction_trace_is_exact(self):
        trace = arithmetic_trace("8/(3-8/3)")

        self.assertEqual(trace, "8/3=8/3; 3-(8/3)=1/3; 8/(1/3)=24")

    def test_build_compact_row_uses_conversational_completion(self):
        row = {
            "target_nums": [1, 1, 1, 8],
            "target_value": 24,
            "solvable": True,
            "prompt": [{"role": "user", "content": "solve"}],
            "answer": "8*(1+1+1)",
            "rejection_source": "deepseek_api",
        }

        compact = build_compact_row(row)

        self.assertEqual(compact["completion"][0]["role"], "assistant")
        self.assertEqual(
            compact["completion"][0]["content"],
            "<think>1+1=2; 2+1=3; 8*3=24</think>\n"
            "<answer>8*(1+1+1)</answer>",
        )

    def test_invalid_source_answer_is_rejected(self):
        row = {
            "target_nums": [1, 1, 1, 8],
            "target_value": 24,
            "solvable": True,
            "prompt": [{"role": "user", "content": "solve"}],
            "answer": "1+1+1+8",
        }

        with self.assertRaises(ValueError):
            build_compact_row(row)


if __name__ == "__main__":
    unittest.main()
