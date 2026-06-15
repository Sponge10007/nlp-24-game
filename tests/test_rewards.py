import unittest

from src.game24 import ILLEGAL_CHARACTER, NUMBER_MISMATCH, WRONG_VALUE, judge_answer
from src.rewards import completion_issue_counts, correctness_reward, protocol_issue_counts, reward_for_judgment


class RewardProtocolTest(unittest.TestCase):
    def test_equal_sign_protocol_issue(self):
        answer = "3*6+7-13=24"
        judgment = judge_answer(answer, [3, 6, 7, 13])
        issues = protocol_issue_counts(answer)

        self.assertEqual(judgment.code, ILLEGAL_CHARACTER)
        self.assertEqual(issues["equal_sign_count"], 1)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), -1.2)

    def test_unicode_operator_protocol_issue(self):
        answer = "(2 × 2 + 2) × 6"
        judgment = judge_answer(answer, [2, 2, 5, 6])
        issues = protocol_issue_counts(answer)

        self.assertEqual(judgment.code, ILLEGAL_CHARACTER)
        self.assertEqual(issues["unicode_operator_count"], 1)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), -1.2)

    def test_fullwidth_paren_protocol_issue(self):
        answer = "（2+2）"
        judgment = judge_answer(answer, [2, 2])
        issues = protocol_issue_counts(answer)

        self.assertEqual(judgment.code, ILLEGAL_CHARACTER)
        self.assertEqual(issues["fullwidth_paren_count"], 1)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), -1.2)

    def test_multiple_answer_protocol_issue(self):
        completion = "<think>x</think><answer>1+2</answer><answer>3+4</answer>"
        issues = completion_issue_counts(completion)

        self.assertEqual(issues["multiple_answer_count"], 1)
        self.assertEqual(correctness_reward([completion], [[1, 2, 3, 4]])[0], -1.2)

    def test_number_mismatch_still_number_mismatch(self):
        answer = "4*6/(1*6)+1"
        judgment = judge_answer(answer, [1, 6, 6, 12])

        self.assertEqual(judgment.code, NUMBER_MISMATCH)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), -0.8)

    def test_copied_prompt_example_is_tracked_and_penalized(self):
        answer = "8/(3-8/3)"
        judgment = judge_answer(answer, [1, 6, 6, 12])
        issues = protocol_issue_counts(answer)

        self.assertEqual(judgment.code, NUMBER_MISMATCH)
        self.assertEqual(issues["copied_prompt_example_count"], 1)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), -0.9)

    def test_bare_target_is_tracked_and_penalized(self):
        answer = "24"
        judgment = judge_answer(answer, [2, 2, 5, 8])
        issues = protocol_issue_counts(answer, target_value=24)

        self.assertEqual(judgment.code, NUMBER_MISMATCH)
        self.assertEqual(issues["bare_target_count"], 1)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer, 24), -0.9)

    def test_legal_wrong_value_gets_small_positive_reward(self):
        answer = "(5*5-11)/2"
        judgment = judge_answer(answer, [2, 5, 5, 11])

        self.assertEqual(judgment.code, WRONG_VALUE)
        self.assertEqual(judgment.value, 7.0)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), 0.1)


if __name__ == "__main__":
    unittest.main()
