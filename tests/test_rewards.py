import unittest

from src.game24 import ILLEGAL_CHARACTER, NUMBER_MISMATCH, WRONG_VALUE, judge_answer
from src.rewards import protocol_issue_counts, reward_for_judgment


class RewardProtocolTest(unittest.TestCase):
    def test_equal_sign_protocol_issue(self):
        answer = "3*6+7-13=24"
        judgment = judge_answer(answer, [3, 6, 7, 13])
        issues = protocol_issue_counts(answer)

        self.assertEqual(judgment.code, ILLEGAL_CHARACTER)
        self.assertEqual(issues["equal_sign_count"], 1)
        self.assertLess(reward_for_judgment(judgment.code, judgment.value, answer), -0.5)

    def test_unicode_operator_protocol_issue(self):
        answer = "(2 × 2 + 2) × 6"
        judgment = judge_answer(answer, [2, 2, 5, 6])
        issues = protocol_issue_counts(answer)

        self.assertEqual(judgment.code, ILLEGAL_CHARACTER)
        self.assertEqual(issues["unicode_operator_count"], 1)
        self.assertLess(reward_for_judgment(judgment.code, judgment.value, answer), -0.5)

    def test_number_mismatch_still_number_mismatch(self):
        answer = "4*6/(1*6)+1"
        judgment = judge_answer(answer, [1, 6, 6, 12])

        self.assertEqual(judgment.code, NUMBER_MISMATCH)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), -0.4)

    def test_legal_wrong_value_gets_small_positive_reward(self):
        answer = "(5*5-11)/2"
        judgment = judge_answer(answer, [2, 5, 5, 11])

        self.assertEqual(judgment.code, WRONG_VALUE)
        self.assertEqual(judgment.value, 7.0)
        self.assertEqual(reward_for_judgment(judgment.code, judgment.value, answer), 0.1)


if __name__ == "__main__":
    unittest.main()
