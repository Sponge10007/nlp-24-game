import unittest

from src.rewards import correctness_reward, wrong_value_distance_reward


class RewardTest(unittest.TestCase):
    def test_wrong_value_distance_reward_is_smoother_for_closer_values(self):
        close_reward = wrong_value_distance_reward(23, 24)
        far_reward = wrong_value_distance_reward(4, 24)

        self.assertLess(close_reward, 0)
        self.assertGreater(close_reward, far_reward)

    def test_correct_answer_reward_stays_positive(self):
        rewards = correctness_reward(
            ["<think>ok</think>\n<answer>8/(3-(8/3))</answer>"],
            [[3, 3, 8, 8]],
        )

        self.assertEqual(rewards, [2.0])

    def test_wrong_value_reward_uses_distance_penalty(self):
        rewards = correctness_reward(
            [
                "<think>close</think>\n<answer>8+8+3+3</answer>",
                "<think>far</think>\n<answer>(3+3)*(8-8)</answer>",
            ],
            [[3, 3, 8, 8], [3, 3, 8, 8]],
        )

        self.assertGreater(rewards[0], rewards[1])
        self.assertLess(rewards[0], 0)


if __name__ == "__main__":
    unittest.main()
