import unittest

from data.prepare_data import can_make_target, parse_solved_rate, select_tot_splits


class PrepareDataTest(unittest.TestCase):
    def test_parse_solved_rate_percent(self):
        self.assertAlmostEqual(parse_solved_rate("99.20%"), 0.992)
        self.assertAlmostEqual(parse_solved_rate({"Solved rate": "82.40%"}), 0.824)
        self.assertAlmostEqual(parse_solved_rate({"solved_rate": 0.988}), 0.988)

    def test_select_tot_hard_split(self):
        samples = [
            {
                "target_nums": [1, 1, 1, index + 1],
                "source_index": index,
                "solved_rate": 1.0 - index / 2000,
            }
            for index in range(1005)
        ]

        hard_samples, low_solved_rate, selected_tests = select_tot_splits(samples, low_solved_rate_size=10)

        self.assertEqual(len(hard_samples), 100)
        self.assertEqual(hard_samples[0]["source_index"], 900)
        self.assertEqual(hard_samples[-1]["source_index"], 999)
        self.assertEqual(len(low_solved_rate), 10)
        self.assertEqual(low_solved_rate[0]["source_index"], 1004)
        self.assertEqual(len(selected_tests), 105)

    def test_can_make_target(self):
        self.assertTrue(can_make_target([3, 3, 8, 8]))
        self.assertFalse(can_make_target([1, 1, 1, 1]))


if __name__ == "__main__":
    unittest.main()
