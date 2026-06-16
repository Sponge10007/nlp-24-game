import unittest

from data.prepare_data import (
    can_make_target,
    generate_random_unsolvable_samples,
    normalize_countdown_row,
    parse_solved_rate,
    select_tot_splits,
    split_countdown_samples,
    split_countdown_tinyzero,
)


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
        self.assertTrue(can_make_target([2, 3, 7], target_value=17))
        self.assertFalse(can_make_target([1, 1, 1], target_value=99))

    def test_normalize_countdown_row(self):
        sample = normalize_countdown_row({"nums": [2, 3, 7], "target": 17}, source_index=5)

        self.assertEqual(sample["target_nums"], [2, 3, 7])
        self.assertEqual(sample["target_value"], 17)
        self.assertTrue(sample["solvable"])
        self.assertEqual(sample["source_index"], 5)

    def test_split_countdown_samples_is_disjoint_by_case(self):
        samples = [
            {"target_nums": [1, 2, 3], "target_value": 6, "solvable": True, "source": "test"},
            {"target_nums": [3, 2, 1], "target_value": 6, "solvable": True, "source": "test"},
            {"target_nums": [2, 3, 7], "target_value": 17, "solvable": True, "source": "test"},
            {"target_nums": [4, 5, 6], "target_value": 30, "solvable": True, "source": "test"},
        ]

        train_samples, test_samples = split_countdown_samples(samples, test_size=1, seed=7)
        train_keys = {(tuple(sorted(s["target_nums"])), s["target_value"]) for s in train_samples}
        test_keys = {(tuple(sorted(s["target_nums"])), s["target_value"]) for s in test_samples}

        self.assertEqual(len(train_samples) + len(test_samples), 3)
        self.assertFalse(train_keys & test_keys)

    def test_tinyzero_split_is_sequential(self):
        samples = [
            {"target_nums": [1, 2, index], "target_value": index + 10, "solvable": True, "source": "test"}
            for index in range(10)
        ]

        train_samples, test_samples = split_countdown_tinyzero(samples, train_size=6, test_size=3)

        self.assertEqual(train_samples, samples[:6])
        self.assertEqual(test_samples, samples[6:9])

    def test_default_countdown_data_has_no_unsolvable_train_samples(self):
        samples = [
            {"target_nums": [1, 2, index], "target_value": index + 10, "solvable": True, "source": "test"}
            for index in range(30)
        ]
        train_samples, _ = split_countdown_tinyzero(samples, train_size=20, test_size=5)

        self.assertTrue(all(sample["solvable"] for sample in train_samples))

    def test_generate_random_unsolvable_samples(self):
        samples = generate_random_unsolvable_samples(5, seed=123, min_target=50, max_target=60)

        self.assertEqual(len(samples), 5)
        for sample in samples:
            self.assertFalse(sample["solvable"])
            self.assertFalse(can_make_target(sample["target_nums"], sample["target_value"]))


if __name__ == "__main__":
    unittest.main()
