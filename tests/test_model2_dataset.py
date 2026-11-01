"""Reproducibility and protocol checks for the approved Model 2 dataset."""

import copy
import json
from pathlib import Path
import unittest

import numpy as np

from opttheory.datasets.synthetic import generate_model1, generate_model2
from opttheory.models.logistic import predict


ROOT = Path(__file__).resolve().parents[1]


class Model2DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config" / "model2.json").read_text(encoding="utf-8"))

    def test_approved_truth_and_range(self):
        data = generate_model2(self.config)
        self.assertEqual(data.true_parameters, {"L": 10.0, "k": 1.5, "x0": 2.0})
        self.assertEqual(len(data.x), 120)
        self.assertEqual(float(data.x[0]), 0.0)
        self.assertEqual(float(data.x[-1]), 4.0)
        np.testing.assert_allclose(data.y_clean,
                                   predict(data.x, L=10.0, k=1.5, x0=2.0), rtol=0, atol=0)
        self.assertAlmostEqual(float(data.y_clean[0] + data.y_clean[-1]), 10.0)
        self.assertTrue(np.all(np.diff(data.y_clean) > 0))

    def test_noise_is_added_and_reproducible(self):
        first = generate_model2(self.config)
        second = generate_model2(self.config)
        for field in ("x", "y_clean", "noise", "y_observed", "split"):
            np.testing.assert_array_equal(getattr(first, field), getattr(second, field))
        np.testing.assert_array_equal(first.y_observed, first.y_clean + first.noise)

    def test_approved_split_sizes_and_no_overlap(self):
        data = generate_model2(self.config)
        self.assertEqual(data.split_counts(), {"train": 72, "validation": 24, "test": 24})
        self.assertEqual(set(data.split), {"train", "validation", "test"})
        self.assertEqual(len(set(np.nonzero(data.split == "train")[0])
                             | set(np.nonzero(data.split == "validation")[0])
                             | set(np.nonzero(data.split == "test")[0])), 120)

    def test_different_noise_seed_changes_noise_but_not_split(self):
        altered = copy.deepcopy(self.config)
        altered["noise"]["seed"] = 1101
        first = generate_model2(self.config)
        second = generate_model2(altered)
        np.testing.assert_array_equal(first.x, second.x)
        np.testing.assert_array_equal(first.y_clean, second.y_clean)
        np.testing.assert_array_equal(first.split, second.split)
        self.assertFalse(np.array_equal(first.noise, second.noise))

    def test_split_assignment_identical_to_model1(self):
        model1_config = json.loads((ROOT / "config" / "model1.json").read_text(encoding="utf-8"))
        np.testing.assert_array_equal(generate_model2(self.config).split,
                                      generate_model1(model1_config).split)

    def test_zero_noise_recovers_exact_curve(self):
        altered = copy.deepcopy(self.config)
        altered["noise"]["sigma"] = 0.0
        data = generate_model2(altered)
        np.testing.assert_array_equal(data.y_observed, data.y_clean)
        np.testing.assert_array_equal(data.noise, np.zeros(120))

    def test_invalid_ground_truth_rejected(self):
        for key, value in (("L", -2), ("k", 0), ("x0", 5), ("k", float("nan"))):
            with self.subTest(key=key, value=value):
                altered = copy.deepcopy(self.config)
                altered["true_parameters"][key] = value
                with self.assertRaises(ValueError):
                    generate_model2(altered)


if __name__ == "__main__":
    unittest.main()
