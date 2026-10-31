"""Small scientific checks using the Python standard-library test runner."""

import copy
import json
import math
from pathlib import Path
import unittest

import numpy as np

from opttheory.datasets.synthetic import generate_model1
from opttheory.models.exponential import predict


ROOT = Path(__file__).resolve().parents[1]


class Model1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / "config" / "model1.json").read_text(encoding="utf-8"))

    def test_known_curve_points(self):
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        prediction = predict(x, a=4.0, b=math.log(0.5), c=1.0)
        np.testing.assert_allclose(prediction, [5.0, 3.0, 2.0, 1.5, 1.25],
                                   atol=1e-12, rtol=0)

    def test_120_observations_and_endpoints(self):
        data = generate_model1(self.config)
        self.assertEqual(len(data.x), 120)
        self.assertEqual(float(data.x[0]), 0.0)
        self.assertEqual(float(data.x[-1]), 4.0)
        self.assertAlmostEqual(float(data.y_clean[0]), 5.0)
        self.assertAlmostEqual(float(data.y_clean[-1]), 1.25)

    def test_noise_is_added_exactly(self):
        data = generate_model1(self.config)
        np.testing.assert_array_equal(data.y_observed, data.y_clean + data.noise)

    def test_repeated_seed_produces_same_dataset(self):
        one = generate_model1(self.config)
        two = generate_model1(self.config)
        for name in ("x", "y_clean", "noise", "y_observed", "split"):
            np.testing.assert_array_equal(getattr(one, name), getattr(two, name))

    def test_noise_seed_does_not_change_split(self):
        changed = copy.deepcopy(self.config)
        changed["noise"]["seed"] += 1
        one = generate_model1(self.config)
        two = generate_model1(changed)
        self.assertFalse(np.array_equal(one.noise, two.noise))
        np.testing.assert_array_equal(one.split, two.split)

    def test_split_counts_and_labels(self):
        data = generate_model1(self.config)
        self.assertEqual(data.split_counts(), {"train": 72, "validation": 24, "test": 24})
        self.assertEqual(set(data.split), {"train", "validation", "test"})

    def test_zero_noise_matches_clean_curve(self):
        clean = copy.deepcopy(self.config)
        clean["noise"]["sigma"] = 0.0
        data = generate_model1(clean)
        np.testing.assert_array_equal(data.noise, np.zeros(120))
        np.testing.assert_array_equal(data.y_observed, data.y_clean)


if __name__ == "__main__":
    unittest.main()

