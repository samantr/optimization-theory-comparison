"""Test the MSE objective independently of either optimization algorithm."""

import json
import math
from pathlib import Path
import unittest

import numpy as np

from opttheory.datasets.synthetic import generate_model1
from opttheory.objectives.mse import objective


ROOT = Path(__file__).resolve().parents[1]


class ObjectiveTests(unittest.TestCase):
    def test_single_observation_from_lesson(self):
        # a=3, b=-1, c=1 and x=0 give prediction 4, not 3/e + 1.
        result = objective([3, -1, 1], np.array([0.0]), np.array([5.1]))
        self.assertAlmostEqual(result, 1.21, places=12)

    def test_two_signed_residuals(self):
        # Prediction is [4,4]; residuals are [-0.2,+0.1].
        result = objective([3, 0, 1], np.array([0.0, 1.0]),
                           np.array([4.2, 3.9]))
        self.assertAlmostEqual(result, 0.025, places=12)

    def test_true_parameters_have_nonzero_noisy_training_loss(self):
        config = json.loads((ROOT / "config" / "model1.json").read_text(encoding="utf-8"))
        data = generate_model1(config)
        mask = data.split == "train"
        t = [4.0, math.log(0.5), 1.0]
        value = objective(t, data.x[mask], data.y_observed[mask])
        self.assertAlmostEqual(value, 0.0057756361, places=9)

    def test_candidate_parameters_change_loss(self):
        x = np.array([0.0, 1.0])
        observed = np.array([5.1, 3.0])
        self.assertNotEqual(objective([4, math.log(0.5), 1], x, observed),
                            objective([3, -1, 1], x, observed))

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):
            objective([1, 2], [0, 1], [1, 2])
        with self.assertRaises(ValueError):
            objective([1, 2, 3], [0, 1], [1])


if __name__ == "__main__":
    unittest.main()

