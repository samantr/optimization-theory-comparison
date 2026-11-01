"""Test the MSE objective independently of either optimization algorithm."""

import json
import math
from pathlib import Path
import unittest

import numpy as np

from opttheory.datasets.synthetic import generate_model1
from opttheory.objectives.mse import gradient, objective


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


class GradientTests(unittest.TestCase):
    def test_hand_calculated_two_observation_gradient(self):
        # Current lesson: x=[0,1], theta=[3,ln(0.5),1], y=[5.1,2.7].
        # Residuals [-1.1,-0.2] yield gradient [-1.2,-0.3,-1.3].
        actual = gradient([3.0, math.log(0.5), 1.0], [0.0, 1.0], [5.1, 2.7])
        np.testing.assert_allclose(actual, [-1.2, -0.3, -1.3], atol=1e-12)

    def test_gradient_zero_when_predictions_match_observations(self):
        theta = [3.0, math.log(0.5), 1.0]
        x = np.array([0.0, 1.0, 2.0])
        y_exact = theta[0] * np.exp(theta[1] * x) + theta[2]
        np.testing.assert_allclose(gradient(theta, x, y_exact), np.zeros(3), atol=1e-14)

    def test_b_component_zero_when_all_x_are_zero(self):
        actual = gradient([3.0, -0.8, 1.0], [0.0, 0.0], [5.1, 4.7])
        self.assertEqual(actual[1], 0.0)

    def test_central_finite_differences_independently_check_gradient(self):
        x = np.array([0.0, 0.3, 1.0, 1.8, 3.7])
        y = np.array([4.9, 4.5, 2.8, 2.3, 1.1])
        step = 1e-6
        for theta in (np.array([2.4, -0.4, 0.7]),
                      np.array([4.0, math.log(0.5), 1.0])):
            approx = np.empty(3)
            for j in range(3):
                positive = theta.copy()
                negative = theta.copy()
                positive[j] += step
                negative[j] -= step
                approx[j] = (objective(positive, x, y)
                             - objective(negative, x, y)) / (2.0 * step)
            np.testing.assert_allclose(gradient(theta, x, y), approx,
                                       rtol=1e-6, atol=1e-7)

    def test_gradient_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            gradient([1, 2], [0], [3])
        with self.assertRaises(ValueError):
            gradient([1, 2, 3], [0, 1], [3])


if __name__ == "__main__":
    unittest.main()

