"""Independent checks for the first-order optimizer (no library solver)."""

import math
import unittest

import numpy as np

from opttheory.objectives.mse import gradient, objective
from opttheory.optimizers import gradient_descent


class GradientDescentTests(unittest.TestCase):
    def test_one_step_matches_manual_update(self):
        # From our lesson: gradient [-1.2, -.3, -1.3], alpha = 0.1.
        theta = np.array([3.0, math.log(0.5), 1.0])
        x = [0.0, 1.0]
        y = [5.1, 2.7]
        alpha = 0.1
        result = gradient_descent(theta, x, y, learning_rate=alpha,
                                  max_iterations=1)
        expected = theta - alpha * gradient(theta, x, y)
        np.testing.assert_allclose(result.theta, expected, atol=1e-14)
        np.testing.assert_allclose(result.theta,
                                   [3.12, math.log(0.5) + 0.03, 1.13], atol=1e-14)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.reason, "max_iterations")
        self.assertLess(result.loss_history[-1], result.loss_history[0])
        self.assertAlmostEqual(result.loss_history[-1], objective(result.theta, x, y))
        self.assertEqual((result.objective_evaluations, result.gradient_evaluations), (2, 2))
        np.testing.assert_array_equal(theta, [3.0, math.log(0.5), 1.0])

    def test_can_converge_on_singular_hessian_example(self):
        # x=0 makes the Hessian singular, but gradient descent can still fit.
        result = gradient_descent([0.0, 0.0, 0.0], [0.0], [5.0],
                                  learning_rate=0.25, max_iterations=10)
        np.testing.assert_allclose(result.theta, [2.5, 0.0, 2.5], atol=1e-14)
        self.assertEqual(result.iterations, 1)
        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gradient_tolerance")
        self.assertAlmostEqual(result.loss_history[-1], 0.0)
        self.assertEqual(len(result.loss_history), result.iterations + 1)
        self.assertEqual(len(result.gradient_norm_history), result.iterations + 1)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_stationary_start_requires_no_iterations(self):
        result = gradient_descent([2.0, -0.5, 3.0], [0.0], [5.0])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.loss_history, (0.0,))

    def test_max_iterations_and_history(self):
        result = gradient_descent([0.0, 0.0, 0.0], [0.0], [5.0],
                                  learning_rate=0.01, max_iterations=2,
                                  gradient_tolerance=1e-12)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "max_iterations")
        self.assertEqual(result.iterations, 2)
        self.assertEqual(len(result.loss_history), 3)
        self.assertEqual(result.objective_evaluations, 3)

    def test_numerical_failure_is_reported(self):
        result = gradient_descent([0.0, 0.0, 0.0], [0.0], [5.0],
                                  learning_rate=1e308, max_iterations=5)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "numerical_failure")
        self.assertEqual(result.iterations, 0)
        np.testing.assert_array_equal(result.theta, [0.0, 0.0, 0.0])

    def test_invalid_configuration_rejected(self):
        for alpha in (0.0, -0.1, float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                gradient_descent([0, 0, 0], [0], [5], learning_rate=alpha)
        for budget in (0, -1, 1.5, True):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                gradient_descent([0, 0, 0], [0], [5], max_iterations=budget)
        with self.assertRaises(ValueError):
            gradient_descent([0, 0, 0], [0], [5], gradient_tolerance=0)


if __name__ == "__main__":
    unittest.main()
