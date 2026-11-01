"""Verify exact Newton's update, failure logging, and stopping semantics."""

import math
import unittest

import numpy as np

from opttheory.objectives.mse import gradient, hessian, objective
from opttheory.optimizers import newton


class NewtonTests(unittest.TestCase):
    def test_hand_checkable_three_parameter_step(self):
        # x=[0,1,2], theta=[1,0,1], y=[1,2,3]:
        # grad=[0,-4/3,0], Hessian as below, solve(H,grad)=[1,0,-1].
        theta0 = np.array([1.0, 0.0, 1.0])
        x = [0.0, 1.0, 2.0]
        y = [1.0, 2.0, 3.0]
        expected_h = [[2.0, 2.0 / 3.0, 2.0],
                      [2.0 / 3.0, 2.0 / 3.0, 2.0],
                      [2.0, 2.0, 2.0]]
        np.testing.assert_allclose(hessian(theta0, x, y), expected_h, atol=1e-14)
        np.testing.assert_allclose(gradient(theta0, x, y), [0.0, -4.0 / 3.0, 0.0], atol=1e-14)
        result = newton(theta0, x, y, max_iterations=1)
        np.testing.assert_allclose(result.theta, [0.0, 0.0, 2.0], atol=1e-13)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.hessian_evaluations, 1)
        self.assertEqual((result.objective_evaluations, result.gradient_evaluations), (2, 2))
        np.testing.assert_array_equal(theta0, [1.0, 0.0, 1.0])
        # A stationary point is not necessarily an improved fit.
        self.assertAlmostEqual(result.loss_history[0], result.loss_history[-1])

    def test_update_uses_actual_full_hessian(self):
        theta0 = np.array([3.0, -0.5, 1.0])
        x = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0])
        y = 4.0 * np.exp(math.log(0.5) * x) + 1.0
        expected = theta0 - np.linalg.solve(hessian(theta0, x, y),
                                            gradient(theta0, x, y))
        result = newton(theta0, x, y, max_iterations=1,
                        gradient_tolerance=1e-12)
        np.testing.assert_allclose(result.theta, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(len(result.loss_history), 2)
        self.assertAlmostEqual(result.loss_history[-1], objective(result.theta, x, y))

    def test_singular_hessian_is_reported_without_a_step(self):
        result = newton([0.0, -0.5, 0.0], [0.0], [5.0])
        self.assertFalse(result.converged)
        self.assertIn(result.reason, {"singular_hessian", "ill_conditioned_hessian"})
        self.assertEqual(result.iterations, 0)
        self.assertEqual(len(result.loss_history), 1)
        np.testing.assert_array_equal(result.theta, [0.0, -0.5, 0.0])

    def test_stationary_start_skips_hessian(self):
        result = newton([2.0, -0.5, 3.0], [0.0], [5.0])
        self.assertTrue(result.converged)
        self.assertEqual(result.reason, "gradient_tolerance")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.hessian_evaluations, 0)

    def test_budget_reached(self):
        theta0 = [3.0, -0.5, 1.0]
        x = [0.0, 1.0, 2.0, 3.0]
        y = [5.0, 3.0, 2.0, 1.5]
        result = newton(theta0, x, y, max_iterations=1,
                        gradient_tolerance=1e-15)
        self.assertEqual(result.iterations, 1)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "max_iterations")
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)

    def test_invalid_configuration_rejected(self):
        for budget in (0, -1, True, 2.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                newton([1, 0, 1], [0], [5], max_iterations=budget)
        with self.assertRaises(ValueError):
            newton([1, 0, 1], [0], [5], gradient_tolerance=0.0)
        with self.assertRaises(ValueError):
            newton([1, 0, 1], [0], [5], condition_limit=0.1)


if __name__ == "__main__":
    unittest.main()
