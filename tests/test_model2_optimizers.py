"""Integration checks for Model 2 functions in our existing solvers.

Only one-step test updates are made; this is not the controlled experiment.
"""

import json
from pathlib import Path
import unittest

import numpy as np

from opttheory.datasets.synthetic import generate_model2
from opttheory.objectives import logistic_mse
from opttheory.optimizers import gradient_descent, newton, safeguarded_newton


ROOT = Path(__file__).resolve().parents[1]


class Model2OptimizerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads((ROOT / "config" / "model2.json").read_text(encoding="utf-8"))
        dataset = generate_model2(config)
        train = dataset.split == "train"
        cls.x = dataset.x[train]
        cls.y = dataset.y_observed[train]
        cls.initial = np.array([9.0, 1.0, 2.0])  # approved Model 2 S1

    def test_gd_uses_model2_gradient_for_a_single_update(self):
        alpha = 0.001
        expected = self.initial - alpha * logistic_mse.gradient(self.initial, self.x, self.y)
        result = gradient_descent(
            self.initial, self.x, self.y, learning_rate=alpha,
            max_iterations=1, gradient_tolerance=1e-12,
            objective_fn=logistic_mse.objective,
            gradient_fn=logistic_mse.gradient,
        )
        np.testing.assert_allclose(result.theta, expected, rtol=0, atol=1e-14)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.hessian_evaluations, 0)
        self.assertAlmostEqual(result.loss_history[0],
                               logistic_mse.objective(self.initial, self.x, self.y))

    def test_exact_newton_step_uses_model2_full_hessian(self):
        expected = self.initial - np.linalg.solve(
            logistic_mse.hessian(self.initial, self.x, self.y),
            logistic_mse.gradient(self.initial, self.x, self.y),
        )
        result = newton(
            self.initial, self.x, self.y, max_iterations=1,
            gradient_tolerance=1e-12, objective_fn=logistic_mse.objective,
            gradient_fn=logistic_mse.gradient, hessian_fn=logistic_mse.hessian,
        )
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.hessian_evaluations, 1)
        np.testing.assert_allclose(result.theta, expected, rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(result.loss_history[1],
                               logistic_mse.objective(expected, self.x, self.y))

    def test_safeguarded_variant_accepts_a_model2_step(self):
        result = safeguarded_newton(
            self.initial, self.x, self.y, max_iterations=1,
            gradient_tolerance=1e-12, objective_fn=logistic_mse.objective,
            gradient_fn=logistic_mse.gradient, hessian_fn=logistic_mse.hessian,
        )
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.hessian_evaluations, 1)
        self.assertLessEqual(result.loss_history[-1], result.loss_history[0])

    def test_negative_k_start_is_not_silently_excluded(self):
        # Only verify that the approved difficult S4 start can reach the
        # optimizer interface; the study must log any subsequent failures.
        theta0 = np.array([10.0, -1.0, 2.0])
        result = gradient_descent(
            theta0, self.x, self.y, learning_rate=0.001,
            max_iterations=1, gradient_tolerance=1e-12,
            objective_fn=logistic_mse.objective,
            gradient_fn=logistic_mse.gradient,
        )
        self.assertEqual(result.iterations, 1)
        self.assertAlmostEqual(result.loss_history[0],
                               logistic_mse.objective(theta0, self.x, self.y))
        np.testing.assert_array_equal(theta0, [10.0, -1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
