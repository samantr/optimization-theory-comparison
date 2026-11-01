"""Safeguarded-Newton tests; pure full-step Newton remains an independent baseline."""

import math
import unittest

import numpy as np

from opttheory.optimizers import newton, safeguarded_newton


class SafeguardedNewtonTests(unittest.TestCase):
    def test_singular_hessian_uses_explicit_gradient_fallback(self):
        # At x=0, pure Newton cannot invert H. Backtracked -gradient fits y=5.
        original = newton([0.0, 0.0, 0.0], [0.0], [5.0])
        self.assertFalse(original.converged)
        self.assertIn(original.reason, {"singular_hessian", "ill_conditioned_hessian"})

        protected = safeguarded_newton([0.0, 0.0, 0.0], [0.0], [5.0],
                                       max_iterations=10)
        self.assertTrue(protected.converged)
        self.assertEqual(protected.iterations, 1)
        self.assertEqual(protected.gradient_fallback_steps, 1)
        self.assertEqual(protected.line_search_backtracks, 2)
        np.testing.assert_allclose(protected.theta, [2.5, 0.0, 2.5])
        self.assertEqual(protected.loss_history, (25.0, 0.0))
        self.assertEqual(protected.step_sizes, (0.25,))

    def test_indefinite_hessian_triggers_fallback(self):
        result = safeguarded_newton([1.0, 0.0, 1.0],
                                    [0.0, 1.0, 2.0], [1.0, 2.0, 3.0],
                                    max_iterations=1)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.gradient_fallback_steps, 1)
        self.assertLess(result.loss_history[1], result.loss_history[0])

    def test_exact_hessian_steps_with_line_search_fit_clean_data(self):
        x = np.linspace(0.0, 4.0, 16)
        y = 4.0 * np.exp(math.log(0.5) * x) + 1.0
        result = safeguarded_newton([3.0, -0.5, 1.0], x, y,
                                    max_iterations=100, gradient_tolerance=1e-7)
        self.assertTrue(result.converged)
        self.assertLess(result.loss_history[-1], 1e-8)
        self.assertTrue(all(after <= before + 1e-15 for before, after
                            in zip(result.loss_history, result.loss_history[1:])))
        self.assertEqual(len(result.loss_history), result.iterations + 1)
        self.assertEqual(len(result.step_sizes), result.iterations)
        self.assertEqual(result.hessian_evaluations, result.iterations)
        self.assertTrue(all(0 < alpha <= 1 for alpha in result.step_sizes))

    def test_rejected_step_does_not_modify_candidate(self):
        result = safeguarded_newton([0.0, 0.0, 0.0], [0.0], [5.0],
                                    max_iterations=10, max_backtracks=0)
        self.assertFalse(result.converged)
        self.assertEqual(result.reason, "line_search_failed")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.loss_history, (25.0,))
        np.testing.assert_array_equal(result.theta, [0.0, 0.0, 0.0])

    def test_initial_stationary_point_skips_hessian(self):
        result = safeguarded_newton([2.0, -0.5, 3.0], [0.0], [5.0])
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.hessian_evaluations, 0)

    def test_invalid_configuration_rejected(self):
        overrides = (
            {"max_iterations": 0},
            {"gradient_tolerance": 0},
            {"condition_limit": 0.5},
            {"armijo_constant": 1},
            {"armijo_constant": 0},
            {"backtrack_factor": 1},
            {"backtrack_factor": 0},
            {"max_backtracks": -1},
            {"max_backtracks": True},
        )
        for arguments in overrides:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                safeguarded_newton([0, 0, 0], [0], [5], **arguments)


if __name__ == "__main__":
    unittest.main()
