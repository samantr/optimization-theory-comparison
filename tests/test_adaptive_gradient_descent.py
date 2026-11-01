"""Small deterministic examples separate from the approved two-way study."""

import unittest

import numpy as np

from opttheory.optimizers.adaptive_gradient_descent import adaptive_gradient_descent


def quadratic(theta, _x, _y):
    return float((theta[0] - 3.0) ** 2)


def quadratic_gradient(theta, _x, _y):
    return np.array([2.0 * (theta[0] - 3.0)])


def solve(*, strategy, alpha=1.0, steps=10):
    return adaptive_gradient_descent(
        [0.0], [0.0], [0.0], strategy=strategy,
        initial_learning_rate=alpha, max_iterations=steps,
        gradient_tolerance=1e-10, objective_fn=quadratic,
        gradient_fn=quadratic_gradient,
    )


class AdaptiveGradientDescentTests(unittest.TestCase):
    def test_gradient_reversal_halves_after_crossing_even_if_mse_drops(self):
        result = solve(strategy="gradient_reversal", alpha=0.75, steps=2)
        np.testing.assert_allclose(result.theta, [3.375])
        self.assertEqual(result.step_sizes, (0.75, 0.375))
        self.assertAlmostEqual(result.loss_history[1], 2.25)
        self.assertEqual(result.line_search_backtracks, 0)
        self.assertEqual(result.objective_evaluations, 3)
        self.assertEqual(result.gradient_evaluations, 3)

    def test_gradient_reversal_accepts_worse_step_before_halving(self):
        result = solve(strategy="gradient_reversal", alpha=1.2, steps=1)
        np.testing.assert_allclose(result.theta, [7.2])
        self.assertAlmostEqual(result.loss_history[1], 17.64)
        self.assertEqual(result.step_sizes, (1.2,))

    def test_backtracking_rejects_worse_step_then_retries_from_start(self):
        result = solve(strategy="backtracking", alpha=1.2, steps=1)
        np.testing.assert_allclose(result.theta, [3.6])
        self.assertAlmostEqual(result.loss_history[-1], 0.36)
        self.assertEqual(result.step_sizes, (0.6,))
        self.assertEqual(result.line_search_backtracks, 1)
        self.assertEqual(result.objective_evaluations, 3)
        self.assertEqual(result.gradient_evaluations, 2)

    def test_backtracking_never_accepts_increased_training_mse(self):
        result = solve(strategy="backtracking", alpha=1.0, steps=10)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertEqual(result.step_sizes, (0.5,))
        self.assertAlmostEqual(result.loss_history[-1], 0.0)

    def test_recovery_raises_rate_halfway_after_acceptance(self):
        # At theta=0, alpha=1.2 fails; alpha=.6 succeeds at theta=3.6.
        # The next proposed rate is (.6+1.2)/2=.9; it too succeeds.
        result = solve(strategy="recovering_backtracking", alpha=1.2, steps=2)
        np.testing.assert_allclose(result.theta, [2.52])
        np.testing.assert_allclose(result.step_sizes, (0.6, 0.9))
        self.assertEqual(result.line_search_backtracks, 1)
        self.assertTrue(all(b < a for a, b in zip(result.loss_history, result.loss_history[1:])))

    def test_recovery_shrinks_again_after_another_bad_trial(self):
        result = solve(strategy="recovering_backtracking", alpha=1.2, steps=3)
        # Third proposed rate 1.05 is worse; halve to .525, then accept.
        np.testing.assert_allclose(result.step_sizes, (0.6, 0.9, 0.525))
        self.assertEqual(result.line_search_backtracks, 2)
        self.assertLess(result.loss_history[-1], result.loss_history[-2])

    def test_failed_search_is_recorded_without_moving(self):
        result = adaptive_gradient_descent(
            [0.0], [0], [0], strategy="backtracking", initial_learning_rate=1,
            min_learning_rate=0.25, max_backtracks_per_iteration=5,
            objective_fn=lambda theta, x, y: 2.0,
            gradient_fn=lambda theta, x, y: np.array([1.0]),
        )
        self.assertEqual(result.reason, "line_search_failed")
        self.assertEqual(result.iterations, 0)
        self.assertEqual(result.line_search_backtracks, 3)
        self.assertEqual(result.objective_evaluations, 4)

    def test_stationary_start_requires_no_updates(self):
        for strategy in ("gradient_reversal", "backtracking", "recovering_backtracking"):
            with self.subTest(strategy=strategy):
                result = adaptive_gradient_descent(
                    [3.0], [0], [0], strategy=strategy,
                    objective_fn=quadratic, gradient_fn=quadratic_gradient,
                )
                self.assertTrue(result.converged)
                self.assertEqual(result.iterations, 0)
                self.assertEqual(result.step_sizes, ())

    def test_invalid_parameters(self):
        for overrides in ({"strategy": "none"},
                          {"strategy": "backtracking", "initial_learning_rate": 0},
                          {"strategy": "backtracking", "min_learning_rate": -1},
                          {"strategy": "backtracking", "max_iterations": 0},
                          {"strategy": "backtracking", "max_backtracks_per_iteration": 0}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                adaptive_gradient_descent([0], [0], [0], **overrides)


if __name__ == "__main__":
    unittest.main()
