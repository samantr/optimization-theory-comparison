"""Checks for the Model 2 logistic prediction function only.

The synthetic dataset parameters and optimizer derivatives are not fixed yet.
"""

import unittest

import numpy as np

from opttheory.models.logistic import predict


class Model2PredictionTests(unittest.TestCase):
    def test_midpoint_is_half_of_capacity(self):
        self.assertAlmostEqual(float(predict(2.0, L=10.0, k=1.5, x0=2.0)), 5.0)

    def test_increasing_curve_and_approach_to_limit(self):
        x = np.array([-20.0, 0.0, 2.0, 4.0, 20.0])
        y = predict(x, L=10.0, k=1.5, x0=2.0)
        self.assertTrue(np.all(np.diff(y) > 0))
        self.assertTrue(np.all((y > 0.0) & (y < 10.0)))
        self.assertAlmostEqual(float(y[-1]), 10.0, places=9)

    def test_stable_for_extreme_inputs(self):
        y = predict(np.array([-1000.0, 0.0, 1000.0]), L=10.0, k=1.0, x0=0.0)
        np.testing.assert_allclose(y, [0.0, 5.0, 10.0], atol=1e-12, rtol=0)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_rate_parameter_changes_steepness_not_midpoint(self):
        x = np.array([1.0, 2.0, 3.0])
        slow = predict(x, L=10.0, k=0.5, x0=2.0)
        fast = predict(x, L=10.0, k=2.0, x0=2.0)
        self.assertLess(float(fast[0]), float(slow[0]))
        self.assertEqual(float(fast[1]), float(slow[1]))
        self.assertGreater(float(fast[2]), float(slow[2]))


if __name__ == "__main__":
    unittest.main()
