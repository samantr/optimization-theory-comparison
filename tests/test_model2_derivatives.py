"""Independent finite-difference checks for Model 2 exact MSE derivatives."""

import unittest

import numpy as np

from opttheory.models.logistic import predict
from opttheory.objectives.logistic_mse import objective, gradient, hessian


class LogisticObjectiveDerivativeTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0.0, 0.35, 1.15, 1.9, 2.4, 3.1, 4.0])
        self.y = predict(self.x, L=10.0, k=1.5, x0=2.0) + np.array([
            0.03, -0.09, 0.07, 0.04, -0.02, 0.10, -0.05,
        ])

    def test_hand_checked_midpoint_mse_gradient(self):
        # At x=x0=2, f=5, r=5-5.1=-0.1, grad(f)=[0.5,0,-3.75].
        theta = [10.0, 1.5, 2.0]
        self.assertAlmostEqual(objective(theta, [2.0], [5.1]), 0.01)
        np.testing.assert_allclose(gradient(theta, [2.0], [5.1]),
                                   [-0.1, 0.0, 0.75], rtol=0, atol=1e-14)

    def test_zero_residual_gives_zero_gradient(self):
        theta = [9.0, 1.2, 1.9]
        y_exact = predict(self.x, L=theta[0], k=theta[1], x0=theta[2])
        self.assertAlmostEqual(objective(theta, self.x, y_exact), 0.0)
        np.testing.assert_allclose(gradient(theta, self.x, y_exact),
                                   np.zeros(3), atol=1e-13)

    def test_gradient_matches_independent_loss_differences(self):
        step = 1e-6
        for theta in (np.array([9.0, 1.2, 2.2]),
                      np.array([10.0, -1.0, 2.0]),
                      np.array([7.0, 0.15, 0.0])):
            with self.subTest(theta=theta.tolist()):
                approx = np.zeros(3)
                for j in range(3):
                    direction = np.zeros(3)
                    direction[j] = step
                    approx[j] = (objective(theta + direction, self.x, self.y)
                                 - objective(theta - direction, self.x, self.y)) / (2 * step)
                np.testing.assert_allclose(gradient(theta, self.x, self.y),
                                           approx, rtol=1e-6, atol=1e-7)

    def test_exact_hessian_matches_independent_gradient_differences(self):
        step = 1e-5
        for theta in (np.array([9.0, 1.2, 2.2]),
                      np.array([10.0, -1.0, 2.0]),
                      np.array([7.0, 0.15, 0.0])):
            with self.subTest(theta=theta.tolist()):
                approx = np.zeros((3, 3))
                for j in range(3):
                    direction = np.zeros(3)
                    direction[j] = step
                    approx[:, j] = (gradient(theta + direction, self.x, self.y)
                                    - gradient(theta - direction, self.x, self.y)) / (2 * step)
                actual = hessian(theta, self.x, self.y)
                np.testing.assert_allclose(actual, approx, rtol=2e-7, atol=1e-7)
                np.testing.assert_allclose(actual, actual.T, rtol=0, atol=0)

    def test_residual_dependent_hessian_is_not_gauss_newton(self):
        theta = [9.0, 1.2, 2.2]
        fitted = predict(self.x, L=theta[0], k=theta[1], x0=theta[2])
        h_exact_fit = hessian(theta, self.x, fitted)
        h_with_residuals = hessian(theta, self.x, self.y)
        self.assertFalse(np.allclose(h_exact_fit, h_with_residuals))

    def test_stable_extreme_exponents(self):
        theta = [10.0, 1.0, 0.0]
        x = [-1000.0, 0.0, 1000.0]
        y = [0.0, 5.0, 10.0]
        self.assertTrue(np.isfinite(objective(theta, x, y)))
        self.assertTrue(np.all(np.isfinite(gradient(theta, x, y))))
        self.assertTrue(np.all(np.isfinite(hessian(theta, x, y))))

    def test_input_shapes_and_nonfinite_values_rejected(self):
        for function in (objective, gradient, hessian):
            with self.subTest(function=function.__name__):
                with self.assertRaises(ValueError):
                    function([10, 1], [0.0], [5.0])
                with self.assertRaises(ValueError):
                    function([10, 1, 2], [0.0, 1.0], [5.0])
                with self.assertRaises(ValueError):
                    function([10, float("nan"), 2], [0.0], [5.0])


if __name__ == "__main__":
    unittest.main()
