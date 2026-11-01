"""Model 2 logistic-regression MSE, analytical gradient, and *exact* Hessian.

This is continuous-output nonlinear regression, not logistic classification.
All functions use candidate theta=[L,k,x0] and observed data supplied by the
caller. True synthetic parameters and held-out labels are never accessed.
"""

import numpy as np

from opttheory.models.logistic import predict


def _validated_inputs(theta, x, y_observed):
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y_observed = np.asarray(y_observed, dtype=np.float64)
    if theta.shape != (3,):
        raise ValueError("theta must have exactly three parameters: [L, k, x0].")
    if x.ndim != 1 or y_observed.ndim != 1 or x.shape != y_observed.shape or not x.size:
        raise ValueError("x and y_observed must be equally sized, nonempty 1-D arrays.")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(x))
            and np.all(np.isfinite(y_observed))):
        raise ValueError("Inputs must contain only finite values.")
    return theta, x, y_observed


def objective(theta, x, y_observed) -> float:
    """J(theta) = mean((L*sigmoid(k*(x-x0)) - y_observed)**2)."""
    theta, x, y_observed = _validated_inputs(theta, x, y_observed)
    with np.errstate(over="raise", invalid="raise"):
        residual = predict(x, L=theta[0], k=theta[1], x0=theta[2]) - y_observed
        result = float(np.mean(residual * residual))
    if not np.isfinite(result):
        raise FloatingPointError("Model 2 MSE is not finite at this candidate.")
    return result


def _prediction_derivatives(theta, x, *, second_order=False):
    """Return prediction, Jacobian rows, and optional exact second derivatives.

    Let t=x-x0, s=sigmoid(k*t), q=s*(1-s), u=1-2*s.
    Jacobian columns [L,k,x0]: [s, L*q*t, -L*k*q].
    """
    L, k, x0 = theta
    t = x - x0
    z = k * t
    s = np.exp(-np.logaddexp(0.0, -z))  # numerically stable sigmoid
    q = s * (1.0 - s)
    prediction = L * s

    jacobian = np.column_stack((s, L * q * t, -L * k * q))
    if not second_order:
        # Gradient Descent must not pay for Hessian calculations.
        return prediction, jacobian, None

    u = 1.0 - 2.0 * s
    second = np.zeros((len(x), 3, 3), dtype=np.float64)
    second[:, 0, 1] = second[:, 1, 0] = q * t
    second[:, 0, 2] = second[:, 2, 0] = -k * q
    second[:, 1, 1] = L * q * u * t**2
    second[:, 1, 2] = second[:, 2, 1] = -L * q * (1.0 + k * u * t)
    second[:, 2, 2] = L * k**2 * q * u
    # d²f/dL² is zero because the prediction is linear in L for fixed k,x0.
    return prediction, jacobian, second


def gradient(theta, x, y_observed) -> np.ndarray:
    """Analytical gradient: (2/n) * sum_i residual_i * grad(f_i)."""
    theta, x, y_observed = _validated_inputs(theta, x, y_observed)
    with np.errstate(over="raise", invalid="raise"):
        prediction, jacobian, _ = _prediction_derivatives(theta, x)
        residual = prediction - y_observed
        result = (2.0 / len(x)) * (jacobian.T @ residual)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Model 2 gradient is not finite at this candidate.")
    return result


def hessian(theta, x, y_observed) -> np.ndarray:
    """Exact Hessian, including residual-dependent second derivatives.

    H = (2/n) * sum_i [grad(f_i) grad(f_i)^T + residual_i Hessian(f_i)].
    Retaining the second term is essential: omitting it is Gauss-Newton,
    which is not the exact Newton method in the approved comparison.
    """
    theta, x, y_observed = _validated_inputs(theta, x, y_observed)
    with np.errstate(over="raise", invalid="raise"):
        prediction, jacobian, second = _prediction_derivatives(theta, x, second_order=True)
        residual = prediction - y_observed
        result = (2.0 / len(x)) * (
            jacobian.T @ jacobian + np.einsum("i,ijk->jk", residual, second)
        )
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Model 2 Hessian is not finite at this candidate.")
    return result
