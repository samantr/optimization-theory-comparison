"""Model 1 objective J(theta), evaluated at a *candidate* parameter vector."""

import numpy as np

from opttheory.models.exponential import predict


def _validated_inputs(theta, x, y_observed):
    """Check the shared input contract for objective and its derivatives."""
    theta = np.asarray(theta, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y_observed = np.asarray(y_observed, dtype=np.float64)

    if theta.shape != (3,):
        raise ValueError("theta must be a vector of three parameters: [a, b, c].")
    if x.ndim != 1 or y_observed.ndim != 1 or x.shape != y_observed.shape or not x.size:
        raise ValueError("x and y_observed must be equally sized, nonempty 1-D arrays.")
    if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(x))
            and np.all(np.isfinite(y_observed))):
        raise ValueError("Inputs must contain only finite values.")
    return theta, x, y_observed


def objective(theta, x, y_observed) -> float:
    """Mean squared error of a candidate [a, b, c] on supplied observations.

    J(theta) = mean((a*exp(b*x_i) + c - y_i)^2).

    Only x and y_observed should be passed from the *training* subset during
    optimization. Neither theta_true nor y_clean is read or used here.
    """
    theta, x, y_observed = _validated_inputs(theta, x, y_observed)

    a, b, c = theta
    with np.errstate(over="raise", invalid="raise"):
        predictions = predict(x, a=a, b=b, c=c)
        residuals = predictions - y_observed
        mse = float(np.mean(residuals ** 2))
    if not np.isfinite(mse):
        raise FloatingPointError("MSE is not finite at this parameter vector.")
    return mse


def gradient(theta, x, y_observed) -> np.ndarray:
    """Analytical MSE gradient in parameter order [a, b, c].

    With r_i = a*exp(b*x_i) + c - y_i and n observations:
      dJ/da = (2/n) * sum(r_i * exp(b*x_i))
      dJ/db = (2/n) * sum(r_i * a*x_i*exp(b*x_i))
      dJ/dc = (2/n) * sum(r_i)

    Uses only candidate parameters and supplied observations. During fitting,
    supply training data only; this function does not access true parameters.
    """
    theta, x, y_observed = _validated_inputs(theta, x, y_observed)
    a, b, c = theta

    with np.errstate(over="raise", invalid="raise"):
        exp_bx = np.exp(b * x)
        residuals = a * exp_bx + c - y_observed
        derivatives = np.array([
            2.0 * np.mean(residuals * exp_bx),
            2.0 * np.mean(residuals * a * x * exp_bx),
            2.0 * np.mean(residuals),
        ], dtype=np.float64)

    if not np.all(np.isfinite(derivatives)):
        raise FloatingPointError("Gradient is not finite at this parameter vector.")
    return derivatives


def hessian(theta, x, y_observed) -> np.ndarray:
    """Exact 3x3 Hessian of the Model 1 MSE in parameter order [a, b, c].

    For E_i = exp(b*x_i) and r_i = a*E_i + c - y_i:

        H_aa = 2 mean(E_i**2)
        H_ab = 2 mean(a*x_i*E_i**2 + r_i*x_i*E_i)
        H_ac = 2 mean(E_i)
        H_bb = 2 mean((a*x_i*E_i)**2 + r_i*a*x_i**2*E_i)
        H_bc = 2 mean(a*x_i*E_i)
        H_cc = 2

    The residual-dependent terms make this the *exact* Hessian, not the
    Gauss-Newton approximation. Only supplied observations are used.
    """
    theta, x, y_observed = _validated_inputs(theta, x, y_observed)
    a, b, c = theta

    with np.errstate(over="raise", invalid="raise"):
        exp_bx = np.exp(b * x)
        residuals = a * exp_bx + c - y_observed
        x_exp_bx = x * exp_bx
        a_x_exp_bx = a * x_exp_bx

        h_aa = 2.0 * np.mean(exp_bx ** 2)
        h_ab = 2.0 * np.mean(a_x_exp_bx * exp_bx + residuals * x_exp_bx)
        h_ac = 2.0 * np.mean(exp_bx)
        h_bb = 2.0 * np.mean(a_x_exp_bx ** 2 + residuals * a * x * x_exp_bx)
        h_bc = 2.0 * np.mean(a_x_exp_bx)

        result = np.array([
            [h_aa, h_ab, h_ac],
            [h_ab, h_bb, h_bc],
            [h_ac, h_bc, 2.0],
        ], dtype=np.float64)

    if not np.all(np.isfinite(result)):
        raise FloatingPointError("Hessian is not finite at this parameter vector.")
    return result

