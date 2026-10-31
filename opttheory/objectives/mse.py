"""Model 1 objective J(theta), evaluated at a *candidate* parameter vector."""

import numpy as np

from opttheory.models.exponential import predict


def objective(theta, x, y_observed) -> float:
    """Mean squared error of a candidate [a, b, c] on supplied observations.

    J(theta) = mean((a*exp(b*x_i) + c - y_i)^2).

    Only x and y_observed should be passed from the *training* subset during
    optimization. Neither theta_true nor y_clean is read or used here.
    """
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

    a, b, c = theta
    with np.errstate(over="raise", invalid="raise"):
        predictions = predict(x, a=a, b=b, c=c)
        residuals = predictions - y_observed
        mse = float(np.mean(residuals ** 2))
    if not np.isfinite(mse):
        raise FloatingPointError("MSE is not finite at this parameter vector.")
    return mse

