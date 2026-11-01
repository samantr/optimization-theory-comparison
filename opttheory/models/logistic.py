"""Model 2: three-parameter logistic (S-shaped) growth regression.

This is a continuous-output regression curve, not a binary classifier.
"""

import numpy as np


def predict(x: np.ndarray, L: float, k: float, x0: float) -> np.ndarray:
    """Evaluate f(x) = L / (1 + exp(-k*(x-x0))).

    For L > 0 and k > 0, the curve rises from zero toward capacity L,
    and reaches L/2 at x=x0. ``logaddexp`` avoids overflow when the
    exponential argument has large magnitude.

    This function intentionally does not constrain candidate parameters:
    the optimization methods will need to handle difficult starting points.
    """
    x = np.asarray(x, dtype=np.float64)
    z = k * (x - x0)
    sigmoid = np.exp(-np.logaddexp(0.0, -z))
    return L * sigmoid
