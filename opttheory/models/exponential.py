"""Model 1: three-parameter exponential regression."""

import numpy as np


def predict(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Evaluate f(x) = a * exp(b*x) + c at scalar or array inputs."""
    return a * np.exp(b * np.asarray(x, dtype=np.float64)) + c

