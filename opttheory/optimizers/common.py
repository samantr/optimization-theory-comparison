"""Small shared result contract for our from-scratch optimizers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OptimizationResult:
    """One optimization run; histories include the initial parameter vector.

    `converged` means the gradient-norm stopping rule was met, *not* that a
    global minimum has been proven. Iterations count completed updates.
    """

    theta: np.ndarray
    iterations: int
    converged: bool
    reason: str
    loss_history: tuple[float, ...]
    gradient_norm_history: tuple[float, ...]
    elapsed_seconds: float
    objective_evaluations: int
    gradient_evaluations: int
    hessian_evaluations: int = 0


