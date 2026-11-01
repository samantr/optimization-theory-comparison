"""Transparent, fixed-step batch Gradient Descent for Model 1 regression."""

from time import perf_counter

import numpy as np

from opttheory.objectives.mse import gradient, objective
from opttheory.optimizers.common import OptimizationResult


def gradient_descent(theta0, x, y_observed, *, learning_rate=0.01,
                     max_iterations=10_000, gradient_tolerance=1e-6) -> OptimizationResult:
    """Minimize training MSE with theta <- theta - learning_rate * gradient.

    Uses the analytical gradient, not a library optimizer. `learning_rate`
    is constant: there is no hidden line search, momentum, or adaptation.

    The run terminates on a small gradient, an iteration budget, or a numerical
    failure. Histories include the initial point and every completed update.
    A loss increase is recorded rather than hidden; fixed-step GD is not
    guaranteed to decrease loss for every learning rate.
    """
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be finite and positive.")

    theta = np.asarray(theta0, dtype=np.float64).copy()
    start = perf_counter()
    loss = objective(theta, x, y_observed)
    grad = gradient(theta, x, y_observed)
    loss_history = [loss]
    norm_history = [float(np.linalg.norm(grad))]
    objective_evaluations = 1
    gradient_evaluations = 1
    iterations = 0
    reason = "max_iterations"

    if norm_history[-1] <= gradient_tolerance:
        reason = "gradient_tolerance"
    else:
        for _ in range(max_iterations):
            try:
                with np.errstate(over="raise", invalid="raise"):
                    next_theta = theta - learning_rate * grad
                if not np.all(np.isfinite(next_theta)):
                    raise FloatingPointError("Nonfinite parameter update.")
                next_loss = objective(next_theta, x, y_observed)
                objective_evaluations += 1
                next_grad = gradient(next_theta, x, y_observed)
                gradient_evaluations += 1
                next_norm = float(np.linalg.norm(next_grad))
                if not np.isfinite(next_norm):
                    raise FloatingPointError("Nonfinite gradient norm.")
            except (FloatingPointError, OverflowError):
                reason = "numerical_failure"
                break

            theta = next_theta
            grad = next_grad
            iterations += 1
            loss_history.append(next_loss)
            norm_history.append(next_norm)
            if next_norm <= gradient_tolerance:
                reason = "gradient_tolerance"
                break

    return OptimizationResult(
        theta=theta,
        iterations=iterations,
        converged=(reason == "gradient_tolerance"),
        reason=reason,
        loss_history=tuple(loss_history),
        gradient_norm_history=tuple(norm_history),
        elapsed_seconds=perf_counter() - start,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
    )
