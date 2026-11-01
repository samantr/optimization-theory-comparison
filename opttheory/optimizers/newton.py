"""Full-step exact Newton optimizer for nonlinear regression."""

from time import perf_counter

import numpy as np

from opttheory.objectives.mse import (
    gradient as model1_gradient,
    hessian as model1_hessian,
    objective as model1_objective,
)
from opttheory.optimizers.common import OptimizationResult


def newton(theta0, x, y_observed, *, max_iterations=100,
           gradient_tolerance=1e-6, condition_limit=1e12,
           objective_fn=model1_objective, gradient_fn=model1_gradient,
           hessian_fn=model1_hessian) -> OptimizationResult:
    """Apply theta <- theta - solve(exact_Hessian, analytical_gradient).

    This is *exact*, full-step Newton, not Gauss-Newton or a black-box solver.
    The condition-number check avoids blindly solving nearly singular systems.
    An indefinite Hessian is allowed: its Newton step can be an ascent step,
    and any loss increase is recorded rather than silently corrected.

    `converged` means only that the gradient norm met the tolerance; a zero
    gradient alone is not proof of a local or global minimum.

    Supply the same model's objective, gradient, and exact Hessian callables.
    Model 1 remains the backwards-compatible default.
    """
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be finite and positive.")
    if not np.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("condition_limit must be finite and at least 1.")

    theta = np.asarray(theta0, dtype=np.float64).copy()
    start = perf_counter()
    loss = objective_fn(theta, x, y_observed)
    grad = gradient_fn(theta, x, y_observed)
    loss_history = [loss]
    norm_history = [float(np.linalg.norm(grad))]
    objective_evaluations = 1
    gradient_evaluations = 1
    hessian_evaluations = 0
    iterations = 0
    reason = "max_iterations"

    if norm_history[-1] <= gradient_tolerance:
        reason = "gradient_tolerance"
    else:
        for _ in range(max_iterations):
            try:
                exact_hessian = hessian_fn(theta, x, y_observed)
                hessian_evaluations += 1
                condition = float(np.linalg.cond(exact_hessian))
                if not np.isfinite(condition):
                    reason = "singular_hessian"
                    break
                if condition > condition_limit:
                    reason = "ill_conditioned_hessian"
                    break

                direction = np.linalg.solve(exact_hessian, grad)
                with np.errstate(over="raise", invalid="raise"):
                    next_theta = theta - direction
                if not np.all(np.isfinite(next_theta)):
                    raise FloatingPointError("Nonfinite Newton step.")

                next_loss = objective_fn(next_theta, x, y_observed)
                objective_evaluations += 1
                next_grad = gradient_fn(next_theta, x, y_observed)
                gradient_evaluations += 1
                next_norm = float(np.linalg.norm(next_grad))
                if not np.isfinite(next_norm):
                    raise FloatingPointError("Nonfinite gradient norm.")
            except np.linalg.LinAlgError:
                reason = "singular_hessian"
                break
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
        hessian_evaluations=hessian_evaluations,
    )
