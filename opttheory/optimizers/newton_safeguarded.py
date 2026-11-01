"""Clearly labelled safeguarded Newton variant; pure Newton stays separate."""

from time import perf_counter

import numpy as np

from opttheory.objectives.mse import gradient, hessian, objective
from opttheory.optimizers.common import OptimizationResult


def safeguarded_newton(theta0, x, y_observed, *, max_iterations=100,
                       gradient_tolerance=1e-6, condition_limit=1e12,
                       armijo_constant=1e-4, backtrack_factor=0.5,
                       max_backtracks=30) -> OptimizationResult:
    """Exact-Hessian Newton with a descent fallback and Armijo backtracking.

    A positive-definite, well-conditioned Hessian gives the Newton direction
    p = -solve(H,g). Otherwise p = -g, explicitly counted as a fallback.
    A trial step is accepted only if J(theta + alpha*p) satisfies the Armijo
    sufficient-decrease condition; alpha starts at 1 and shrinks geometrically.

    This hybrid is **not** the original full-step Newton baseline. It neither
    changes the original solver nor replaces the exact Hessian with an
    unreported approximation. Histories contain accepted updates only.
    """
    if isinstance(max_iterations, (bool, np.bool_)) or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer.")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be finite and positive.")
    if not np.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("condition_limit must be finite and at least 1.")
    if not np.isfinite(armijo_constant) or not 0 < armijo_constant < 1:
        raise ValueError("armijo_constant must lie strictly between 0 and 1.")
    if not np.isfinite(backtrack_factor) or not 0 < backtrack_factor < 1:
        raise ValueError("backtrack_factor must lie strictly between 0 and 1.")
    if isinstance(max_backtracks, (bool, np.bool_)) or not isinstance(max_backtracks, (int, np.integer)) or max_backtracks < 0:
        raise ValueError("max_backtracks must be a nonnegative integer.")

    theta = np.asarray(theta0, dtype=np.float64).copy()
    start = perf_counter()
    loss = objective(theta, x, y_observed)
    grad = gradient(theta, x, y_observed)
    loss_history = [loss]
    norm_history = [float(np.linalg.norm(grad))]
    objective_evaluations = 1
    gradient_evaluations = 1
    hessian_evaluations = 0
    line_search_backtracks = 0
    gradient_fallback_steps = 0
    step_sizes = []
    iterations = 0
    reason = "max_iterations"

    if norm_history[-1] <= gradient_tolerance:
        reason = "gradient_tolerance"
    else:
        for _ in range(max_iterations):
            try:
                H = hessian(theta, x, y_observed)
                hessian_evaluations += 1
                condition = float(np.linalg.cond(H))
                use_fallback = not np.isfinite(condition) or condition > condition_limit

                if not use_fallback:
                    try:
                        np.linalg.cholesky(H)  # Newton direction is descent for SPD H.
                        direction = -np.linalg.solve(H, grad)
                    except np.linalg.LinAlgError:
                        use_fallback = True

                if use_fallback:
                    direction = -grad

                directional_derivative = float(np.dot(grad, direction))
                if not np.isfinite(directional_derivative):
                    raise FloatingPointError("Nonfinite directional derivative.")
                if directional_derivative >= 0:
                    # Defensive check in case of numerical cancellation.
                    direction = -grad
                    directional_derivative = -float(np.dot(grad, grad))
                    use_fallback = True
                if not np.isfinite(directional_derivative) or directional_derivative >= 0:
                    raise FloatingPointError("No finite descent direction.")

                alpha = 1.0
                accepted = False
                for backtracks in range(max_backtracks + 1):
                    try:
                        with np.errstate(over="raise", invalid="raise"):
                            candidate = theta + alpha * direction
                        if not np.all(np.isfinite(candidate)):
                            raise FloatingPointError("Nonfinite trial parameters.")
                        candidate_loss = objective(candidate, x, y_observed)
                        objective_evaluations += 1
                        if candidate_loss <= loss + armijo_constant * alpha * directional_derivative:
                            accepted = True
                            break
                    except (FloatingPointError, OverflowError):
                        pass  # Invalid trial is rejected; a shorter step may work.

                    if backtracks < max_backtracks:
                        alpha *= backtrack_factor
                        line_search_backtracks += 1

                if not accepted:
                    reason = "line_search_failed"
                    break

                next_grad = gradient(candidate, x, y_observed)
                gradient_evaluations += 1
                next_norm = float(np.linalg.norm(next_grad))
                if not np.isfinite(next_norm):
                    raise FloatingPointError("Nonfinite gradient norm.")

            except np.linalg.LinAlgError:
                reason = "numerical_failure"
                break
            except (FloatingPointError, OverflowError):
                reason = "numerical_failure"
                break

            theta = candidate
            loss = candidate_loss
            grad = next_grad
            iterations += 1
            loss_history.append(loss)
            norm_history.append(next_norm)
            step_sizes.append(float(alpha))
            if use_fallback:
                gradient_fallback_steps += 1
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
        line_search_backtracks=line_search_backtracks,
        gradient_fallback_steps=gradient_fallback_steps,
        step_sizes=tuple(step_sizes),
    )
