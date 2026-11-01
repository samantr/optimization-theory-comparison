"""Three exploratory learning-rate adaptations, separate from the frozen study.

gradient_reversal: accept every finite step, then halve the global learning
rate if successive full gradient vectors have a negative dot product. This is
an operationalization of an overshoot heuristic, not proof of crossing a min.

backtracking: reject a trial unless it strictly reduces *training* MSE;
halve the global rate and retry from the previous accepted point. This is a
simple monotone line search, not the Armijo sufficient-decrease condition.

recovering_backtracking: use the same trial rejection as backtracking, but
after accepting a step, recover half the gap to the most recently rejected
rate: alpha_next = (alpha_accepted + alpha_rejected_upper) / 2.
This experimental heuristic remembers a rate from a previous theta; its
"upper" bound is not a mathematically valid bracket at the current theta.
"""

from time import perf_counter

import numpy as np

from opttheory.objectives.mse import gradient as model1_gradient, objective as model1_objective
from opttheory.optimizers.common import OptimizationResult


def adaptive_gradient_descent(
    theta0, x, y_observed, *, strategy,
    initial_learning_rate=1.0, max_iterations=10_000,
    gradient_tolerance=1e-4, min_learning_rate=1e-12,
    max_backtracks_per_iteration=50,
    objective_fn=model1_objective, gradient_fn=model1_gradient,
) -> OptimizationResult:
    """Minimize MSE with one of three explicit, first-order step-size rules.

    `iterations` counts accepted updates, not rejected trials.
    `line_search_backtracks` counts rejected trials of backtracking.
    `step_sizes` stores the rate actually used for each accepted update.
    """
    if strategy not in {"gradient_reversal", "backtracking", "recovering_backtracking"}:
        raise ValueError("strategy must be gradient_reversal, backtracking, or recovering_backtracking.")
    if not np.isfinite(initial_learning_rate) or initial_learning_rate <= 0:
        raise ValueError("initial_learning_rate must be finite and positive.")
    if not np.isfinite(min_learning_rate) or min_learning_rate <= 0:
        raise ValueError("min_learning_rate must be finite and positive.")
    if initial_learning_rate < min_learning_rate:
        raise ValueError("initial_learning_rate must be >= min_learning_rate.")
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1):
        raise ValueError("max_iterations must be a positive integer.")
    if (isinstance(max_backtracks_per_iteration, (bool, np.bool_))
            or not isinstance(max_backtracks_per_iteration, (int, np.integer))
            or max_backtracks_per_iteration < 1):
        raise ValueError("max_backtracks_per_iteration must be a positive integer.")
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError("gradient_tolerance must be finite and positive.")

    theta = np.asarray(theta0, dtype=np.float64).copy()
    start = perf_counter()
    loss = objective_fn(theta, x, y_observed)
    grad = gradient_fn(theta, x, y_observed)
    norm = float(np.linalg.norm(grad))
    if not np.isfinite(norm):
        raise FloatingPointError("Initial gradient norm must be finite.")

    losses = [loss]
    norms = [norm]
    step_sizes = []
    objective_evaluations = 1
    gradient_evaluations = 1
    backtracks = 0
    rate = float(initial_learning_rate)
    last_rejected_rate = None
    reason = "gradient_tolerance" if norm <= gradient_tolerance else "max_iterations"

    if reason != "gradient_tolerance":
        for _ in range(max_iterations):
            # Gradient reversal accepts all finite candidates even if MSE
            # increases. Backtracking tests MSE before computing new gradient.
            trial_rate = rate
            accepted = False
            trial_count = (max_backtracks_per_iteration + 1
                           if strategy != "gradient_reversal" else 1)
            for _ in range(trial_count):
                if trial_rate < min_learning_rate:
                    break
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        candidate = theta - trial_rate * grad
                    if not np.all(np.isfinite(candidate)):
                        raise FloatingPointError("Nonfinite parameter update.")
                    objective_evaluations += 1
                    candidate_loss = objective_fn(candidate, x, y_observed)
                    if not np.isfinite(candidate_loss):
                        raise FloatingPointError("Nonfinite objective.")
                except (FloatingPointError, OverflowError):
                    if strategy == "gradient_reversal":
                        reason = "numerical_failure"
                        break
                else:
                    if strategy == "gradient_reversal" or candidate_loss < loss:
                        next_theta, next_loss = candidate, candidate_loss
                        accepted = True
                        break

                # Backtracking retries from theta, not the rejected candidate.
                backtracks += 1
                if strategy == "recovering_backtracking":
                    last_rejected_rate = trial_rate
                trial_rate *= 0.5

            if not accepted:
                if strategy != "gradient_reversal":
                    reason = "line_search_failed"
                break

            try:
                next_grad = gradient_fn(next_theta, x, y_observed)
                gradient_evaluations += 1
                next_norm = float(np.linalg.norm(next_grad))
                if not np.isfinite(next_norm):
                    raise FloatingPointError("Nonfinite gradient norm.")
            except (FloatingPointError, OverflowError):
                reason = "numerical_failure"
                break

            reversal = float(np.dot(grad, next_grad)) < 0.0
            theta, loss, grad, norm = next_theta, next_loss, next_grad, next_norm
            losses.append(loss)
            norms.append(norm)
            step_sizes.append(trial_rate)
            rate = trial_rate
            if strategy == "gradient_reversal" and reversal:
                rate *= 0.5
            elif (strategy == "recovering_backtracking" and
                  last_rejected_rate is not None and last_rejected_rate > rate):
                # Saman's rule: regain half of what was previously removed.
                # This influences the *next* attempt, not the accepted step.
                rate = (rate + last_rejected_rate) * 0.5
            if norm <= gradient_tolerance:
                reason = "gradient_tolerance"
                break
            if rate < min_learning_rate:
                reason = "step_size_too_small"
                break

    return OptimizationResult(
        theta=theta,
        iterations=len(step_sizes),
        converged=(reason == "gradient_tolerance"),
        reason=reason,
        loss_history=tuple(losses),
        gradient_norm_history=tuple(norms),
        elapsed_seconds=perf_counter() - start,
        objective_evaluations=objective_evaluations,
        gradient_evaluations=gradient_evaluations,
        line_search_backtracks=backtracks,
        step_sizes=tuple(step_sizes),
    )
