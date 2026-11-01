"""Failure-aware selection of a GD learning rate from validation-only results.

This module aggregates results handed to it; it does not generate data,
run optimizers, choose based on test labels, or tune any parameters itself.
"""

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Any, Mapping, Sequence

from opttheory.experiments.plan import CalibrationTrial, build_gd_calibration_plan


@dataclass(frozen=True)
class CalibrationResult:
    """Recorded output of one planned GD training/validation attempt.

    `validation_mse` is measured on the validation split only. If the
    optimizer fails numerically, it may be absent; a finite last-iterate
    validation MSE is mandatory for gradient/budget termination.
    """

    trial: CalibrationTrial
    termination_reason: str
    validation_mse: float | None


@dataclass(frozen=True)
class RateScore:
    learning_rate: float
    numerical_failures: int
    budget_exhaustions: int
    finite_validation_cases: int
    mean_validation_mse: float


def select_gd_learning_rate(
    config: Mapping[str, Any], model: str, results: Sequence[CalibrationResult]
) -> tuple[float, tuple[RateScore, ...]]:
    """Choose a model-specific rate by a fully predeclared, failure-first rule.

    1. Prefer the rate with fewer numerical failures (divergence/overflow).
    2. On ties, prefer lower mean validation MSE of finite nonfailed runs.
       Budget-exhausted runs count: their last finite validation loss matters.
    3. Break remaining ties in favor of the smaller learning rate.

    All three seeds × five starts × eight candidate rates must be present.
    Failed cases remain counted, not cherry-picked out of the trial list.
    The full diagnostic scores are returned for transparent reporting.
    """
    ranking = config["gradient_descent_learning_rate_tuning"]["selection_policy"]["ranking"]
    if ranking != [
        "fewest_numerical_failures",
        "lowest_mean_validation_mse_among_finite_nonfailed_trials",
        "smaller_learning_rate_on_exact_tie",
    ]:
        raise ValueError("Code and preregistered calibration ranking must match.")
    expected = build_gd_calibration_plan(config, model)
    if len(results) != len(expected):
        raise ValueError("Calibration results must cover the complete registered grid.")
    by_trial = {}
    for row in results:
        if not isinstance(row, CalibrationResult):
            raise ValueError("Calibration rows must have the approved result format.")
        if row.trial in by_trial:
            raise ValueError("Duplicate calibration condition is not allowed.")
        by_trial[row.trial] = row
    if set(by_trial) != set(expected):
        raise ValueError("Calibration results differ from the approved seed/start/rate grid.")

    valid_reasons = {"gradient_tolerance", "max_iterations", "numerical_failure"}
    for row in results:
        if row.termination_reason not in valid_reasons:
            raise ValueError("Unknown GD termination reason in calibration results.")
        if row.termination_reason != "numerical_failure":
            value = row.validation_mse
            if value is None or not isfinite(value) or value < 0:
                raise ValueError("Finite nonnegative validation MSE is required for nonfailed GD runs.")

    rates = config["gradient_descent_learning_rate_tuning"]["candidate_learning_rates"]
    scores = []
    for rate in rates:
        rows = [by_trial[t] for t in expected if t.learning_rate == rate]
        failures = sum(row.termination_reason == "numerical_failure" for row in rows)
        exhausted = sum(row.termination_reason == "max_iterations" for row in rows)
        finite_values = [float(row.validation_mse) for row in rows
                         if row.termination_reason != "numerical_failure"]
        scores.append(RateScore(
            learning_rate=rate,
            numerical_failures=failures,
            budget_exhaustions=exhausted,
            finite_validation_cases=len(finite_values),
            mean_validation_mse=mean(finite_values) if finite_values else float("inf"),
        ))

    winner = min(scores, key=lambda s: (
        s.numerical_failures, s.mean_validation_mse, s.learning_rate,
    ))
    if not isfinite(winner.mean_validation_mse):
        raise ValueError("No finite calibration validation MSE is available.")
    return winner.learning_rate, tuple(scores)
