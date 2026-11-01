"""Paired, failure-preserving evaluation runner (NOT executed on import).

All numerical test outcomes are produced only by the explicitly gated CLI.
The default CLI mode prints a manifest without generating any evaluation data.
"""

from copy import deepcopy
from collections import defaultdict
import hashlib
from math import isfinite
from statistics import median
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from opttheory.datasets.synthetic import generate_model1, generate_model2
from opttheory.experiments.plan import build_model1_plan, build_model2_plan, PlannedRun
from opttheory.experiments.policy import optimizer_stopping_kwargs
from opttheory.experiments.run_calibration import canonical_json_hash
from opttheory.objectives import mse, logistic_mse
from opttheory.optimizers import gradient_descent, newton, safeguarded_newton


MODELS = {
    "exponential_decay": (build_model1_plan, generate_model1, mse),
    "logistic": (build_model2_plan, generate_model2, logistic_mse),
}
METHODS = {
    "gradient_descent": gradient_descent,
    "newton": newton,
    "safeguarded_newton": safeguarded_newton,
}
TERMINATION_REASONS = {
    "gradient_tolerance", "max_iterations", "numerical_failure",
    "singular_hessian", "ill_conditioned_hessian", "line_search_failed",
}


def balanced_method_order(trials: tuple[PlannedRun, ...]) -> tuple[PlannedRun, ...]:
    """Alternate GD-first and Newton-first to limit fixed-order timing bias.

    Method order is deterministic and chosen from trial identifiers *before*
    generating data or knowing any performance results. The safeguarded
    exploratory variant remains clearly separate and always last.
    """
    grouped = defaultdict(dict)
    for trial in trials:
        if trial.method in grouped[trial.pairing_key]:
            raise ValueError("Duplicate paired method condition.")
        grouped[trial.pairing_key][trial.method] = trial
    output = []
    for index, methods in enumerate(grouped.values()):
        if set(methods) != set(METHODS):
            raise ValueError("Every paired condition must include all three methods.")
        primary = ("gradient_descent", "newton") if index % 2 == 0 else (
            "newton", "gradient_descent")
        output.extend(methods[name] for name in (*primary, "safeguarded_newton"))
    return tuple(output)


def validate_evaluation_protocol(
    plan: Mapping[str, Any], frozen: Mapping[str, Any], protocol: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate exactly the frozen models, seeds, starts, rates and policies.

    This is safe for a dry run: neither synthetic data nor a solver is called.
    """
    if canonical_json_hash(plan) != frozen["approved_experiment_plan_sha256"]:
        raise ValueError("Approved experiment plan changed after GD calibration.")
    if frozen["final_test_or_newton_results_used"] is not False:
        raise ValueError("Calibration provenance must exclude evaluation results.")
    if protocol["evaluation"]["primary_methods"] != plan["primary_methods"]:
        raise ValueError("Primary method definitions differ from the registered plan.")
    if protocol["evaluation"]["exploratory_methods"] != plan["exploratory_methods"]:
        raise ValueError("Exploratory method definitions differ from the registered plan.")
    if protocol["evaluation"]["gradient_tolerance"] != plan["optimizer_stopping"]["gradient_norm_tolerance"]:
        raise ValueError("Evaluation stopping tolerance must be the approved tolerance.")
    timing = protocol["timing"]
    if timing["repetitions_per_condition"] != 5:
        raise ValueError("Exactly five approved timing repetitions are required.")
    if timing["scope"] != "optimizer_invocation_only_no_dataset_generation_or_posthoc_metrics":
        raise ValueError("Timing must isolate optimizer invocations.")
    if timing["per_condition_summary"] != "median_of_five_elapsed_seconds":
        raise ValueError("The approved timing summary is the median of five runs.")
    if timing["repeat_initialization"] != "same_data_same_theta0_every_repetition_no_warm_start":
        raise ValueError("Every timing repetition must restart at the same theta.")
    if protocol["evaluation"]["same_data_and_initial_theta_for_all_methods"] is not True:
        raise ValueError("Paired methods must share data and initial parameters.")
    if protocol["evaluation"]["held_out_test"] != "score_once_after_final_parameters_are_frozen_never_optimize_or_tune_on_test":
        raise ValueError("Evaluation test-label isolation policy changed.")
    if not isfinite(protocol["evaluation"]["curvature_eigenvalue_tolerance"]) or protocol["evaluation"]["curvature_eigenvalue_tolerance"] <= 0:
        raise ValueError("Hessian curvature tolerance must be positive and finite.")
    if protocol["analysis"]["unit_of_independence"] != "dataset_noise_seed":
        raise ValueError("Dataset seeds, not starting vectors, are independent units.")
    if (protocol["analysis"]["starts_per_dataset"] != plan["starting_points_per_dataset"]
            or protocol["analysis"]["dataset_seeds_per_model"] != plan["dataset_seeds_per_model"]
            or protocol["analysis"]["primary_time_difference"] != "newton_minus_gradient_descent"):
        raise ValueError("Dataset-level paired analysis contradicts the approved grid.")
    analysis = protocol["analysis"]
    bootstrap = analysis["bootstrap"]
    if (analysis["confidence_interval_method"] != "paired_dataset_cluster_percentile_bootstrap"
            or bootstrap["resampling_unit"] != "whole_dataset_noise_seed_with_all_five_paired_starts"
            or bootstrap["statistic"] != "mean_of_ten_dataset_level_mean_newton_minus_gd_seconds"
            or bootstrap["confidence_level"] != 0.95
            or bootstrap["resamples"] != 10000
            or bootstrap["quantile_method"] != "linear"
            or type(bootstrap["base_seed"]) is not int or bootstrap["base_seed"] < 0
            or bootstrap["model_seed_offset"] != {"exponential_decay": 0, "logistic": 1}):
        raise ValueError("Approved cluster-bootstrap inference settings changed.")
    if protocol["final_evaluation_performed"] is not False:
        raise ValueError("The protocol is a frozen pre-execution document.")

    rates = frozen["learning_rates_by_model"]
    if set(rates) != set(plan["model_families"]) or set(rates) != set(MODELS):
        raise ValueError("Exactly the two calibrated model-specific rates are required.")
    allowed_rates = plan["gradient_descent_learning_rate_tuning"]["candidate_learning_rates"]
    if any(not isfinite(rate) or rate not in allowed_rates for rate in rates.values()):
        raise ValueError("A frozen rate is not one of the approved calibration candidates.")

    summary = {}
    for model, (builder, _, _) in MODELS.items():
        runs = builder(plan)
        n_datasets = len(set(run.dataset_id for run in runs))
        expected = (plan["dataset_seeds_per_model"] * plan["starting_points_per_dataset"]
                    * (len(plan["primary_methods"]) + len(plan["exploratory_methods"])))
        if len(runs) != expected or n_datasets != plan["dataset_seeds_per_model"]:
            raise ValueError("Incomplete approved evaluation trial manifest.")
        summary[model] = {
            "datasets": n_datasets,
            "starting_points_per_dataset": plan["starting_points_per_dataset"],
            "method_conditions": len(runs),
            "timed_invocations": len(runs) * timing["repetitions_per_condition"],
            "frozen_gd_learning_rate": rates[model],
        }
    return summary


def _model_and_method(trial: PlannedRun):
    if trial.model not in MODELS or trial.method not in METHODS:
        raise ValueError("Unregistered model or optimization method.")
    objective_module = MODELS[trial.model][2]
    kwargs = {"objective_fn": objective_module.objective,
              "gradient_fn": objective_module.gradient}
    if trial.method != "gradient_descent":
        kwargs["hessian_fn"] = objective_module.hessian
    return objective_module, METHODS[trial.method], kwargs


def _curvature(hessian_fn, theta, x, y, tolerance: float) -> dict:
    """Post-timing Hessian diagnostic of the *training* objective only."""
    try:
        H = hessian_fn(theta, x, y)
        eigenvalues = np.linalg.eigvalsh(H)
        if not np.all(np.isfinite(eigenvalues)):
            raise FloatingPointError("Nonfinite Hessian eigenvalues.")
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        return {"curvature": "unavailable", "hessian_eigenvalues": None,
                "curvature_error": f"{type(exc).__name__}: {exc}"}
    if np.all(eigenvalues > tolerance):
        label = "strict_local_minimum"
    elif np.all(eigenvalues < -tolerance):
        label = "strict_local_maximum"
    elif np.any(eigenvalues > tolerance) and np.any(eigenvalues < -tolerance):
        label = "indefinite_saddle"
    else:
        label = "inconclusive_near_zero_curvature"
    return {"curvature": label, "hessian_eigenvalues": eigenvalues.tolist(),
            "curvature_error": None}


def _safe_mse(objective_fn, theta, x, y) -> tuple[float | None, str | None]:
    try:
        result = objective_fn(theta, x, y)
        if not isfinite(result):
            raise FloatingPointError("Nonfinite MSE.")
        return float(result), None
    except (FloatingPointError, OverflowError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def evaluate_condition(
    data, trial: PlannedRun, plan: Mapping[str, Any],
    frozen: Mapping[str, Any], protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Five fresh optimizer runs on identical training data, then score once.

    Test/validation observations are never provided to the optimizer.
    This function is usable in tests with non-evaluation *pilot* datasets.
    """
    objective_module, optimizer, kwargs = _model_and_method(trial)
    rate = frozen["learning_rates_by_model"][trial.model]
    if trial.method == "gradient_descent":
        kwargs["learning_rate"] = rate
    kwargs.update(optimizer_stopping_kwargs(plan, trial.method))
    nrep = protocol["timing"]["repetitions_per_condition"]
    if nrep != 5:
        raise ValueError("Exactly five repetitions are required.")

    train = data.split == "train"
    validation = data.split == "validation"
    test = data.split == "test"
    x_train, y_train = data.x[train], data.y_observed[train]
    if not (train.any() and validation.any() and test.any()):
        raise ValueError("All three disjoint data splits must be present.")

    timings = []
    reference = None
    reference_signature = None
    for index in range(nrep):
        result = None
        error = None
        start = perf_counter()
        try:
            result = optimizer(trial.theta0, x_train, y_train, **kwargs)
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        elapsed = perf_counter() - start

        reason = result.reason if result is not None else "numerical_failure"
        if reason not in TERMINATION_REASONS:
            raise ValueError(f"Undocumented termination reason: {reason}")
        signature = (reason, None if result is None else result.iterations,
                     None if result is None else result.converged,
                     None if result is None else tuple(result.theta.tolist()),
                     None if result is None else result.loss_history[-1])
        if reference_signature is None:
            reference_signature = signature
            reference = result
        elif signature != reference_signature:
            raise RuntimeError("Repeated deterministic optimization changed its outcome.")
        timings.append({
            "model": trial.model, "dataset_id": trial.dataset_id,
            "noise_seed": trial.noise_seed, "start_id": trial.start_id,
            "method": trial.method, "repetition": index + 1,
            "elapsed_seconds": elapsed, "termination_reason": reason,
            "error": error,
        })

    reason = reference.reason if reference is not None else "numerical_failure"
    stationary = bool(reference is not None and reference.converged
                      and reason == "gradient_tolerance")
    row = {
        "model": trial.model, "dataset_id": trial.dataset_id,
        "noise_seed": trial.noise_seed, "split_seed": trial.split_seed,
        "start_id": trial.start_id, "theta_initial": list(trial.theta0),
        "method": trial.method,
        "gd_learning_rate": rate if trial.method == "gradient_descent" else None,
        "repetitions": nrep,
        "median_optimizer_seconds": float(median(t["elapsed_seconds"] for t in timings)),
        "termination_reason": reason, "stationary_at_tolerance": stationary,
        "iterations": reference.iterations if reference is not None else 0,
        "final_gradient_norm": (reference.gradient_norm_history[-1]
                                if reference is not None else None),
        "objective_evaluations": reference.objective_evaluations if reference is not None else 0,
        "gradient_evaluations": reference.gradient_evaluations if reference is not None else 0,
        "hessian_evaluations": reference.hessian_evaluations if reference is not None else 0,
        "line_search_backtracks": reference.line_search_backtracks if reference is not None else 0,
        "gradient_fallback_steps": reference.gradient_fallback_steps if reference is not None else 0,
        "final_theta": reference.theta.tolist() if reference is not None else None,
        "train_mse": None, "validation_mse": None, "test_mse": None,
        "diagnostic_error": None,
        "curvature": "not_stationary", "hessian_eigenvalues": None,
        "curvature_error": None,
    }
    if reference is not None:
        for name, mask in (("train", train), ("validation", validation), ("test", test)):
            score, error = _safe_mse(
                objective_module.objective, reference.theta,
                data.x[mask], data.y_observed[mask]
            )
            row[f"{name}_mse"] = score
            if error:
                row["diagnostic_error"] = (row["diagnostic_error"] or "") + f"{name}: {error}; "
        if stationary:
            row.update(_curvature(objective_module.hessian, reference.theta,
                                  x_train, y_train,
                                  protocol["evaluation"]["curvature_eigenvalue_tolerance"]))
    return row, timings


def run_full_model_evaluation(
    model: str, plan: Mapping[str, Any], frozen: Mapping[str, Any],
    protocol: Mapping[str, Any], prototype: Mapping[str, Any]
) -> dict[str, Any]:
    """Execute the full registered evaluation grid. CLI --execute ONLY.

    This is intentionally NOT called by automated tests or preview mode.
    """
    validation = validate_evaluation_protocol(plan, frozen, protocol)
    if model not in MODELS or prototype["model"] != model:
        raise ValueError("Model configuration and registered model must match.")
    if prototype["n_observations"] != plan["observations_per_dataset"]:
        raise ValueError("Dataset size does not match preregistration.")
    if float(prototype["noise"]["sigma"]) != float(plan["primary_noise"]["sigma"]):
        raise ValueError("Dataset noise does not match preregistration.")
    builder, generator, _ = MODELS[model]
    trials = balanced_method_order(builder(plan))
    cached_data = {}
    hashes = {}
    summary_rows, timing_rows = [], []
    for trial in trials:
        if trial.noise_seed not in cached_data:
            cfg = deepcopy(prototype)
            cfg["noise"]["seed"] = trial.noise_seed
            cfg["split"]["seed"] = trial.split_seed
            data = generator(cfg)
            expected_counts = {name: plan["split"][f"{name}_count"]
                               for name in ("train", "validation", "test")}
            if data.split_counts() != expected_counts:
                raise ValueError("Generated evaluation split counts changed.")
            cached_data[trial.noise_seed] = data
            digest = hashlib.sha256()
            digest.update(data.x.tobytes())
            digest.update(data.y_observed.tobytes())
            digest.update("|".join(data.split.tolist()).encode("ascii"))
            hashes[str(trial.noise_seed)] = digest.hexdigest()
        row, timings = evaluate_condition(cached_data[trial.noise_seed], trial,
                                          plan, frozen, protocol)
        summary_rows.append(row)
        timing_rows.extend(timings)
    if len(summary_rows) != validation[model]["method_conditions"] or len(timing_rows) != validation[model]["timed_invocations"]:
        raise RuntimeError("Incomplete registered evaluation results.")
    return {"model": model, "run_records": summary_rows, "timing_records": timing_rows,
            "dataset_sha256_by_noise_seed": hashes}
