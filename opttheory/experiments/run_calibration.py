"""Execute pre-registered GD calibration on training/validation data only.

This module does NOT use evaluation noise seeds or final test observations.
It does not run either Newton method or the final GD-vs-Newton study.
"""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from math import isfinite
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from opttheory.datasets.synthetic import generate_model1, generate_model2
from opttheory.experiments.calibration import CalibrationResult, select_gd_learning_rate
from opttheory.experiments.plan import build_gd_calibration_plan
from opttheory.experiments.policy import optimizer_stopping_kwargs
from opttheory.objectives import logistic_mse, mse
from opttheory.optimizers import gradient_descent


def _dataset_digest(data) -> str:
    """Hash calibration inputs, training/validation targets and split labels.

    Deliberately exclude held-out test targets even from calibration metadata.
    """
    digest = hashlib.sha256()
    used = data.split != "test"
    for values in (data.x, data.y_observed[used]):
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    digest.update("|".join(data.split.tolist()).encode("ascii"))
    return digest.hexdigest()


def run_model_calibration(
    plan: Mapping[str, Any], model: str, prototype: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one model's complete 3-seed × 5-start × 8-rate GD calibration.

    The optimizer receives **only training** x/y. MSE for choosing a rate
    is computed from **validation** x/y. Nothing reads test x/y labels.
    Numerical failures remain in the raw records; no run is silently dropped.
    """
    trials = build_gd_calibration_plan(plan, model)
    if prototype["model"] != model:
        raise ValueError("Prototype model does not match the trial manifest.")
    if int(prototype["n_observations"]) != plan["observations_per_dataset"]:
        raise ValueError("Observation count does not match approved design.")
    if float(prototype["noise"]["sigma"]) != float(plan["primary_noise"]["sigma"]):
        raise ValueError("Noise sigma differs from the approved primary experiment.")
    for key in ("train_fraction", "validation_fraction"):
        if float(prototype["split"][key]) != float(plan["split"][key]):
            raise ValueError("Split differs from approved design.")

    if model == "exponential_decay":
        generate = generate_model1
        objective_fn, gradient_fn = mse.objective, mse.gradient
    elif model == "logistic":
        generate = generate_model2
        objective_fn, gradient_fn = logistic_mse.objective, logistic_mse.gradient
    else:
        raise ValueError("Unknown approved regression model.")

    stopping = optimizer_stopping_kwargs(plan, "gradient_descent")
    cached_data = {}
    data_hashes = {}
    outcomes = []
    raw_records = []
    started = perf_counter()

    for trial in trials:
        if trial.noise_seed not in cached_data:
            cfg = deepcopy(prototype)
            cfg["noise"]["seed"] = trial.noise_seed
            cfg["split"]["seed"] = trial.split_seed
            data = generate(cfg)
            expected_counts = {
                "train": int(plan["split"]["train_count"]),
                "validation": int(plan["split"]["validation_count"]),
                "test": int(plan["split"]["test_count"]),
            }
            if data.split_counts() != expected_counts:
                raise ValueError("Generated dataset violates approved split counts.")
            cached_data[trial.noise_seed] = data
            data_hashes[str(trial.noise_seed)] = _dataset_digest(data)

        data = cached_data[trial.noise_seed]
        is_train = data.split == "train"
        is_validation = data.split == "validation"
        x_train, y_train = data.x[is_train], data.y_observed[is_train]
        x_val, y_val = data.x[is_validation], data.y_observed[is_validation]

        result = None
        error = None
        try:
            result = gradient_descent(
                trial.theta0, x_train, y_train,
                learning_rate=trial.learning_rate,
                objective_fn=objective_fn,
                gradient_fn=gradient_fn,
                **stopping,
            )
            reason = result.reason
            if reason not in {"gradient_tolerance", "max_iterations", "numerical_failure"}:
                raise ValueError(f"Unexpected GD termination: {reason}")
            if reason != "numerical_failure":
                validation_mse = objective_fn(result.theta, x_val, y_val)
                if not isfinite(validation_mse) or validation_mse < 0:
                    raise FloatingPointError("Nonfinite validation loss.")
            else:
                validation_mse = None
        except (FloatingPointError, OverflowError) as exc:
            reason = "numerical_failure"
            validation_mse = None
            error = f"{type(exc).__name__}: {exc}"

        outcomes.append(CalibrationResult(trial, reason, validation_mse))
        raw_records.append({
            "model": trial.model,
            "dataset_id": trial.dataset_id,
            "noise_seed": trial.noise_seed,
            "split_seed": trial.split_seed,
            "start_id": trial.start_id,
            "initial_theta": list(trial.theta0),
            "learning_rate": trial.learning_rate,
            "termination_reason": reason,
            "solver_reason": result.reason if result is not None else None,
            "converged": bool(result.converged) if result is not None else False,
            "iterations": result.iterations if result is not None else 0,
            "initial_train_mse": result.loss_history[0] if result is not None else None,
            "final_train_mse": result.loss_history[-1] if result is not None else None,
            "final_gradient_norm": (result.gradient_norm_history[-1]
                                    if result is not None else None),
            "validation_mse": validation_mse,
            "theta_final": result.theta.tolist() if result is not None else None,
            "optimizer_elapsed_seconds": result.elapsed_seconds if result is not None else None,
            "objective_evaluations": result.objective_evaluations if result is not None else 0,
            "gradient_evaluations": result.gradient_evaluations if result is not None else 0,
            "error": error,
        })

    chosen_rate, scores = select_gd_learning_rate(plan, model, outcomes)
    return {
        "model": model,
        "status": "calibration_only_no_test_or_evaluation_data",
        "chosen_learning_rate": chosen_rate,
        "registered_conditions": len(trials),
        "dataset_hashes_by_noise_seed": data_hashes,
        "model_calibration_elapsed_seconds": perf_counter() - started,
        "rate_scores": [
            {**asdict(score),
             "mean_validation_mse": score.mean_validation_mse
             if isfinite(score.mean_validation_mse) else None}
            for score in scores
        ],
        "runs": raw_records,
    }


def canonical_json_hash(value: Any) -> str:
    """Hash exactly the experiment settings used for this calibration."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
