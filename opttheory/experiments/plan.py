"""Enumerate planned evaluation and calibration trials without fitting models.

Evaluation dataset IDs D01..D10 and calibration IDs C01..C03 identify
pre-registered noise seeds. No observations are drawn by this module.
"""

from dataclasses import dataclass
from math import isfinite
from typing import Mapping, Any


@dataclass(frozen=True)
class PlannedRun:
    model: str
    dataset_id: str
    noise_seed: int
    split_seed: int
    start_id: str
    theta0: tuple[float, ...]
    method: str

    @property
    def pairing_key(self) -> tuple[str, str, str]:
        """All methods sharing this key must use the same data and theta0."""
        return self.model, self.dataset_id, self.start_id


@dataclass(frozen=True)
class CalibrationTrial:
    """One independent training/validation-only GD rate-selection attempt."""

    model: str
    dataset_id: str
    noise_seed: int
    split_seed: int
    start_id: str
    theta0: tuple[float, ...]
    learning_rate: float
    method: str = "gradient_descent"


def build_gd_calibration_plan(
    config: Mapping[str, Any], model: str
) -> tuple[CalibrationTrial, ...]:
    """Enumerate 3 calibration seeds × 5 starts × 8 rates for one model.

    No fitting occurs. Seeds must be disjoint from the ten evaluation seeds
    and from the other model's calibration seeds to avoid hidden tuning
    on evaluation data. Stopping and failure policies are not yet approved.
    """
    model_numbers = {"exponential_decay": 1, "logistic": 2}
    if model not in config["model_families"] or model not in model_numbers:
        raise ValueError("Unknown or inactive model for GD calibration.")
    model_number = model_numbers[model]
    tuning = config["gradient_descent_learning_rate_tuning"]
    rates = [float(rate) for rate in tuning["candidate_learning_rates"]]
    seeds = tuning["calibration_noise_seeds_by_model"][model]
    expected_datasets = int(tuning["calibration_datasets_per_model"])
    split_seed = int(tuning["split_seed"])
    starts = config[f"model{model_number}_starting_points"]
    parameter_count = len(config[f"model{model_number}_parameter_order"])

    if not rates or len(set(rates)) != len(rates) or not all(
        isfinite(rate) and rate > 0.0 for rate in rates
    ):
        raise ValueError("GD calibration rates must be distinct, positive and finite.")
    if (len(seeds) != expected_datasets or len(set(seeds)) != len(seeds)
            or any(not isinstance(s, int) or isinstance(s, bool) or s < 0
                   for s in seeds)):
        raise ValueError("GD calibration seeds must be distinct nonnegative integers.")

    all_evaluation_seeds = {
        s for group in config["seed_schedule"]["noise_seeds_by_model"].values()
        for s in group
    }
    all_calibration_seeds = [
        s for group in tuning["calibration_noise_seeds_by_model"].values()
        for s in group
    ]
    if (set(seeds) & all_evaluation_seeds
            or len(all_calibration_seeds) != len(set(all_calibration_seeds))):
        raise ValueError("Calibration seeds must be separate from all evaluation seeds.")
    if split_seed != int(config["seed_schedule"]["split_seed"]):
        raise ValueError("Calibration and evaluation must follow the approved split rule.")
    if (len(starts) != config["starting_points_per_dataset"]
            or len({start["id"] for start in starts}) != len(starts)):
        raise ValueError("Calibrate using all approved, distinct starting points.")
    vectors = [tuple(float(value) for value in start["theta"]) for start in starts]
    if (any(len(v) != parameter_count or not all(isfinite(value) for value in v)
            for v in vectors) or len(set(vectors)) != len(vectors)):
        raise ValueError("All calibration starts must be valid approved vectors.")

    return tuple(
        CalibrationTrial(
            model=model,
            dataset_id=f"C{index:02d}",
            noise_seed=seed,
            split_seed=split_seed,
            start_id=start["id"],
            theta0=vector,
            learning_rate=rate,
        )
        for index, seed in enumerate(seeds, start=1)
        for start, vector in zip(starts, vectors)
        for rate in rates
    )


def build_model1_plan(config: Mapping[str, Any]) -> tuple[PlannedRun, ...]:
    """Enumerate 10 datasets × 5 fixed starts × 3 labelled methods (150 rows).

    This is only a *manifest*. It contains pre-registered seeds but no
    observations, runtimes, MSE values, or fabricated experimental results.
    """
    return _build_plan(config, model="exponential_decay", model_number=1)


def build_model2_plan(config: Mapping[str, Any]) -> tuple[PlannedRun, ...]:
    """Enumerate Model 2's 10 datasets × 5 approved starts × 3 methods.

    A negative initial k is deliberately retained, even if a method fails.
    This constructs a manifest only; no data or results are produced.
    """
    return _build_plan(config, model="logistic", model_number=2)


def _build_plan(
    config: Mapping[str, Any], *, model: str, model_number: int
) -> tuple[PlannedRun, ...]:
    n_datasets = int(config["dataset_seeds_per_model"])
    n_starts = int(config["starting_points_per_dataset"])
    starts = config[f"model{model_number}_starting_points"]
    n_params = len(config[f"model{model_number}_parameter_order"])
    seeds = config["seed_schedule"]["noise_seeds_by_model"][model]
    split_seed = int(config["seed_schedule"]["split_seed"])
    methods = list(config["primary_methods"]) + list(config["exploratory_methods"])
    if n_datasets <= 0 or n_starts <= 0 or len(starts) != n_starts:
        raise ValueError("Dataset count or starting-vector configuration is inconsistent.")
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("Method identifiers must be distinct and nonempty.")
    if len({item["id"] for item in starts}) != n_starts:
        raise ValueError("Starting-point identifiers must be distinct.")
    vectors = [tuple(float(value) for value in entry["theta"]) for entry in starts]
    if (any(len(v) != n_params or not all(isfinite(value) for value in v)
            for v in vectors) or len(set(vectors)) != n_starts):
        raise ValueError("Starting vectors must be finite, distinct, and match parameter count.")
    if (len(seeds) != n_datasets or len(set(seeds)) != n_datasets
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                   for seed in seeds)):
        raise ValueError("Need exactly one distinct nonnegative integer noise seed per dataset.")
    if split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer.")

    return tuple(
        PlannedRun(
            model=model,
            dataset_id=f"D{dataset_number:02d}",
            noise_seed=seeds[dataset_number - 1],
            split_seed=split_seed,
            start_id=entry["id"],
            theta0=tuple(float(value) for value in entry["theta"]),
            method=method,
        )
        for dataset_number in range(1, n_datasets + 1)
        for entry in starts
        for method in methods
    )
