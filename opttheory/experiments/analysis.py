"""Dataset-aware paired summaries and a preregistered cluster bootstrap.

The ten independent noise datasets (NOT their 50 starting-point trials)
are resampled to obtain a cautious uncertainty interval for mean timing.
"""

from collections import defaultdict
from math import isfinite
from statistics import mean, median
from typing import Any, Mapping, Sequence

import numpy as np


def summarize_paired_primary(
    rows: Sequence[Mapping[str, Any]], *, expected_starts: int = 5,
    expected_datasets: int | None = 10,
) -> dict[str, Any]:
    """Average five *paired* start-wise time differences within each seed.

    Time difference is Newton minus GD: negative means Newton was faster.
    Failures still contribute wall-clock time and convergence denominators;
    no unsuccessful or cap-exhausted condition is silently removed.
    """
    primary = [r for r in rows if r["method"] in {"gradient_descent", "newton"}]
    if not primary:
        raise ValueError("Paired primary method records are required.")
    pairs = defaultdict(dict)
    for row in primary:
        key = (row["model"], row["dataset_id"], row["start_id"])
        method = row["method"]
        if method in pairs[key]:
            raise ValueError("Duplicate optimizer record in a paired condition.")
        runtime = float(row["median_optimizer_seconds"])
        if not isfinite(runtime) or runtime < 0:
            raise ValueError("Execution time must be finite and nonnegative.")
        pairs[key][method] = row

    datasets = defaultdict(list)
    dataset_identifiers = {}
    for key, methods in pairs.items():
        if set(methods) != {"gradient_descent", "newton"}:
            raise ValueError(f"A matched primary run is missing for {key}.")
        gd, nt = methods["gradient_descent"], methods["newton"]
        if (gd["noise_seed"] != nt["noise_seed"]
                or gd["split_seed"] != nt["split_seed"]
                or gd["theta_initial"] != nt["theta_initial"]):
            raise ValueError("Methods must share noise, split seed and initial theta.")
        ds = key[:2]
        seed_identity = (gd["noise_seed"], gd["split_seed"])
        if ds in dataset_identifiers and dataset_identifiers[ds] != seed_identity:
            raise ValueError("All starts in one dataset must share noise/split seeds.")
        dataset_identifiers[ds] = seed_identity
        datasets[key[:2]].append({
            "start_id": key[2],
            "newton_minus_gd_seconds": nt["median_optimizer_seconds"] - gd["median_optimizer_seconds"],
            "gd_stationary": bool(gd["stationary_at_tolerance"]),
            "newton_stationary": bool(nt["stationary_at_tolerance"]),
        })

    model_datasets = defaultdict(list)
    for (model, dataset_id), starts in sorted(datasets.items()):
        if len(starts) != expected_starts or len({x["start_id"] for x in starts}) != expected_starts:
            raise ValueError("Each dataset must contribute every approved distinct start.")
        model_datasets[model].append({
            "dataset_id": dataset_id,
            "paired_starts": len(starts),
            "mean_newton_minus_gd_seconds": mean(x["newton_minus_gd_seconds"] for x in starts),
            "gd_stationary_count": sum(x["gd_stationary"] for x in starts),
            "newton_stationary_count": sum(x["newton_stationary"] for x in starts),
        })

    model_summaries = {}
    for model, records in model_datasets.items():
        if expected_datasets is not None and len(records) != expected_datasets:
            raise ValueError("Expected all registered independent dataset seeds.")
        values = [r["mean_newton_minus_gd_seconds"] for r in records]
        n_conditions = len(records) * expected_starts
        model_summaries[model] = {
            "independent_dataset_count": len(records),
            "paired_condition_count": n_conditions,
            "dataset_mean_newton_minus_gd_seconds": mean(values),
            "median_of_dataset_time_differences_seconds": median(values),
            "gd_stationary_fraction": sum(r["gd_stationary_count"] for r in records) / n_conditions,
            "newton_stationary_fraction": sum(r["newton_stationary_count"] for r in records) / n_conditions,
            "dataset_records": records,
            "confidence_intervals": "computed_separately_by_cluster_bootstrap",
        }
    return model_summaries


def bootstrap_dataset_time_difference(
    model: str, model_summary: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Percentile 95% CI for the *mean of dataset-level paired differences*.

    Each resampling draw chooses whole dataset means with replacement.
    All five paired starts were already combined inside each dataset mean;
    they are never incorrectly treated as fifty independent replicates.
    The seed stream is fixed independently for each model, regardless of
    which model gets processed first. This method supplies no p-value.
    """
    settings = protocol["analysis"]
    boot = settings["bootstrap"]
    if settings["confidence_interval_method"] != "paired_dataset_cluster_percentile_bootstrap":
        raise ValueError("Unexpected unregistered confidence interval method.")
    if boot["resampling_unit"] != "whole_dataset_noise_seed_with_all_five_paired_starts":
        raise ValueError("Bootstrap must resample whole dataset units.")
    if boot["statistic"] != "mean_of_ten_dataset_level_mean_newton_minus_gd_seconds":
        raise ValueError("Bootstrap statistic has changed after registration.")
    if (boot["resamples"] != 10000 or boot["confidence_level"] != 0.95
            or boot["quantile_method"] != "linear"):
        raise ValueError("Approved 10,000-draw 95% linear-percentile bootstrap is required.")
    if (type(boot["base_seed"]) is not int or boot["base_seed"] < 0
            or set(boot["model_seed_offset"]) != {"exponential_decay", "logistic"}
            or any(type(offset) is not int or offset < 0
                   for offset in boot["model_seed_offset"].values())):
        raise ValueError("Bootstrap seed schedule must use fixed nonnegative integers.")
    if model not in boot["model_seed_offset"]:
        raise ValueError("Unregistered model for bootstrap analysis.")

    records = sorted(model_summary["dataset_records"], key=lambda r: r["dataset_id"])
    n_datasets = len(records)
    if (n_datasets != settings["dataset_seeds_per_model"]
            or model_summary["independent_dataset_count"] != n_datasets
            or len({r["dataset_id"] for r in records}) != n_datasets):
        raise ValueError("Bootstrap requires all distinct independent dataset units.")
    values = np.array([r["mean_newton_minus_gd_seconds"] for r in records],
                      dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Every dataset needs a finite paired timing difference.")
    if not np.isclose(float(np.mean(values)),
                      model_summary["dataset_mean_newton_minus_gd_seconds"],
                      rtol=1e-12, atol=1e-12):
        raise ValueError("Reported dataset-level mean differs from its source rows.")

    rng = np.random.default_rng(np.random.SeedSequence([
        boot["base_seed"], boot["model_seed_offset"][model],
    ]))
    choices = rng.integers(0, n_datasets, size=(boot["resamples"], n_datasets))
    replicate_means = values[choices].mean(axis=1)
    alpha = 1.0 - boot["confidence_level"]
    lower, upper = np.quantile(replicate_means, [alpha / 2, 1 - alpha / 2],
                               method=boot["quantile_method"])
    return {
        "method": settings["confidence_interval_method"],
        "statistic": boot["statistic"],
        "point_estimate_seconds": float(np.mean(values)),
        "confidence_level": boot["confidence_level"],
        "lower_seconds": float(lower),
        "upper_seconds": float(upper),
        "independent_dataset_count": n_datasets,
        "starting_points_resampled_as_independent_units": False,
        "resamples": boot["resamples"],
        "base_seed": boot["base_seed"],
        "model_seed_offset": boot["model_seed_offset"][model],
        "quantile_method": boot["quantile_method"],
        "small_cluster_caution": "Only ten independent datasets: percentile CI coverage may be imperfect.",
        "p_value": None,
    }
