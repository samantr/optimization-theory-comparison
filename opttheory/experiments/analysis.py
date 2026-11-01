"""Paired dataset-level descriptive summaries, without pseudoreplication.

No confidence intervals or significance tests are claimed here; those
require a separately approved statistical-inference protocol.
"""

from collections import defaultdict
from math import isfinite
from statistics import mean, median
from typing import Any, Mapping, Sequence


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
            "confidence_intervals": "not_computed_inference_protocol_pending",
        }
    return model_summaries
