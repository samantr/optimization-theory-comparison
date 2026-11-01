"""Enumerate *planned* paired trials without drawing data or fitting models.

Dataset IDs D01..D10 identify the pre-registered noise seeds. This module
never draws observations or runs optimizers.
"""

from dataclasses import dataclass
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


def build_model1_plan(config: Mapping[str, Any]) -> tuple[PlannedRun, ...]:
    """Enumerate 10 datasets × 5 fixed starts × 3 labelled methods (150 rows).

    This is only a *manifest*. It contains pre-registered seeds but no
    observations, runtimes, MSE values, or fabricated experimental results.
    """
    n_datasets = int(config["dataset_seeds_per_model"])
    n_starts = int(config["starting_points_per_dataset"])
    starts = config["model1_starting_points"]
    seeds = config["seed_schedule"]["noise_seeds_by_model"]["exponential_decay"]
    split_seed = int(config["seed_schedule"]["split_seed"])
    methods = list(config["primary_methods"]) + list(config["exploratory_methods"])
    if n_datasets <= 0 or n_starts <= 0 or len(starts) != n_starts:
        raise ValueError("Dataset count or starting-vector configuration is inconsistent.")
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("Method identifiers must be distinct and nonempty.")
    if len({item["id"] for item in starts}) != n_starts:
        raise ValueError("Starting-point identifiers must be distinct.")
    if (len(seeds) != n_datasets or len(set(seeds)) != n_datasets
            or any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
                   for seed in seeds)):
        raise ValueError("Need exactly one distinct nonnegative integer noise seed per dataset.")
    if split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer.")

    return tuple(
        PlannedRun(
            model="exponential_decay",
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
