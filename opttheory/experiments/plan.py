"""Enumerate *planned* paired trials without drawing data or fitting models.

Dataset IDs D01..D10 are placeholders until an independent noise/split seed
schedule is approved. This module never chooses random seeds or runs solvers.
"""

from dataclasses import dataclass
from typing import Mapping, Any


@dataclass(frozen=True)
class PlannedRun:
    model: str
    dataset_id: str
    start_id: str
    theta0: tuple[float, ...]
    method: str

    @property
    def pairing_key(self) -> tuple[str, str, str]:
        """All methods sharing this key must use the same data and theta0."""
        return self.model, self.dataset_id, self.start_id


def build_model1_plan(config: Mapping[str, Any]) -> tuple[PlannedRun, ...]:
    """Enumerate 10 datasets × 5 fixed starts × 3 labelled methods (150 rows).

    This is only a *manifest skeleton*. It intentionally contains no seeds,
    observations, runtimes, MSE values, or fabricated experimental results.
    """
    n_datasets = int(config["dataset_seeds_per_model"])
    n_starts = int(config["starting_points_per_dataset"])
    starts = config["model1_starting_points"]
    methods = list(config["primary_methods"]) + list(config["exploratory_methods"])
    if n_datasets <= 0 or n_starts <= 0 or len(starts) != n_starts:
        raise ValueError("Dataset count or starting-vector configuration is inconsistent.")
    if len(methods) != len(set(methods)) or not methods:
        raise ValueError("Method identifiers must be distinct and nonempty.")
    if len({item["id"] for item in starts}) != n_starts:
        raise ValueError("Starting-point identifiers must be distinct.")

    return tuple(
        PlannedRun(
            model="exponential_decay",
            dataset_id=f"D{dataset_number:02d}",
            start_id=entry["id"],
            theta0=tuple(float(value) for value in entry["theta"]),
            method=method,
        )
        for dataset_number in range(1, n_datasets + 1)
        for entry in starts
        for method in methods
    )
