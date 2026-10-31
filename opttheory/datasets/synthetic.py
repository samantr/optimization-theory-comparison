"""Synthetic observations for Model 1, with separate random streams."""

from dataclasses import dataclass
from math import log
from typing import Any, Mapping

import numpy as np

from opttheory.models.exponential import predict


@dataclass(frozen=True)
class SyntheticDataset:
    x: np.ndarray
    y_clean: np.ndarray
    noise: np.ndarray
    y_observed: np.ndarray
    split: np.ndarray
    true_parameters: dict[str, float]

    def split_counts(self) -> dict[str, int]:
        return {
            name: int(np.count_nonzero(self.split == name))
            for name in ("train", "validation", "test")
        }


def generate_model1(config: Mapping[str, Any]) -> SyntheticDataset:
    """Generate one fixed Model 1 dataset, independently of any optimizer.

    Important: call this once and reuse the returned dataset for both methods.
    A distinct noise seed creates a different experimental dataset.
    """
    n = int(config["n_observations"])
    x_min = float(config["x_min"])
    x_max = float(config["x_max"])
    if n < 3 or not x_min < x_max:
        raise ValueError("Need at least 3 observations and x_min < x_max.")

    truth = config["true_parameters"]
    a = float(truth["a"])
    half_life = float(truth["half_life"])
    c = float(truth["c"])
    if half_life <= 0:
        raise ValueError("The exponential decay half-life must be positive.")
    b = -log(2.0) / half_life

    noise_options = config["noise"]
    sigma = float(noise_options["sigma"])
    if sigma < 0:
        raise ValueError("Noise standard deviation cannot be negative.")

    split_options = config["split"]
    train_fraction = float(split_options["train_fraction"])
    validation_fraction = float(split_options["validation_fraction"])
    test_fraction = 1.0 - train_fraction - validation_fraction
    if min(train_fraction, validation_fraction, test_fraction) <= 0:
        raise ValueError("Training, validation, and test fractions must be positive.")

    n_train = round(n * train_fraction)
    n_validation = round(n * validation_fraction)
    n_test = n - n_train - n_validation
    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("Each split must contain at least one observation.")

    x = np.linspace(x_min, x_max, n, dtype=np.float64)
    y_clean = predict(x, a=a, b=b, c=c)

    # Separate generators let us change noise seeds without changing the split.
    noise_rng = np.random.default_rng(int(noise_options["seed"]))
    epsilon = noise_rng.normal(loc=0.0, scale=sigma, size=n)
    y_observed = y_clean + epsilon

    split_rng = np.random.default_rng(int(split_options["seed"]))
    shuffled_indices = split_rng.permutation(n)
    labels = np.empty(n, dtype="<U10")
    labels[shuffled_indices[:n_train]] = "train"
    labels[shuffled_indices[n_train:n_train + n_validation]] = "validation"
    labels[shuffled_indices[n_train + n_validation:]] = "test"

    return SyntheticDataset(
        x=x,
        y_clean=y_clean,
        noise=epsilon,
        y_observed=y_observed,
        split=labels,
        true_parameters={"a": a, "b": b, "c": c},
    )

