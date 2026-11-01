"""Seeded synthetic regression observations, with separate noise/split streams."""

from dataclasses import dataclass
from math import log
from typing import Any, Mapping

import numpy as np

from opttheory.models.exponential import predict as exponential_predict
from opttheory.models.logistic import predict as logistic_predict


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


def _input_grid(config: Mapping[str, Any]) -> np.ndarray:
    n = int(config["n_observations"])
    x_min = float(config["x_min"])
    x_max = float(config["x_max"])
    if n < 3 or not (np.isfinite(x_min) and np.isfinite(x_max) and x_min < x_max):
        raise ValueError("Need at least 3 observations and x_min < x_max.")
    return np.linspace(x_min, x_max, n, dtype=np.float64)


def _add_noise_and_split(
    x: np.ndarray,
    y_clean: np.ndarray,
    true_parameters: dict[str, float],
    config: Mapping[str, Any],
) -> SyntheticDataset:
    """Reuse identical seeded-noise/split rules for every regression model."""
    n = len(x)

    noise_options = config["noise"]
    sigma = float(noise_options["sigma"])
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Noise standard deviation must be finite and nonnegative.")

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
        true_parameters=true_parameters,
    )


def generate_model1(config: Mapping[str, Any]) -> SyntheticDataset:
    """Generate one fixed Model 1 dataset, independently of any optimizer.

    Important: call this once and reuse the returned dataset for both methods.
    A distinct noise seed creates a different experimental dataset.
    """
    x = _input_grid(config)

    truth = config["true_parameters"]
    a = float(truth["a"])
    half_life = float(truth["half_life"])
    c = float(truth["c"])
    if half_life <= 0:
        raise ValueError("The exponential decay half-life must be positive.")
    b = -log(2.0) / half_life

    y_clean = exponential_predict(x, a=a, b=b, c=c)
    return _add_noise_and_split(x, y_clean, {"a": a, "b": b, "c": c}, config)


def generate_model2(config: Mapping[str, Any]) -> SyntheticDataset:
    """Generate one reproducible S-curve dataset with the approved protocol.

    True parameters are used *only* to create synthetic observations. All
    future regression fits must estimate them from training inputs/targets.
    """
    x = _input_grid(config)
    truth = config["true_parameters"]
    L = float(truth["L"])
    k = float(truth["k"])
    x0 = float(truth["x0"])
    if not np.all(np.isfinite([L, k, x0])) or L <= 0 or k <= 0:
        raise ValueError("Model 2 truth must have finite L>0, k>0, and x0.")
    if not x[0] < x0 < x[-1]:
        raise ValueError("Model 2 midpoint x0 must lie inside the input range.")

    y_clean = logistic_predict(x, L=L, k=k, x0=x0)
    return _add_noise_and_split(x, y_clean, {"L": L, "k": k, "x0": x0}, config)

