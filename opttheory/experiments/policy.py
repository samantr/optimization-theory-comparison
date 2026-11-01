"""Validate approved stopping settings before any future optimizer run.

Keeping this adapter explicit avoids silently using the historical Model 1
pilot defaults, which have a different gradient-norm tolerance.
"""

from math import isfinite
from typing import Any, Mapping


def optimizer_stopping_kwargs(config: Mapping[str, Any], method: str) -> dict[str, Any]:
    """Return the approved `max_iterations` and `gradient_tolerance` kwargs.

    Iteration limits are safety caps: both optimizers stop early when the
    gradient norm reaches the tolerance. That is stationarity, not
    proof that the final point is a minimum.
    """
    allowed = list(config["primary_methods"]) + list(config["exploratory_methods"])
    if method not in allowed or len(allowed) != len(set(allowed)):
        raise ValueError("Stopping policy requires a registered, distinct method.")

    rules = config["optimizer_stopping"]
    configured = rules["max_iterations_by_method"]
    if set(configured) != set(allowed):
        raise ValueError("Stopping policy must cover exactly all registered methods.")

    tolerance = float(rules["gradient_norm_tolerance"])
    if not isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Shared gradient tolerance must be finite and positive.")

    budget = configured[method]
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError("Maximum iteration count must be a positive integer.")

    return {"max_iterations": budget, "gradient_tolerance": tolerance}
