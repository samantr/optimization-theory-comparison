"""From-scratch optimizers for the paired nonlinear regression experiments.

Gradient Descent is implemented; exact Newton will be added separately.
"""

from opttheory.optimizers.gradient_descent import gradient_descent

__all__ = ["gradient_descent"]

