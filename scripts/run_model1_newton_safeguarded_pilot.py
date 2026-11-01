"""Run a reproducible training-only *safeguarded* Newton pilot.

From the repository root:
    python -m scripts.run_model1_newton_safeguarded_pilot

This is a separate, explicitly hybrid variant, NOT pure full-step Newton.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from opttheory.datasets.synthetic import generate_model1
from opttheory.objectives.mse import hessian
from opttheory.optimizers import safeguarded_newton


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model1.json")
    parser.add_argument("--theta0", nargs=3, type=float, default=[3.0, -0.5, 1.0],
                        metavar=("A", "B", "C"))
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    parser.add_argument("--condition-limit", type=float, default=1e12)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = generate_model1(config)
    train = data.split == "train"
    result = safeguarded_newton(
        args.theta0, data.x[train], data.y_observed[train],
        max_iterations=args.max_iterations,
        gradient_tolerance=args.gradient_tolerance,
        condition_limit=args.condition_limit,
    )
    eigenvalues = np.linalg.eigvalsh(hessian(result.theta,
                                            data.x[train], data.y_observed[train]))

    print("Safeguarded Newton pilot (training subset only; separate hybrid variant)")
    print(f"Start theta: {args.theta0}")
    print(f"Final theta: {result.theta.tolist()}")
    print(f"Train MSE: {result.loss_history[0]:.9g} -> {result.loss_history[-1]:.9g}")
    print(f"Gradient norm: {result.gradient_norm_history[-1]:.9g}")
    print(f"Iterations: {result.iterations} | Status: {result.reason}")
    print(f"Gradient fallback steps: {result.gradient_fallback_steps}")
    print(f"Line-search backtracks: {result.line_search_backtracks}")
    print(f"Objective evaluations: {result.objective_evaluations}")
    print(f"Final Hessian eigenvalues: {eigenvalues.tolist()}")
    print(f"Elapsed: {result.elapsed_seconds:.6f} seconds (single pilot run)")


if __name__ == "__main__":
    main()
