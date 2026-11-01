"""Run a reproducible *training-only* Model 1 Gradient Descent pilot.

From the repository root:
    python -m scripts.run_model1_gd_pilot

This is a teaching sanity check, not a paired GD-vs-Newton experiment.
"""

import argparse
import json
from pathlib import Path

from opttheory.datasets.synthetic import generate_model1
from opttheory.optimizers import gradient_descent


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model1.json")
    parser.add_argument("--theta0", nargs=3, type=float, default=[3.0, -0.5, 1.0],
                        metavar=("A", "B", "C"), help="Candidate starting parameters.")
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-iterations", type=int, default=10_000)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    data = generate_model1(config)
    train = data.split == "train"
    result = gradient_descent(
        args.theta0, data.x[train], data.y_observed[train],
        learning_rate=args.learning_rate,
        max_iterations=args.max_iterations,
        gradient_tolerance=args.gradient_tolerance,
    )

    print("Model 1 Gradient Descent pilot (training subset only; not final results)")
    print(f"Start theta: {args.theta0}")
    print(f"Final theta: {result.theta.tolist()}")
    print(f"Train MSE: {result.loss_history[0]:.9g} -> {result.loss_history[-1]:.9g}")
    print(f"Gradient norm: {result.gradient_norm_history[-1]:.9g}")
    print(f"Iterations: {result.iterations} | Status: {result.reason}")
    print(f"Elapsed: {result.elapsed_seconds:.6f} seconds (single pilot run)")


if __name__ == "__main__":
    main()
