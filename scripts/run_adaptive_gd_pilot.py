"""Separate adaptive-GD pilot on NON-evaluation seeds; no test data accessed.

Use `python -m scripts.run_adaptive_gd_pilot` to preview the 30 runs.
Use `python -m scripts.run_adaptive_gd_pilot --execute` to generate local CSV.
This pilot is descriptive, not a preregistered extension or a timing study.
"""

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np

from opttheory.datasets.synthetic import generate_model1, generate_model2
from opttheory.objectives import mse, logistic_mse
from opttheory.optimizers import gradient_descent, adaptive_gradient_descent


ROOT = Path(__file__).resolve().parents[1]
PILOT_SEEDS = {"exponential_decay": 3000, "logistic": 3100}
VARIANTS = ("fixed_gd", "gradient_reversal", "backtracking")


def execute_pilot():
    plan = json.loads((ROOT / "config/experiment_plan.json").read_text())
    frozen = json.loads((ROOT / "config/selected_gd_rates.json").read_text())
    model_settings = {
        "exponential_decay": ("model1.json", generate_model1, mse, 1),
        "logistic": ("model2.json", generate_model2, logistic_mse, 2),
    }
    rows = []
    for model, (config_name, generator, functions, model_number) in model_settings.items():
        config = deepcopy(json.loads((ROOT / "config" / config_name).read_text()))
        config["noise"]["seed"] = PILOT_SEEDS[model]
        data = generator(config)
        train = data.split == "train"
        validation = data.split == "validation"
        x_train, y_train = data.x[train], data.y_observed[train]
        x_valid, y_valid = data.x[validation], data.y_observed[validation]
        for start in plan[f"model{model_number}_starting_points"]:
            for variant in VARIANTS:
                kwargs = dict(objective_fn=functions.objective,
                              gradient_fn=functions.gradient,
                              gradient_tolerance=1e-4,
                              max_iterations=10_000)
                if variant == "fixed_gd":
                    result = gradient_descent(
                        start["theta"], x_train, y_train,
                        learning_rate=frozen["learning_rates_by_model"][model],
                        **kwargs,
                    )
                else:
                    result = adaptive_gradient_descent(
                        start["theta"], x_train, y_train,
                        strategy=variant, initial_learning_rate=1.0, **kwargs,
                    )
                rows.append({
                    "model": model, "noise_seed": PILOT_SEEDS[model],
                    "start": start["id"], "method": variant,
                    "reason": result.reason, "converged": result.converged,
                    "iterations": result.iterations,
                    "objective_evaluations": result.objective_evaluations,
                    "gradient_evaluations": result.gradient_evaluations,
                    "rejected_trials": result.line_search_backtracks,
                    "initial_rate": (frozen["learning_rates_by_model"][model]
                                     if variant == "fixed_gd" else 1.0),
                    "last_accepted_rate": (result.step_sizes[-1]
                                           if result.step_sizes else ""),
                    "train_mse": result.loss_history[-1],
                    "validation_mse": functions.objective(result.theta, x_valid, y_valid),
                })
    out = ROOT / "results" / "adaptive_gd_pilot.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Pilot complete: {len(rows)} paired model/start/method runs")
    print("All three methods used the same training observations and start within each condition.")
    print("Only training and validation labels were accessed; no evaluation seeds or test labels.")
    print(f"Results: {out}")
    for model in model_settings:
        for method in VARIANTS:
            selected = [r for r in rows if r["model"] == model and r["method"] == method]
            converged = sum(r["converged"] for r in selected)
            med_mse = float(np.median([r["validation_mse"] for r in selected]))
            med_iterations = float(np.median([r["iterations"] for r in selected]))
            med_objectives = float(np.median([r["objective_evaluations"] for r in selected]))
            print(f"{model:18s} {method:20s} gradient_tol={converged}/5 "
                  f"median_val_mse={med_mse:.6g} median_updates={med_iterations:.0f} "
                  f"median_MSE_evals={med_objectives:.0f}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the separate adaptive GD pilot")
    args = parser.parse_args()
    if args.execute:
        execute_pilot()
    else:
        print("Dry-run: 2 models x 1 NEW noise seed x 5 original starts x 3 GD methods = 30 runs.")
        print("Pilot-only seeds: Model 1=3000, Model 2=3100 (no frozen evaluation/test data).")
        print("Original fixed GD rate vs two variants initialized at alpha=1.0.")
        print("Add --execute to run and save results/adaptive_gd_pilot.csv.")


if __name__ == "__main__":
    main()
