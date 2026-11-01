"""Inspect frozen GD candidate rates/seeds without running calibration.

From the repository root:
    python -m scripts.preview_gd_calibration_plan
"""

import json
from pathlib import Path

from opttheory.experiments import build_gd_calibration_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "config" / "experiment_plan.json").read_text(encoding="utf-8"))
    for model in config["model_families"]:
        trials = build_gd_calibration_plan(config, model)
        print(f"{model}: {len(trials)} planned GD calibration attempts")
        print(f"  Calibrated on seeds: {sorted({trial.noise_seed for trial in trials})}")
        print(f"  Candidate learning rates: {sorted({trial.learning_rate for trial in trials})}")
        first = trials[0]
        print(f"  First: {first.dataset_id}/{first.start_id}, "
              f"theta0={first.theta0}, rate={first.learning_rate}")
    print("Manifest preview only: no learning rates chosen and no optimizers run.")


if __name__ == "__main__":
    main()
