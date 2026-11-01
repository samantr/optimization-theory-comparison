"""Run the preregistered 240-condition GD learning-rate calibration.

Usage: python -m scripts.calibrate_gd [--output-dir results/calibration]

Only the separate calibration seeds' train/validation observations enter GD
or learning-rate selection. The final evaluation datasets are not generated.
"""

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from opttheory.experiments.run_calibration import canonical_json_hash, run_model_calibration


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "exponential_decay": ("model1", "model1.json"),
    "logistic": ("model2", "model2.json"),
}
HASHED_SOURCE_FILES = (
    "scripts/calibrate_gd.py",
    "opttheory/experiments/run_calibration.py",
    "opttheory/experiments/calibration.py",
    "opttheory/experiments/plan.py",
    "opttheory/experiments/policy.py",
    "opttheory/datasets/synthetic.py",
    "opttheory/optimizers/gradient_descent.py",
    "opttheory/objectives/mse.py",
    "opttheory/objectives/logistic_mse.py",
)


def _source_hashes() -> dict[str, str]:
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in HASHED_SOURCE_FILES
    }


def _write_runs_csv(path: Path, runs: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
        writer.writeheader()
        for run in runs:
            writer.writerow({
                key: json.dumps(value, allow_nan=False) if isinstance(value, list)
                else value for key, value in run.items()
            })


def _write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True,
                  allow_nan=False)
        handle.write("\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/calibration")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly permit replacing existing calibration artifacts")
    parser.add_argument("--source-commit", default=None,
                        help="Frozen GitHub commit for the calibration implementation")
    parser.add_argument("--model", choices=[*MODELS, "both"], default="both")
    args = parser.parse_args(argv)

    plan = json.loads((ROOT / "config/experiment_plan.json").read_text(encoding="utf-8"))
    selected = list(MODELS) if args.model == "both" else [args.model]
    out = args.output_dir.resolve()
    targets = [out / f"{MODELS[model][0]}_{suffix}"
               for model in selected for suffix in ("runs.csv", "summary.json")]
    existing = [path for path in targets if path.exists()]
    if existing and not args.overwrite:
        parser.error("Calibration artifacts already exist; use --overwrite explicitly: "
                     + ", ".join(str(p) for p in existing))

    out.mkdir(parents=True, exist_ok=True)
    sources = _source_hashes()
    started = perf_counter()
    for model in selected:
        prefix, prototype_name = MODELS[model]
        prototype = json.loads((ROOT / "config" / prototype_name).read_text(encoding="utf-8"))
        print(f"Starting {prefix}: 3 seeds × 5 starts × 8 rates = 120 runs", flush=True)
        result = run_model_calibration(plan, model, prototype)
        runs = result.pop("runs")
        summary = {
            **result,
            "calibration_plan_sha256": canonical_json_hash(plan),
            "prototype_sha256": canonical_json_hash(prototype),
            "source_file_sha256": sources,
            "source_commit": args.source_commit,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "selection_uses": "train_for_fitting_validation_for_ranking_test_excluded",
        }
        _write_runs_csv(out / f"{prefix}_runs.csv", runs)
        _write_json(out / f"{prefix}_summary.json", summary)
        print(f"Finished {prefix}: rate={result['chosen_learning_rate']:.4g}, "
              f"time={result['model_calibration_elapsed_seconds']:.2f}s, "
              f"conditions={len(runs)}", flush=True)

    print(f"Calibration completed: {len(selected) * 120} attempts; "
          f"total elapsed {perf_counter() - started:.2f}s. Outputs in {out}",
          flush=True)


if __name__ == "__main__":
    main()
