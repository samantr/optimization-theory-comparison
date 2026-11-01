"""Preview or explicitly execute the approved GD-vs-Newton study.

Default is PREVIEW ONLY: no data generation, optimization, or test access.
Actual final evaluation requires --execute, after human review/approval.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from opttheory.experiments.analysis import (
    bootstrap_dataset_time_difference, summarize_paired_primary,
)
from opttheory.experiments.evaluation import (
    run_full_model_evaluation, validate_evaluation_protocol,
)
from opttheory.experiments.run_calibration import canonical_json_hash


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIGS = {"exponential_decay": "model1", "logistic": "model2"}
SOURCE_FILES = (
    "scripts/run_comparison.py",
    "opttheory/experiments/analysis.py",
    "opttheory/experiments/evaluation.py",
    "opttheory/experiments/plan.py",
    "opttheory/experiments/policy.py",
    "opttheory/datasets/synthetic.py",
    "opttheory/optimizers/gradient_descent.py",
    "opttheory/optimizers/newton.py",
    "opttheory/optimizers/newton_safeguarded.py",
    "opttheory/optimizers/common.py",
    "opttheory/objectives/mse.py",
    "opttheory/objectives/logistic_mse.py",
    "opttheory/models/exponential.py",
    "opttheory/models/logistic.py",
)


def _csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, allow_nan=False) if isinstance(v, list)
                             else v for k, v in row.items()})


def _json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True,
                  ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                        help="RUN the full final evaluation, including held-out test scoring")
    parser.add_argument("--model", choices=[*MODEL_CONFIGS, "both"], default="both")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/evaluation")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit replacing existing final evaluation artifacts")
    args = parser.parse_args(argv)

    plan = json.loads((ROOT / "config/experiment_plan.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "config/selected_gd_rates.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "config/evaluation_protocol.json").read_text(encoding="utf-8"))
    manifests = validate_evaluation_protocol(plan, frozen, protocol)
    selected = list(MODEL_CONFIGS) if args.model == "both" else [args.model]

    print("PREREGISTERED PAIRED EXPERIMENT PREVIEW:")
    for model in selected:
        print(f"  {model}: {json.dumps(manifests[model], sort_keys=True)}")
    if not args.execute:
        print("PREVIEW ONLY. No datasets were generated, no optimizers were run, "
              "and no held-out labels were accessed.")
        print("Final evaluation is gated by --execute after methodological review.")
        return

    output = args.output_dir.resolve()
    filenames = ("runs.csv", "timings.csv", "summary.json")
    targets = [output / f"{MODEL_CONFIGS[model]}_{suffix}"
               for model in selected for suffix in filenames]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.overwrite:
        parser.error("Evaluation artifacts exist; pass --overwrite to replace: "
                     + ", ".join(existing))

    output.mkdir(parents=True, exist_ok=True)
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
              for path in SOURCE_FILES}
    for model in selected:
        prefix = MODEL_CONFIGS[model]
        prototype = json.loads((ROOT / f"config/{prefix}.json").read_text(encoding="utf-8"))
        print(f"Starting {prefix}: full registered paired evaluation", flush=True)
        result = run_full_model_evaluation(model, plan, frozen, protocol, prototype)
        rows = result["run_records"]
        timings = result["timing_records"]
        descriptive = summarize_paired_primary(
            rows, expected_starts=plan["starting_points_per_dataset"],
            expected_datasets=plan["dataset_seeds_per_model"],
        )
        interval = bootstrap_dataset_time_difference(model, descriptive[model], protocol)
        summary = {
            "model": model,
            "method_conditions": len(rows),
            "timed_invocations": len(timings),
            "dataset_sha256_by_noise_seed": result["dataset_sha256_by_noise_seed"],
            "dataset_aware_primary_summary": descriptive[model],
            "paired_time_difference_95pct_cluster_bootstrap": interval,
            "experiment_plan_sha256": canonical_json_hash(plan),
            "frozen_rates_sha256": canonical_json_hash(frozen),
            "evaluation_protocol_sha256": canonical_json_hash(protocol),
            "model_config_sha256": canonical_json_hash(prototype),
            "source_file_sha256": hashes,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "calibration_is_excluded_from_timing": True,
            "inference_status": "approved_dataset_cluster_bootstrap_95pct_ci_no_p_value",
        }
        _csv(output / f"{prefix}_runs.csv", rows)
        _csv(output / f"{prefix}_timings.csv", timings)
        _json(output / f"{prefix}_summary.json", summary)
        print(f"Finished {prefix}: {len(rows)} conditions, "
              f"{len(timings)} timing measurements", flush=True)
    print(f"Output directory: {output}")


if __name__ == "__main__":
    main()
