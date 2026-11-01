"""Preview Model 2 matched trials, without running any optimizer.

Run from repository root:
    python -m scripts.preview_model2_plan
"""

import json
from pathlib import Path

from opttheory.experiments import build_model2_plan


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = json.loads((ROOT / "config" / "experiment_plan.json").read_text(encoding="utf-8"))
    runs = build_model2_plan(config)
    print(f"Model 2 planned conditions: {len({run.pairing_key for run in runs})}")
    print(f"Model 2 planned method runs: {len(runs)}")
    print("First paired condition:")
    for row in runs[:3]:
        print(f"  {row.dataset_id}/{row.start_id}: {row.method}, "
              f"noise_seed={row.noise_seed}, split_seed={row.split_seed}, "
              f"theta0={row.theta0}")
    print("Only a manifest preview. No fitting or evaluation occurred.")


if __name__ == "__main__":
    main()
