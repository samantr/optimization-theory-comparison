"""Export the approved Model 2 logistic synthetic dataset and preview.

Run from the repository root:
    python -m scripts.generate_model2

The output uses registered dataset seed 1100; no optimizer is fitted here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from opttheory.datasets.synthetic import generate_model2
from scripts.generate_model1 import write_csv, write_plot


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model2.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model2")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace existing generated outputs.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = generate_model2(config)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "model2_dataset.csv"
    metadata_path = args.output / "model2_metadata.json"
    plot_path = args.output / "model2_preview.png"
    targets = [csv_path, metadata_path] + ([] if args.no_plot else [plot_path])
    existing = [str(path) for path in targets if path.exists()]
    if existing and not args.overwrite:
        parser.error("Generated files already exist; use --overwrite if intended: "
                     + ", ".join(existing))

    write_csv(csv_path, dataset)
    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "status": "model2_dataset_only_no_optimizer_fit",
        "model": "logistic",
        "n_observations": len(dataset.x),
        "x_range": [float(dataset.x[0]), float(dataset.x[-1])],
        "true_parameters": dataset.true_parameters,
        "noise": config["noise"],
        "split_config": config["split"],
        "split_counts": dataset.split_counts(),
        "dataset_sha256": sha256,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        write_plot(plot_path, dataset, title="Model 2: logistic synthetic observations")

    print(f"Generated {len(dataset.x)} observations: {dataset.split_counts()}")
    print(f"True parameters (generation only): {dataset.true_parameters}")
    print(f"Dataset SHA-256: {sha256}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Metadata: {metadata_path.resolve()}")
    if not args.no_plot:
        print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()
