"""Generate a reproducible Model 1 CSV, metadata, and preview figure.

Run from the repository root:
    python -m scripts.generate_model1
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from opttheory.datasets.synthetic import generate_model1


ROOT = Path(__file__).resolve().parents[1]


def write_csv(path: Path, dataset) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "x", "y_clean", "noise", "y_observed", "split"])
        for i in range(len(dataset.x)):
            writer.writerow([
                i,
                format(float(dataset.x[i]), ".17g"),
                format(float(dataset.y_clean[i]), ".17g"),
                format(float(dataset.noise[i]), ".17g"),
                format(float(dataset.y_observed[i]), ".17g"),
                str(dataset.split[i]),
            ])


def write_plot(path: Path, dataset, *, title: str = "Model 1: reproducible synthetic observations (pilot)") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(dataset.x, dataset.y_clean, color="#1f4e79", linewidth=2.3,
            label="Known noise-free curve")
    colors = {"train": "#2b8a3e", "validation": "#e67700", "test": "#8d4dc4"}
    for split_name in ("train", "validation", "test"):
        mask = dataset.split == split_name
        ax.scatter(dataset.x[mask], dataset.y_observed[mask], s=18, alpha=0.8,
                   color=colors[split_name], label=split_name.title())
    ax.set(xlabel="Input x", ylabel="Output y",
           title=title)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config" / "model1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "model1")
    parser.add_argument("--no-plot", action="store_true", help="Only export CSV and JSON.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace existing generated files in the output directory.")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    dataset = generate_model1(config)
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "model1_dataset.csv"
    manifest_path = args.output / "model1_metadata.json"
    plot_path = args.output / "model1_preview.png"
    targets = [csv_path, manifest_path]
    if not args.no_plot:
        targets.append(plot_path)
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.overwrite:
        parser.error("Generated files already exist. Use --overwrite only if intended: "
                     + ", ".join(existing))

    write_csv(csv_path, dataset)
    sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    metadata = {
        "status": "pilot_not_final_experiment",
        "model": config["model"],
        "n_observations": len(dataset.x),
        "x_range": [float(dataset.x[0]), float(dataset.x[-1])],
        "true_parameters": dataset.true_parameters,
        "noise": config["noise"],
        "split_config": config["split"],
        "split_counts": dataset.split_counts(),
        "dataset_sha256": sha256,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    manifest_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if not args.no_plot:
        write_plot(plot_path, dataset)

    print(f"Generated {len(dataset.x)} observations: {dataset.split_counts()}")
    print(f"True parameters: {dataset.true_parameters}")
    print(f"Dataset SHA-256: {sha256}")
    print(f"CSV: {csv_path.resolve()}")
    print(f"Metadata: {manifest_path.resolve()}")
    if not args.no_plot:
        print(f"Plot: {plot_path.resolve()}")


if __name__ == "__main__":
    main()

