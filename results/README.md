# Generated outputs

Run the dataset generator from the repository root. It writes CSV, JSON
metadata, and a preview figure under `results/model1/` or `results/model2/`.

Other generated artifacts:

- `results/calibration/`: 240 actually executed GD calibration attempts,
  with raw CSV results and JSON reproducibility metadata. Frozen selected
  rates and checksums are **tracked** in `config/selected_gd_rates.json`.
- `results/evaluation/`: reserved for paired optimizer records, five raw
  timing repetitions per condition, dataset-aware descriptive summaries,
  and 95% cluster-bootstrap intervals for mean paired timing differences.
  The first fully approved evaluation has been executed and **archived**
  under tracked `evidence/evaluation_2026-11-01/`. The safe default command
  without `--execute` still previews the plan only. Re-execution should be
  documented as a separately labelled replication, not an overwrite.

Outputs are reproducible from the frozen configurations and seeds. Do not
commit generated output by accident; preserve important finalized experiments
in an explicitly named archive and record their hashes in the report.

