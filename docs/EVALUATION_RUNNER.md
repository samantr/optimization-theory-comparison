# Paired evaluation runner — implementation and review checkpoint

**Status: implementation and pilot tests complete; FINAL evaluation NOT executed.**
This is the next step after `docs/CALIBRATION_RESULTS.md`. The study's
precalibration experiment plan remains unchanged, and its SHA-256 is checked
against the frozen GD rate record at startup.

## Safe command for PyCharm / PowerShell

From the project root:

```powershell
python -m unittest discover -s tests -q
python -m scripts.run_comparison
```

The second command is **preview only**. It does not generate final datasets,
start optimizers, score held-out test data, or create result files. The full
comparison requires an explicit `--execute` flag **after** methodological
review. Please do not run it yet; we are reviewing the implementation first.

The preview checks `config/evaluation_protocol.json` against
`config/experiment_plan.json` and the already calibrated rates in
`config/selected_gd_rates.json`.

## Agreed experimental design

| Design element | Frozen behavior |
| --- | --- |
| Core models | Exponential decay, logistic growth |
| Calibration | Completed earlier, excluded from all evaluation timings |
| Fixed GD rates | 0.1 for Model 1; 0.03 for Model 2 |
| Dataset seeds | 10 per model, fixed preregistered noise seeds |
| Starts | S1–S5 per dataset, identical across all methods |
| Methods | GD and **full-step exact** Newton primary; safeguarded Newton exploratory |
| Stopping | Gradient norm ≤ 1e-4 or approved method-specific budget |
| Timing | Five fresh runs per condition; median of five optimizer-only times |
| Final evaluations | 100 paired model/dataset/start conditions, 200 primary method runs, 100 exploratory runs |
| Timing invocations | 300 method conditions × 5 repetitions = **1,500** |

For each repetition the optimizer receives **only** the training x/y and the
same initial parameter vector. The timer begins immediately before the
optimizer call and stops when it returns or raises an expected numerical
error. Dataset generation and post-hoc scoring are outside the timed region.
Newton's Hessian evaluations and linear-system solves are **inside** it.
Across the 50 paired conditions per model, method order alternates
**GD-first** and **Newton-first** (25 each) to reduce systematic timing
drift; the safeguarded exploratory method is always recorded separately.

After all five repetitions, the final parameter vector is evaluated once
on the training, validation and held-out test sets. The test labels **never**
determine the GD rate, Newton steps, stopping, or selection of a starting
vector. Repeated deterministic runs must yield the same parameters and
termination; otherwise the runner raises an error rather than silently
averaging inconsistent fits.

## Recording a result without hiding failures

Each method/seed/start record includes:

- All five individual elapsed times and their median.
- Iterations and termination reason (`gradient_tolerance`, `max_iterations`,
  `numerical_failure`, singular/ill-conditioned Hessian, or failed line search).
- Final parameter vector, gradient norm, and counts of objective, gradient,
  and Hessian evaluations.
- GD's fixed learning rate or, for the **separate exploratory** safeguarded
  Newton, fallback counts and line-search backtracks.
- Training, validation and final test MSE, with diagnostic errors recorded
  if a last finite solution cannot be scored.
- **Stationarity** distinguished from **solution quality**. Only at
  gradient-norm termination do we inspect the training-objective Hessian
  eigenvalues and label a strict local minimum, strict local maximum,
  indefinite saddle, or inconclusive/near-zero curvature.

A tiny gradient alone does **not** prove minimum loss; a low MSE does **not**
prove that the stationarity stopping rule was reached. A nonconverged or
failed run is always retained in counts and runtime summaries.

## Dataset-aware descriptive statistics

Primary GD-versus-exact-Newton methods are paired on each
`(model, dataset_id, start_id)` key. Define a timing difference:

\[
\Delta t_{d,s}=t_{\mathrm{Newton},d,s}-t_{\mathrm{GD},d,s}.
\]

For a single independent dataset `d`, average its **five** start-specific
time differences before combining across the ten independent dataset seeds:

\[
\overline{\Delta t}_{d}=\tfrac{1}{5}\sum_{s=1}^{5}\Delta t_{d,s}.
\]

Negative differences indicate Newton was faster for that paired condition.
The descriptive output reports all ten dataset-level differences, mean and
median, and each method's fraction reaching the gradient tolerance. Five
starts **do not count as five independent datasets**.

### Approved 95% dataset-level bootstrap interval

We additionally report a percentile interval for the **mean of those ten
dataset-level paired timing differences**. For each of **10,000** bootstrap
draws, sample ten complete dataset units *with replacement* and average
their ten values of `mean_newton_minus_gd_seconds`. The original five
starting-point results always stay within their own dataset unit; they are
**never** resampled as independent observations. Use quantiles 0.025 and
0.975 of the 10,000 resulting means (NumPy's `linear` quantile rule).

The bootstrap has a fixed base seed `20261101`, with separate deterministic
streams for the two models (offset 0 for exponential decay; 1 for logistic).
The model-specific seed, sample count, method, point estimate, and interval
endpoints are saved alongside the descriptive statistics. A negative
time difference means Newton was faster, but we do **not** automatically
translate a confidence interval into an unconditional superiority claim.

**Small-sample caveat:** There are only **ten independent datasets** per
model. The percentile bootstrap is transparent but its nominal 95% interval
may not attain 95% coverage with so few groups. We will report all ten
dataset-level differences and explicitly describe this limitation. We do
not calculate p-values or treat five starts as independent. See MacKinnon,
Nielsen, and Webb (2023), [*Cluster-robust inference: A guide to empirical
practice*](https://doi.org/10.1016/j.jeconom.2022.04.001), on the challenges
of inference with clustered and limited independent data.

## Files created only by an explicitly approved final run

Generated output is Git-ignored under `results/evaluation/`:

- `model1_runs.csv`, `model2_runs.csv`: one row per method condition.
- `model1_timings.csv`, `model2_timings.csv`: every one of the five raw times.
- `model1_summary.json`, `model2_summary.json`: dataset-level descriptive
  statistics, **seed-cluster bootstrap 95% timing intervals**, configuration
  hashes, data hashes, Python/NumPy versions, and source-file SHA-256 hashes.

Existing outputs are protected unless `--overwrite` is explicitly supplied.
No evaluation files or final metrics were generated while implementing and
testing this runner. Tests use pilot noise seed `42`, never evaluation seeds
`1000`–`1009` or `1100`–`1109`.
