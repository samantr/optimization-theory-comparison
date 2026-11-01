# Optimization Theory: Gradient Descent vs Newton's Method

Course: Optimizasyon Teorisi, Kocaeli University  
Instructor: Prof. Dr. Yaşar Becerikli  
Student: Saman Tarighpeima

Approved title: **Doğrusal Olmayan Regresyon Problemlerinde Gradyan İnişi ve Newton Optimizasyon Yöntemlerinin Karşılaştırmalı Analizi**

## Scope of this starter

This is a **guided-learning scaffold, not a completed research project**.

Already implemented and tested:

- Model 1: the noise-free function f(x) = a exp(bx) + c.
- Model 2: stable logistic function f(x) = L/(1+exp[-k(x-x0)]) and an
  approved, independently seeded synthetic dataset (truth [10,1.5,2]).
- Model 2: five approved starting vectors and a reproducible manifest of
  50 paired dataset/start conditions, now evaluated with all three methods.
- Model 2: analytical MSE gradient and **exact Hessian** (including
  residual-dependent curvature), checked by numerical differentiation.
- Both models: explicit model-specific loss/gradient/Hessian callables accepted
  by the same optimizers, with Model 2 one-step integration tests.
- GD calibration **executed**: eight preregistered candidate rates, three
  independent calibration noise seeds per model, all five starting points,
  and 240 recorded attempts. Frozen rates: exponential decay **0.1**,
  logistic growth **0.03**; see `docs/CALIBRATION_RESULTS.md`.
- Shared stopping rules for future experiments: gradient norm <= 1e-4,
  maximum 10,000 GD or 100 Newton iterations, with early stopping and
  explicit nonconvergence/failure logging.
- Failure-aware GD rate selection from a **complete** calibration grid:
  fewest numerical failures first, then lowest finite validation MSE;
  independently unit tested and applied to all 240 actual outcomes.
- A **preview-only-by-default paired evaluation runner** enforcing the
  frozen GD rates, five optimizer-only timing repetitions, Hessian-based
  stationary-point diagnostics, failure logging, dataset-level paired
  descriptive statistics, and approved 10,000-draw 95% whole-dataset
  bootstrap timing intervals. The **full final evaluation has now run and
  passed an independent audit**; the preview command remains safe by default.
- Original raw evidence (300 method conditions, 1,500 timing repetitions)
  archived in `evidence/evaluation_2026-11-01/`, with SHA-256 digests in
  `config/evaluation_provenance.json`; the first results and limitations are
  documented in `docs/FINAL_EVALUATION_RESULTS.md`.
- Synthetic Gaussian observations from known true parameters.
- Parameter-dependent Model 1 training MSE, explicitly using candidate [a,b,c]
  and supplied observed training data (never known true parameters).
- Model 1 analytical MSE gradient in parameter order [a,b,c], checked against
  manual calculations and independent central finite differences.
- Model 1 **exact Hessian** (all nine entries), verified against independent
  central differences of the analytical gradient and a singular test case.
- A from-scratch, fixed-step **batch Gradient Descent** pilot recording
  iteration count, runtime, loss/gradient histories, and numerical failures.
- A from-scratch **full-step exact Newton** pilot using a Hessian linear-system
  solve, singular/ill-conditioned detection, and explicit failure logging.
- A separate, labelled **safeguarded Newton** pilot with Armijo backtracking,
  an explicit gradient fallback, and detailed extra-work counters.
- Separate random seeds for observation noise and train/validation/test assignment.
- Deterministic CSV, metadata (including a SHA-256 digest), and an optional preview plot.
- Unit checks for the mathematical endpoints, seed reproducibility, and split counts.

**Completed:** the registered paired optimizer comparison and independent
numeric audit. Plots, interpretation exercises, the verified literature
review, and academic report are still pending. The **official scope is Models 1 and 2**. The Gaussian peak
and oscillatory trend ideas are optional future work, not required models.

The current Model 1 scripts are **pilots**, not final experimental results.
The dataset size, noise level, 60/20/20 split, and random-seed schedule are
recorded. Model-specific GD learning-rate tuning is complete, using only
separate calibration training/validation data. The two selected rates are
frozen in `config/selected_gd_rates.json`. The recorded timing and evaluation
policies are in `config/evaluation_protocol.json`. The dataset-cluster
percentile bootstrap was applied to the real study; its small-sample
limitations are disclosed in the results document. The preregistered
experiment plan and evaluation-protocol files remain **unchanged** after
execution to preserve their SHA-256 provenance.

### Academic report preparation

The report's reference-first preparation is recorded in
`docs/REPORT_REFERENCE_REGISTER.md`: 22 scholarly candidate sources, links
to publisher/proceedings records, planned citations by report section, and
the professor's minimum 10-page / 20-source requirements. This is a
selection register, not yet a completed literature review. The original
optimizer study and its archived evidence remain unchanged.
The professor-aligned page budget, section-by-section outline, lecture-slide
links, planned figures, and evidence checklist are in
`docs/ACADEMIC_REPORT_OUTLINE.md`.

### Separate adaptive GD pilot (not part of the final study)

Saman's gradient-reversal step-halving idea and a training-MSE backtracking
variant have been implemented and tested against the original fixed-rate GD
on **fresh pilot seeds**, using all five original starts per model.
See `docs/ADAPTIVE_GD_EXPLORATORY_PILOT.md` for measured outcomes,
failure cases, and limitations. The original final-evaluation code and
pre-registered settings are unchanged.

```powershell
python -m scripts.run_adaptive_gd_pilot
python -m scripts.run_adaptive_gd_pilot --execute
```

After selecting **rejection of harmful steps**, Saman additionally proposed
recovering half the lost learning rate after successful steps. This was
implemented as a **separate** second pilot with new noise seeds (3200/3300)
and a fourth method, `recovering_backtracking`. See
`docs/RECOVERING_ADAPTIVE_GD_PILOT.md` and run:

```powershell
python -m scripts.run_recovering_gd_pilot
python -m scripts.run_recovering_gd_pilot --execute
```

## Folder structure

~~~text
optimization-theory-comparison/
  README.md
  requirements.txt
  .gitignore
  config/
    model1.json
    model2.json
    experiment_plan.json
    selected_gd_rates.json
    evaluation_protocol.json
    evaluation_provenance.json
  opttheory/
    experiments/
      plan.py
      policy.py
      calibration.py
      run_calibration.py
      evaluation.py
      analysis.py
    models/
      exponential.py
      logistic.py
    datasets/
      synthetic.py
    objectives/
      mse.py
      logistic_mse.py
    optimizers/
      __init__.py
      common.py
      gradient_descent.py
      newton.py
      newton_safeguarded.py
  scripts/
    generate_model1.py
    generate_model2.py
    preview_gd_calibration_plan.py
    calibrate_gd.py
    run_comparison.py
    preview_model2_plan.py
    preview_model1_plan.py
    run_model1_gd_pilot.py
    run_model1_newton_pilot.py
    run_model1_newton_safeguarded_pilot.py
  tests/
    test_model1.py
    test_model2.py
    test_model2_dataset.py
    test_model2_derivatives.py
    test_model2_optimizers.py
    test_stopping_policy.py
    test_calibration_selection.py
    test_run_calibration.py
    test_frozen_rates.py
    test_evaluation_runner.py
    test_cluster_bootstrap.py
    test_objective.py
    test_gradient_descent.py
    test_newton.py
    test_newton_safeguarded.py
  docs/
    MODELS.md
    MODEL1_HESSIAN.md
    MODEL2_DERIVATIVES.md
    GRADIENT_DESCENT.md
    NEWTON.md
    NEWTON_SAFEGUARDED.md
    RESEARCH_PROTOCOL.md
    CALIBRATION_RESULTS.md
    EVALUATION_RUNNER.md
    FINAL_EVALUATION_RESULTS.md
    PROGRESS.md
  results/
    README.md
  evidence/
    evaluation_2026-11-01/
      model1_runs.csv, model1_timings.csv, model1_summary.json
      model2_runs.csv, model2_timings.csv, model2_summary.json
~~~

## Run on Windows (PowerShell)

Install Python 3.12, then open PowerShell in this folder:

~~~powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m scripts.generate_model1
.\.venv\Scripts\python.exe -m scripts.generate_model2
.\.venv\Scripts\python.exe -m scripts.preview_model2_plan
.\.venv\Scripts\python.exe -m scripts.preview_gd_calibration_plan
.\.venv\Scripts\python.exe -m scripts.calibrate_gd
.\.venv\Scripts\python.exe -m scripts.run_comparison
.\.venv\Scripts\python.exe -m scripts.run_model1_gd_pilot
.\.venv\Scripts\python.exe -m scripts.run_model1_newton_pilot
.\.venv\Scripts\python.exe -m scripts.run_model1_newton_safeguarded_pilot
.\.venv\Scripts\python.exe -m scripts.preview_model1_plan
~~~

The commands run the tests and create:

- results/model1/model1_dataset.csv
- results/model1/model1_metadata.json
- results/model1/model1_preview.png
- results/model2/model2_dataset.csv
- results/model2/model2_metadata.json
- results/model2/model2_preview.png

These generated files are intentionally ignored by Git; the script regenerates them from the configuration and seed. To rerun into the same folder, deliberately add the --overwrite flag. Otherwise, use a fresh --output directory to preserve previous outputs.

`scripts.calibrate_gd` **reruns the full 240-condition calibration**, which
may take minutes. `scripts.run_comparison` is **safe preview-only by default**:
it prints the 1,500-invocation plan without fitting or reading evaluation
observations. The original approved `--execute` study is **already archived**;
use the flag again only to perform a separately labelled replication on
your computer. See `docs/EVALUATION_RUNNER.md` and
`docs/FINAL_EVALUATION_RESULTS.md`.

The newest automated tests also verify our manually solved example:
theta = [3,-1,1], x = [0], observed y = [5.1], giving MSE = 1.21.

They also verify the analytical gradient at candidate theta =
[3, ln(0.5), 1] on two illustrative observations x = [0, 1],
y = [5.1, 2.7]. The expected gradient is [-1.2, -0.3, -1.3].

The function `opttheory.objectives.mse.gradient(theta, x, y_observed)` returns
a NumPy array with shape `(3,)`, ordered `[dJ/da, dJ/db, dJ/dc]`.
The objective and gradient share input validation. A separate test estimates
each derivative numerically using a centered difference of MSE evaluations.

The function `opttheory.objectives.mse.hessian(theta, x, y_observed)` returns
the exact symmetric `(3, 3)` Hessian, again in `[a, b, c]` order. Tests verify
its entries against finite differences of the gradient. The derivation and a
singular-Hessian example are in `docs/MODEL1_HESSIAN.md`.

The GD teaching pilot uses the analytical gradient on **training data only**.
It reports final candidate parameters, training MSE, gradient norm, iteration
count, runtime, and termination reason; it does not inspect the test split.
The formula and hand-worked update are in `docs/GRADIENT_DESCENT.md`.

The exact-Newton pilot uses `numpy.linalg.solve` with the analytical Hessian,
records all full-step losses, and checks for singular or badly conditioned
Hessians. It can reach a stationary **saddle** rather than a minimum; see
`docs/NEWTON.md`. A small gradient alone does not imply a good fit.

The separately named safeguarded Newton variant combines the exact-Hessian
direction with a declared negative-gradient fallback and sufficient-decrease
line search. Its fallback and extra objective evaluations are reported; it is
**not** the full-step Newton baseline. See `docs/NEWTON_SAFEGUARDED.md`.

## Run on macOS or Linux

~~~bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python -m scripts.generate_model1
./.venv/bin/python -m scripts.generate_model2
./.venv/bin/python -m scripts.preview_model2_plan
./.venv/bin/python -m scripts.preview_gd_calibration_plan
./.venv/bin/python -m scripts.calibrate_gd
./.venv/bin/python -m scripts.run_comparison
./.venv/bin/python -m scripts.run_model1_gd_pilot
./.venv/bin/python -m scripts.run_model1_newton_pilot
./.venv/bin/python -m scripts.run_model1_newton_safeguarded_pilot
./.venv/bin/python -m scripts.preview_model1_plan
~~~

## How the code relates to the mathematics

The approved true curve is:

\[
f(x)=4e^{\ln(0.5)x}+1=4(0.5)^x+1.
\]

Thus the exact parameter b is computed as -ln(2), not stored as a rounded -0.69.

The program creates 120 equally spaced x-values from 0 through 4. Then:

\[
y_{\mathrm{clean},i}=f(x_i),\qquad
\epsilon_i\sim \mathcal N(0,\sigma^2),\qquad
y_i=y_{\mathrm{clean},i}+\epsilon_i.
\]

The current pilot uses sigma = 0.1, noise seed 42, and split seed 2026. They are independent random streams. A single generated dataset must be reused by both algorithms in each paired comparison.

The approved study breadth is **two model families, ten dataset seeds per
model, and five paired starting points per dataset**. This is recorded in
`config/experiment_plan.json` and `docs/RESEARCH_PROTOCOL.md`. Both models'
starting vectors, Gaussian noise sigma=0.1, dataset size 120,
and the 72/24/24 seeded train/validation/test split are approved. Timing
rules are recorded in `config/evaluation_protocol.json`. The noise seed schedule is explicitly recorded
(Model 1: 1000–1009; Model 2: 1100–1109), with split seed 2026.
The plan-preview scripts list matched conditions without fitting models.
One GD learning rate per model was tuned on separate calibration data and
frozen for the evaluation grid; no test data was used for tuning. The
approved candidate rates are `0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,
0.1, 0.3`. Calibration seeds are `2000–2002` (Model 1) and `2100–2102`
(Model 2); the preregistered 240 attempts across all five starts were
**executed and audited**. The selected rates are `0.1` and `0.03`,
respectively. Full per-rate summaries are in `docs/CALIBRATION_RESULTS.md`.

All study runs will use the approved `optimizer_stopping_kwargs` helper:
stop once the gradient norm is at most `1e-4`, with iteration ceilings
of `10000` for GD and `100` for each Newton variant. The helper prevents
accidental use of older pilot defaults. Gradient-based termination
is **not** proof of reaching a minimum; keep loss and Hessian checks.

The calibration selector is also registered before evaluation: choose the
rate with the fewest numerical failures across all 15 calibration
conditions, then the lowest average finite nonfailed validation MSE,
then the smaller rate if exactly tied. A run that reaches its iteration
cap but retains finite parameters contributes its last validation MSE
and is flagged as not converged. All 240 results were present before
selecting the model-specific rates. Generated raw CSV/JSON remain ignored
by Git, but are reproducible with `python -m scripts.calibrate_gd`;
frozen decisions and SHA-256 digests are tracked in `config/selected_gd_rates.json`.

The CSV retains the clean curve and the actual noise values for auditability. The metadata records configuration, true parameters, split counts, package/runtime versions, and a file hash.

## Next steps

1. Pull the archived full-run evidence and inspect
   `docs/FINAL_EVALUATION_RESULTS.md`.
2. Explain why exact Newton's very fast stationary points were often saddles
   while GD found lower-loss minima. Use actual Hessian and MSE records.
3. Build verified, clearly labelled figures from the archived CSV files.
4. Draft the academic report incrementally, adding verified references and
   stating the small-sample limitations of the timing bootstrap.

See docs/PROGRESS.md for the cumulative checkpoint and docs/RESEARCH_PROTOCOL.md for controls.

## GitHub

Canonical repository: https://github.com/samantr/optimization-theory-comparison

For a first-time checkout, clone into a **new folder** (or select **Get from VCS** in PyCharm):

~~~powershell
git clone https://github.com/samantr/optimization-theory-comparison.git
cd optimization-theory-comparison
~~~

Afterwards, use `git pull` from inside this cloned folder to get future lessons and changes. If you already ran the starter ZIP in a separate local folder, keep that folder as a backup until you have checked the clone; simply running `git pull` inside a folder that was never cloned or initialized with Git will not work.

Generated datasets, plots, and virtual environments must not be committed; `.gitignore` excludes these local artifacts.

