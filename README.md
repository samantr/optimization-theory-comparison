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
  50 paired dataset/start conditions (no optimization runs yet).
- Model 2: analytical MSE gradient and **exact Hessian** (including
  residual-dependent curvature), checked by numerical differentiation.
- Both models: explicit model-specific loss/gradient/Hessian callables accepted
  by the same optimizers, with Model 2 one-step integration tests.
- GD calibration: eight approved candidate rates, three independent noise
  seeds per model, all five starting points, and a validated manifest;
  no learning-rate selection has run yet.
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

**Not yet implemented:** controlled paired optimizer experiments or the
final report. The **official scope is Models 1 and 2**. The Gaussian peak
and oscillatory trend ideas are optional future work, not required models.

The current Model 1 scripts are **pilots**, not final experimental results.
The dataset size, noise level, 60/20/20 split, and random-seed schedule are
recorded. GD learning-rate tuning will be model-specific using separate
calibration data; the candidate grid and seeds are approved, but stopping
rules, calibration failure handling, and timing protocol still need to
be frozen.

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
  opttheory/
    experiments/
      plan.py
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
    PROGRESS.md
  results/
    README.md
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
rules are **not yet frozen**. The noise seed schedule is explicitly recorded
(Model 1: 1000–1009; Model 2: 1100–1109), with split seed 2026.
The plan-preview scripts list matched conditions without fitting models.
One GD learning rate per model will be tuned on separate calibration data and
frozen for the evaluation grid; no test data will be used for tuning. The
approved candidate rates are `0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03,
0.1, 0.3`. Calibration seeds are `2000–2002` (Model 1) and `2100–2102`
(Model 2); this creates 240 planned calibration attempts across all
five starts, separate from the final evaluation grid. These are only
manifest entries, not calibration results.

The CSV retains the clean curve and the actual noise values for auditability. The metadata records configuration, true parameters, split counts, package/runtime versions, and a file hash.

## Next steps

1. Inspect and understand the generated noisy observations.
2. Confirm the independent random-seed schedule, optimization budgets, and timing protocol.
3. Review and confirm the implemented training MSE J(theta).
4. Review the independently tested Model 1 analytic gradient and exact Hessian.
5. Review both implemented pilot solvers, especially exact Newton's singular
   and indefinite-Hessian behavior; do not silently substitute Gauss-Newton.
6. Evaluate the separately labelled safeguarded Newton variant without
   replacing or disguising failures of the original full-step method.
7. Keep any extra model families as optional future work after the two-model
   controlled experiment has been completed and interpreted.
8. Run paired comparisons and record failures as well as successes.

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

