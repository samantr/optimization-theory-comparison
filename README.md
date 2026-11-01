# Optimization Theory: Gradient Descent vs Newton's Method

Course: Optimizasyon Teorisi, Kocaeli University  
Instructor: Prof. Dr. Yaşar Becerikli  
Student: Saman Tarighpeima

Approved title: **Doğrusal Olmayan Regresyon Problemlerinde Gradyan İnişi ve Newton Optimizasyon Yöntemlerinin Karşılaştırmalı Analizi**

## Scope of this starter

This is a **guided-learning scaffold, not a completed research project**.

Already implemented and tested:

- Model 1: the noise-free function f(x) = a exp(bx) + c.
- Synthetic Gaussian observations from known true parameters.
- Parameter-dependent Model 1 training MSE, explicitly using candidate [a,b,c]
  and supplied observed training data (never known true parameters).
- Model 1 analytical MSE gradient in parameter order [a,b,c], checked against
  manual calculations and independent central finite differences.
- Model 1 **exact Hessian** (all nine entries), verified against independent
  central differences of the analytical gradient and a singular test case.
- Separate random seeds for observation noise and train/validation/test assignment.
- Deterministic CSV, metadata (including a SHA-256 digest), and an optional preview plot.
- Unit checks for the mathematical endpoints, seed reproducibility, and split counts.

**Not yet implemented:** Gradient Descent, exact Newton optimizer, Model 2 (logistic), Model 3 (Gaussian peak), Model 4 (oscillatory trend), optimizer experiments, or the final report. These will be added after mathematical checkpoints.

The current configuration is a **pilot**, not a frozen experiment protocol. The noise level and 60/20/20 split remain provisional until we discuss them.

## Folder structure

~~~text
optimization-theory-comparison/
  README.md
  requirements.txt
  .gitignore
  config/
    model1.json
  opttheory/
    models/
      exponential.py
    datasets/
      synthetic.py
    objectives/
      mse.py
    optimizers/
      __init__.py
  scripts/
    generate_model1.py
  tests/
    test_model1.py
    test_objective.py
  docs/
    MODELS.md
    MODEL1_HESSIAN.md
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
~~~

The commands run the tests and create:

- results/model1/model1_dataset.csv
- results/model1/model1_metadata.json
- results/model1/model1_preview.png

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

## Run on macOS or Linux

~~~bash
python3 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m unittest discover -s tests -v
./.venv/bin/python -m scripts.generate_model1
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

The CSV retains the clean curve and the actual noise values for auditability. The metadata records configuration, true parameters, split counts, package/runtime versions, and a file hash.

## Next steps

1. Inspect and understand the generated noisy observations.
2. Confirm dataset size, noise levels, split strategy, and multiple-seed protocol.
3. Review and confirm the implemented training MSE J(theta).
4. Review the independently tested Model 1 analytic gradient and exact Hessian.
5. Implement Gradient Descent and **exact Newton**, explicitly; do not silently substitute Gauss-Newton.
6. Add the other three model families only after the core method is verified.
7. Run paired comparisons and record failures as well as successes.

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

