# Guided project progress register

## Phase status

| Phase | Status | Evidence / next action |
| --- | --- | --- |
| 1. Research question and foundations | Completed | Worked residuals, MSE, derivatives, curvature, local/global extrema, fairness |
| 2. Model and data | Completed for Model 1 pilot | Model, data, MSE, analytical gradient, and exact Hessian implemented and verified; other models and experimental settings remain for later phases |
| 3. Gradient Descent | Pilot implemented | Fixed-step batch GD, hand-checked update, loss/gradient histories, runtime and failure logging; experimental settings remain provisional |
| 4. Exact Newton | Pilot implemented | Full-step exact Newton preserved; separate safeguarded exact-Hessian hybrid with Armijo backtracking and explicit gradient fallback added |
| 5. Validation | Partially completed | Finite-difference derivative checks and unit tests for all three pilot variants; broader empirical validation pending |
| 6. Controlled comparisons | Size, Model 1 starts, noise, split, and seeds fixed; no runs yet | 120 observations, 72/24/24 fixed split seed 2026, disjoint model-specific noise seeds, and 150 Model 1 method placeholders; tuning/timing pending |
| 7. Results and plots | Not started | No controlled-comparison results exist yet; pilot diagnostics only |
| 8. Literature and interpretation | Not started | Verified sources only |
| 9. Report and presentation | Not started | Use validated results only |

## Confirmed mathematical choices

- Four nonlinear model families: exponential, logistic, Gaussian peak, oscillatory trend.
- The oscillatory model is mathematical and sketch-inspired, not an exact trace.
- Model 1 true curve: f(x) = 4(0.5)^x + 1.
- Exact true parameters: a = 4, b = ln(0.5), c = 1.
- f(0) = 5, f(1) = 3, f(2) = 2, f(3) = 1.5, f(4) = 1.25.
- Both algorithms must use the same generated data and initial parameters.
- Five Model 1 start vectors S1–S5 and the fixed-start-per-seed strategy are
  approved and recorded in `config/experiment_plan.json`.
- Primary Gaussian noise sigma=0.1 across the ten noise seeds per model is
  approved; other noise levels are deferred to robustness analysis.
- Exactly 120 observations per dataset and a seeded random 72/24/24
  training/validation/test split are approved across all four models.
- One fixed split seed (2026) across all datasets, plus pre-registered
  nonoverlapping noise seeds (1000–1009, 1100–1109, 1200–1209, 1300–1309)
  by model, is now recorded in the study configuration.

## Pilot settings and remaining confirmations

- 120 x-values per dataset and the 60/20/20 seeded random split are approved;
  Model 1 currently uses x over [0,4].
- Initial pilot sigma_noise = 0.1 is now also the **approved primary noise
  level**; additional levels may be tested later.
- Pilot noise seed = 42 and split seed = 2026.

## Implemented in scaffold

- Model 1 forward prediction and synthetic Gaussian data generation.
- Independent noise and split random generators.
- CSV/JSON/plot export script.
- Model 1 objective J(theta) using candidate parameters and training observations.
- Model 1 analytical gradient [dJ/da, dJ/db, dJ/dc], using only candidate
  parameters and observed input/target arrays.
- Independent central-difference gradient checks and the hand-worked two-point
  example, giving [-1.2, -0.3, -1.3] at [3, ln(0.5), 1].
- Exact analytical Hessian with all mixed second derivatives, including the
  residual-dependent terms required by exact Newton (not Gauss–Newton).
- Independent finite-difference Hessian checks, symmetry tests, and a singular
  one-observation example; see `docs/MODEL1_HESSIAN.md`.
- A first-order fixed-step batch GD optimizer with a common result contract,
  a reproducible training-only pilot command, and independent tests for the
  hand-computed step, singular Hessian independence, stopping, and overflow;
  see `docs/GRADIENT_DESCENT.md`.
- A full-step **exact** Newton optimizer with an explicit linear system solve,
  condition-number/singularity safeguards, history/counter logging, and tests
  for a manually checkable three-parameter update; see `docs/NEWTON.md`.
- A separate **safeguarded exact-Hessian Newton** variant with positive-definite
  Hessian check, explicit gradient fallback, Armijo backtracking, and counters
  for the extra computational work; see `docs/NEWTON_SAFEGUARDED.md`.
- A deterministic Model 1 *manifest builder* lists 50 paired conditions and
  150 method placeholders, including their noise/split seed values, without
  running any solver.
- Unit tests and mathematical endpoint checks.

## Current learning checkpoint

The two primary Model 1 optimizers and one explicitly separate safeguarded
Newton variant are implemented, with a growing passing unit-test suite. Reproducible
training-only pilot scripts exist for all three. On one illustrative start,
full-step Newton reached a poor saddle while the safeguarded variant found
a good training fit; neither is comparative evidence. Next: confirm
starting-vector selection for Models 2–4, learning-rate/line-search
policies, success criteria, and timing before the approved 10-seed × 5-start
paired grid is executed.
The dataset size, noise/split seeds, and 60/20/20 split are recorded; solver
tuning and evaluation policies remain pending. No **controlled** optimizer
grid has run.

