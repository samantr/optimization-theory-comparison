# Guided project progress register

## Phase status

| Phase | Status | Evidence / next action |
| --- | --- | --- |
| 1. Research question and foundations | Completed | Worked residuals, MSE, derivatives, curvature, local/global extrema, fairness |
| 2. Model and data | Model 1 pilot and Model 2 derivatives verified | Both models' synthetic data, analytical MSE gradients and exact Hessians tested; Model 2 optimizer integration tests pass |
| 3. Gradient Descent | Pilot implemented | Fixed-step batch GD, hand-checked update, loss/gradient histories, runtime and failure logging; experimental settings remain provisional |
| 4. Exact Newton | Pilot implemented | Full-step exact Newton preserved; separate safeguarded exact-Hessian hybrid with Armijo backtracking and explicit gradient fallback added |
| 5. Validation | Partially completed | Finite-difference derivative checks and unit tests for all three pilot variants; broader empirical validation pending |
| 6. Controlled comparisons | Two-model scope, GD calibration grid/seeds, stopping and failure-aware selection rules approved; no runs yet | 100 matched evaluation conditions, 200 primary and 100 exploratory planned method runs; 240 GD calibration attempts planned; timing and final analysis rules pending |
| 7. Results and plots | Not started | No controlled-comparison results exist yet; pilot diagnostics only |
| 8. Literature and interpretation | Not started | Verified sources only |
| 9. Report and presentation | Not started | Use validated results only |

## Confirmed mathematical choices

- The **official** study includes exponential decay and logistic growth only.
  Gaussian peak and the mathematical, sketch-inspired oscillatory model are
  optional future work, not part of the approved experiment.
- Model 1 true curve: f(x) = 4(0.5)^x + 1.
- Exact true parameters: a = 4, b = ln(0.5), c = 1.
- f(0) = 5, f(1) = 3, f(2) = 2, f(3) = 1.5, f(4) = 1.25.
- Both algorithms must use the same generated data and initial parameters.
- Five Model 1 start vectors S1–S5 and the fixed-start-per-seed strategy are
  approved and recorded in `config/experiment_plan.json`.
- Five Model 2 start vectors S1–S5 are also approved and frozen, including
  a deliberately negative k for an initially decreasing S-curve.
- Primary Gaussian noise sigma=0.1 across the ten noise seeds per model is
  approved; other noise levels are deferred to robustness analysis.
- Exactly 120 observations per dataset and a seeded random 72/24/24
  training/validation/test split are approved across both models.
- One fixed split seed (2026) across all datasets, plus pre-registered
  nonoverlapping noise seeds (1000–1009, 1100–1109) by model, is recorded
  in the study configuration.
- Model-specific GD learning rates will be selected once from independent
  calibration training/validation data, then frozen for the ten evaluation
  datasets; tuning is excluded from execution-time comparisons.
- Eight GD learning rates (0.0001–0.3) and disjoint calibration seeds
  (Model 1: 2000–2002; Model 2: 2100–2102) are approved. The calibration
  trial manifest contains 240 placeholders; no rate has been selected yet.
- The common stopping criterion is gradient norm <= 1e-4, with iteration
  ceilings of 10,000 (GD), 100 (exact Newton), and 100 (safeguarded Newton).
  Reaching tolerance is stationarity, not necessarily a minimum.
- GD calibration rate selection prioritizes fewest numerical failures,
  then lowest mean finite nonfailed validation MSE, then smaller rate.
  Budget-exhausted trials contribute their last finite validation loss
  and are separately flagged as nonconverged. This policy was selected
  after Saman delegated the choice to the assistant.

## Pilot settings and remaining confirmations

- 120 x-values per dataset and the 60/20/20 seeded random split are approved;
  Model 1 currently uses x over [0,4].
- Initial pilot sigma_noise = 0.1 is now also the **approved primary noise
  level**; additional levels may be tested later.
- Pilot noise seed = 42 and split seed = 2026.

## Implemented in scaffold

- Model 1 forward prediction and synthetic Gaussian data generation.
- Model 2 overflow-resistant logistic prediction, approved 3-parameter truth,
  deterministic noise and split, metadata/CSV/plot export, analytical MSE
  gradient/exact Hessian, and independent finite-difference tests.
  See `docs/MODEL2_DERIVATIVES.md`.
- Model 2's objective/gradient/Hessian are now connected through explicit
  callables to the existing GD, pure exact Newton, and separately labelled
  safeguarded Newton. One-step unit checks validate each connection; no
  controlled optimizer comparison has yet been performed.
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
- Deterministic Model 1 and Model 2 *manifest builders* each list 50 paired
  conditions and 150 method placeholders, including noise/split seeds and
  approved starting parameters, without running any solver.
- A deterministic GD *calibration manifest* lists 120 seed/start/rate
  combinations per model and checks that calibration seeds are disjoint
  from evaluation seeds. It does not run or tune GD.
- Validated `optimizer_stopping_kwargs` ensures future experiments apply
  the approved shared tolerance and method-specific iteration caps rather
  than historical Model 1 pilot defaults.
- Tested failure-aware `select_gd_learning_rate` requires the complete
  registered calibration grid and matches its ordering to the frozen
  configuration. It has been tested with *hypothetical* outcomes only.
- Unit tests and mathematical endpoint checks.

## Current learning checkpoint

The two primary Model 1 optimizers and one explicitly separate safeguarded
Newton variant are implemented, with a growing passing unit-test suite. Reproducible
training-only pilot scripts exist for all three. On one illustrative start,
full-step Newton reached a poor saddle while the safeguarded variant found
a good training fit; neither is comparative evidence. The official study
now consists of two model families only. The GD calibration rates and seeds
and stopping and calibration-selection rules are frozen. Next: approve
success criteria, timing and statistical summaries before the 10-seed × 5-start
paired grid is executed.
The dataset size, noise/split seeds, and 60/20/20 split are recorded; solver
tuning and evaluation policies remain pending. No **controlled** optimizer
grid has run.

