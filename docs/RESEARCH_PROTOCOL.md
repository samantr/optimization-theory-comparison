# Research protocol — working draft

## Main question

Under matched nonlinear regression problems and starting conditions, how do
first-order Gradient Descent and exact second-order Newton optimization differ
in convergence, iteration count, elapsed time, loss, stability, and sensitivity
to initialization?

## Current working hypotheses

- H1: Newton may require fewer iterations near a well-conditioned minimum,
  without necessarily requiring less wall-clock time.
- H2: Both algorithms may depend on their starting vectors, but in different ways.
- Neither hypothesis is an established experimental result.

## Preliminary Model 1 dataset

- Function: f(x) = a exp(bx) + c
- True a = 4; b = ln(0.5); c = 1
- x in [0, 4], 120 equally spaced observations
- Pilot Gaussian noise standard deviation 0.1
- Noise seed 42, random split seed 2026
- Pilot split 72 train, 24 validation, 24 test
- These settings require final confirmation before the comparative experiment.

## Approved experimental breadth (2026-11-01)

- **Four model families:** exponential decay, logistic, Gaussian peak, and
  sketch-inspired oscillatory trend.
- **Ten independently seeded datasets per model.**
- **Observation count approved:** exactly 120 input/target observations per
  dataset for all four models. Each model may have a different appropriate
  input domain, but the count remains matched.
- **Random split approved:** seeded random allocation of 60% training (72),
  20% validation (24), and 20% untouched final test (24) observations,
  with no overlap.
- **Seed strategy approved:** the split assignment is fixed by seed `2026`
  across all ten replicates, isolating the effect of different noise draws.
  A pre-registered nonoverlapping noise-seed schedule is stored explicitly:
  Model 1 `1000`–`1009`; logistic `1100`–`1109`; Gaussian `1200`–`1209`;
  oscillatory `1300`–`1309`. The exact numeric seeds are fixed by this
  project configuration and must not be selected based on test results.
- **Primary noise setting approved:** additive zero-mean Gaussian noise with
  standard deviation \(\sigma_{noise}=0.1\) for all ten datasets per model;
  the noise realization changes with the dataset seed. Additional noise
  intensities are deferred to a separate, explicitly labelled robustness
  study rather than tripling the initial experimental grid.
- **Five starting parameter vectors per dataset.**
- **Starting-vector strategy approved:** five fixed, deliberately varied,
  model-specific vectors reused across the ten dataset seeds; all methods
  receive the same vectors within every matched condition.
- **Same dataset and same start across methods within each paired condition.**
- **Two primary methods:** fixed-step batch Gradient Descent and full-step
  exact Newton. **One separately labelled exploratory variant:** safeguarded
  exact-Hessian Newton with gradient fallback.
- This yields **50 paired conditions per model** or **200 conditions across
  four models**. If all three methods are run on each condition, the planned
  initial grid contains 400 primary method runs plus 200 safeguarded runs,
  before timing repetitions or tuning.
- Dataset seeds are grouping units: five starts on the same dataset are **not
  five independent datasets**. Summaries and uncertainty estimates must
  respect this hierarchy.
- **Model 1 starting vectors approved** (parameter order `[a,b,c]`):
  `S1=[3,-0.5,1]`, `S2=[1,-1.2,0]`, `S3=[6,-0.2,2]`,
  `S4=[-1,-0.5,3]`, `S5=[2,0.4,0]`. These probe normal, extreme,
  sign-reversed, and growth-like initial conditions. They are kept even if
  an algorithm fails, and are not optimized against the test set.
- **Not approved/frozen yet:** starting vectors for Models 2–4, optimizer
  tuning budgets, timing
  protocol, and success/failure thresholds. No controlled grid has been run.

Machine-readable approved counts are in `config/experiment_plan.json`.
The `opttheory.experiments.plan` module enumerates 50 matched Model 1
conditions (150 method-run placeholders), including the fixed noise/split
seed for each condition, **without** generating data or fitting any method.
`D01`–`D10` are dataset labels. Preview with:

```powershell
python -m scripts.preview_model1_plan
```

## Required controls (future phases)

1. Hold the data, training split, and initialization fixed within every paired
   Gradient Descent vs Newton comparison.
2. Use separate random streams for dataset noise, splitting, and initialization.
3. Tune algorithm settings on training/validation only. Keep final test data untouched.
4. Report training convergence separately from unseen-data predictive performance.
5. Define consistent stopping and success criteria, budgets, and failure categories.
6. Derive analytic gradients and Hessians, checking them numerically.
7. Implement **exact Newton** with a linear system solve and documented safeguards.
   Do not silently substitute Gauss-Newton, BFGS, or library optimizers.
8. Run multiple dataset seeds and multiple paired starting vectors; do not present a
   single seed as evidence of general superiority.
9. Log every failed, divergent, or numerically unstable attempt.
10. Repeat and summarize timing measurements; one Newton iteration can cost more
    than one Gradient Descent iteration.
11. Separate parameter recovery error, training MSE, test MSE, and gradient norm.
12. Preserve code version, package versions, config, seeds, and generated data hash.
13. A small gradient is *not* automatically successful optimization: inspect
    final loss and Hessian curvature, and record saddle/poor stationary points.
14. Clearly distinguish unsafeguarded full-step Newton from any later
    safeguarded or damped variant; do not silently alter the baseline.
15. Keep the primary two-way analysis as GD versus full-step exact Newton.
    Treat safeguarded Newton with gradient fallback as a **separate third
    variant**, reporting its extra line-search/function evaluations and
    fallback steps instead of using its iteration count alone.

## Report target

Approximately 10-15 pages, at least 20 verified academic references, separate
code, and final-week presentation, following Prof. Dr. Yaşar Becerikli's latest
course requirements.

No experiment, result, figure, or academic conclusion should be claimed until
the corresponding computation has actually been performed and checked.

