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

## Report target

Approximately 10-15 pages, at least 20 verified academic references, separate
code, and final-week presentation, following Prof. Dr. Yaşar Becerikli's latest
course requirements.

No experiment, result, figure, or academic conclusion should be claimed until
the corresponding computation has actually been performed and checked.

