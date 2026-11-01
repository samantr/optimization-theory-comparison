# Exploratory pilot: Saman's adaptive learning-rate GD proposal

**Date:** 2026-11-01. **Status:** pilot only, not the registered evaluation.
This work starts *after* the audited GD-versus-Newton comparison. The frozen
experiment plan, selected GD rates, evaluation protocol, source provenance,
and archived final evaluation remain unchanged.

## Question and motivation

Could a dynamically reduced Gradient Descent step size improve optimization
behavior relative to the original fixed model-specific learning rate?

Saman proposed: start at a large rate; continue while the gradient direction
is consistent; halve the rate after apparent overshooting. We tested a
precise vector version of this suggestion and a loss-based alternative.

### Three algorithms in the paired pilot

1. **Fixed GD (reference):** unchanged original solver and frozen learning
   rates 0.1 for exponential decay, 0.03 for logistic growth.
2. **Gradient-reversal GD (original idea):** initialize the global rate at
   1.0. Accept *every finite update*. If successive full gradient vectors
   have a **negative dot product**, halve the rate for the *next* step.
   Otherwise keep the rate. A direction reversal is a heuristic; it does
   **not** establish that a local minimum was crossed in 3D parameter space.
3. **Simple backtracking GD:** initialize the rate at 1.0. Propose a GD step,
   **reject** it if training MSE does not strictly decrease, and retry from
   the last accepted theta with half the rate. Keep the accepted rate for
   subsequent steps. Up to 50 retries per iteration and a 1e-12 step floor.
   This is *not* the Armijo line search (which requires sufficient decrease).

All three use the analytical gradient. Only the fixed reference inherits its
previously calibrated model-specific rate. All methods stop at gradient norm
`<=1e-4` or 10,000 **accepted** updates, logging nonconvergence and failures.
`iterations` counts accepted updates; a rejected backtracking candidate
increases the MSE evaluation and rejected-trial counters, not iterations.

## Pilot design (not tuned on the frozen evaluation)

- **Separate noise seeds:** Model 1 = 3000, Model 2 = 3100. These differ
  from registered evaluation seeds 1000-1009 / 1100-1109 and original rate
  calibration seeds 2000-2002 / 2100-2102.
- Same approved 120 input coordinates, Gaussian sigma 0.1, split seed 2026,
  72 train / 24 validation / 24 test, and all five original starting
  parameter vectors for each method within a model.
- **Only training observations** enter the optimizers; validation labels are
  used for descriptive pilot fit checks; test labels are not accessed.
- Total **30 runs** (2 models x 1 new seed x 5 starts x 3 GD methods).
- Starting rate 1.0 is Saman's suggested exploratory setting, **not** a
  calibrated or claimed optimal rate. No hyperparameter tuning was run.
- Timing wasn't repeated and therefore **no computational speed comparison
  is claimed**. Update and function-evaluation counts are descriptive only.

## Actual pilot results

| Model | Method | Reached gradient tolerance | Median validation MSE (5 starts) | Median accepted updates | Median training MSE evaluations |
|---|---|---:|---:|---:|---:|
| Exponential | Fixed GD | 5/5 | 0.0117737 | 1188 | 1189 |
| Exponential | Gradient reversal | 4/5 | 0.962215 | 623 | 624 |
| Exponential | Backtracking | 3/5 | 0.0117738 | 1739 | 1742 |
| Logistic | Fixed GD | 5/5 | 0.0074957 | 2873 | 2874 |
| Logistic | Gradient reversal | 5/5 | 11.0846 | 41 | 42 |
| Logistic | Backtracking | 5/5 | 0.0074957 | 1377 | 1382 |

**Post-hoc fit description:** Using validation MSE <0.02 as a simple visual
marker of a fit near the fixed-GD results (not a registered success metric),
the counts are Model 1: fixed 5/5, reversal 2/5, backtracking 3/5;
Model 2: fixed 5/5, reversal 2/5, backtracking 5/5. These are five starts
on **one dataset per model**, not independent noise replicates or inference.

Important individual cases:

- Logistic S1 and S3: reversal GD met gradient tolerance after **2** updates,
  but validation MSE was approximately **39.013**, versus approximately
  **0.0075** from fixed GD. *Stationarity is not proof of a good fit.*
- Exponential S3: reversal GD exhausted all 10,000 updates with validation
  MSE **0.9622**, while fixed GD fitted at approximately **0.0118**.
- Exponential S4 and S5: backtracking exhausted 10,000 updates after its
  rate shrank; validation MSE remained **0.2987** and **1.2089**. Monotone
  decrease does not guarantee a good minimum or fast convergence.
- Logistic S4: backtracking needed **9,248** updates versus fixed GD's
  **154**, despite reaching similar final validation MSE. Adaptive is **not
  consistently faster** even by update count.

## Interpretation and next decision

The pure sign-reversal rule can produce very short runs that nevertheless
end at poor stationary points. Backtracking is more consistently good in this
small pilot, but it can also shrink the rate too far and make poor progress.
The pilot cannot establish algorithm superiority, and counts of iterations
are not equivalent to wall-clock cost: backtracking adds rejected MSE trials.

For a formal follow-up, **freeze the extension's rules first**, select any
hyperparameters with new training/validation-only calibration data, and use
new nonoverlapping evaluation seeds plus repeated timing. Report all starts,
termination reasons, Hessian curvature classifications, and validation/test
performance; aggregate independent noise-dataset units instead of treating
the five starts as independent samples. Do not merge the results into the
original preregistered GD-versus-Newton comparison.

## Reproduction

From the repository root:

```powershell
python -m unittest tests.test_adaptive_gradient_descent -v
python -m scripts.run_adaptive_gd_pilot
python -m scripts.run_adaptive_gd_pilot --execute
```

The execution writes the exact 30 rows to `results/adaptive_gd_pilot.csv`.
An identical snapshot is committed under
`evidence/adaptive_gd_pilot_2026-11-01.csv` for auditability.
The SHA-256 of both LF-terminated CSV copies is
`29d97e98e15be926dee62f6559511e65a3d27507eedb2f96eb04da03d2325453`.

### Prior methods

- Jacobs, R. A. (1988). *Increased rates of convergence through learning
  rate adaptation*. Neural Networks 1(4), 295–307.
  https://doi.org/10.1016/0893-6080(88)90003-2
- Riedmiller, M. & Braun, H. (1993). *A direct adaptive method for faster
  backpropagation learning: The RPROP algorithm*. ICNN, pp. 586–591.
  https://doi.org/10.1109/ICNN.1993.298623
- Boyd, S. & Vandenberghe, L. (2004). *Convex Optimization*, sec. 9.2.
  https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf
