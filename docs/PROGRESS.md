# Guided project progress register

## Phase status

| Phase | Status | Evidence / next action |
| --- | --- | --- |
| 1. Research question and foundations | Completed | Worked residuals, MSE, derivatives, curvature, local/global extrema, fairness |
| 2. Model and data | Learning | Model 1 data, MSE, and analytical gradient verified; Hessian derivation next |
| 3. Gradient Descent | Not started | Derive and manually test updates before coding |
| 4. Exact Newton | Not started | Study second-order Taylor model, Hessian solve and safeguards |
| 5. Validation | Not started | Derivative checks and solver tests |
| 6. Controlled comparisons | Not started | Paired seeds, initializations, timing and failure logging |
| 7. Results and plots | Not started | No optimizer results exist yet |
| 8. Literature and interpretation | Not started | Verified sources only |
| 9. Report and presentation | Not started | Use validated results only |

## Confirmed mathematical choices

- Four nonlinear model families: exponential, logistic, Gaussian peak, oscillatory trend.
- The oscillatory model is mathematical and sketch-inspired, not an exact trace.
- Model 1 true curve: f(x) = 4(0.5)^x + 1.
- Exact true parameters: a = 4, b = ln(0.5), c = 1.
- f(0) = 5, f(1) = 3, f(2) = 2, f(3) = 1.5, f(4) = 1.25.
- Both algorithms must use the same generated data and initial parameters.

## Provisional settings pending confirmation

- 120 x-values over [0,4].
- 60% training, 20% validation, 20% test using a fixed random split.
- Initial pilot sigma_noise = 0.1; later noise levels may be tested.
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
- Unit tests and mathematical endpoint checks.

## Current learning checkpoint

The Model 1 gradient has been derived by hand, implemented, and verified with
17 passing unit tests. Next: learn what second derivatives mean, derive the
exact Hessian of MSE, and verify its implementation before coding Newton.
The noise/split settings remain provisional and need confirmation before
controlled comparisons. No optimizer has been implemented yet.

