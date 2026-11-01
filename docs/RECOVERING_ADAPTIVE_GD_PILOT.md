# Exploratory follow-up: learning-rate recovery after backtracking

**2026-11-01.** This is a second, explicitly *post-hoc*, exploratory pilot.
It is NOT part of the already completed/frozen Gradient Descent vs Newton
study, nor an independent confirmation of an untuned optimization method.
It was motivated by the results of the earlier adaptive-GD pilot.

## Saman's amended rule and the accepted decision

The initial proposal halved the learning rate on suspected overshooting.
Saman then proposed **giving back half the amount previously removed** after
a successful update. In discussion, he chose option B: **reject proposed
steps that raise training MSE** and retry from the previous accepted theta,
instead of taking an extra step to return to it.

We operationalize this as **recovering backtracking GD**:

1. Initialize a shared global step size at alpha = 1.0.
2. Compute a trial theta = theta - alpha * gradient(theta).
3. If the training MSE fails to **strictly decrease**, leave theta unchanged,
   record the rejected rate as `last_rejected_rate`, halve alpha, and retry
   from the same accepted theta. Also reject nonfinite proposals.
4. When the trial reduces MSE, accept the new theta and its gradient.
5. If there has been a previous rejected rate greater than the just-accepted
   rate, try the following rate **on the next iteration**:

   `next_alpha = (accepted_alpha + last_rejected_rate)/2`

   Otherwise retain the accepted rate. Each new rejection updates the
   recorded upper value. This is a *heuristic*, **not a valid mathematical
   bisection bracket**: the recorded failure may be from a different theta,
   where the loss geometry was different.
6. Stop at gradient norm <= 1e-4, 10,000 accepted updates, exhausted search,
   or a numerical failure. Like the previous pilot, line search is strict
   decrease, **not** Armijo sufficient decrease.

Example: reject alpha = 1; accept alpha = 0.5; next try 0.75. If that
also reduces MSE, next try 0.875. If it fails, retry without moving at
0.4375, and attempt recovery from there after an accepted step.

This method uses only analytic first derivatives and MSE evaluations.
It neither estimates nor uses a Hessian inside the optimization loop.

## New pilot design and isolation

- **Fresh, separate pilot noise seeds:** Model 1 3200, Model 2 3300;
  different from the original evaluation, GD calibration, and first adaptive
  pilot seeds. They were selected before executing this second pilot.
- The same 120 x-values, sigma = 0.1, 72/24/24 random split (seed 2026),
  and all five original starting vectors as in the original study.
- **Four paired methods** within each model/start: original frozen-rate GD,
  gradient-reversal GD, original halving-only MSE backtracking, and the new
  recovering-backtracking variant. All receive identical training data and
  initial parameters within each matched condition.
- Adaptive rates start at 1.0, but original fixed GD retains its frozen
  rate (0.1 Model 1, 0.03 Model 2). 50-retry cap per update, minimum rate
  1e-12, and 10,000-update maximum for each adaptive method.
- **40 total method runs:** 2 models x 1 new noise seed x 5 starts x 4
  methods. Only training and validation labels are accessed; held-out test
  labels and registered evaluation seeds are not used for this experiment.
- This experiment does **not** repeatedly measure runtime; update and MSE
  evaluation counts are not a valid substitute for timed speed measurements.

## Measured outcomes

| Model | Method | Small-gradient stops | Median validation MSE | Median accepted updates | Median training MSE evaluations |
|---|---|---:|---:|---:|---:|
| Exponential | Frozen fixed GD | 5/5 | 0.0102511 | 922 | 923 |
| Exponential | Original gradient reversal | 4/5 | 0.954935 | 531 | 532 |
| Exponential | Halving-only backtracking | 3/5 | 0.0102527 | 1543 | 1546 |
| Exponential | **Recovering backtracking** | **4/5** | **0.0102500** | **812** | **832** |
| Logistic | Frozen fixed GD | 5/5 | 0.0081019 | 2731 | 2732 |
| Logistic | Original gradient reversal | 5/5 | 11.0452 | 44 | 45 |
| Logistic | Halving-only backtracking | 5/5 | 0.0081019 | 1309 | 1314 |
| Logistic | **Recovering backtracking** | **5/5** | **0.0081010** | **654** | **659** |

**Failures matter.** On exponential Model 1, recovering backtracking still
exhausted 10,000 updates from S4, with validation MSE **0.27479**, rather
than the approximately **0.01025** achieved by fixed GD. From S5, however,
the halving-only version also exhausted its budget with MSE **1.07262**;
the recovering version met gradient tolerance with MSE **0.01025** after
5,619 accepted updates. On logistic Model 2, recovery reached a small
gradient and low validation MSE for all five starts, but required 4,179
updates from S4, versus 1,713 for frozen fixed GD.

The original sign-reversal idea remains unsafe as a stand-alone convergence
criterion: it reached a tiny gradient after just **two updates** from two
logistic starts, yet their validation MSE was approximately **38.7**.

These counts are **descriptive**: each model has one independent new noise
dataset; five starts on that dataset are not five independent datasets.
The recovery rule is informed by the earlier pilot, so the present pilot is
not an unbiased confirmatory assessment. No p-values, confidence intervals,
or claims of overall superiority are justified by these data.

## Reproduction

```powershell
python -m unittest discover -s tests -q
python -m scripts.run_recovering_gd_pilot
python -m scripts.run_recovering_gd_pilot --execute
```

The source writes `results/recovering_gd_pilot.csv` with the 40 raw records.
An identical LF-terminated evidence snapshot is committed at
`evidence/recovering_gd_pilot_2026-11-01.csv`.
**SHA-256:**
`f6cd6387831773c35797ff4a7875ed84bfbfa87bbd424814b2b624957933f011`.

## Cautious next steps

The modified learning rate may be faster and more reliable on some starts,
but the only defensible next step toward a formal claim is an independently
registered extension with fresh calibration/evaluation seeds, repeated
optimizer-only timing, complete failure logging and separate test scoring.
The original GD-versus-Newton study must not be retroactively changed.
