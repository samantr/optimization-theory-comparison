# Gradient Descent learning-rate calibration — executed and frozen

This document reports **actual calibration**, not the final optimizer
comparison. All 240 pre-registered GD runs were executed using the approved
independent calibration seeds, all five starts per model, and the fixed
10,000-iteration ceiling / gradient tolerance `1e-4`. The ten final evaluation
seeds and held-out test labels were not used for fitting or rate selection.

**Frozen model-specific learning rates:**

| Regression model | Selected fixed GD rate | Numerical failures at that rate | Cap-exhausted runs at that rate | Mean validation MSE (15 conditions) |
| --- | ---: | ---: | ---: | ---: |
| Exponential decay | **0.1** | 0/15 | 0/15 | 0.0119727047 |
| Logistic growth | **0.03** | 0/15 | 0/15 | 0.0081189996 |

## Full registered grid, not cherry-picked

Each row has 3 calibration noise seeds × 5 fixed starts = 15 actual GD
attempts. A cap-exhausted run contributed its **last finite** validation MSE;
it was also recorded as not converged. The predeclared ranking minimizes
numerical failures first, mean nonfailed validation MSE second, and learning
rate third (exact ties only).

### Model 1: exponential decay

| Learning rate | Numerical failures | Cap-exhausted | Mean validation MSE |
| ---: | ---: | ---: | ---: |
| 0.0001 | 0 | 15 | 0.347665724 |
| 0.0003 | 0 | 15 | 0.200378975 |
| 0.001 | 0 | 15 | 0.107110978 |
| 0.003 | 0 | 15 | 0.076489178 |
| 0.01 | 0 | 6 | 0.078434165 |
| 0.03 | 0 | 3 | 0.117880960 |
| **0.1** | **0** | **0** | **0.011972705** |
| 0.3 | 0 | 15 | 0.357070873 |

### Model 2: logistic growth

| Learning rate | Numerical failures | Cap-exhausted | Mean validation MSE |
| ---: | ---: | ---: | ---: |
| 0.0001 | 0 | 15 | 0.706567245 |
| 0.0003 | 0 | 15 | 0.262495275 |
| 0.001 | 0 | 15 | 0.090617634 |
| 0.003 | 0 | 15 | 0.037804710 |
| 0.01 | 0 | 3 | 0.008149117 |
| **0.03** | **0** | **0** | **0.008119000** |
| 0.1 | 0 | 0 | 0.008121805 |
| 0.3 | 0 | 2 | 10.896162230 |

The logistic mean validation MSEs for 0.03 and 0.1 differ by only about
`0.000002805`. Selecting 0.03 follows the pre-registered ranking, **not**
an assertion that it is statistically superior to 0.1.

## Scientific interpretation

- All 240 attempts were included; **no numerical failures** occurred in
  these particular attempts. There were 84 cap-exhausted runs across all
  Model 1 rates and 65 across all Model 2 rates. Thus a *finite result*
  must not be confused with proof of convergence.
- Mean calibration validation MSE is a **selection metric**, not a final
  test MSE or evidence that GD beats Newton.
- Calibration elapsed about 50.31 seconds (Model 1) plus 43.51 seconds
  (Model 2) in this environment. These times are tuning overhead and will
  **not** be mixed into GD-vs-Newton optimizer execution-time comparisons.
- The reported values apply to these particular two models, starts,
  training/validation splits, seeds, and fixed-step GD implementation. They
  are not general optimal learning rates.

## Audit, reproduction, and code provenance

- Preregistered manifest: `config/experiment_plan.json`; its status fields
  intentionally remain at the **pre-execution** snapshot. Do not rewrite
  that historical manifest merely to mark this calibration complete.
- Frozen decisions and file digests: `config/selected_gd_rates.json`.
- Source code commit **`344154564ef380c800413426cc81141cde6ec82e`**.
- Runtime: Python `3.12.14`, NumPy `2.3.5`.
- Each run's recorded noise seed, start, rate, final parameters, training
  loss, validation loss, termination reason, iteration/evaluation counts,
  and elapsed time are stored in the **locally regenerated**, Git-ignored
  `results/calibration/model1_runs.csv` and `model2_runs.csv`.
- Additional provenance (configuration/source hashes, calibration dataset
  digests and per-rate summaries) is in the locally regenerated
  `results/calibration/model1_summary.json` and `model2_summary.json`.

Reproduce on your local checkout, in the project root:

```powershell
python -m unittest discover -s tests -q
python -m scripts.calibrate_gd
```

The second command performs **240 new calibration attempts** and takes time;
the generated results directory is ignored by Git. Use `--overwrite` only if
you intend to replace existing calibration files. Floating-point scores should
reproduce under the recorded software stack, but CPU timing will vary by host.
Do not retune the chosen rates based on evaluation/test results.

**Next gate:** freeze timing repetitions, formal outcome classifications,
and statistical aggregation *before* running the 10-dataset × 5-start paired
GD-versus-exact-Newton evaluation grid. No final grid has been executed.
