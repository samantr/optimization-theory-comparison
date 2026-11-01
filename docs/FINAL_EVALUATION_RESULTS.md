# First full paired evaluation — completed, audited, and archived

**Execution date:** 2026-11-01 (UTC). **Code frozen before execution:**
`f3c1310e5e096670eaf7df8c755494fc5361c931`.

The 10 independently seeded datasets × 5 fixed starts per model were
evaluated with fixed-step GD and pure full-step exact Newton. Safeguarded
Newton is a separately labelled **exploratory** third variant. Every
method/condition was timed five times on the same training observations
and same initial parameters; median of the five optimizer-only times was
recorded. No final-test observation was used for tuning or optimizer updates.

**Audit:** 300/300 unique method conditions, 1,500/1,500 timing repetitions,
all 20 dataset hashes, all recorded timing medians, and both bootstrap
intervals independently recomputed. Raw CSVs and JSON metadata are preserved
in `evidence/evaluation_2026-11-01/`. SHA-256 digests and the code/config
versions are in `config/evaluation_provenance.json`.

## Primary comparison — these are measured results

| Metric | Model 1 GD | Model 1 exact Newton | Model 2 GD | Model 2 exact Newton |
| --- | ---: | ---: | ---: | ---: |
| Registered attempts | 50 | 50 | 50 | 50 |
| Gradient-tolerance termination | 50 | 48 | 50 | 48 |
| Hessian confirms strict local minimum | **50** | **4** | **50** | **22** |
| Indefinite saddle at stationary point | 0 | 44 | 0 | 24 |
| Other stationary/inconclusive curvature | 0 | 0 | 0 | 2 |
| Failure before stationarity | 0 | 2 | 0 | 2 |
| Median of per-condition median optimizer time (s) | 0.066688 | **0.001096** | 0.147090 | **0.001128** |
| Median final *test* MSE | **0.008850** | 1.155205 | **0.009147** | 10.805764 |

In the two Model 1 Newton nonstationary cases, there was one
`ill_conditioned_hessian` and one `numerical_failure`. In Model 2, both
were `ill_conditioned_hessian`. Importantly, Newton's **48/50** small-gradient
terminations in *each* model do **not** mean 48 successful minimizations:
the Hessian indicates saddles or inconclusive curvature in many of them.

On these fixed starts and datasets, GD reached a strict local minimum in
all 100 primary attempts. Its test MSE remained near the synthetic noise
variance, while exact full-step Newton frequently returned a poor fit.
Newton had a lower test MSE than GD on **3/50** paired Model 1 conditions and
**14/50** paired Model 2 conditions (including small differences). Those
fractions are descriptive, **not** independent replicate counts for tests.

Newton also produced a very large finite outlier on Model 1 (test MSE
exceeding `1e12`). We show the **median** test error in the main table
because the arithmetic mean is dominated by that outlier, while retaining
the complete raw record. None of the failed or cap-exhausted results was
discarded or retuned after viewing the test set.

## Dataset-paired runtime differences and uncertainty

Let `Δt = Newton time − GD time`, with each method's time being the median
of five runs. First average the *five* paired differences within each of
the ten datasets; then average those ten dataset values. A negative result
means Newton was faster **per attempted optimization invocation**.

| Model | Mean paired `Δt` | 95% percentile dataset-cluster bootstrap CI |
| --- | ---: | ---: |
| Exponential decay | **−83.24 ms** | [−91.72, −75.36] ms |
| Logistic growth | **−149.54 ms** | [−160.81, −138.33] ms |

The predeclared bootstrap used 10,000 resamples, seed `20261101`, and whole
dataset units; all five starts of each dataset stayed together. The dataset
is the independent sampling unit (`n = 10`), not the 50 paired starts.
With only ten independent datasets, the nominal 95% percentile intervals
may not achieve ideal coverage. See `docs/EVALUATION_RUNNER.md` for the
methodological caution and academic source. **No p-values** are asserted.

**Crucial distinction:** These intervals describe *time spent per attempt*,
including short runs that stopped at saddles or with Hessian failures.
They do **not** estimate Newton's time to achieve an equally good minimum
or equal prediction quality. Consequently, the timing result must not be
presented as evidence of universally superior usable performance.

## Exploratory safeguarded exact-Hessian Newton

The independently labelled safeguarded method is **not** silently replacing
the full-step exact Newton baseline.

| Metric | Model 1 | Model 2 |
| --- | ---: | ---: |
| Hessian-confirmed strict minima | 20/50 | 38/50 |
| Reached iteration cap without gradient-tolerance termination | 30/50 | 12/50 |
| Median optimizer time per condition | 0.011027 s | 0.002589 s |
| Median test MSE | 0.134525 | 0.010743 |

The 100-iteration cap and fallback/line-search costs are preserved exactly
as registered. These exploratory outcomes provide context for why full-step
Newton can behave poorly, but are not mixed into the two-method primary
paired time differences or their confidence intervals.

## Interpretation tied to the mathematics

The prediction functions may be smooth and nonlinear, but minimizing MSE
over three parameters produces a **multidimensional, nonconvex objective**.
In that parameter space, `∇J(θ) ≈ 0` can describe a minimum **or** a saddle.
Exact Newton solves `H(θ) p = −∇J(θ)`; when the Hessian is indefinite, the
direction is not necessarily a descent direction. The recorded saddles
demonstrate why iteration count, gradient tolerance, local curvature, and
prediction error must be discussed separately.

**Conditional findings:** Under these two synthetic nonlinear regressions,
particular noise realizations, frozen starting vectors, GD rates, and
software environment, GD found low-loss strict minima more reliably,
whereas unsafeguarded Newton terminated much sooner but often at poor
stationary points. Neither method is universally preferable; larger model
families, different initial conditions, damping strategies, and repeated
environments could change the results.

## How to inspect and reproduce

The raw archived records include every method/seed/start, failure,
final parameter vector, training/validation/test MSE, gradient norm,
Hessian curvature label, and all 1,500 raw timing values. Source and
configuration hashes appear in each JSON summary and
`config/evaluation_provenance.json`.

From your checkout, `python -m scripts.run_comparison` safely previews the
plan. Re-execution requires the explicit `--execute` flag and will produce
*new host-dependent timing data* in Git-ignored `results/evaluation/`;
those new times should **not** be substituted for the archived original
without documenting a new independent replication. Do not change rates or
other optimizer settings based on the held-out test results.

**Next guided checkpoint:** inspect a representative Newton saddle beside
a GD minimum and explain how the Hessian eigenvalues distinguish them.
Figures and the full academic report remain to be prepared step by step.
