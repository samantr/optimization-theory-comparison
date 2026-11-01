# Academic report outline (approved reference-first workflow)

**Working English title:** Comparative Analysis of Gradient Descent and Newton Optimization Methods in Nonlinear Regression Problems

**Course:** Optimization Theory (BLM 599), Kocaeli University  
**Instructor:** Prof. Dr. Yaşar Becerikli  
**Student:** Saman Tarighpeima  
**Status:** Planning outline, **not** a completed manuscript. The original experiment code, protocol and archived results are frozen.

## Format and compliance

- The professor's posted project announcement requires an individual report of **at least 10 pages** covering the title/cover information, introduction, theories, methods, algorithms, application, and sources; a class presentation takes place in the final two weeks.
- Include **at least 20 genuine references**. Fewer than 20% may be standalone direct web/document-address sources. The 22 candidates in `docs/REPORT_REFERENCE_REGISTER.md` are scholarly books/articles/proceedings, but the final bibliography must contain **at least 20 works actually cited** in the text.
- Working format: English, approximately **15 pages** including cover and references; use numbered equations, cited tables/figures, consistent notation, and an IEEE-style numbered reference list. Page allocations below are flexible and must be checked after typesetting.
- Course slides are pedagogical anchors, **not replacements for the academic bibliography**. Reference IDs below refer to the scholarly register, not the lecture files.

## Proposed structure and page budget

| Part | Approx. pages | Main content and relevant scholarly references |
|---|---:|---|
| Cover | 1.0 | University, department/course, student, professor, approved title, date. |
| Abstract and keywords | 0.5 | Objective, models, paired design, principal success-versus-timing result; write **last**. |
| **1. Introduction** | 1.0 | Nonlinear regression as optimization; why first- versus second-order methods; precise research question, contribution and scope. **R01, R03, R04.** |
| **2. Related Work** | 1.0 | Classical gradient/Newton, safeguards and nonlinear least-squares alternatives; established saddle-point and complexity results; explicitly avoid claiming novelty. **R02, R06–R08, R10–R19, R22** (cite selectively). |
| **3. Mathematical Foundations** | 1.5 | Residuals and MSE; vector gradient; exact Hessian and residual-curvature term; Taylor updates; local minima, saddles, positive definiteness, stationarity. **R01–R06, R10, R17–R18.** |
| **4. Regression Models and Data** | 1.0 | Exponential decay and logistic growth equations and parameter meaning; additive Gaussian noise; 120 observations, 72/24/24 split; true parameters for synthetic generation only. **R03–R05.** |
| **5. Optimization Algorithms** | 1.5 | From-scratch fixed-step batch GD and full-step exact Newton, update equations, pseudocode, stopping caps, singular/indefinite-Hessian issues. Distinguish Gauss–Newton, LM and separately labelled safeguarded Newton. **R01, R05–R07, R11–R16, R22.** |
| **6. Experimental Methodology and Reproducibility** | 1.5 | Ten datasets/model, five fixed paired starts/dataset, rates frozen via independent calibration, identical data/starts, timing protocol, success classification, test isolation, dataset-cluster bootstrap and hashes. **R01, R05, R09, R20–R21.** |
| **7. Results** | 2.0 | Measured convergence outcomes, Hessian labels, train/test loss, failures, iterations, per-attempt runtime and bootstrap CIs; carefully labelled figures and tables. **Archived measured evidence, not hypothetical results.** |
| **8. Discussion and Limitations** | 1.5 | Newton's small-gradient terminations at saddles; why faster per attempt does not mean faster to an acceptable minimum; different starts/model geometry; effect of safeguards; limited synthetic models and only 10 independent dataset clusters. **R01, R07–R09, R17–R22.** |
| **9. Conclusion and Future Work** | 0.5 | Conditional answer to the research question; fixed-GD versus exact Newton tradeoff; brief optional future work on line searches/adaptive steps, **without** mixing exploratory pilot outcomes into the primary study. |
| References | 2.0 | At least 20 **used** scholarly sources, fully formatted; DOI/links as location aids, not web-page substitutes. |
| **Estimated total** | **15.0** | Actual length depends on figures, typography and final citation formatting. |

The main text (Introduction through Conclusion, excluding cover, abstract and references) is budgeted at roughly **11.5 pages**, so the ten-page requirement is not being satisfied only by front matter.

## Explicit connection to Prof. Becerikli's lecture materials

| Uploaded lecture / PDF page | Topic demonstrably shown | Report connection |
|---|---|---|
| `yOptmT-h1.pdf`, pp. 2–4 | Course scope: gradient methods, Newton, modified Newton, Marquardt, least squares; instructor's book suggestions | Sections 1, 2 and 5. Do not imply all algorithms were implemented. |
| `yOptmT-h2.pdf`, pp. 4–6 | Local/global minima and dependence on the starting point; method classifications | Sections 1, 3 and 8: initialization sensitivity and what a local minimum means. |
| `yOptmT-h3.pdf`, pp. 6, 9 | Concrete optimization objective example; Rosenbrock function | Section 3: translating an applied problem into minimization and contrasting objective landscapes. |
| `yOptmT-h4.pdf`, pp. 2–6 and 12–14 | Stationary points, multimodality, contour plots and gradient vector | Sections 3, 5 and 8: stationary does **not** imply minimum. |
| `yOptmT-h5.pdf`, pp. 2–9 | Partial derivatives, gradient, Hessian/curvature matrix, extrema conditions | Sections 3, 5 and 7: second-order classification of Newton outcomes. |

These are **page references to the supplied lecture PDFs**, not claims about unprovided later course weeks. General Newton/least-squares derivations are independently supported by the academic bibliography.

## Core equations to number and explain

1. **Nonlinear prediction:** `f_1(x; a,b,c) = a exp(bx) + c`, and `f_2(x; L,k,x_0) = L / (1 + exp[-k(x-x_0)])`.
2. **Residual and cost:** `r_i(theta) = f(x_i; theta) - y_i`; `J(theta) = (1/n) sum_i r_i(theta)^2` (training observations only during optimization).
3. **Exact gradient and Hessian:** `grad J = (2/n) sum_i r_i grad r_i`; `H_J = (2/n) sum_i [(grad r_i)(grad r_i)^T + r_i H_{r_i}]`. **Keep the second residual-curvature term; omitting it changes the method.**
4. **Batch GD:** `theta_(t+1) = theta_t - alpha grad J(theta_t)`.
5. **Exact full-step Newton:** solve `H_J(theta_t) p_t = -grad J(theta_t)`; `theta_(t+1) = theta_t + p_t` (a linear solve, not an explicit inverse).
6. **Curvature check:** zero gradient indicates stationarity; a positive-definite Hessian supports a strict local minimum, an indefinite Hessian establishes a saddle, and semidefinite/near-singular curvature can be inconclusive.

## Experimental facts the report MUST preserve

- **Model 1:** exponential decay, `[a,b,c]`; **Model 2:** logistic growth, `[L,k,x_0]`.
- Each model used ten separately noisy datasets, five shared fixed starts per dataset, 120 observations/dataset with 72 train / 24 validation / 24 test, Gaussian noise SD 0.1; calibration and evaluation seeds were disjoint.
- Frozen GD rates: **0.1** (exponential) and **0.03** (logistic); common gradient-norm tolerance `1e-4`; iteration caps **10,000 GD** versus **100 full-step exact Newton**. Report these differences transparently rather than calling the iteration budgets equal.
- **Primary comparison:** 50 GD and 50 exact Newton attempts *per model*, with five optimizer-only timing repetitions each. A **separate exploratory safeguarded Newton** adds 100 method attempts; all 300 original method conditions and 1,500 raw timing measurements were archived/audited.
- GD reached Hessian-confirmed strict local minima in **50/50 for each model**. Exact Newton reached strict minima in **4/50** for Model 1 and **22/50** for Model 2; Model 1 had **44** saddle terminations, Model 2 **24**, with additional failures/inconclusive cases. Never mislabel all small-gradient terminations as successful minimizations.
- Mean paired exact-Newton-minus-GD runtime per *attempt*: **-83.24 ms** for Model 1 (95% dataset-bootstrap interval [-91.72, -75.36] ms); **-149.54 ms** for Model 2 ([-160.81, -138.33] ms). These **do not** estimate time to achieve equal-quality minima. No p-values were computed.
- Main median test MSE: Model 1 GD **0.008850**, Newton **1.155205**; Model 2 GD **0.009147**, Newton **10.805764**. Disclose the extreme Model 1 Newton outlier; never hide divergent/failing runs or infer universal superiority.
- Independent sampling for the uncertainty calculation is **ten noise datasets/model**, not 50 starts. No parameter or optimizer was tuned using final-test observations.

## Figures and tables to prepare (not yet claimed as generated)

1. **Figure 1:** two example synthetic data plots with model curves, training/validation/test observations; explain that the known curve was used only to generate synthetic data.
2. **Figure 2:** representative paired GD versus exact-Newton training-MSE trajectories versus iteration and elapsed optimizer time, annotated with final Hessian classification; use archived runs if trace data exist, otherwise **rerun transparently as a separate illustrative diagnostic** without replacing original measured results.
3. **Table 1:** dataset/optimizer configuration, parameter vectors, learning rates and stopping rules.
4. **Table 2:** audited outcomes per model and method (minima, saddles, inconclusive, failures, median test MSE and median runtime).
5. **Figure 3 or Table 3:** paired per-dataset runtime differences with whole-dataset bootstrap intervals, making the per-attempt interpretation unmistakable.
6. **Optional Figure 4:** surface/contour or Hessian eigenvalue example illustrating saddle-versus-minimum classification, tied to the professor's gradient/Hessian lecture content.

Figures should be kept only if they make a concrete result easier to interpret within the page budget. No unverified plot or new benchmark is to be passed off as part of the archived original experiment.

## Drafting and verification sequence

1. Obtain approval of this outline and title; leave references in `docs/REPORT_REFERENCE_REGISTER.md` for now.
2. Draft **Section 1: Introduction** with the precise research question and 3–5 genuine citations; check claims against the cited works.
3. Draft the theoretical sections; trace equations to the implemented analytical derivatives and the course's Hessian/grading material.
4. Draft the methodology directly from the frozen manifests and results, then generate only needed figures from the archived evidence.
5. Draft Results, Discussion, Conclusion, then Abstract last. Keep the safeguards and adaptive-GD pilots explicitly secondary.
6. Compile at least 20 actually cited scholarly entries; inspect citations, equations, reproducibility and page layout; prepare the professor's in-class presentation afterward.

**Canonical evidence:** `docs/FINAL_EVALUATION_RESULTS.md`, `docs/RESEARCH_PROTOCOL.md`, `config/evaluation_provenance.json`, `evidence/evaluation_2026-11-01/`. The later adaptive-GD experiments are separate exploratory work.
