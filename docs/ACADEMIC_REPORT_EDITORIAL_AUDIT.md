# Editorial audit of the complete Optimization Theory report

**Date:** 2026-11-21  
**Report:** `docs/ACADEMIC_REPORT_DRAFT.md`  
**Status:** Working academic manuscript, pending final layout and scholarly QA.

## Completed structural checks

- Nine report sections present, numbered 1, 2, 3, 4, 5, 6, 7, 8, 9.
- Abstract and keywords precede Section 1.
- 21 scholarly references in the numbered bibliography; all 21 are cited in the body.
- Missing citations: 0; uncited bibliography entries: 0.
- Numbered equations: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16.
- Table numbers: 1, 2, 3, 4, 5; figure numbers: 1, 2.
- Approximate manuscript-body whitespace-token count (including equations and table labels, excluding bibliography): 7055.
- The archived source files and experimental records have not been changed.

## Final-submission checks still outstanding

1. Render PDF/DOCX, inspect page-by-page, ensure professor's **at least 10-page** requirement, readable equations, figures and intact tables, and a suitable cover.
2. Recheck DOI/edition information and source-specific theoretical claims where verification so far relied on abstracts, publisher listings or limited paper excerpts rather than full-text close reading.
3. Verify final reference-list punctuation, typography, cross-references, numbering and consistent technical terminology.
4. Do not imply a small gradient or positive-definite *local* Hessian implies a global optimum.
5. Keep only 10 noise datasets/model as independent bootstrap units; preserve the small-cluster warning.
6. Keep safeguarded Newton exploratory, and exclude subsequent adaptive-GD pilots from the primary comparison.
7. Retain source freeze commit `f3c1310e5e096670eaf7df8c755494fc5361c931`; archived timings remain host-dependent.
8. Add formal title/cover metadata at typesetting, including any university template requirements.

## Historical editorial notes moved out of the report

### Editorial verification (not part of the final manuscript)

- Section 1 was retained as drafted. Section 2 newly uses sources [6]–[14], giving **14 distinct scholarly works actually cited across the first two sections**, not yet the professor's required 20.
- [6] Original Armijo article, pp. 1–3, reviewed; [7] Wolfe publisher abstract confirmed; [8] Barzilai–Borwein abstract/intro and first paper page directly reviewed.
- [9] Conn–Gould–Toint publisher book/chapter description reviewed; [10] Dennis–Schnabel Chapter 10 publisher description reviewed (not the complete chapter).
- [11] Levenberg original article pp. 164–168 checked, including the introductory problem and damped normal-equation derivation; [12] Marquardt publisher publication metadata confirmed (complete primary paper not yet read); [13] Moré original manuscript text checked, including scaled regularized least-squares derivation.
- [14] Dauphin et al. original NeurIPS text and opening analysis checked; claims concerning high-dimensional neural networks are explicitly not generalized to our low-dimensional models.
- Existing sources [4], [5] describe **conditional strict-saddle avoidance** and **modified Newton**, respectively, not universal guarantees for our exact implementations.
- Mathematical Foundations (Section 3) will present and derive the exact Hessian formula, including its residual-weighted second-derivative contribution, and connect it to the source code.
- The primary experiment's frozen GD rates and exact full-step Newton remain unchanged. The safeguarded Newton and two adaptive-GD pilots are separate exploratory analyses.
- The final bibliography is provisional and must be consolidated with at least 20 sources cited in the complete report, with accurate DOI/edition details. The abstract is drafted last.


- Section 3 equations (1)–(10) are **locally numbered in this draft** and must be renumbered consistently across the full report at typesetting.
- Mathematical derivations checked directly against `opttheory/objectives/mse.py`, `opttheory/objectives/logistic_mse.py`, `opttheory/optimizers/gradient_descent.py`, and `opttheory/optimizers/newton.py`.
- Professor Becerikli's `yOptmT-h5.pdf` p. 7 demonstrates \\(q(x_1,x_2)=2x_1^2-3x_1x_2+2x_2^2\\) and Hessian eigenvalues \\(1,7\\); pp. 8–9 cover stationary points and Hessian classification. `yOptmT-h6.pdf` p. 12 derives Newton from the quadratic/Taylor approximation. `yOptmT-h8.pdf` p. 3 lists various stopping criteria.
- Source [15] (Björck) was checked via SIAM's 1996 edition and its published Chapter 9 description; full technical chapter was not read. Source [16] (Boyd–Vandenberghe) author-hosted text verifies the convexity terminology; source [17] (Gill–Murray–Wright) is catalogued as an **original 1981** book, despite the 1986 date on the professor's recommended-books slide. Avoid attributing precise theorem/page numbers to a book whose full relevant chapter has not been inspected.
- References newly added in this section: [15] Björck, [16] Boyd–Vandenberghe, [17] Gill–Murray–Wright. **17 distinct scholarly items** are now cited in the Sections 1–3 draft; at least 20 actually used references are needed in the finished manuscript.

- Section 4 checked against `opttheory/models/exponential.py`, `opttheory/models/logistic.py`, `opttheory/datasets/synthetic.py`, `config/model1.json`, `config/model2.json`, `config/experiment_plan.json`, and `opttheory/experiments/evaluation.py`.
- **Important source nuance:** `config/model1.json` retains an initial pilot noise seed (`42`), but the **final evaluation runner explicitly overrides** it with the preregistered trial seeds 1000–1009. The logistic initial seed 1100 is likewise replaced by the trial's registered seed for each dataset. We report only the **executed evaluation conditions** in Table 1.
- Table 1 and equations (11)–(13) must be typeset and numbered consistently in the final paper; data-split statistics are sourced from project code, not falsely attributed to references [1]/[15].
- No invented measured outcomes or figures; Section 4 describes the already registered and audited data-generation protocol.

- Section 5 source-code cross-check: `opttheory/optimizers/gradient_descent.py`, `opttheory/optimizers/newton.py`, `opttheory/optimizers/newton_safeguarded.py`, `opttheory/experiments/policy.py`, `opttheory/experiments/evaluation.py`, `config/selected_gd_rates.json` and `config/evaluation_protocol.json`. No changes to optimizer code, selected rates, or frozen evaluation evidence.
- One implementation nuance: the Newton condition-number check applies the default limit `1e12`; **indefinite matrices are not automatically rejected** in the exact Newton baseline. The diagnostic uses `eigvalsh` with curvature tolerance `1e-8` after optimizer timing and only labels stationary endpoints.
- Professor's source alignment: `yOptmT-h6.pdf`, pp. 12–16 (quadratic approximation and Newton update); `yOptmT-h7.pdf`, pp. 2–3 (modified Newton discussion); `yOptmT-h8.pdf`, p. 3 (stopping criteria); consult original PDFs for final typography.
- New scholarly bibliography entry **[18]**: Nesterov (2004), verified via Springer bibliographic metadata. Its convex optimization results are introduced only with **explicit qualification** and are not treated as guarantees for our nonconvex regression models.
- The algorithm pseudocode deliberately states both unequal iteration caps and does not imply the same wall-clock budget. `converged` refers to gradient tolerance, not Hessian-confirmed minimization. Combined report citation count is **18 distinct academic works**; at least 20 genuinely cited works are required before submission.

- Section 6 verified against `config/experiment_plan.json`, `config/selected_gd_rates.json`, `config/evaluation_protocol.json`, `config/evaluation_provenance.json`, `docs/CALIBRATION_RESULTS.md`, `docs/EVALUATION_RUNNER.md`, `docs/FINAL_EVALUATION_RESULTS.md`, `opttheory/experiments/evaluation.py`, and `opttheory/experiments/plan.py`. The two preregistration files retain their **historical pre-execution status fields**; actual execution is established by the archived outputs and post-execution provenance. Do not rewrite the frozen files.
- New distinct scholarly references genuinely cited in this section: [19] Efron (1979), [20] Davison & Hinkley (1997), [21] Cameron et al. (2008), bringing the report to **21 in-text-cited scholarly works** (minimum 20). Source [21] is used for dependence/few-cluster concerns; it does not validate the percentile interval employed in this study.
- Section 6 deliberately excludes calibration overhead from optimization times, does not collapse convergence into test quality, and maintains the **GD 10,000 / Newton 100** unequal iteration caps. Refer to original records for timings and failure classifications in Section 7 rather than inserting results prematurely.

- Section 7 derives **only from audited original evidence**: `evidence/evaluation_2026-11-01/model1_runs.csv`, `model2_runs.csv`, `model1_summary.json`, `model2_summary.json`, and `docs/FINAL_EVALUATION_RESULTS.md`. The summary's dataset differences were recomputed from all paired GD/Newton run-level medians before generating the figures.
- Figures 1 and 2 are descriptive graphics computed from the archived counts and dataset-level timing summaries; no new benchmark results or optimizer trajectories were generated. Original method-specific runtime, failure cases, and held-out errors remain unchanged.

- Section 8's Table 5 was independently tabulated from all 50 exact-Newton conditions per model in `evidence/evaluation_2026-11-01/model1_runs.csv` and `model2_runs.csv`; starting-vector groups are not independent datasets. Counts: Exponential [0,0,4,0,0], logistic [10,8,4,0,0].
- The saddle-point mechanism is a local Taylor/eigenspace interpretation grounded in the mathematics and relevant literature [3], [14], not a numerical re-execution or a claim of guaranteed finite-iteration behavior.
- Citations for the statistical caution [19]–[21] and modified-Newton context [5], [9], [13] are **qualified** to distinguish their published methods from the present code. No additional scholarly sources were padded into the list.
- Limitations explicitly include two small synthetic models, five selected initializations, independent GD-only calibration, unequal caps, host-dependent timing, ten bootstrap clusters, approximate stationarity, and lack of global-optimality proof.

