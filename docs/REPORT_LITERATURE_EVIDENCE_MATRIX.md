# Literature evidence matrix — first close-reading pass

**Updated:** 2026-11-21 (second source-reading pass)  
**Project:** Official fixed-rate GD versus full-step exact Newton nonlinear-regression study  
**Stage:** Reference collection and claim verification; **the report's Introduction has not been drafted**.

## Source access and review status

- **Read directly from the original scholarly source:** source text, specified sections/pages or publisher metadata were opened. A scholarly abstract is not equivalent to full-paper review.
- **Publisher abstract verified:** publication and abstract checked, but technical proof or complete article not verified.
- **Needs library copy:** the publisher listing is identifiable, but close reading is not complete.
- See `docs/REPORT_REFERENCE_REGISTER.md` for all **24 scholarly candidates** and `docs/REPORT_FULLTEXT_ACCESS.md` for accessible source links.
- The professor's Moodle screenshot requests **at least 20 academic sources** and **fewer than 20% direct internet-document references**. It does **not** request copies of every cited PDF. A DOI or publisher link used to locate a scholarly book/article is not a separate web-only reference.

## Evidence matrix: initial six high-priority sources

| ID | Source / checked location | Evidence from the source | Intended report use | Important limit / do not claim | Status |
|---|---|---|---|---|---|
| **R18** | Lee, Simchowitz, Jordan & Recht (2016), *Gradient Descent Only Converges to Minimizers*. [PMLR paper](https://proceedings.mlr.press/v49/lee16.html), [PDF](https://proceedings.mlr.press/v49/lee16.pdf), pp. 1, 3, 5, Theorem 4 and §2 | For a twice continuously differentiable objective with Lipschitz gradient, small fixed step `0 < alpha < 1/L`, and random initialization distributed absolutely continuously, the set of starting points that converge to a **given strict saddle** has probability zero. §4 includes the extension to countable strict saddle sets. | Context for why first-order GD can avoid strict saddles; use in Discussion to interpret—but not prove—our observed GD successes. | **Do not attribute this theorem to our exact experiment as a guarantee.** We used a small fixed list of starting vectors, not random initialization as assumed in the theorem, and did not establish a global Lipschitz gradient / `alpha < 1/L` for the nonlinear objectives. The theorem concerns asymptotic behavior, not a finite 10,000-update cap. | **Close-read** title, introductory theorem statement, §§2 and 4 |
| **R17** | Cartis, Gould & Toint (2010), *SIAM Journal on Optimization* 20(6), 2833–2852, [DOI](https://doi.org/10.1137/090774100); abstract and journal author's accessible [PDF](https://www.numerical.rl.ac.uk/media/people/nick-gould/CartGoulToin10_siopt.pdf), pp. 2833–2834 | Standard steepest descent and Newton methods can each display a worst-case stationarity evaluation complexity close to order `epsilon^(-2)`; cubic regularization has a better established worst-case bound. Paper explains the difficulty of full Newton with singular or indefinite Hessian and its locally favorable behavior under positive-definite curvature. | Background on why iteration count and Newton curvature information do not guarantee global reliability; carefully distinguish worst-case theoretical evaluation complexity from our measured runtime. | **Do not** claim that the `epsilon^(-2)` result predicts our observed number of saddle terminations or average runtime. The methods and assumptions of worst-case construction differ from our fixed-step GD / naive full-step Newton implementation. | **Publisher abstract + introductory source text examined**; full proofs not yet reviewed |
| **R22** | Paternain, Mokhtari & Ribeiro (2019), *SIAM Journal on Optimization* 29(1), [published paper](https://doi.org/10.1137/17M1150116), [author preprint](https://arxiv.org/abs/1707.08028), abstract / opening discussion | Proposed Newton-based nonconvex method modifies Hessian treatment by turning negative eigenvalues into their absolute values (with truncation), to facilitate saddle escape. | Motivation for **separately labeled modified Newton methods** in Related Work or Future Work; explain why naive full-step exact Newton may aim toward non-minimum stationary solutions. | **Not the Newton method we evaluated.** Do not transfer the paper's saddle-escape guarantees, iteration bounds, or eigenvalue-handling procedure to our exact Newton baseline or to our ad-hoc safeguarded fallback. | **Publisher abstract verified**; preprint opened, further technical reading pending |
| **R21** | Cameron, Gelbach & Miller (2008), *Review of Economics and Statistics* 90(3), 414–427, [journal metadata](https://doi.org/10.1162/rest.90.3.414); [2007 NBER working-paper version](https://www.nber.org/papers/t0344), pp. 2, 4–6, §§1–2 | Dependence within clusters invalidates simplistic independent-observation standard errors; conventional large-cluster asymptotics can perform poorly with few clusters. Working paper examines **cluster bootstrap-t** approaches and notes whole-cluster resampling. | Support the decision to treat **10 independent noisy datasets/model**, not 50 starts, as units for uncertainty. Show the limitation of a nominal 95% CI based on only 10 datasets. | **Their procedure is not our procedure.** The paper focuses on clustered regression inference and bootstrap-t, whereas our analysis uses a much simpler percentile bootstrap of paired dataset-level timing summaries. Cite for caution and motivation only, never for a claimed validated coverage rate of our intervals. | **NBER abstract, introduction and method overview read**; published-version text not yet checked line-by-line |
| **R06** | Dennis & Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*, Ch. 10, [SIAM chapter listing](https://epubs.siam.org/doi/abs/10.1137/1.9781611971200.ch10) | The chapter explicitly distinguishes the Gauss–Newton and Levenberg–Marquardt family, arising from nonlinear-equation solving, from Newton and secant approaches based on unconstrained minimization. | Explain why we calculate the **exact residual-dependent Hessian**, rather than use the Gauss–Newton approximation. | Publisher chapter abstract is **not sufficient** for precise derivations or theorem attribution; obtain chapter/full text before assigning internal page/equation references. | **Publisher chapter description checked; needs full chapter** |
| **R01** | Nocedal & Wright (2006), *Numerical Optimization*, 2nd ed., [Springer book](https://doi.org/10.1007/978-0-387-40065-5), contents including nonlinear least squares and Newton methods | Authoritative, relevant second edition identified, with chapters on unconstrained optimization and nonlinear least squares. | Main mathematical and implementation reference for objective/gradient/Hessian, Newton solves and numerical stability. | No precise page-specific guarantee or equation is attributed until the relevant chapter is read. | **Bibliographic/contents verified; needs full chapter** |

## Three literature-informed boundaries for the official report

1. **The model being exponential does not make the parameter-space MSE convex.** Our empirical study minimizes the MSE over three parameters; use a local Hessian check to classify each final point. The noise-free prediction curve and optimization landscape are different functions.
2. **Newton reached a small gradient in many runs, but many of those points were saddles.** This is our own archived experimental observation; do not claim the research articles measured our specific rates or counts.
3. **Faster per-attempt Newton time is not faster time-to-good-minimum.** Archive the failure cases and poor fits; do not trim to only favorable starts when producing comparisons.

## Unresolved source work before drafting

- Obtain readable chapter text for R01 (Nocedal & Wright: Newton/line-search chapter and Ch. 10), R03 (Bates & Watts), R06 (Dennis & Schnabel Ch. 10), R23 (Gill–Murray–Wright) and R24 (Rao) as needed.
- Read R19 (Dauphin et al.) and R14–R16 (Levenberg–Marquardt) for correctly framed related-work comparisons.
- Verify whether exact nonlinear-loss gradients have globally bounded Lipschitz constants on our **unconstrained** parameter domains before interpreting R18; do not assume they do.
- Cite at least 20 **actually used** publications in the final report. The 24-item selection pool is not a guarantee that every entry will make it into the paper.
- Page references in this matrix are grounded in inspected full texts, publisher pages, or the named publicly accessible working-paper version. Do not silently convert NBER working-paper page numbers into published-journal page numbers.

## Course alignment

Professor Becerikli's Week 5 slides (`yOptmT-h5.pdf`, pp. 4–9) introduce the Hessian and critical-point classification. Week 6 (`yOptmT-h6.pdf`, pp. 12–16) develops Newton updates and the one-step quadratic case. The literature provides broader context; the **professor's definitions and examples** remain the teaching anchor for this course.

## Second source-reading pass — additional primary papers

These are additions to the six items above. "Text examined" means the original
PDF's extracted text or an original author-hosted accessible HTML rendering
was inspected; **it does not mean every page and proof has been read**.
The web viewer's PDF screenshot endpoint was attempted; some snapshots were
unavailable, so do not claim a visual inspection of those pages.

### R19 — Dauphin et al. (2014): saddle-free Newton

**Full source:** [original NeurIPS proceedings paper](https://proceedings.neurips.cc/paper_files/paper/2014/file/17e23e50bedc63b4095e3d8204ce063b-Paper.pdf).

- **Checked:** abstract and §§1, 4–5, 7; particularly PDF pp. 1, 4–7;
  Section 4 explains the eigenvector-direction behavior of Newton near an
  indefinite critical point.
- **Evidence:** Write locally `g_i=lambda_i delta_i` in the Hessian
  eigenbasis. GD's displacement `-alpha*g_i` points away from a critical
  point in a negative-curvature direction (`lambda_i<0`), whereas
  full-step exact Newton gives `-g_i/lambda_i=-delta_i`, which points
  **toward** the critical point even when it is a saddle.
- **Report claim:** This explains why a numerically converged exact Newton
  iteration is not automatically a minimization success; connect to our
  **measured Hessian eigenvalues** and poor-MSE Newton outcomes.
- **Limit:** The paper focuses on **high-dimensional neural networks**;
  statistical claims about the prevalence of saddles in high dimensions
  are not measurements of our three-parameter exponential/logistic models.
  Do not generalize their reported network accuracy or speed to our project.
- **Status:** Direct primary paper text examined, selected sections.

### R16 — Moré (1978): Levenberg–Marquardt derivation

**Source:** [1977 original Argonne report / prepublication
version at UNT](https://digital.library.unt.edu/ark:/67531/metadc1448594/),
[archived readable scan](https://hwbdocs.env.nm.gov/Los%20Alamos%20National%20Labs/General/32166.pdf).
**Final citation:** Moré, J. J. (1978), *Numerical Analysis*, LNM 630,
pp. 105–116, DOI 10.1007/BFb0067700.
The **1977 report date** belongs to the earlier manuscript, not to the
1978 Springer proceedings entry.

- **Checked:** paper's printed pp. 105–108, §1 and §2 derivation
  (including eqs. 2.3–2.5); §3, equations 3.1 and 3.2.
- **Evidence:** For residual vector `F(theta)`, linearize
  `F(theta+p) ~ F(theta)+J p`. A constrained/scaled subproblem is
  solved via a parameterized system of the form
  `(J^T J + lambda D^T D) p = -J^T F`; the implementation also
  **rejects** candidate updates that fail to reduce the residual norm.
  It discusses potential numerical problems when forming normal
  equations near rank deficiency.
- **Report claim:** LM introduces Jacobian-based regularization and
  scaling, offering a useful **related-work comparison** when Newton's
  exact Hessian is indefinite or ill-conditioned.
- **Limit:** LM's normal-equation matrix is **not** the full nonlinear
  least-squares Hessian `(2/n)(J^T J + sum_i r_i H_i)` used by our
  exact Newton solver. Our explicitly labelled safeguarded Newton is
  also **not** Moré's LM algorithm.
- **Status:** Direct primary-source sections read.

### R11 — Armijo (1966): modified steepest descent

**Full source:** [Pacific Journal of Mathematics, volume 16(1), printed
pp. 1–3](https://msp.org/pjm/1966/16-1/pjm-v16-n1-p01-s.pdf).

- **Checked:** printed pp. 1–3; conditions I–IV, convergence theorem,
  Corollary 2 and discussion.
- **Evidence:** A modified steepest-descent method tries an initial
  step and geometrically reduces it until a **sufficient-decrease**
  condition holds. The paper gives convergence statements under
  assumptions including Lipschitz continuity on the starting-level
  set and extra conditions implying a unique minimizer.
- **Report claim:** Context for standard backtracking and for the course's
  modified methods.
- **Limit:** An arbitrary **nonconvex** regression MSE does not necessarily
  meet those global conditions. Our final **fixed-step GD** did not use
  Armijo backtracking, and our later personal adaptive-GD pilots are
  outside the official comparison.
- **Status:** Direct original text read; first source-page screenshot
  successfully displayed, remaining screenshots not available.

### R20 — Efron (1979): bootstrap

**Source:** [The Annals of Statistics 7(1), 1–26](https://doi.org/10.1214/aos/1176344552);
[public reading copy](https://ledaliang.github.io/journalclub/_static/efron1979/Efron1979.pdf).

- **Checked:** printed pp. 1–2 (abstract/introduction), §8 discussion
  of resampling.
- **Evidence:** The paper introduces the nonparametric bootstrap to
  estimate a statistic's sampling behavior using new samples drawn
  with replacement from the empirical observations. The original
  paper discusses examples and the jackknife's relation to the
  bootstrap.
- **Report claim:** Foundational citation for the idea of bootstrap
  resampling.
- **Limit:** This is **not** a validation of our specific 10-cluster,
  10,000-resample **percentile** interval. R21 and a limitations paragraph
  are needed to discuss within-dataset dependence and small numbers
  of independent noise seeds.
- **Status:** Original article text examined in part.

### R22 — Paternain, Mokhtari and Ribeiro (2019): truncated positive Hessian

**Author version:** [arXiv:1707.08028](https://arxiv.org/abs/1707.08028),
[readable HTML](https://ar5iv.labs.arxiv.org/html/1707.08028).
**Final version:** *SIAM Journal on Optimization* 29(1), 343–368,
DOI 10.1137/17M1150116.

- **Checked:** abstract, introductory description and Definition 2.1.
- **Evidence:** A Newton-like update uses the **positive-definite
  truncated inverse** rather than the unmodified inverse Hessian:
  negative curvature eigenvalues are transformed to their absolute
  values and small-magnitude modes are truncated.
- **Report claim:** A concrete literature-motivated *alternative* for
  negative-curvature saddle points.
- **Limit:** This paper's proved escape properties require its own
  assumptions and **do not apply** to our baseline exact Newton or
  separately labelled safeguarded Newton implementation.
- **Status:** Author-manuscript abstract, definition and opening section
  read; proofs remain to examine if cited beyond the method overview.

### R14 / R15 — early historical methods (metadata-only)

- **R14:** Levenberg (1944), *Quarterly of Applied Mathematics*
  **2(2), 164–168**, DOI 10.1090/qam/10666; publisher metadata
  confirms publication, but **no original full-text analysis yet**.
- **R15:** Marquardt (1963), *Journal of the Society for Industrial
  and Applied Mathematics* **11(2), 431–441**,
  DOI 10.1137/0111030; SIAM journal metadata confirms publication,
  but **no original full-text analysis yet**. Search-indexed summaries
  can conflate this paper with other least-absolute-deviation papers;
  do not use an unverified secondary summary as the source of equations.
- **Next:** inspect full historical texts only if we need a historical
  mechanism claim beyond the publication and title.

## Updated readiness

- **Original candidate pool:** 24 scholarly works (unchanged).
- **Source-specific claim notes:** 10 unique sources with substantive
  evidence notes (R01, R06, R11, R16–R22). Depth varies; book-only
  notes (R01/R06) are explicitly flagged as needing full chapters.
- **Two extra historical references checked only for metadata:** R14/R15.
- **Introduction:** not started, per student instruction.
- **Priority next:** review nonlinear-regression textbook chapters,
  the exact-Hessian versus Gauss–Newton derivation, and source sections
  on small-cluster uncertainty; then finalize the bibliography based
  on references actually used, not merely collected.
