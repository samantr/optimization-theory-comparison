# Academic report: reference register (before drafting)

**Project:** Comparative Analysis of Gradient Descent and Newton Optimization
Methods in Nonlinear Regression Problems

**Student:** Saman Tarighpeima — Kocaeli University, Prof. Dr. Yaşar Becerikli

**Report language:** English (approved 2026-11-01).
**Status:** Bibliographic metadata and relevance checked against authoritative
publisher, original-proceedings, or author/institution records; this is a
source-selection register, **not yet a complete critical reading of all
full texts**. We will verify theorem assumptions and page-level quotations
before using specific claims in the manuscript.

## Professor's requirements and source policy

The provided course announcement screenshot specifies at least **10 pages**,
at least **20 sources**, a cover, topic introduction, methods, theories,
algorithms, and explanation of the application. Books, technical journal
articles, conference papers, and theses are acceptable. It also restricts
**direct document/web-address sources to strictly less than 20%** of the
bibliography. To be safe, the sources below are identifiable published
scholarly works, **not unsourced web pages**; the DOI/publisher URLs are
verification and access links, not a replacement for bibliographic entries.
If exactly 20 items are used, at most 3 should be standalone web-only
sources. We currently plan zero such sources.

## Verified candidates and how each will be used

All 22 items below are different scholarly works. The numeric IDs are
provisional; final reference numbering will follow the manuscript's citation
order (likely IEEE numbered references). The stated use is a *plan*, not a
claim that every paper has already been analyzed in full.

### A. Core optimization and nonlinear regression books (10)

**R01.** Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*,
2nd ed. Springer. https://doi.org/10.1007/978-0-387-40065-5

- Use: GD versus exact Newton, Hessian, stopping rules, conditioning,
  convergence and line-search caveats. **Core source** for theory/methodology.

**R02.** Boyd, S., and Vandenberghe, L. (2004). *Convex Optimization*.
Cambridge University Press. https://web.stanford.edu/~boyd/cvxbook/

- Use: definitions of convexity, descent, backtracking; explicitly explain
  why our nonlinear parameter-MSE objectives are **not presumed convex**.

**R03.** Bates, D. M., and Watts, D. G. (1988). *Nonlinear Regression Analysis
and Its Applications*. Wiley. https://doi.org/10.1002/9780470316757

- Use: parameter-dependent nonlinear least-squares, data-fitting geometry,
  initialization sensitivity, and iterative estimation.

**R04.** Seber, G. A. F., and Wild, C. J. (2003). *Nonlinear Regression*.
Wiley. https://www.wiley.com/en-us/nonlinear-regression-p-9780471471356

- Use: regression-model definitions, residuals/noise, and statistical
  interpretation of nonlinear fitting.

**R05.** Björck, Å. (1996). *Numerical Methods for Least Squares Problems*.
SIAM. https://doi.org/10.1137/1.9781611971484

- Use: numerical properties of least-squares objectives and the difference
  between nonlinear least squares and simple linear normal equations.

**R06.** Dennis, J. E., Jr., and Schnabel, R. B. (1996 SIAM republication;
original 1983). *Numerical Methods for Unconstrained Optimization and
Nonlinear Equations*. SIAM.
https://doi.org/10.1137/1.9781611971200

- Use: exact Newton for minimization versus Gauss-Newton for nonlinear
  least squares (especially Chapter 10), and globally convergent variants.

**R07.** Conn, A. R., Gould, N. I. M., and Toint, Ph. L. (2000).
*Trust-Region Methods*. SIAM.
https://doi.org/10.1137/1.9780898719857

- Use: safeguard context when a full exact Newton step is unreliable; not
  a claim that our primary implementation used a trust region.

**R08.** Cartis, C., Gould, N. I. M., and Toint, Ph. L. (2022).
*Evaluation Complexity of Algorithms for Nonconvex Optimization: Theory,
Computation and Perspectives*. SIAM.
https://doi.org/10.1137/1.9781611976991

- Use: first/second-order optimization complexity and careful distinction
  between reaching stationarity and achieving second-order optimality.

**R09.** Davison, A. C., and Hinkley, D. V. (1997). *Bootstrap Methods
and Their Application*. Cambridge University Press.
https://www.cambridge.org/core/books/bootstrap-methods-and-their-application/ED2FD043579F27952363566DC09CBD6A

- Use: resampling, percentile intervals, and interval limitations. Explain
  our actual **dataset-cluster** resampling rather than generic row bootstrap.

**R10.** Nesterov, Y. (2004). *Introductory Lectures on Convex Optimization:
A Basic Course*. Kluwer/Springer.
https://doi.org/10.1007/978-1-4419-8853-9

- Use: GD convergence theory *under convex/smooth assumptions*; do not
  transfer those guarantees unqualified to our nonconvex MSE objectives.

### B. First-order steps and least-squares algorithms (6)

**R11.** Armijo, L. (1966). “Minimization of Functions Having Lipschitz
Continuous First Partial Derivatives.” *Pacific Journal of Mathematics*,
16(1), 1–3. https://doi.org/10.2140/pjm.1966.16.1

- Use: background on sufficient-decrease line searches; primary full-step
  Newton baseline is **not** Armijo-damped.

**R12.** Wolfe, P. (1969). “Convergence Conditions for Ascent Methods.”
*SIAM Review*, 11(2), 226–235.
https://doi.org/10.1137/1011036

- Use: search conditions in established optimization methods; acknowledge
  our fixed-GD and exact full-step-Newton method definitions.

**R13.** Barzilai, J., and Borwein, J. M. (1988). “Two-Point Step Size Gradient
Methods.” *IMA Journal of Numerical Analysis*, 8(1), 141–148.
https://doi.org/10.1093/imanum/8.1.141

- Use: related-work contrast: adaptive first-order steps already exist;
  they are **not** substituted for our frozen constant-rate GD baseline.

**R14.** Levenberg, K. (1944). “A Method for the Solution of Certain Non-linear
Problems in Least Squares.” *Quarterly of Applied Mathematics*, 2(2),
164–168. https://doi.org/10.1090/qam/10666

- Use: historical nonlinear least-squares safeguards; explicitly distinct
  from our exact Newton and safeguarded exact-Hessian experiment.

**R15.** Marquardt, D. W. (1963). “An Algorithm for Least-Squares Estimation
of Nonlinear Parameters.” *Journal of the Society for Industrial and
Applied Mathematics*, 11(2), 431–441.
https://doi.org/10.1137/0111030

- Use: nonlinear least-squares optimization context; Levenberg-Marquardt
  **was not** the primary method we evaluated.

**R16.** Moré, J. J. (1978). “The Levenberg-Marquardt Algorithm:
Implementation and Theory.” In *Numerical Analysis*, Lecture Notes in
Mathematics, vol. 630, pp. 105–116. Springer.
https://doi.org/10.1007/BFb0067700

- Use: difference between stabilized nonlinear least-squares methods and
  pure full-step Newton; avoid confusing the two.

### C. Nonconvex stationary points, experimental interpretation (6)

**R17.** Cartis, C., Gould, N. I. M., and Toint, Ph. L. (2010). “On the
Complexity of Steepest Descent, Newton’s and Regularized Newton’s Methods
for Nonconvex Unconstrained Optimization Problems.” *SIAM Journal on
Optimization*, 20(6), 2833–2852.
https://doi.org/10.1137/090774100

- Use: theoretical warning that Newton's worst-case iteration complexity
  need not beat first-order descent for nonconvex problems; separate theory
  from our measured wall-clock times.

**R18.** Lee, J. D., Simchowitz, M., Jordan, M. I., and Recht, B. (2016).
“Gradient Descent Only Converges to Minimizers.” *Proceedings of the 29th
Annual Conference on Learning Theory*, PMLR 49, pp. 1246–1257.
https://proceedings.mlr.press/v49/lee16.html

- Use: strict-saddle avoidance under **specific assumptions including random
  initialization**; this theorem does not directly certify our fixed,
  deliberately varied starting vectors.

**R19.** Dauphin, Y. N., Pascanu, R., Gulcehre, C., Cho, K., Ganguli, S., and
Bengio, Y. (2014). “Identifying and Attacking the Saddle Point Problem in
High-Dimensional Non-Convex Optimization.” *Advances in Neural Information
Processing Systems*, 27.
https://proceedings.neurips.cc/paper/2014/hash/04192426585542c54b96ba14445be996-Abstract.html

- Use: independent discussion of saddle points in other nonconvex settings;
  its neural-network results are **context**, not direct evidence for our
  three-parameter regression examples.

**R20.** Efron, B. (1979). “Bootstrap Methods: Another Look at the Jackknife.”
*The Annals of Statistics*, 7(1), 1–26.
https://doi.org/10.1214/aos/1176344552

- Use: foundational bootstrap reasoning, with appropriate caveats for our
  small number (ten) of independent dataset-seed clusters.

**R21.** Cameron, A. C., Gelbach, J. B., and Miller, D. L. (2008).
“Bootstrap-Based Improvements for Inference with Clustered Errors.”
*The Review of Economics and Statistics*, 90(3), 414–427.
https://doi.org/10.1162/rest.90.3.414

- Use: motivation for respecting within-dataset dependence and few-cluster
  limitations. **Their cluster bootstrap-t is not our simple percentile
  cluster bootstrap; do not imply we implemented their procedure.**

**R22.** Paternain, S., Mokhtari, A., and Ribeiro, A. (2019). “A Newton-Based
Method for Nonconvex Optimization with Fast Evasion of Saddle Points.”
*SIAM Journal on Optimization*, 29(1), 343–368.
https://doi.org/10.1137/17M1150116

- Use: an independently published motivation for handling negative Hessian
  eigenvalues; contrast the paper's modified Newton step against our
  **unmodified exact full-step Newton** baseline.

## Planned report mapping

| Report section | Reference IDs | Intended claim types |
|---|---|---|
| Introduction | R01, R03, R04, R06 | Nonlinear regression as parameter estimation via minimization |
| Literature review | R02, R07, R08, R10–R19, R22 | First/second-order tradeoffs, alternatives, nonconvex pitfalls |
| Mathematical foundations | R01–R06, R10, R14–R16 | Residuals/MSE, gradients, exact Hessian, stationary points |
| Experimental methodology | R01, R05, R06, R09, R20, R21 | Paired starts, validation/test isolation, timing and clustered uncertainty |
| Results & discussion | R01, R07, R08, R17–R22 | Interpret measured speed, saddle cases, initial conditions and limits |

## Experimental evidence to integrate later

Use ONLY the archived original evaluation in
`docs/FINAL_EVALUATION_RESULTS.md` and
`evidence/evaluation_2026-11-01/` for official results:

- Two model families, ten independent noise datasets/model, five starts,
  original fixed GD versus full-step exact Newton, with separately identified
  safeguarded Newton.
- 300 run conditions, 1,500 timing repetitions, audited code/data hashes.
- GD reached strict minima in 50/50 conditions on each model. Newton reached
  strict minima in 4/50 (Model 1) and 22/50 (Model 2), and many stationary
  saddles. Runtime advantage **per attempted run** is not time to successful
  minimization.
- Bootstrap intervals are conditional on ten clusters and do not prove
  universal optimizer superiority.

The adaptive GD pilots developed later belong only in an optional, clearly
labelled appendix/future-work note, **not** in the primary comparison.

## Before drafting the literature review

1. Read/confirm relevant passages or abstracts, distinguishing theorems
   from experiments and from our own inferences.
2. Retain at least 20 **actually cited** scholarly entries in the final
   report; do not pad a bibliography with uncited items.
3. Cite exact publications in the report (author, year, venue/publisher,
   DOI), not generic search-results pages.
4. Use a manageable set of references per paragraph rather than citing
   all 22 for every broad statement.
5. Verify the page budget after formatting: the professor's **minimum** is
   ten pages; the working target is approximately 10–15 including a proper
   cover and reference pages unless further formatting guidance is supplied.
