# Comparative Analysis of Gradient Descent and Newton Optimization Methods in Nonlinear Regression Problems

**Working academic report — English**  
**Course:** Optimization Theory (BLM 599), Kocaeli University  
**Instructor:** Prof. Dr. Yaşar Becerikli  
**Status:** Sections 1–2 drafted, 2026-11-21. Full academic report is not yet complete. Reference numbering is provisional.

## 1. Introduction

Estimating unknown model parameters from observed data is a fundamental task in scientific modeling and statistical analysis. In nonlinear regression, the predicted response depends nonlinearly on one or more unknown parameters, so fitting the model generally requires the iterative minimization of a loss function rather than a direct linear least-squares solution [1]. A common choice is the mean squared error (MSE), which quantifies the average squared difference between model predictions and observed responses. Because the objective is defined over the model parameters, the behavior of an optimization method depends not only on the form of the prediction function but also on the geometry of the resulting loss surface.

Gradient Descent (GD) and Newton's Method represent two established approaches to this optimization problem. GD uses first-order information, moving the parameter vector in the direction opposite to the gradient with a selected step size. Newton's Method additionally uses the Hessian matrix to account for local curvature and determines its update by solving a linear system [2]. This second-order information can provide rapid local progress near a suitable minimum, but each iteration requires additional derivative calculations and a matrix solve. Moreover, when the Hessian is indefinite or poorly conditioned, an unrestricted Newton step is not guaranteed to be a descent step. Analyses of nonconvex optimization therefore distinguish favorable local behavior from the reliability of a method across different initial conditions [2], [3].

This distinction is especially relevant to nonlinear regression because a smooth prediction function does not necessarily produce a convex loss surface in parameter space. The condition that the gradient is close to zero identifies approximate stationarity, not necessarily successful minimization: the corresponding point may be a local minimum, a saddle point, or a case of inconclusive curvature. Published research investigates both the behavior of GD near strict saddle points and Newton-type modifications designed to handle negative curvature [4], [5]. These findings motivate examining solution quality and Hessian curvature separately from a gradient-based stopping criterion. Their theoretical guarantees, however, should not be assumed to apply automatically to a finite experiment with fixed starting vectors and prespecified algorithm settings.

The objective of this study is to compare fixed-step batch GD with full-step exact Newton optimization in two three-parameter nonlinear regression problems: an exponential-decay model and a logistic-growth model. The central research question is: **Under matched data and initialization conditions, how do these first- and second-order methods differ in convergence behavior, execution time, solution quality, numerical stability, and sensitivity to starting parameters?** For each model, ten independently noise-seeded synthetic datasets and five fixed starting vectors are used. Both methods receive identical observations and initial parameters for each paired condition. GD learning rates are calibrated separately using training and validation data and then held fixed; final test observations are excluded from optimization and tuning.

The study contributes a reproducible, controlled comparison based on explicitly implemented analytical derivatives, documented stopping and failure criteria, repeated optimizer-only timing measurements, and Hessian-based classification of final stationary points. Rather than equating a small gradient or short runtime with optimization success, the analysis distinguishes termination status, local curvature, predictive error, and computational cost. The observed outcomes illustrate an important trade-off in this experimental setting: full-step Newton often terminates faster per attempt, while GD more consistently reaches low-error local minima. These conclusions are deliberately limited to the two synthetic model families, registered initializations, and computational conditions examined here.


## 2. Related Work

### 2.1. First-order optimization and step-size selection

Gradient-based minimization remains an important baseline because its parameter updates require first-order derivatives rather than the solution of a Hessian system [2]. Its performance, however, depends strongly on the selection of a step size. Armijo [6] studied a modified steepest-descent scheme in which a candidate step must satisfy a decrease condition, while Wolfe [7] developed convergence conditions for step selection in descent methods. Barzilai and Borwein [8] later proposed two-point gradient step sizes derived from an approximation to a secant relation, illustrating that useful curvature information can be incorporated into a first-order update without forming a full Hessian. These works establish several alternatives to a constant learning rate. **The present study does not implement those alternatives in its primary comparison:** it calibrates one constant GD rate per regression model on separate data and freezes that rate before final evaluation.

### 2.2. Newton updates and safeguards

Newton's Method uses local second-order information and can exhibit rapid convergence in a neighborhood where the Hessian is appropriately conditioned [2]. Such local advantages do not imply that a full Newton update is globally reliable. Cartis, Gould, and Toint [3] investigate worst-case evaluation complexity in nonconvex optimization and demonstrate why the theoretical relationship between steepest descent, Newton, and regularized Newton cannot be reduced to a simple ranking by derivative order. Trust-region techniques, treated systematically by Conn, Gould, and Toint [9], instead limit the region in which a local model is trusted. These safeguards provide valuable context, but they are methodologically distinct from the **unrestricted full-step exact Newton** algorithm used as the primary second-order baseline in this project.

### 2.3. Methods tailored to nonlinear least squares

Nonlinear regression leads naturally to a nonlinear least-squares objective. Dennis and Schnabel [10, Ch. 10] distinguish approaches based on local linearization of the residual vector, notably Gauss–Newton and Levenberg–Marquardt, from methods that treat the full minimization problem. In the original damped least-squares formulation, Levenberg [11] addressed the risk that an excessively large update derived from a linearized model could increase the actual sum of squared residuals. Marquardt [12] further developed an algorithm specifically for nonlinear least-squares parameter estimation, and Moré [13] analyzed implementation issues such as scaling and selection of the regularization parameter.

A distinction is essential for interpreting these methods. With residual Jacobian (J_r), Gauss–Newton uses a matrix proportional to (J_r^T J_r), whereas the **exact Hessian** of an MSE objective also contains terms involving each residual and its second derivatives. Those additional terms need not preserve positive semidefiniteness. Our primary Newton implementation retains them; it must therefore not be described as Gauss–Newton or Levenberg–Marquardt.

### 2.4. Stationary points and negative curvature

Nonconvex optimization introduces another complication: reaching a small gradient does not imply reaching a minimum. Dauphin and colleagues [14] explain how indefinite Hessian curvature can make ordinary Newton updates approach saddle points, and propose a saddle-free alternative in the context of neural-network optimization. Lee and colleagues [4] provide a complementary result concerning gradient descent and strict saddles, but their avoidance theorem requires particular smoothness, step-size, and initialization assumptions. Paternain, Mokhtari, and Ribeiro [5] investigate a separate Newton-based approach that modifies the treatment of negative Hessian eigenvalues.

These studies motivate evaluating **gradient-based termination, Hessian curvature, and final prediction error as separate outcomes**. Their theoretical guarantees and high-dimensional applications do not automatically transfer to our two three-parameter models. Accordingly, this project's contribution is not a new optimizer or a universal performance theorem; it is a reproducible paired experimental comparison of established algorithms, explicitly retaining unsuccessful stationary solutions and numerical failures.

## Provisional references for Sections 1–2 (renumber after completing the report)
[1] D. M. Bates and D. G. Watts, *Nonlinear Regression Analysis and Its Applications*. New York: Wiley, 1988. doi: [10.1002/9780470316757](https://doi.org/10.1002/9780470316757).

[2] J. Nocedal and S. J. Wright, *Numerical Optimization*, 2nd ed. New York: Springer, 2006. doi: [10.1007/978-0-387-40065-5](https://doi.org/10.1007/978-0-387-40065-5).

[3] C. Cartis, N. I. M. Gould, and Ph. L. Toint, “On the Complexity of Steepest Descent, Newton's and Regularized Newton's Methods for Nonconvex Unconstrained Optimization Problems,” *SIAM Journal on Optimization*, vol. 20, no. 6, pp. 2833–2852, 2010. doi: [10.1137/090774100](https://doi.org/10.1137/090774100).

[4] J. D. Lee, M. Simchowitz, M. I. Jordan, and B. Recht, “Gradient Descent Only Converges to Minimizers,” in *Proceedings of the 29th Conference on Learning Theory*, PMLR, vol. 49, pp. 1246–1257, 2016. [Publisher entry](https://proceedings.mlr.press/v49/lee16.html).

[5] S. Paternain, A. Mokhtari, and A. Ribeiro, “A Newton-Based Method for Nonconvex Optimization with Fast Evasion of Saddle Points,” *SIAM Journal on Optimization*, vol. 29, no. 1, pp. 343–368, 2019. doi: [10.1137/17M1150116](https://doi.org/10.1137/17M1150116).


[6] L. Armijo, “Minimization of Functions Having Lipschitz Continuous First Partial Derivatives,” *Pacific Journal of Mathematics*, vol. 16, no. 1, pp. 1–3, 1966. doi: [10.2140/pjm.1966.16.1](https://doi.org/10.2140/pjm.1966.16.1).

[7] P. Wolfe, “Convergence Conditions for Ascent Methods,” *SIAM Review*, vol. 11, no. 2, pp. 226–235, 1969. doi: [10.1137/1011036](https://doi.org/10.1137/1011036).

[8] J. Barzilai and J. M. Borwein, “Two-Point Step Size Gradient Methods,” *IMA Journal of Numerical Analysis*, vol. 8, no. 1, pp. 141–148, 1988. doi: [10.1093/imanum/8.1.141](https://doi.org/10.1093/imanum/8.1.141).

[9] A. R. Conn, N. I. M. Gould, and Ph. L. Toint, *Trust-Region Methods*. Philadelphia: SIAM, 2000. doi: [10.1137/1.9780898719857](https://doi.org/10.1137/1.9780898719857).

[10] J. E. Dennis Jr. and R. B. Schnabel, *Numerical Methods for Unconstrained Optimization and Nonlinear Equations*. SIAM, 1996 reprint of the 1983 edition. doi: [10.1137/1.9781611971200](https://doi.org/10.1137/1.9781611971200).

[11] K. Levenberg, “A Method for the Solution of Certain Non-Linear Problems in Least Squares,” *Quarterly of Applied Mathematics*, vol. 2, no. 2, pp. 164–168, 1944. doi: [10.1090/qam/10666](https://doi.org/10.1090/qam/10666).

[12] D. W. Marquardt, “An Algorithm for Least-Squares Estimation of Nonlinear Parameters,” *Journal of the Society for Industrial and Applied Mathematics*, vol. 11, no. 2, pp. 431–441, 1963. doi: [10.1137/0111030](https://doi.org/10.1137/0111030).

[13] J. J. Moré, “The Levenberg–Marquardt Algorithm: Implementation and Theory,” in *Numerical Analysis*, Lecture Notes in Mathematics, vol. 630, Springer, pp. 105–116, 1978. doi: [10.1007/BFb0067700](https://doi.org/10.1007/BFb0067700).

[14] Y. N. Dauphin, R. Pascanu, Ç. Gülçehre, K. Cho, S. Ganguli, and Y. Bengio, “Identifying and Attacking the Saddle Point Problem in High-Dimensional Non-Convex Optimization,” in *Advances in Neural Information Processing Systems*, vol. 27, pp. 2933–2941, 2014. [Proceedings](https://proceedings.neurips.cc/paper/2014/hash/04192426585542c54b96ba14445be996-Abstract.html).

---

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
