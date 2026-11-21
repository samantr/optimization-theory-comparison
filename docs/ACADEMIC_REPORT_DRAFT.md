# Comparative Analysis of Gradient Descent and Newton Optimization Methods in Nonlinear Regression Problems

**Working academic report — English**  
**Course:** Optimization Theory (BLM 599), Kocaeli University  
**Instructor:** Prof. Dr. Yaşar Becerikli  
**Status:** Sections 1–3 drafted, 2026-11-21. Full academic report is not yet complete. Reference numbering is provisional.

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

## 3. Mathematical Foundations

### 3.1. Nonlinear regression as a minimization problem

Let $x_i,y_i)\, $i=1,\ldots,n$, denote observed input–response pairs and let $f(x_i;\boldsymbol{\theta})$ be a nonlinear regression prediction with parameter vector $\boldsymbol{\theta}\in\mathbb{R}^{p}$. Define the residual as the difference between the prediction and the observed response:

$
r_i(\boldsymbol{\theta})=f(x_i;\boldsymbol{\theta})-y_i.
\tag{1}
$

In this study, the training objective is the mean squared error (MSE) [1], [15]:

$
J(\boldsymbol{\theta})=\frac{1}{n}\sum_{i=1}^{n}r_i(\boldsymbol{\theta})^2
=\frac{1}{n}\mathbf r(\boldsymbol{\theta})^T\mathbf r(\boldsymbol{\theta}).
\tag{2}
$

Here, $n$ counts **training observations**, not all observations in a dataset. The optimizers evaluate (2) using a *candidate* parameter vector and observed training responses. The known parameters used to generate synthetic examples are never passed into the loss function. Although the error in (2) is a sum of squares, it need not be convex in the parameters when the residuals are nonlinear functions of those parameters [15], [16].

### 3.2. The gradient and exact Hessian of MSE

The **gradient** (gradient vector, *gradyan vektörü*) collects the first partial derivatives of the scalar objective. Applying the chain rule to each squared residual gives

$
\nabla J(\boldsymbol{\theta})
=\frac{2}{n}\sum_{i=1}^{n}r_i(\boldsymbol{\theta})\nabla r_i(\boldsymbol{\theta})
=\frac{2}{n}J_r^T\mathbf r,
\tag{3}
$

where $J_r\in\mathbb{R}^{n\times p}$ denotes the residual Jacobian, whose $i$th row is $\nabla r_i^T$. Differentiating (3) once more, using the product rule, yields the **exact Hessian** (Hessian or curvature matrix, *Hessian/eğrilik matrisi*):

$
H_J(\boldsymbol{\theta})
=\nabla^2J(\boldsymbol{\theta})
=\frac{2}{n}\left[
J_r^TJ_r+\sum_{i=1}^{n}r_i(\boldsymbol{\theta})\nabla^2r_i(\boldsymbol{\theta})
\right].
\tag{4}
$

The first term $J_r^TJ_r$ is positive semidefinite. The second, **residual-weighted curvature term**, can alter the signs of the Hessian eigenvalues. Omitting this second term gives the Gauss–Newton approximation rather than the full Hessian used by exact Newton [10], [15]. This is not merely a distinction in terminology: the implemented algorithm's step direction depends on the complete matrix.

For example, the exponential prediction considered later is $f(x;a,b,c)=ae^{bx}+c$. Writing $E_i=e^{bx_i}$, its prediction derivative is

$
\nabla_{(a,b,c)}f(x_i;a,b,c)=
\begin{bmatrix}E_i & ax_iE_i & 1\end{bmatrix}^T.
\tag{5}
$

Substitution of this vector into (3) gives the model's analytical MSE gradient. Its exact Hessian follows from (4), including terms involving $r_i$. The logistic model follows the same general formulas but requires different prediction derivatives. Both sets of analytical derivatives were independently checked in the Python project against numerical finite differences.

### 3.3. First- and second-order parameter updates

A first-order Taylor approximation at the current parameter vector $\boldsymbol{\theta}_k$ is

$
J(\boldsymbol{\theta}_k+\mathbf p)
\approx J(\boldsymbol{\theta}_k)
+\nabla J(\boldsymbol{\theta}_k)^T\mathbf p.
\tag{6}
$

Consequently, **fixed-step batch Gradient Descent** updates parameters in the opposite direction to the gradient:

$
\boldsymbol{\theta}_{k+1}
=\boldsymbol{\theta}_k-\alpha\nabla J(\boldsymbol{\theta}_k),\qquad\alpha>0.
\tag{7}
$

The learning rate $\alpha$ controls the step size. It remains constant within each model in the official experiment; neither line search nor adaptive rescaling is hidden in the GD baseline.

Newton's Method instead uses a second-order Taylor approximation [2], [17]:

$
J(\boldsymbol{\theta}_k+\mathbf p)
\approx J(\boldsymbol{\theta}_k)
+\mathbf g_k^T\mathbf p
+\tfrac12\mathbf p^TH_k\mathbf p,
\tag{8}
$

with $\mathbf g_k=\nabla J(\boldsymbol{\theta}_k)$ and $H_k=\nabla^2J(\boldsymbol{\theta}_k)$. Setting the derivative of this quadratic approximation with respect to $\mathbf p$ to zero leads to $H_k\mathbf p_k=-\mathbf g_k$, giving the **full-step exact Newton** update

$
H_k\mathbf p_k=-\mathbf g_k,\qquad
\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k+\mathbf p_k.
\tag{9}
$

When $H_k$ is positive definite, $\mathbf g_k^T\mathbf p_k=-\mathbf g_k^TH_k^{-1}\mathbf g_k<0$ for nonzero $\mathbf g_k$, so Newton's direction is a descent direction. An indefinite Hessian provides no such general guarantee. The implementation therefore checks for singular or ill-conditioned systems and uses a linear-system solve, not an explicitly computed matrix inverse [17]. It does **not** replace an indefinite Hessian with a positive-definite approximation in the primary comparison.

### 3.4. Stationary points and curvature classification

A point $\boldsymbol{\theta}^{*}$ satisfying $\nabla J(\boldsymbol{\theta}^{*})=\mathbf0$ is a **stationary point** (*durağan nokta*), but stationarity alone is insufficient to establish minimization. For a twice continuously differentiable objective at an interior point, a positive-definite Hessian at a stationary point is sufficient for a **strict local minimum**, a negative-definite Hessian indicates a strict local maximum, and an indefinite Hessian identifies a **saddle point**. When the Hessian is only semidefinite, the usual second-order test can be inconclusive [2], [16].

A course-aligned two-parameter illustration is $q(x_1,x_2)=2x_1^2-3x_1x_2+2x_2^2$. Its Hessian is

$
H_q=\begin{bmatrix}4&-3\\-3&4\end{bmatrix},
\qquad\lambda_1=1,\quad\lambda_2=7.
\tag{10}
$

Both eigenvalues are positive; hence the stationary point at $(0,0)$ is a strict minimum. In contrast, a Hessian with both positive and negative eigenvalues is indefinite. In the present study, this local classification is computed separately from gradient-tolerance termination and final test MSE. **A small gradient is not, by itself, evidence of a successful minimum.**

## Provisional references for Sections 1–3 (renumber after completing the report)
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

[15] Å. Björck, *Numerical Methods for Least Squares Problems*. Philadelphia: SIAM, 1996. doi: [10.1137/1.9781611971484](https://doi.org/10.1137/1.9781611971484).

[16] S. Boyd and L. Vandenberghe, *Convex Optimization*. Cambridge: Cambridge University Press, 2004. [Author-hosted book](https://web.stanford.edu/~boyd/cvxbook/).

[17] P. E. Gill, W. Murray, and M. H. Wright, *Practical Optimization*. London: Academic Press, 1981. [SIAM reprint DOI](https://doi.org/10.1137/1.9781611975604).

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


- Section 3 equations (1)–(10) are **locally numbered in this draft** and must be renumbered consistently across the full report at typesetting.
- Mathematical derivations checked directly against `opttheory/objectives/mse.py`, `opttheory/objectives/logistic_mse.py`, `opttheory/optimizers/gradient_descent.py`, and `opttheory/optimizers/newton.py`.
- Professor Becerikli's `yOptmT-h5.pdf` p. 7 demonstrates \\(q(x_1,x_2)=2x_1^2-3x_1x_2+2x_2^2\\) and Hessian eigenvalues \\(1,7\\); pp. 8–9 cover stationary points and Hessian classification. `yOptmT-h6.pdf` p. 12 derives Newton from the quadratic/Taylor approximation. `yOptmT-h8.pdf` p. 3 lists various stopping criteria.
- Source [15] (Björck) was checked via SIAM's 1996 edition and its published Chapter 9 description; full technical chapter was not read. Source [16] (Boyd–Vandenberghe) author-hosted text verifies the convexity terminology; source [17] (Gill–Murray–Wright) is catalogued as an **original 1981** book, despite the 1986 date on the professor's recommended-books slide. Avoid attributing precise theorem/page numbers to a book whose full relevant chapter has not been inspected.
- References newly added in this section: [15] Björck, [16] Boyd–Vandenberghe, [17] Gill–Murray–Wright. **17 distinct scholarly items** are now cited in the Sections 1–3 draft; at least 20 actually used references are needed in the finished manuscript.
