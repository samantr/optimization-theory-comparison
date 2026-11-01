# Phase 4 checkpoint: full-step exact Newton

For the same Model 1 training loss \(J(\theta)\) used by Gradient Descent,
Newton's Method considers both the gradient \(g=\nabla J(\theta)\) and the
**exact** Hessian \(H=\nabla^2 J(\theta)\).

It solves the linear system \(Hd=g\) and takes the update

\[
\boxed{\theta_{k+1}=\theta_k-d,\qquad H(\theta_k)d=\nabla J(\theta_k).}
\]

The code uses `numpy.linalg.solve(H, g)` and does **not** explicitly form
the matrix inverse. It does **not** substitute a Gauss–Newton approximation.

## Small example

For \(J(t)=(t-3)^2\), at \(t=0\) the gradient is \(-6\) and second
derivative is \(2\). Thus \(t_{new}=0-(-6/2)=3\). One exact step reaches
the minimum of this quadratic. Nonlinear Model 1 does not inherit this
one-step guarantee.

## Failure modes and interpretation

- **Singular Hessian:** no unique Newton direction. The solver reports
  `singular_hessian` instead of attempting an inverse.
- **Ill-conditioned Hessian:** numerical solve may be unreliable. A configurable
  condition-number threshold (default \(10^{12}\)) stops the pilot with
  `ill_conditioned_hessian`.
- **Indefinite Hessian:** full Newton can step *uphill* or approach a saddle.
  This first implementation intentionally does not hide that behavior with a
  line search or an unreported Hessian approximation. Loss increases remain
  in the history.
- **Small gradient:** the `gradient_tolerance` status alone does **not** prove
  that the final point is a minimum. Examine both the loss and Hessian
  eigenvalues at the final candidate.

The optimizer records the same basic result fields as Gradient Descent,
plus its Hessian evaluation count. Both implementations use training data
only during updates.

Run the single-dataset training diagnostic:

```powershell
python -m scripts.run_model1_newton_pilot
```

With the pilot start `[3, -0.5, 1]` and current synthetic dataset, the full
Newton trajectory can reach a poor stationary point with an indefinite final
Hessian, despite using fewer updates. This is useful diagnostic evidence that
we must record saddle/instability cases and consider **clearly labelled**
safeguards in later comparisons. It is **not** a statistical comparison or
proof of Newton's general performance.

Next: decide and document the treatment of indefinite Hessians (for example,
pure full-step baseline versus a separately labelled damped/line-search
variant) before freezing the paired experimental design.
