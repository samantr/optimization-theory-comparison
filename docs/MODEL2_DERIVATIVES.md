# Model 2: logistic MSE, gradient, and exact Hessian

We now fit the continuous-output nonlinear regression curve

\[
\hat y_i = L s_i,\qquad
s_i=\frac{1}{1+e^{-z_i}},\qquad
z_i=k(x_i-x_0).
\]

The **unknown candidate parameters** are \(\theta=[L,k,x_0]\). For an
observed training point, \(r_i=\hat y_i-y_i\). As for Model 1, minimize

\[
J(\theta)=\frac{1}{n}\sum_i r_i^2.
\]

The mean-square *formula* is unchanged, but its derivatives must be
recalculated because the prediction model changed.

## Prediction derivatives: chain rule

Write \(t_i=x_i-x_0\) and \(q_i=s_i(1-s_i)\). The sigmoid derivative is
\(ds/dz=s(1-s)=q\). Since \(dz/dk=t\) and \(dz/dx_0=-k\),

\[
\nabla_\theta\hat y_i =
\begin{bmatrix}
s_i\\
L q_i t_i\\
-L k q_i
\end{bmatrix}.
\]

The MSE gradient follows the same *chain-rule pattern* as Model 1:

\[
\boxed{\nabla J(\theta)=\frac{2}{n}\sum_i r_i\nabla\hat y_i.}
\]

For the prediction's second derivatives, put \(u_i=1-2s_i\).
Differentiating \(q=s(1-s)\) gives \(dq/dz=q(1-2s)=qu\). The six
distinct entries of the symmetric matrix \(\nabla^2_\theta\hat y_i\) are:

\[
\begin{aligned}
f_{LL} &= 0, \\
f_{Lk} &= q_i t_i, \\
f_{Lx_0} &= -kq_i, \\
f_{kk} &= Lq_i u_i t_i^2, \\
f_{kx_0} &= -Lq_i(1+k u_i t_i), \\
f_{x_0x_0} &= Lk^2q_i u_i.
\end{aligned}
\]

The **exact** MSE Hessian is

\[
\boxed{
\nabla^2 J=\frac{2}{n}\sum_i
\left[(\nabla\hat y_i)(\nabla\hat y_i)^T
+r_i\nabla^2\hat y_i\right].
}
\]

The second summand contains the actual residual \(r_i\). Dropping it
would produce the *Gauss–Newton* Hessian approximation. Our approved
primary Newton comparison uses this **full exact** Hessian.

## A small example at the S-curve midpoint

Let the current **candidate** be \([L,k,x_0]=[10,1.5,2]\), with one
training observation \(x=2\), \(y=5.1\):

- \(t=2-2=0\), \(s=0.5\), \(q=0.25\).
- Predicted value: \(\hat y=10(0.5)=5\).
- Residual: \(r=5-5.1=-0.1\).
- Prediction derivative: \(\nabla\hat y=[0.5,0,-3.75]\).
- MSE: \(J=(-0.1)^2=0.01\).
- MSE gradient: \(2r\nabla\hat y=[-0.1,0,0.75]\).

At **exactly the midpoint**, changing \(k\) while holding \(L,x_0\)
fixed does not change the prediction. Observations at other \(x\) values
are required to identify steepness \(k\) reliably.

## Code and verification

The functions in `opttheory.objectives.logistic_mse` are:

- `objective(theta, x, y_observed)` — training MSE;
- `gradient(theta, x, y_observed)` — order `[L,k,x0]`;
- `hessian(theta, x, y_observed)` — exact symmetric 3x3 matrix.

They use supplied observations and candidate parameters, never the
synthetic true parameters. The gradient does not construct a Hessian;
this is important for a fair Gradient Descent versus Newton runtime study.

Tests independently estimate each gradient coordinate using central
differences of MSE and each Hessian column using central differences of
the gradient. They include the approved negative-`k` starting condition,
zero-residual cases, nonzero-residual Hessian checks, and large-magnitude
exponential arguments. Passing tests validate formulas but **do not**
establish optimizer performance; no Model 2 optimizations have run.
