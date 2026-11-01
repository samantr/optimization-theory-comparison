# Model 1: exact Hessian checkpoint

For the nonlinear regression model

\[
\hat y_i = a e^{b x_i} + c,\qquad
r_i = \hat y_i - y_i,\qquad
J(a,b,c) = \frac{1}{n}\sum_{i=1}^{n}r_i^2,
\]

we optimize a **candidate** parameter vector \(\theta=(a,b,c)\), using observed
training data only. True synthetic parameters are never inputs to the objective
or its derivatives.

Let \(E_i=e^{b x_i}\). The gradient is

\[
\nabla J = \frac{2}{n}\sum_i r_i
\begin{bmatrix} E_i\\a x_i E_i\\1\end{bmatrix}.
\]

## Deriving the six distinct Hessian entries

The Hessian is the matrix of derivatives of the gradient. By the product rule,
\(\partial_b E_i=x_iE_i\),
\(\partial_a r_i=E_i\),
\(\partial_b r_i=a x_iE_i\), and
\(\partial_c r_i=1\).

\[
\begin{aligned}
H_{aa} &= 2\operatorname{mean}(E_i^2),\\
H_{ab} &= 2\operatorname{mean}(a x_i E_i^2+r_i x_iE_i),\\
H_{ac} &= 2\operatorname{mean}(E_i),\\
H_{bb} &= 2\operatorname{mean}((a x_iE_i)^2+r_i a x_i^2E_i),\\
H_{bc} &= 2\operatorname{mean}(a x_iE_i),\\
H_{cc} &= 2.
\end{aligned}
\]

The other entries follow from symmetry:
\(H_{ba}=H_{ab}\), \(H_{ca}=H_{ac}\), \(H_{cb}=H_{bc}\).

In particular, \(H_{ab}\) and \(H_{bb}\) contain terms involving the
**residuals**. Keeping these terms yields the **exact** Hessian required by our
Newton comparison. Dropping them would yield a Gauss–Newton approximation,
which is a different algorithm and is not our primary comparison.

## One-observation example

For \(x=0\) and \(y=5\),

\[
J=(a+c-5)^2,\qquad
H=\begin{bmatrix}2&0&2\\0&0&0\\2&0&2\end{bmatrix}.
\]

The Hessian is singular: \(b\) cannot affect the prediction at \(x=0\),
and infinitely many \((a,c)\) pairs satisfy \(a+c=5\). A direct Newton solve
must detect and handle such a case rather than silently invert the matrix.

## Verification and use

The Python function `opttheory.objectives.mse.hessian(theta, x, y_observed)`
uses the parameter order `[a, b, c]`. Independent tests compare every
component against central finite differences of the previously verified
analytical gradient. Other tests check symmetry, the singular example, and
the residual-dependent terms.

To run all tests from the repository root:

```powershell
python -m unittest discover -s tests -v
```

There is **no Newton optimizer yet**. It will solve a linear system involving
this Hessian at each iteration and record numerical failures transparently.
