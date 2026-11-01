# Phase 3 checkpoint: batch Gradient Descent

The Model 1 training objective is

\[
J(\theta)=\frac{1}{n}\sum_i(ae^{bx_i}+c-y_i)^2,
\qquad \theta=(a,b,c).
\]

At each iteration, **fixed-step batch Gradient Descent** updates all three
parameters using the gradient of training MSE:

\[
\boxed{\theta_{k+1}=\theta_k-\alpha\nabla J(\theta_k)},
\]

where \(\alpha>0\) is the learning rate. The negative gradient is the local
direction of steepest decrease. An excessively large learning rate can jump
past the minimum or even increase MSE.

## One manually checkable step

Using our earlier two-point example,

- \(\theta_0=[3,\ln(0.5),1]\)
- \(x=[0,1]\), \(y=[5.1,2.7]\)
- \(\nabla J(\theta_0)=[-1.2,-0.3,-1.3]\)
- \(\alpha=0.1\)

the next candidate is

\[
\theta_1=[3+0.12,\ \ln(0.5)+0.03,\ 1+0.13]
        =[3.12,\ \ln(0.5)+0.03,\ 1.13].
\]

The code includes a unit test comparing this hand step with the computed
update. This is an illustration, not a final experiment or a chosen learning
rate for all models.

## Implementation and verification

`opttheory.optimizers.gradient_descent.gradient_descent` uses the verified
analytical gradient, not a black-box optimizer. It records completed
iterations, candidate parameters, MSE history, gradient-norm history, elapsed
time, status, and evaluation counts.

Stopping conditions:

1. Gradient norm at or below the chosen tolerance.
2. Maximum iteration count.
3. A detected numerical failure (e.g., overflow).

The code does **not** assume every update decreases MSE. Reaching the gradient
tolerance means *stationarity to that tolerance*, not proof of a global
minimum. This first solver uses a fixed step with no hidden adaptation.

For a reproducible teaching pilot on the configured **training split only**:

```powershell
python -m scripts.run_model1_gd_pilot
```

This pilot intentionally does not inspect or fit against validation/test
observations or true synthetic parameters. Timing from a single pilot is not
evidence of superiority; fair comparisons come after exact Newton is added
and the research protocol is frozen.
