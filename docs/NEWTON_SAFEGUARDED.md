# Optional third method: safeguarded exact-Hessian Newton

**Status:** exploratory pilot, separate from the approved two-algorithm
comparison (batch Gradient Descent versus full-step exact Newton).

We preserve `opttheory.optimizers.newton.newton` unchanged. The extra
`safeguarded_newton` method tests whether explicit safeguards avoid large
uphill steps and singular-Hessian failures.

## Update rule and safeguards

At each candidate \(\theta\), calculate the **exact** Hessian \(H\) and
gradient \(g\) of training MSE.

1. If \(H\) is positive definite and its condition number is below the
   configurable threshold, choose the Newton direction
   \(p=-H^{-1}g\), computed using a **linear solve**.
2. Otherwise use the negative gradient direction \(p=-g\), explicitly
   counted as a **gradient fallback**. This makes the method a labelled
   hybrid, not pure Newton or Gauss–Newton.
3. Test \(\theta_{new}=\theta+\alpha p\), starting with \(\alpha=1\).
   Accept only if the **Armijo sufficient-decrease** condition holds:

   \[
   J(\theta+\alpha p) \le J(\theta)+c_1\alpha g^T p,
   \quad c_1=10^{-4}.
   \]

4. Otherwise halve \(\alpha\) and retry (default limit: 30 reductions).
   A rejected trial never updates the accepted parameter vector.

The result records accepted steps, training MSE history, gradient-norm
history, accepted step sizes, Hessian evaluations, objective evaluations,
gradient fallbacks, and line-search reductions. A small gradient still does
**not** prove that a global minimum has been found.

Run the teaching pilot on the training split:

```powershell
python -m scripts.run_model1_newton_safeguarded_pilot
```

With the **provisional** Model 1 dataset and start `[3,-0.5,1]`, the variant
reaches training MSE near `0.005671` in six accepted steps, using one fallback
and three line-search reductions. The full-step Newton pilot from the same
start reached a poor indefinite-Hessian stationary point. These are only
diagnostics on one dataset and one initialization: no statistical or timing
superiority claim is supported yet.

## Rules for later reporting

- Report **batch GD** and **full-step exact Newton** as the primary comparison.
- Report **safeguarded exact-Hessian Newton + gradient fallback** as an
  explicitly separate third variant, never merge its outcomes into Newton's.
- Use the same training dataset and paired initial parameters for all
  methods; tune settings on training/validation only, then seal final test.
- Count backtracking objective evaluations, fallback steps, unsuccessful
  trials, wall-clock time, and failures as well as iterations and final MSE.
- Distinguish stopping due to a small gradient from a verified good fit;
  inspect Hessian curvature and held-out generalization separately.
