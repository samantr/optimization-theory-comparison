# Four planned nonlinear regression families

Model 1 has synthetic data and analytical optimization derivatives. Model 2
currently has only a tested prediction function; its true synthetic parameters,
dataset, gradient, and Hessian have **not** been selected or implemented.
Models 3-4 remain proposals.

| Model | Function | Free parameters | Status |
| --- | --- | --- | --- |
| 1. Exponential decay | a exp(bx) + c | a, b, c | Data generation implemented |
| 2. Logistic growth | L / (1 + exp[-k(x-x0)]) | L, k, x0 | Prediction function tested; dataset and derivatives pending |
| 3. Gaussian peak | A exp[-(x-mu)^2/(2 sigma_model^2)] | A, mu, sigma_model | Planned |
| 4. Oscillatory upward trend | mx + A sin(omega x) + B sin(2 omega x) | m, A, omega, B | Planned |

**Naming note:** sigma_model is the width of the Gaussian regression curve and
must not be confused with sigma_noise, the standard deviation of observation noise.

For Model 2, `L` is the saturation level, `k > 0` gives an increasing
S-curve, and `x0` is the midpoint: `f(x0) = L/2`. The prediction uses a
numerically stable sigmoid computation, avoiding exponential overflow at
extreme inputs. It predicts *continuous outcomes*, so it is not the same
task as binary logistic classification. The values used in unit tests are
illustrative, **not approved ground-truth dataset parameters**.

Model 4 is inspired by Saman's hand-drawn rising, multi-peak curve. It is not
intended to trace that image pixel-for-pixel. Its unknown frequency omega can
create a multimodal loss landscape, making it a potential initialization-sensitivity
stress test.

Within each model we will compare both algorithms on **identical data and
initial parameter vectors**. Raw performance between different model families
must not be interpreted as an algorithm effect without accounting for their
different complexity and parameter count.

