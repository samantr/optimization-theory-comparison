# Four planned nonlinear regression families

Model 1 has synthetic data and analytical optimization derivatives. Model 2
has an approved true synthetic curve and tested, seeded data generator;
its gradient, Hessian, and optimizers have **not** been implemented.
Models 3-4 remain proposals.

| Model | Function | Free parameters | Status |
| --- | --- | --- | --- |
| 1. Exponential decay | a exp(bx) + c | a, b, c | Data generation implemented |
| 2. Logistic growth | L / (1 + exp[-k(x-x0)]) | L, k, x0 | Prediction and seeded dataset tested; derivatives pending |
| 3. Gaussian peak | A exp[-(x-mu)^2/(2 sigma_model^2)] | A, mu, sigma_model | Planned |
| 4. Oscillatory upward trend | mx + A sin(omega x) + B sin(2 omega x) | m, A, omega, B | Planned |

**Naming note:** sigma_model is the width of the Gaussian regression curve and
must not be confused with sigma_noise, the standard deviation of observation noise.

For Model 2, `L` is the saturation level, `k > 0` gives an increasing
S-curve, and `x0` is the midpoint: `f(x0) = L/2`. The prediction uses a
numerically stable sigmoid computation, avoiding exponential overflow at
extreme inputs. It predicts *continuous outcomes*, so it is not the same
task as binary logistic classification.

**Approved Model 2 synthetic truth:** `L=10`, `k=1.5`, `x0=2`, over 120
equally spaced inputs from `0` to `4`. Additive Gaussian noise has
`sigma_noise=0.1`, independent noise seeds `1100`–`1109`, and the common
fixed split seed `2026` (72 training, 24 validation, 24 test). The
`config/model2.json` file represents the **first** registered dataset
(seed `1100`). Generate its CSV, JSON metadata with SHA-256, and plot with:

```powershell
python -m scripts.generate_model2
```

No true synthetic parameter is passed to either optimization algorithm.

Model 4 is inspired by Saman's hand-drawn rising, multi-peak curve. It is not
intended to trace that image pixel-for-pixel. Its unknown frequency omega can
create a multimodal loss landscape, making it a potential initialization-sensitivity
stress test.

Within each model we will compare both algorithms on **identical data and
initial parameter vectors**. Raw performance between different model families
must not be interpreted as an algorithm effect without accounting for their
different complexity and parameter count.

