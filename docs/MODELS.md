# Four planned nonlinear regression families

Only Model 1 is implemented. Formulas for Models 2-4 are **proposals**, not yet
validated implementations.

| Model | Function | Free parameters | Status |
| --- | --- | --- | --- |
| 1. Exponential decay | a exp(bx) + c | a, b, c | Data generation implemented |
| 2. Logistic growth | L / (1 + exp[-k(x-x0)]) | L, k, x0 | Planned |
| 3. Gaussian peak | A exp[-(x-mu)^2/(2 sigma_model^2)] | A, mu, sigma_model | Planned |
| 4. Oscillatory upward trend | mx + A sin(omega x) + B sin(2 omega x) | m, A, omega, B | Planned |

**Naming note:** sigma_model is the width of the Gaussian regression curve and
must not be confused with sigma_noise, the standard deviation of observation noise.

Model 4 is inspired by Saman's hand-drawn rising, multi-peak curve. It is not
intended to trace that image pixel-for-pixel. Its unknown frequency omega can
create a multimodal loss landscape, making it a potential initialization-sensitivity
stress test.

Within each model we will compare both algorithms on **identical data and
initial parameter vectors**. Raw performance between different model families
must not be interpreted as an algorithm effect without accounting for their
different complexity and parameter count.

