# Rho Initialization and Vector-Rho Findings

The sweep used initial rho values `0.01, 0.1, 1, 10, 100`, three seeds, 100 ADMM iterations, and 4-bit quantization.

## Takeaways

- Scalar online OGD is fairly stable in final rho: on tanh QAT, `online_ogd` returns near `rho ~= 0.6` across a wide initialization range, but deploy loss still worsens when initialized very high. Its best initial rho was `0.01` and worst was `100`.
- On tiny LLM PTQ, the scalar feasibility-task method is the best online scalar method in this sweep. Its mean deploy relative error across initializations is `0.323`, with best `0.316` at `rho0=1` and worst `0.332` at `rho0=0.01`.
- Vector rho helps only modestly in this implementation. The best vector feasibility-task result is `0.317`, compared with the scalar feasibility-task best of `0.316`. The gain is not enough to beat the oracle scalar fixed `rho=10` result from the earlier suite.
- The value of vector rho is more about block heterogeneity and diagnostics than an automatic win. It is worth keeping as a serious extension, but it needs a stronger per-block task signal or blockwise curvature normalization.

## Research Implication

Initialization robustness is a useful selling point for online tuning: a robust online method should recover similar final penalties from poor starts. Vector rho should be presented as a natural generalization for multi-block ADMM, but the current evidence says it is not sufficient by itself; the online loss matters more than scalar versus vector parameterization.
