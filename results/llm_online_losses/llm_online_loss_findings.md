# LLM Online Loss Sweep Findings

Run configuration: synthetic LLM PTQ, three seeds, 100 ADMM iterations, 4-bit symmetric per-channel quantization.

## Main Results

- Best deploy relative error: `online_ogd_task_feas_aggressive` at `0.254`.
- The aggressive feasibility-task online method beats fixed `rho=10` on deploy error: `0.254` versus `0.278`. The cost is a larger average final dual residual: `37.37` versus `28.21`.
- The milder feasibility-task variants form a useful ladder: base feasibility `0.316`, mid feasibility `0.282`, aggressive feasibility `0.254`.
- The normalized-magnitude high-feasibility variant reaches `0.367`, so this residual-size objective is not competitive here.
- The plain sum-log residual loss is not good here: it reaches `0.364` and drives the final rho down to `0.1087`. This supports the analytic concern that minimizing `log ||r|| + log ||s||` has a one-sided immediate gradient in `log(rho)`.

## Tradeoff

The scatter plots show a real tradeoff: penalties that improve deploy accuracy often do not minimize the combined ADMM residual score. In PTQ, stronger coupling can lower quantization error while leaving larger dual movement. This suggests that faster ADMM residual convergence is not automatically worth it if the objective is deploy accuracy.
