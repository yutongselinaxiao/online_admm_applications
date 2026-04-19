# Nonconvex Quantization Findings

Generated from `nonconvex_quantization_summary.csv` using three seeds, 100 ADMM iterations, and 4-bit symmetric per-channel quantization.

## Implemented Examples

- `tanh_qat_nonconvex`: one-hidden-layer tanh regression with quantized deployable weights. The continuous weight update is an inexact nonconvex gradient step; the quantized block is a low-bit projection.
- `tiny_llm_ptq`: synthetic LLM-style blockwise PTQ for q/k/v/o projections and MLP up/gate/down projections. The continuous block update is an exact least-squares solve; the deployable block is a 4-bit quantized projection.

## Main Observations

- On `tanh_qat_nonconvex`, the best deploy loss among the tested methods is `fixed_rho_0p1` at about `0.289`. `online_ogd` improves over fixed `rho=1` but does not beat the low fixed penalty.
- The new task-aware secant loss (`online_ogd_task_aware`) slightly improves over residual-only `online_ogd` on the tanh QAT example (`0.308` versus `0.311`), but the gain is small and still behind the best fixed low penalty.
- On `tiny_llm_ptq`, the feasibility-guarded task loss (`online_ogd_task_feasibility`) improves the online method substantially: deploy relative error drops from `0.349` for residual-only `online_ogd` to `0.316`. It still does not beat oracle fixed `rho=10` at `0.278`.
- The accept/reject shrink rule is not competitive here. It drives `rho` too low on PTQ and worsens deploy error, so it should be treated as a failed ablation unless redesigned with a two-sided or trust-region acceptance rule.
- The nonconvex/discrete examples therefore show a key limitation: residual balancing is not automatically task-optimal for quantization. The strongest new online result is problem-dependent: it helps the synthetic LLM PTQ case but hurts tanh QAT.
- The no-dual-rescale ablation is less dramatic in these short nonconvex examples than in the convex benchmarks, but it is still not theoretically valid. Keep it as a failure-mode ablation, not as a candidate method.

## Average Metrics

| Problem | Method | Deploy metric | Final primal | Final dual | Final rho |
|---|---:|---:|---:|---:|---:|
| tanh QAT | `fixed_rho_0p1` | deploy loss `0.289` | `0.476` | `0.083` | `0.1` |
| tanh QAT | `online_ogd_task_accept` | deploy loss `0.300` | `0.468` | `0.098` | `0.135` |
| tanh QAT | `online_ogd_task_aware` | deploy loss `0.308` | `0.368` | `0.371` | `0.618` |
| tanh QAT | `online_ogd` | deploy loss `0.311` | `0.363` | `0.372` | `0.588` |
| tanh QAT | `fixed_rho_1` | deploy loss `0.340` | `0.313` | `0.540` | `1.0` |
| tanh QAT | `online_ogd_task_feasibility` | deploy loss `0.382` | `0.252` | `0.604` | `1.39` |
| tanh QAT | `fixed_rho_10` | deploy loss `0.498` | `0.00037` | `0.069` | `10.0` |
| tiny LLM PTQ | `fixed_rho_10` | deploy rel. error `0.278` | `2.19` | `28.21` | `10.0` |
| tiny LLM PTQ | `online_ogd_task_feasibility` | deploy rel. error `0.316` | `5.79` | `15.80` | `2.05` |
| tiny LLM PTQ | `fixed_rho_1` | deploy rel. error `0.339` | `7.07` | `9.82` | `1.0` |
| tiny LLM PTQ | `online_ogd_epoch5` | deploy rel. error `0.343` | `7.31` | `8.73` | `0.853` |
| tiny LLM PTQ | `online_ogd_task_aware` | deploy rel. error `0.347` | `7.20` | `9.96` | `1.00` |
| tiny LLM PTQ | `online_ogd` | deploy rel. error `0.349` | `7.64` | `7.75` | `0.721` |
| tiny LLM PTQ | `online_ogd_task_accept` | deploy rel. error `0.392` | `9.24` | `0.093` | `0.0059` |

## Theorem Implication

These examples support the report's recommendation to change the nonconvex theorem target. A residual-balanced online penalty may control feasibility and dual residuals, but quantization quality is governed by calibration/deploy loss and quantization noise. The new task-aware losses do not invalidate the residual-balancing proof; they are different algorithms with additional black-box feedback terms. Their proof target should be a perturbation or dynamic-regret bound for the scalar online loss, plus stationarity/noise-floor guarantees for the nonconvex ADMM layer.
