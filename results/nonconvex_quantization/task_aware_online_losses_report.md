# Task-Aware Online Loss Experiment

Run configuration: three seeds, 100 ADMM iterations, 4-bit symmetric per-channel quantization, using the nonconvex QAT and synthetic LLM PTQ examples.

## New Online Losses

- `online_ogd_task_aware`: residual-balancing OGD plus a black-box secant estimate of the deploy metric gradient with respect to `log(rho)`.
- `online_ogd_task_feasibility`: the task-aware loss plus a feasibility pressure term that raises `rho` when the primal residual is large relative to the ADMM threshold.
- `online_ogd_task_accept`: a one-sided accept/reject safeguard that shrinks `rho` when the deploy metric worsens too quickly.

## Results

| Problem | Best fixed baseline | Residual OGD | Best new online loss | Takeaway |
|---|---:|---:|---:|---|
| `tanh_qat_nonconvex` deploy loss | `0.289` (`fixed_rho_0p1`) | `0.311` | `0.300` (`online_ogd_task_accept`) | New losses do not beat the low fixed penalty; task secant gives only a small improvement over residual OGD. |
| `tiny_llm_ptq` deploy relative error | `0.278` (`fixed_rho_10`) | `0.349` | `0.316` (`online_ogd_task_feasibility`) | The feasibility-task loss substantially improves over residual OGD, but still trails the oracle fixed high penalty. |

## Interpretation

The strongest new result is the feasibility-task loss on the synthetic LLM PTQ example. It moves the average final penalty to about `rho=2.05`, between residual OGD's `rho=0.72` and oracle fixed `rho=10`, and lowers deploy relative error from `0.349` to `0.316`.

The tanh QAT example points the other way: deploy loss prefers a low fixed penalty, while feasibility pressure over-tightens the continuous/quantized constraint and worsens task loss. This suggests the online loss should be problem-class aware. Residual balancing remains a useful stability objective, but deploy quality in discrete quantization needs task feedback or an oracle-tuned fixed baseline for comparison.

## Next Candidate Losses

- Two-sided trust-region task acceptance: reject updates that worsen deploy loss, but search both larger and smaller `rho`.
- Bandit finite-difference loss: occasionally probe `u_k +/- delta` on copied state or short rollouts to get a less biased task gradient.
- Hybrid schedule: use feasibility-task loss for PTQ-style least-squares blocks and task-aware residual loss for QAT-style nonconvex training.
