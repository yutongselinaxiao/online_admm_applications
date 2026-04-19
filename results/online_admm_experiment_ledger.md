# Online ADMM Experiment Ledger

This file summarizes the experiment sequence and how each modification maps to a possible theorem.

## 1. Convex ADMM Penalty Tuning

Problems: graphical lasso, consensus lasso, TV denoising.

Methods: fixed rho, residual balancing, normalized/relative residual balancing, online OGD on `u = log(rho)`, simplified BB-style step sizes, spectral AADMM baseline.

Theory role: validates the online penalty-tuning layer in convex settings and motivates the bounded-path-length proof for `log(rho)`.

## 2. Nonconvex Quantization ADMM

Problems: one-hidden-layer tanh QAT and synthetic LLM-style blockwise PTQ.

Initial result: residual balancing alone is not task-optimal. On synthetic LLM PTQ, fixed high rho gave better deploy error than residual-only online OGD.

Theory role: convex objective-gap claims should be replaced by nonconvex stationarity, quantization-noise floors, and task/residual Pareto guarantees.

## 3. Task-Aware Online Losses

Methods:

- `online_ogd_task_aware`: residual OGD plus a secant estimate of deploy loss with respect to `log(rho)`.
- `online_ogd_task_feasibility`: task-aware OGD plus primal-feasibility pressure.
- `online_ogd_task_accept`: one-sided reject/shrink ablation.

Key result: on synthetic LLM PTQ, feasibility-task OGD improved deploy relative error from residual OGD's `0.349` to `0.316`, but did not beat fixed `rho=10` at `0.278`.

Theory role: adds a dynamic/linearized online-loss theorem target for task-aware updates.

## 4. Rho Initialization and Vector Rho

Sweep: `rho0 in {0.01, 0.1, 1, 10, 100}` over three seeds.

Key result: scalar online methods are robust in final rho, but robust final rho does not guarantee best deploy accuracy. Vector rho was principled but only modestly helpful in the current implementation.

Theory role: supports a product-space path-length theorem for vector `u in U^m`, but shows that vectorization alone is not enough without better blockwise task/curvature signals.

## 5. Alternative Online Losses and Tradeoffs

Methods:

- sum of log residual magnitudes,
- normalized residual magnitude,
- task-normalized residual magnitude,
- mid and aggressive task-feasibility losses.

Key result: the plain sum-log residual loss failed on LLM PTQ, reaching deploy error `0.364` and driving rho down to about `0.109`. Aggressive task-feasibility reached deploy error `0.254`, beating fixed `rho=10` at `0.278`, but with larger dual residual.

Theory role: suggests a Pareto theorem rather than a pure residual-convergence theorem. The relevant tradeoff is deploy accuracy versus primal/dual residual behavior.

## 6. Local LLM Quantization Proxy Baselines

Baselines: RTN, Hessian diagonal clipping, GPTQ-like sequential compensation, AWQ-like scaling, SmoothQuant-like scaling.

Key result: plain ADMM with uniform projection is roughly RTN-level, but does not beat Hessian-aware PTQ. GPTQ-like proxy reached deploy error `0.205`; aggressive uniform ADMM reached `0.254`.

Theory role: shows that online rho tuning alone is insufficient. The quantized `Z` update must be curvature-aware.

## 7. Curvature-Aware ADMM

Modification: replace Euclidean projection

`Z = Proj_Q(W + U)`

with a curvature-aware quantized update based on the target

`T = (alpha H + rho I)^(-1) (alpha H W_fp + rho (W + U))`.

Tested `Z` updates: Hessian diagonal, GPTQ-like sequential, AWQ-like scaled, and uniform.

Key result: ADMM with GPTQ-like `Z` update plus online rho reached deploy error `0.214`, close to standalone GPTQ-like proxy at `0.205` and better than RTN at `0.252`.

Theory role: motivates a projection-error theorem for curvature-aware approximate `Z` updates, combined with the online penalty path perturbation theorem.

## Overall Conclusion

The online ADMM method has a viable path if it is framed as:

1. a bounded online penalty controller,
2. a task/residual tradeoff mechanism,
3. a curvature-aware quantized projection framework.

It should not be framed as pure residual balancing or as a plain uniform-projection quantizer competing directly with GPTQ/AWQ.

## 8. Real-Model Smoke Test

Script: `run_real_model_smoke.py`.

Local status: the required CPU packages installed successfully, but downloading `sshleifer/tiny-gpt2` from Hugging Face failed because the remote connection was closed. An offline random tiny GPT-2 fallback was added and passed on CPU.

Smoke result: `fp32`, `rtn`, and `admm_gptq` all ran on a random tiny GPT-2 architecture. The script quantized 6 supported projection modules and wrote `results/real_model_smoke/real_model_smoke_summary.csv`.

Theory role: no theorem evidence yet, because the fallback model is random and the corpus is synthetic. The value is engineering readiness: the real-model hook, calibration, module quantization, and evaluation path work.

Next step: run the same script on a cluster or machine with stable Hugging Face access using `sshleifer/tiny-gpt2`, then `gpt2` or `facebook/opt-125m`.
