# Online ADMM Penalty Tuning Experiments

This folder contains a small, pure-NumPy experiment harness for validating the online ADMM method described in your notes.

The code focuses on exact two-block scaled ADMM benchmarks:

- Graphical lasso with an eigendecomposition `X` update and soft-thresholded `Z` update.
- Consensus Lasso with shared scalar `rho` across workers.
- 1D total-variation denoising with an exact linear solve and soft-thresholded difference variable.

The adaptive method updates `u = log(rho)` with projected online gradient descent on the residual-balance loss:

```text
0.5 * (log(rho * dual_base_norm) - log(primal_norm))^2
```

Whenever `rho` changes, each solver rescales the scaled dual variable so that the unscaled dual is preserved:

```text
u_scaled_new = (rho_old / rho_new) * u_scaled_old
```

## Run

From this folder:

```powershell
$py='C:\Users\12058\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py .\run_experiments.py --problem all --seeds 3 --max-iter 150
```

For the burn-in/freeze validation suggested by the theory notes:

```powershell
& $py .\run_experiments.py --problem all --seeds 5 --max-iter 200 --freeze-after 75
```

Outputs are written to `results\summary.csv` and one history CSV per problem/method/seed.

To print grouped averages:

```powershell
& $py .\summarize_results.py
```

To generate SVG charts and an HTML dashboard:

```powershell
& $py .\visualize_results.py
```

Open `results\visualizations\index.html` to inspect the aggregate method comparisons and seed-0 trajectories.

To run the expanded benchmark suite with oracle fixed-rho, normalized and relative residual balancing, Xu-style spectral AADMM, a simplified BB/OCO ablation, freeze/epoch variants, and the no-dual-rescale ablation:

```powershell
& $py .\run_benchmark_suite.py --problem all --seeds 3 --max-iter 150
& $py .\visualize_benchmarks.py
```

Open `results\benchmarks\visualizations\index.html` for the expanded comparison plots, fixed-rho sensitivity figures, and rho trajectories with oracle-rho reference lines.

The LaTeX writeup for the problem formulations, ADMM derivations, closed-form ADMM subproblem solutions, and pseudocode is in `latex\online_admm_formulations.tex`.

For experimental nonconvex ADMM quantization examples:

```powershell
& $py .\run_nonconvex_quantization.py --problem all --seeds 3 --max-iter 100 --bits 4
& $py .\visualize_nonconvex_quantization.py
```

Open `results\nonconvex_quantization\visualizations\index.html` for plots. The companion LaTeX theorem roadmap is in `latex\nonconvex_online_admm_quantization.tex`.

For the LLM PTQ follow-up experiments:

```powershell
& $py .\run_rho_robustness_vector.py --mode all --seeds 3 --rho0 0.01 0.1 1 10 100 --max-iter 100 --bits 4
& $py .\visualize_rho_robustness.py
& $py .\run_llm_online_loss_sweep.py --seeds 3 --max-iter 100 --bits 4
& $py .\visualize_llm_online_loss_sweep.py
& $py .\run_llm_quantization_baselines.py --seeds 3 --max-iter 100 --bits 4
& $py .\visualize_llm_quantization_baselines.py
& $py .\run_llm_curvature_admm.py --seeds 3 --max-iter 100 --bits 4
& $py .\visualize_llm_curvature_admm.py
& $py .\run_real_model_smoke.py --random-tiny --max-batches 3 --max-modules 6 --max-samples 96 --max-admm-iter 6 --block-size 64
```

The cumulative experiment ledger is in `results\online_admm_experiment_ledger.md`. Cluster notes for a real Hugging Face model smoke test are in `CLUSTER_SMOKE_README.md`.

## What To Look For

Use `summary.csv` to compare:

- final primal and dual residuals;
- final objective and best objective;
- number of penalty changes;
- task-specific recovery metrics such as support F1 or relative signal error.

Use the per-iteration history files to inspect:

- whether online `rho` oscillates;
- whether residual spikes align with penalty changes;
- whether freezing `rho` improves convergence stability;
- whether online OGD beats the fixed-rho grid on iteration count or final residuals.
