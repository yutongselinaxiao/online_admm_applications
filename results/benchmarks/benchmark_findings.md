# Expanded Benchmark Findings

Generated from `benchmark_summary.csv` using three seeds and 150 ADMM iterations.

## Main Takeaways

- `online_ogd` is competitive with the oracle fixed-rho grid on graphical lasso and consensus Lasso, and it is much stronger than poorly chosen fixed penalties.
- On TV denoising, the oracle fixed-rho grid selects `rho = 3` and averages about 64 iterations. `online_ogd` averages about 76 iterations without using the oracle grid.
- Relative residual balancing is a strong TV baseline, averaging about 76 iterations, and is much better than raw residual balancing on this problem.
- The no-dual-rescale ablation is consistently bad. It fails to converge within 150 iterations on graphical lasso and TV denoising, and is also much slower on consensus Lasso. This supports the implementation claim that scaled-dual rescaling is required when `rho` changes.
- `spectral_aadmm_xu2017` is the literature-faithful spectral AADMM-style baseline. It is strong on graphical lasso and consensus Lasso, but weaker on TV denoising in this implementation.
- `bb_online_penalty_simplified` is retained only as a simplified BB/OCO ablation. It is not the Xu-Figueiredo-Goldstein AADMM method.

## Average Iterations By Problem

| Problem | Method | Avg. iterations |
|---|---:|---:|
| Graphical lasso | `bb_online_penalty_simplified` | 29.0 |
| Graphical lasso | `spectral_aadmm_xu2017` | 33.0 |
| Graphical lasso | `online_ogd` | 34.7 |
| Graphical lasso | `oracle_fixed_grid` | 36.7 |
| Graphical lasso | `online_ogd_no_dual_rescale` | 150.0 |
| Consensus Lasso | `spectral_aadmm_xu2017` | 24.3 |
| Consensus Lasso | `online_ogd` | 24.7 |
| Consensus Lasso | `oracle_fixed_grid` | 27.0 |
| Consensus Lasso | `online_ogd_no_dual_rescale` | 149.0 |
| TV denoising | `oracle_fixed_grid` | 63.7 |
| TV denoising | `bb_online_penalty_simplified` | 69.3 |
| TV denoising | `online_ogd` | 76.3 |
| TV denoising | `residual_balance_relative` | 76.3 |
| TV denoising | `spectral_aadmm_xu2017` | 131.7 |
| TV denoising | `online_ogd_no_dual_rescale` | 150.0 |

Use `visualizations/index.html` for the full chart dashboard and `benchmark_summary.csv` for all numeric values.

## Sensitivity Figures

The dashboard now includes fixed-rho sensitivity plots for each problem:

- `sensitivity_graphical_lasso_iterations.svg`
- `sensitivity_graphical_lasso_residual.svg`
- `sensitivity_consensus_lasso_iterations.svg`
- `sensitivity_consensus_lasso_residual.svg`
- `sensitivity_tv_denoising_iterations.svg`
- `sensitivity_tv_denoising_residual.svg`

These plots show fixed-rho grid performance with a min-max band across seeds, the oracle fixed-rho reference, and vertical markers for the average final rho reached by adaptive methods. The rho trajectory plots also include a horizontal oracle-rho line.
