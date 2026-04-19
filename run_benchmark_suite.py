from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path

from online_admm_experiments.controllers import (
    BBOnlinePenalty,
    FixedRho,
    NormalizedResidualBalancing,
    OnlineOGD,
    OnlineOGDNoDualRescale,
    RelativeResidualBalancing,
    ResidualBalancing,
    SpectralAADMM,
)
from online_admm_experiments.problems import (
    ExperimentResult,
    run_consensus_lasso,
    run_graphical_lasso,
    run_tv_denoising,
)
from online_admm_experiments.utils import summarize_history, write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "benchmarks"


PROBLEMS = {
    "graphical_lasso": run_graphical_lasso,
    "consensus_lasso": run_consensus_lasso,
    "tv_denoising": run_tv_denoising,
}


def fmt_rho(rho: float) -> str:
    text = f"{rho:g}"
    return text.replace(".", "p").replace("-", "m")


def fixed_grid(rhos: list[float]) -> list[tuple[FixedRho, float]]:
    return [(FixedRho(name=f"fixed_rho_{fmt_rho(rho)}"), rho) for rho in rhos]


def adaptive_controllers(freeze_after: int | None) -> list[tuple[object, float]]:
    return [
        (
            ResidualBalancing(
                rho_min=1e-3,
                rho_max=1e3,
                update_period=2,
                name="residual_balance_raw",
            ),
            1.0,
        ),
        (
            NormalizedResidualBalancing(
                rho_min=1e-3,
                rho_max=1e3,
                update_period=2,
            ),
            1.0,
        ),
        (
            RelativeResidualBalancing(
                rho_min=1e-3,
                rho_max=1e3,
                update_period=2,
            ),
            1.0,
        ),
        (
            BBOnlinePenalty(
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                update_period=1,
                freeze_after=freeze_after,
            ),
            1.0,
        ),
        (
            SpectralAADMM(
                rho_min=1e-3,
                rho_max=1e3,
                update_period=2,
                correlation_threshold=0.2,
            ),
            1.0,
        ),
        (
            OnlineOGD(
                eta0=0.35,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                update_period=1,
                freeze_after=freeze_after,
                name="online_ogd",
            ),
            1.0,
        ),
        (
            OnlineOGD(
                eta0=0.25,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=1.0,
                ema=0.35,
                update_period=5,
                freeze_after=freeze_after,
                name="online_ogd_epoch5",
            ),
            1.0,
        ),
        (
            OnlineOGD(
                eta0=0.25,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=1.0,
                ema=0.35,
                update_period=10,
                freeze_after=freeze_after,
                name="online_ogd_epoch10",
            ),
            1.0,
        ),
        (
            OnlineOGD(
                eta0=0.35,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                update_period=1,
                freeze_after=50,
                name="online_ogd_freeze50",
            ),
            1.0,
        ),
        (
            OnlineOGDNoDualRescale(
                eta0=0.35,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                update_period=1,
                freeze_after=freeze_after,
            ),
            1.0,
        ),
    ]


def run_one(
    runner,
    problem: str,
    seed: int,
    controller,
    rho0: float,
    max_iter: int,
    tol: float,
) -> ExperimentResult:
    start = time.perf_counter()
    result = runner(
        seed=seed,
        controller=controller,
        rho0=rho0,
        max_iter=max_iter,
        tol=tol,
    )
    elapsed = time.perf_counter() - start
    result.metrics["wall_time_sec"] = elapsed
    summary = summarize_history(result.history)
    print(
        f"{problem:17s} {controller.name:28s} seed={seed} "
        f"iters={summary['iterations']:3d} "
        f"pr={summary['final_primal']:.2e} "
        f"du={summary['final_dual']:.2e} "
        f"rho={summary['final_rho']:.3g} "
        f"time={elapsed:.2f}s"
    )
    return result


def oracle_key(result: ExperimentResult) -> tuple[int, float, float]:
    summary = summarize_history(result.history)
    converged = 0 if summary["final_status"] == "converged" else 1
    residual = max(summary["final_primal"], summary["final_dual"])
    return converged, summary["iterations"] if not converged else summary["iterations"], residual


def make_oracle_result(
    problem: str,
    seed: int,
    fixed_results: list[ExperimentResult],
) -> ExperimentResult:
    best = min(fixed_results, key=oracle_key)
    best_summary = summarize_history(best.history)
    oracle_history = [dict(row) for row in best.history]
    metrics = dict(best.metrics)
    metrics["oracle_source_method"] = best.method
    metrics["oracle_rho"] = best_summary["final_rho"]
    metrics["wall_time_sec"] = best.metrics.get("wall_time_sec", 0.0)
    return ExperimentResult(problem, "oracle_fixed_grid", seed, metrics, oracle_history)


def write_result(result: ExperimentResult) -> None:
    history_path = RESULTS / f"{result.problem}_{result.method}_seed{result.seed}_history.csv"
    write_csv(history_path, result.history)


def write_summary(results: list[ExperimentResult]) -> None:
    rows = []
    for result in results:
        row = {
            "problem": result.problem,
            "method": result.method,
            "seed": result.seed,
            **summarize_history(result.history),
            **result.metrics,
        }
        rows.append(row)
    write_csv(RESULTS / "benchmark_summary.csv", rows)


def run_suite(args: argparse.Namespace) -> list[ExperimentResult]:
    selected = list(PROBLEMS) if args.problem == "all" else [args.problem]
    seeds = list(range(args.seed, args.seed + args.seeds))
    rhos = [float(rho) for rho in args.rho_grid.split(",")]
    all_results: list[ExperimentResult] = []

    for problem in selected:
        runner = PROBLEMS[problem]
        for seed in seeds:
            fixed_results = []
            for controller, rho0 in fixed_grid(rhos):
                result = run_one(runner, problem, seed, controller, rho0, args.max_iter, args.tol)
                fixed_results.append(result)
                all_results.append(result)
                write_result(result)

            oracle = make_oracle_result(problem, seed, fixed_results)
            all_results.append(oracle)
            write_result(oracle)

            for controller, rho0 in adaptive_controllers(args.freeze_after):
                result = run_one(runner, problem, seed, controller, rho0, args.max_iter, args.tol)
                all_results.append(result)
                write_result(result)

    return all_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run expanded ADMM penalty-controller benchmark suite."
    )
    parser.add_argument(
        "--problem",
        choices=["all", *PROBLEMS.keys()],
        default="all",
        help="Benchmark to run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--freeze-after", type=int, default=None)
    parser.add_argument(
        "--rho-grid",
        default="0.01,0.03,0.1,0.3,1,3,10,30,100",
        help="Comma-separated fixed-rho grid used for oracle comparison.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = run_suite(args)
    write_summary(results)
    print(f"\nWrote benchmark CSV outputs to {RESULTS}")


if __name__ == "__main__":
    main()
