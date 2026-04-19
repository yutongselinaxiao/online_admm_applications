from __future__ import annotations

import argparse
from pathlib import Path

from online_admm_experiments.controllers import FixedRho, OnlineOGD, ResidualBalancing
from online_admm_experiments.problems import (
    ExperimentResult,
    run_consensus_lasso,
    run_graphical_lasso,
    run_tv_denoising,
)
from online_admm_experiments.utils import summarize_history, write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


PROBLEMS = {
    "graphical_lasso": run_graphical_lasso,
    "consensus_lasso": run_consensus_lasso,
    "tv_denoising": run_tv_denoising,
}


def make_controllers(rho0: float, freeze_after: int | None) -> list:
    return [
        FixedRho(name="fixed_rho_0.1"),
        FixedRho(name="fixed_rho_1.0"),
        FixedRho(name="fixed_rho_10.0"),
        ResidualBalancing(rho_min=1e-3, rho_max=1e3, update_period=2),
        OnlineOGD(
            eta0=0.35,
            rho_min=1e-3,
            rho_max=1e3,
            grad_clip=2.0,
            ema=0.2,
            update_period=1,
            freeze_after=freeze_after,
        ),
        OnlineOGD(
            eta0=0.25,
            rho_min=1e-3,
            rho_max=1e3,
            grad_clip=1.0,
            ema=0.35,
            update_period=3,
            freeze_after=freeze_after,
            name="online_ogd_epoch3",
        ),
    ]


def controller_rho0(method_name: str, default_rho0: float) -> float:
    if method_name.endswith("0.1"):
        return 0.1
    if method_name.endswith("1.0"):
        return 1.0
    if method_name.endswith("10.0"):
        return 10.0
    return default_rho0


def run_suite(args: argparse.Namespace) -> list[ExperimentResult]:
    selected = list(PROBLEMS) if args.problem == "all" else [args.problem]
    seeds = list(range(args.seed, args.seed + args.seeds))
    results = []
    for problem in selected:
        runner = PROBLEMS[problem]
        for seed in seeds:
            for controller in make_controllers(args.rho0, args.freeze_after):
                rho0 = controller_rho0(controller.name, args.rho0)
                result = runner(
                    seed=seed,
                    controller=controller,
                    rho0=rho0,
                    max_iter=args.max_iter,
                    tol=args.tol,
                )
                results.append(result)
                history_path = RESULTS / f"{problem}_{controller.name}_seed{seed}_history.csv"
                write_csv(history_path, result.history)
                summary = summarize_history(result.history)
                print(
                    f"{problem:17s} {controller.name:20s} seed={seed} "
                    f"iters={summary['iterations']:3d} "
                    f"obj={summary['final_objective']:.6g} "
                    f"pr={summary['final_primal']:.2e} "
                    f"du={summary['final_dual']:.2e} "
                    f"rho={summary['final_rho']:.3g} "
                    f"changes={summary['rho_changes']:3d}"
                )
    return results


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
    write_csv(RESULTS / "summary.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact ADMM benchmarks for online rho tuning."
    )
    parser.add_argument(
        "--problem",
        choices=["all", *PROBLEMS.keys()],
        default="all",
        help="Benchmark to run.",
    )
    parser.add_argument("--seed", type=int, default=0, help="First random seed.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of seeds.")
    parser.add_argument("--rho0", type=float, default=1.0, help="Initial adaptive rho.")
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument(
        "--freeze-after",
        type=int,
        default=None,
        help="Freeze online rho after this ADMM iteration for stability experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    results = run_suite(args)
    write_summary(results)
    print(f"\nWrote CSV outputs to {RESULTS}")


if __name__ == "__main__":
    main()

