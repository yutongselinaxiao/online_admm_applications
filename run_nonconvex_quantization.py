from __future__ import annotations

import argparse
import time
from pathlib import Path

from online_admm_experiments.controllers import (
    FixedRho,
    FeasibilityTaskOnlineOGD,
    NormalizedResidualBalancing,
    OnlineOGD,
    OnlineOGDNoDualRescale,
    RelativeResidualBalancing,
    ResidualBalancing,
    TaskAcceptRejectOnlineOGD,
    TaskAwareOnlineOGD,
)
from online_admm_experiments.nonconvex_quantization import (
    run_tanh_qat,
    run_tiny_llm_ptq,
)
from online_admm_experiments.problems import ExperimentResult
from online_admm_experiments.utils import summarize_history, write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nonconvex_quantization"


PROBLEMS = {
    "tanh_qat_nonconvex": run_tanh_qat,
    "tiny_llm_ptq": run_tiny_llm_ptq,
}


def controllers() -> list[tuple[object, float]]:
    return [
        (FixedRho(name="fixed_rho_0p1"), 0.1),
        (FixedRho(name="fixed_rho_1"), 1.0),
        (FixedRho(name="fixed_rho_10"), 10.0),
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
            OnlineOGD(
                eta0=0.35,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                update_period=1,
                name="online_ogd",
            ),
            1.0,
        ),
        (
            TaskAwareOnlineOGD(
                eta0=0.25,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                residual_weight=0.85,
                task_weight=0.45,
                task_grad_clip=1.5,
            ),
            1.0,
        ),
        (
            FeasibilityTaskOnlineOGD(
                eta0=0.22,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                residual_weight=0.75,
                task_weight=0.35,
                task_grad_clip=1.25,
                feasibility_weight=0.3,
            ),
            1.0,
        ),
        (
            TaskAcceptRejectOnlineOGD(
                eta0=0.25,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.25,
                residual_weight=0.85,
                task_weight=0.25,
                task_grad_clip=1.0,
                max_task_log_increase=0.01,
                shrink_on_reject=0.85,
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
                name="online_ogd_epoch5",
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
    bits: int,
) -> ExperimentResult:
    start = time.perf_counter()
    result = runner(
        seed=seed,
        controller=controller,
        rho0=rho0,
        bits=bits,
        max_iter=max_iter,
        tol=tol,
    )
    elapsed = time.perf_counter() - start
    result.metrics["wall_time_sec"] = elapsed
    summary = summarize_history(result.history)
    task_metric = result.metrics.get("deploy_rel_error", result.metrics.get("deploy_loss", 0.0))
    print(
        f"{problem:21s} {controller.name:28s} seed={seed} "
        f"iters={summary['iterations']:3d} "
        f"pr={summary['final_primal']:.2e} "
        f"du={summary['final_dual']:.2e} "
        f"rho={summary['final_rho']:.3g} "
        f"metric={task_metric:.4g} "
        f"time={elapsed:.2f}s"
    )
    return result


def write_result(result: ExperimentResult) -> None:
    write_csv(
        RESULTS / f"{result.problem}_{result.method}_seed{result.seed}_history.csv",
        result.history,
    )


def write_summary(results: list[ExperimentResult]) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "problem": result.problem,
                "method": result.method,
                "seed": result.seed,
                **summarize_history(result.history),
                **result.metrics,
            }
        )
    write_csv(RESULTS / "nonconvex_quantization_summary.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run nonconvex online ADMM quantization examples."
    )
    parser.add_argument(
        "--problem",
        choices=["all", *PROBLEMS.keys()],
        default="all",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--bits", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    selected = list(PROBLEMS) if args.problem == "all" else [args.problem]
    seeds = list(range(args.seed, args.seed + args.seeds))
    results: list[ExperimentResult] = []
    for problem in selected:
        runner = PROBLEMS[problem]
        for seed in seeds:
            for controller, rho0 in controllers():
                result = run_one(
                    runner,
                    problem,
                    seed,
                    controller,
                    rho0,
                    args.max_iter,
                    args.tol,
                    args.bits,
                )
                results.append(result)
                write_result(result)
    write_summary(results)
    print(f"\nWrote nonconvex quantization CSV outputs to {RESULTS}")


if __name__ == "__main__":
    main()
