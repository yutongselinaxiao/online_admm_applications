from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from online_admm_experiments.controllers import (
    FeasibilityTaskOnlineOGD,
    FixedRho,
    NormalizedResidualMagnitudeOGD,
    OnlineOGD,
    SumLogResidualsOGD,
    TaskAwareOnlineOGD,
    TaskNormalizedMagnitudeOGD,
)
from online_admm_experiments.nonconvex_quantization import run_tiny_llm_ptq
from online_admm_experiments.problems import ExperimentResult
from online_admm_experiments.utils import summarize_history, write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llm_online_losses"


def controllers() -> list[tuple[object, float]]:
    return [
        (FixedRho(name="fixed_rho_0p1"), 0.1),
        (FixedRho(name="fixed_rho_1"), 1.0),
        (FixedRho(name="fixed_rho_10"), 10.0),
        (
            OnlineOGD(
                eta0=0.35,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                ema=0.2,
                name="online_ogd_balance",
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
            FeasibilityTaskOnlineOGD(
                eta0=0.2,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.3,
                ema=0.2,
                residual_weight=0.65,
                task_weight=0.25,
                task_grad_clip=1.0,
                feasibility_weight=0.6,
                name="online_ogd_task_feas_mid",
            ),
            1.0,
        ),
        (
            FeasibilityTaskOnlineOGD(
                eta0=0.18,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.5,
                ema=0.2,
                residual_weight=0.55,
                task_weight=0.2,
                task_grad_clip=1.0,
                feasibility_weight=0.9,
                name="online_ogd_task_feas_aggressive",
            ),
            1.0,
        ),
        (
            SumLogResidualsOGD(
                eta0=0.12,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                name="online_ogd_sum_log_residuals",
            ),
            1.0,
        ),
        (
            NormalizedResidualMagnitudeOGD(
                eta0=0.18,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                name="online_ogd_norm_magnitude",
            ),
            1.0,
        ),
        (
            TaskNormalizedMagnitudeOGD(
                eta0=0.18,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                task_weight=0.25,
                task_grad_clip=1.0,
                feasibility_weight=0.2,
            ),
            1.0,
        ),
        (
            TaskNormalizedMagnitudeOGD(
                eta0=0.16,
                rho_min=1e-3,
                rho_max=1e3,
                grad_clip=2.0,
                task_weight=0.15,
                task_grad_clip=1.0,
                feasibility_weight=0.45,
                name="online_ogd_task_norm_mag_feas_high",
            ),
            1.0,
        ),
    ]


def convergence_metrics(history: list[dict]) -> dict:
    residual_max = [max(float(row["primal_norm"]), float(row["dual_norm"])) for row in history]
    primal = [float(row["primal_norm"]) for row in history]
    deploy = [float(row["deploy_rel_error"]) for row in history]
    log_residual_auc = sum(math.log10(max(v, 1e-12)) for v in residual_max) / len(residual_max)
    log_primal_auc = sum(math.log10(max(v, 1e-12)) for v in primal) / len(primal)
    iter_primal_below_3 = next((i + 1 for i, value in enumerate(primal) if value < 3.0), "")
    return {
        "best_deploy_rel_error": min(deploy),
        "final_residual_max": residual_max[-1],
        "log_residual_auc": log_residual_auc,
        "log_primal_auc": log_primal_auc,
        "iter_primal_below_3": iter_primal_below_3,
    }


def run_one(seed: int, controller, rho0: float, max_iter: int, bits: int, tol: float) -> ExperimentResult:
    start = time.perf_counter()
    result = run_tiny_llm_ptq(
        seed=seed,
        controller=controller,
        rho0=rho0,
        bits=bits,
        max_iter=max_iter,
        tol=tol,
    )
    result.metrics["rho0"] = rho0
    result.metrics["wall_time_sec"] = time.perf_counter() - start
    result.metrics.update(convergence_metrics(result.history))
    summary = summarize_history(result.history)
    print(
        f"{controller.name:34s} seed={seed} rho0={rho0:g} "
        f"deploy={result.metrics['deploy_rel_error']:.4f} "
        f"pr={summary['final_primal']:.2e} du={summary['final_dual']:.2e} "
        f"rho={summary['final_rho']:.3g}"
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
    write_csv(RESULTS / "llm_online_loss_summary.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted LLM PTQ online-loss sweep.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []
    for seed in range(args.seed, args.seed + args.seeds):
        for controller, rho0 in controllers():
            result = run_one(seed, controller, rho0, args.max_iter, args.bits, args.tol)
            results.append(result)
            write_result(result)
    write_summary(results)
    print(f"\nWrote LLM online-loss outputs to {RESULTS}")


if __name__ == "__main__":
    main()
