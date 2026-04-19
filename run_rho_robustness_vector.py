from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable

import numpy as np

from online_admm_experiments.controllers import (
    FeasibilityTaskOnlineOGD,
    OnlineOGD,
    TaskAwareOnlineOGD,
)
from online_admm_experiments.nonconvex_quantization import (
    EPS,
    _controller_context,
    combined_norm,
    make_tiny_llm_blocks,
    run_tanh_qat,
    run_tiny_llm_ptq,
    symmetric_uniform_quantize,
)
from online_admm_experiments.problems import ExperimentResult, _maybe_rescale_scaled_dual
from online_admm_experiments.utils import summarize_history, write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "rho_robustness"


def scalar_controller_factories() -> dict[str, Callable[[], object]]:
    return {
        "online_ogd": lambda: OnlineOGD(
            eta0=0.35,
            rho_min=1e-3,
            rho_max=1e3,
            grad_clip=2.0,
            ema=0.2,
            update_period=1,
            name="online_ogd",
        ),
        "online_ogd_task_aware": lambda: TaskAwareOnlineOGD(
            eta0=0.25,
            rho_min=1e-3,
            rho_max=1e3,
            grad_clip=2.0,
            ema=0.2,
            residual_weight=0.85,
            task_weight=0.45,
            task_grad_clip=1.5,
        ),
        "online_ogd_task_feasibility": lambda: FeasibilityTaskOnlineOGD(
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
    }


def vector_controller_factories() -> dict[str, Callable[[], object]]:
    return {
        "vector_online_ogd": lambda: OnlineOGD(
            eta0=0.35,
            rho_min=1e-3,
            rho_max=1e3,
            grad_clip=2.0,
            ema=0.2,
            update_period=1,
            name="online_ogd",
        ),
        "vector_task_aware": lambda: TaskAwareOnlineOGD(
            eta0=0.25,
            rho_min=1e-3,
            rho_max=1e3,
            grad_clip=2.0,
            ema=0.2,
            residual_weight=0.85,
            task_weight=0.45,
            task_grad_clip=1.5,
        ),
        "vector_task_feasibility": lambda: FeasibilityTaskOnlineOGD(
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
    }


def _rho_stats(rhos: list[float]) -> tuple[float, float, float]:
    values = np.asarray(rhos, dtype=float)
    geo = float(np.exp(np.mean(np.log(np.maximum(values, EPS)))))
    return geo, float(np.min(values)), float(np.max(values))


def run_tiny_llm_ptq_vector_rho(
    seed: int,
    controller_factory: Callable[[], object],
    method_name: str,
    rho0: float,
    bits: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4,
    gamma: float = 1e-4,
) -> ExperimentResult:
    blocks = make_tiny_llm_blocks(seed)
    controllers = []
    states = []
    for block in blocks:
        controller = controller_factory()
        controllers.append(controller)
        states.append(controller.init_state(rho0))
        w = block["w_fp"].copy()
        block["w"] = w
        block["z"] = symmetric_uniform_quantize(w, bits=bits, axis=0).value
        block["u"] = np.zeros_like(w)
        block["xtx"] = block["x"].T @ block["x"] / block["x"].shape[0]
        block["xty"] = block["x"].T @ block["target"] / block["x"].shape[0]

    history: list[dict] = []
    for k in range(max_iter):
        primal_arrays = []
        dual_terms = []
        objective = 0.0
        deploy_error_num = 0.0
        deploy_error_den = 0.0
        changed_count = 0

        for i, block in enumerate(blocks):
            rho = states[i].rho
            z_old = block["z"].copy()
            lhs = block["xtx"] + (gamma + rho) * np.eye(block["w_fp"].shape[0])
            rhs = block["xty"] + gamma * block["w_fp"] + rho * (block["z"] - block["u"])
            block["w"] = np.linalg.solve(lhs, rhs)
            block["z"] = symmetric_uniform_quantize(block["w"] + block["u"], bits=bits, axis=0).value
            block["u"] = block["u"] + block["w"] - block["z"]

            primal = block["w"] - block["z"]
            dual_base = block["z"] - z_old
            primal_arrays.append(primal)
            dual_terms.append(rho * dual_base)

            cont_err = block["x"] @ block["w"] - block["target"]
            deploy_err = block["x"] @ block["z"] - block["target"]
            objective += 0.5 * float(np.mean(cont_err * cont_err))
            objective += 0.5 * gamma * float(np.mean((block["w"] - block["w_fp"]) ** 2))
            block_deploy_num = float(np.sum(deploy_err * deploy_err))
            block_deploy_den = float(np.sum(block["target"] * block["target"]))
            deploy_error_num += block_deploy_num
            deploy_error_den += block_deploy_den

            context = _controller_context(
                tol,
                [block["w"]],
                [block["z"]],
                [block["u"]],
                rho,
            )
            context["task_metric"] = float(np.sqrt(block_deploy_num / max(block_deploy_den, EPS)))
            decision = controllers[i].update(
                k,
                states[i],
                float(np.linalg.norm(primal)),
                float(np.linalg.norm(dual_base)),
                context,
            )
            if getattr(controllers[i], "rescale_scaled_dual", True):
                block["u"] = _maybe_rescale_scaled_dual(block["u"], rho, decision.rho)
            states[i] = decision.state
            changed_count += int(decision.changed)

        rhos = [state.rho for state in states]
        rho_geo, rho_min, rho_max = _rho_stats(rhos)
        primal_norm = combined_norm(primal_arrays)
        dual_norm = combined_norm(dual_terms)
        deploy_rel_error = float(np.sqrt(deploy_error_num / max(deploy_error_den, EPS)))
        status = "converged" if max(primal_norm, dual_norm) < tol else "running"
        row = {
            "iter": k + 1,
            "objective": objective,
            "deploy_rel_error": deploy_rel_error,
            "primal_norm": primal_norm,
            "dual_norm": dual_norm,
            "rho": rho_geo,
            "rho_next": rho_geo,
            "rho_min": rho_min,
            "rho_max": rho_max,
            "rho_changed": changed_count > 0,
            "rho_changed_blocks": changed_count,
            "gradient": 0.0,
            "loss": 0.0,
            "decision": "vector",
            "status": status,
        }
        for block, rho in zip(blocks, rhos):
            row[f"rho_{block['name']}"] = rho
        history.append(row)
        if status == "converged":
            break

    weight_num = sum(float(np.sum((block["z"] - block["w_fp"]) ** 2)) for block in blocks)
    weight_den = sum(float(np.sum(block["w_fp"] * block["w_fp"])) for block in blocks)
    metrics = {
        "deploy_rel_error": history[-1]["deploy_rel_error"],
        "weight_rel_error": float(np.sqrt(weight_num / max(weight_den, EPS))),
        "bits": bits,
        "num_blocks": len(blocks),
        "rho_min": history[-1]["rho_min"],
        "rho_max": history[-1]["rho_max"],
    }
    return ExperimentResult("tiny_llm_ptq_vector_rho", method_name, seed, metrics, history)


def run_scalar_robustness(
    seeds: list[int],
    rho0s: list[float],
    max_iter: int,
    bits: int,
    tol: float,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    problems = {
        "tanh_qat_nonconvex": run_tanh_qat,
        "tiny_llm_ptq": run_tiny_llm_ptq,
    }
    for problem, runner in problems.items():
        for seed in seeds:
            for rho0 in rho0s:
                for method, factory in scalar_controller_factories().items():
                    controller = factory()
                    controller.name = f"{method}_rho0_{rho0:g}"
                    start = time.perf_counter()
                    result = runner(
                        seed=seed,
                        controller=controller,
                        rho0=rho0,
                        bits=bits,
                        max_iter=max_iter,
                        tol=tol,
                    )
                    result.metrics["rho0"] = rho0
                    result.metrics["base_method"] = method
                    result.metrics["wall_time_sec"] = time.perf_counter() - start
                    results.append(result)
                    print_result(problem, method, seed, rho0, result)
    return results


def run_vector_ptq(
    seeds: list[int],
    rho0s: list[float],
    max_iter: int,
    bits: int,
    tol: float,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    for seed in seeds:
        for rho0 in rho0s:
            for method, factory in vector_controller_factories().items():
                start = time.perf_counter()
                result = run_tiny_llm_ptq_vector_rho(
                    seed=seed,
                    controller_factory=factory,
                    method_name=f"{method}_rho0_{rho0:g}",
                    rho0=rho0,
                    bits=bits,
                    max_iter=max_iter,
                    tol=tol,
                )
                result.metrics["rho0"] = rho0
                result.metrics["base_method"] = method
                result.metrics["wall_time_sec"] = time.perf_counter() - start
                results.append(result)
                print_result("tiny_llm_ptq_vector_rho", method, seed, rho0, result)
    return results


def print_result(problem: str, method: str, seed: int, rho0: float, result: ExperimentResult) -> None:
    summary = summarize_history(result.history)
    task_metric = result.metrics.get("deploy_rel_error", result.metrics.get("deploy_loss", 0.0))
    print(
        f"{problem:25s} {method:28s} seed={seed} rho0={rho0:<6g} "
        f"pr={summary['final_primal']:.2e} du={summary['final_dual']:.2e} "
        f"rho={summary['final_rho']:.3g} metric={task_metric:.4g}"
    )


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
    write_csv(RESULTS / "rho_robustness_summary.csv", rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rho-initialization and vector-rho robustness experiments.")
    parser.add_argument("--mode", choices=["all", "scalar", "vector"], default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--rho0", nargs="*", type=float, default=[0.01, 0.1, 1.0, 10.0, 100.0])
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seed, args.seed + args.seeds))
    results: list[ExperimentResult] = []
    if args.mode in {"all", "scalar"}:
        results.extend(run_scalar_robustness(seeds, args.rho0, args.max_iter, args.bits, args.tol))
    if args.mode in {"all", "vector"}:
        results.extend(run_vector_ptq(seeds, args.rho0, args.max_iter, args.bits, args.tol))
    for result in results:
        write_result(result)
    write_summary(results)
    print(f"\nWrote rho robustness outputs to {RESULTS}")


if __name__ == "__main__":
    main()
