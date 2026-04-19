from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from online_admm_experiments.controllers import FeasibilityTaskOnlineOGD, FixedRho, PenaltyController
from online_admm_experiments.nonconvex_quantization import (
    EPS,
    _controller_context,
    combined_norm,
    make_tiny_llm_blocks,
    symmetric_uniform_quantize,
)
from online_admm_experiments.problems import ExperimentResult, _maybe_rescale_scaled_dual
from online_admm_experiments.utils import summarize_history, write_csv
from run_llm_quantization_baselines import PTQ_BASELINES, evaluate_blocks


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llm_curvature_admm"


def quantize_with_scale(x: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    safe_scale = np.maximum(scale, EPS)
    codes = np.clip(np.round(x / safe_scale), qmin, qmax)
    return codes * safe_scale


def per_output_scale(w: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    return np.maximum(np.max(np.abs(w), axis=0, keepdims=True) / max(qmax, 1), EPS)


def combined_z_target(block: dict, v: np.ndarray, rho: float, curvature_weight: float) -> tuple[np.ndarray, np.ndarray]:
    hessian = block["xtx"]
    q = curvature_weight * hessian + rho * np.eye(hessian.shape[0])
    rhs = curvature_weight * (hessian @ block["w_fp"]) + rho * v
    target = np.linalg.solve(q, rhs)
    return target, q


def quantize_hessian_diag_target(target: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    diag = np.maximum(np.diag(q).reshape(-1, 1), EPS)
    base = per_output_scale(target, bits)
    multipliers = [0.45, 0.6, 0.75, 0.9, 1.0, 1.15, 1.35]
    best_z = None
    best_loss = math.inf
    for mult in multipliers:
        z = quantize_with_scale(target, base * mult, bits)
        err = z - target
        loss = float(np.sum(diag * err * err))
        if loss < best_loss:
            best_loss = loss
            best_z = z
    assert best_z is not None
    return best_z


def quantize_gptq_target(target: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    damp = 0.001 * float(np.mean(np.diag(q)))
    q = q + (damp + 1e-8) * np.eye(q.shape[0])
    order = np.argsort(-np.diag(q))
    inv_order = np.argsort(order)
    q_perm = q[np.ix_(order, order)]
    w_work = target[order, :].copy()
    q_inv = np.linalg.inv(q_perm)
    scale = per_output_scale(w_work, bits)
    z_perm = np.zeros_like(w_work)
    for i in range(w_work.shape[0]):
        z_i = quantize_with_scale(w_work[i : i + 1, :], scale, bits)[0]
        z_perm[i] = z_i
        err = (w_work[i] - z_i) / max(q_inv[i, i], EPS)
        if i + 1 < w_work.shape[0]:
            w_work[i + 1 :, :] -= q_inv[i + 1 :, i : i + 1] * err.reshape(1, -1)
    return z_perm[inv_order, :]


def quantize_awq_target(block: dict, target: np.ndarray, q: np.ndarray, bits: int) -> np.ndarray:
    act = np.maximum(np.mean(np.abs(block["x"]), axis=0).reshape(-1, 1), EPS)
    diag = np.maximum(np.diag(q).reshape(-1, 1), EPS)
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    best_z = None
    best_loss = math.inf
    for alpha in alphas:
        scale_in = act**alpha
        scale_in = scale_in / np.exp(np.mean(np.log(np.maximum(scale_in, EPS))))
        z = symmetric_uniform_quantize(scale_in * target, bits=bits, axis=0).value / scale_in
        err = z - target
        loss = float(np.sum(diag * err * err))
        if loss < best_loss:
            best_loss = loss
            best_z = z
    assert best_z is not None
    return best_z


def z_update(block: dict, v: np.ndarray, rho: float, bits: int, mode: str, curvature_weight: float) -> np.ndarray:
    if mode == "uniform":
        return symmetric_uniform_quantize(v, bits=bits, axis=0).value
    target, q = combined_z_target(block, v, rho, curvature_weight)
    if mode == "hessian_diag":
        return quantize_hessian_diag_target(target, q, bits)
    if mode == "gptq_like":
        return quantize_gptq_target(target, q, bits)
    if mode == "awq_like":
        return quantize_awq_target(block, target, q, bits)
    raise ValueError(f"Unknown z-update mode: {mode}")


def run_tiny_llm_curvature_admm(
    seed: int,
    controller: PenaltyController,
    rho0: float,
    z_mode: str,
    bits: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4,
    gamma: float = 1e-4,
    curvature_weight: float = 1.0,
) -> ExperimentResult:
    blocks = make_tiny_llm_blocks(seed)
    for block in blocks:
        w = block["w_fp"].copy()
        block["w"] = w
        block["z"] = symmetric_uniform_quantize(w, bits=bits, axis=0).value
        block["u"] = np.zeros_like(w)
        block["xtx"] = block["x"].T @ block["x"] / block["x"].shape[0]
        block["xty"] = block["x"].T @ block["target"] / block["x"].shape[0]

    state = controller.init_state(rho0)
    history: list[dict] = []
    for k in range(max_iter):
        rho = state.rho
        primal_arrays = []
        dual_base_arrays = []
        objective = 0.0
        deploy_error_num = 0.0
        deploy_error_den = 0.0

        for block in blocks:
            z_old = block["z"].copy()
            lhs = block["xtx"] + (gamma + rho) * np.eye(block["w_fp"].shape[0])
            rhs = block["xty"] + gamma * block["w_fp"] + rho * (block["z"] - block["u"])
            block["w"] = np.linalg.solve(lhs, rhs)
            block["z"] = z_update(block, block["w"] + block["u"], rho, bits, z_mode, curvature_weight)
            block["u"] = block["u"] + block["w"] - block["z"]

            primal_arrays.append(block["w"] - block["z"])
            dual_base_arrays.append(block["z"] - z_old)
            cont_err = block["x"] @ block["w"] - block["target"]
            deploy_err = block["x"] @ block["z"] - block["target"]
            objective += 0.5 * float(np.mean(cont_err * cont_err))
            objective += 0.5 * gamma * float(np.mean((block["w"] - block["w_fp"]) ** 2))
            deploy_error_num += float(np.sum(deploy_err * deploy_err))
            deploy_error_den += float(np.sum(block["target"] * block["target"]))

        primal_norm = combined_norm(primal_arrays)
        dual_base_norm = combined_norm(dual_base_arrays)
        dual_norm = rho * dual_base_norm
        deploy_rel_error = float(np.sqrt(deploy_error_num / max(deploy_error_den, EPS)))
        status = "converged" if max(primal_norm, dual_norm) < tol else "running"
        context = _controller_context(
            tol,
            [block["w"] for block in blocks],
            [block["z"] for block in blocks],
            [block["u"] for block in blocks],
            rho,
        )
        context["task_metric"] = deploy_rel_error
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        if getattr(controller, "rescale_scaled_dual", True):
            for block in blocks:
                block["u"] = _maybe_rescale_scaled_dual(block["u"], rho, decision.rho)
        state = decision.state

        history.append(
            {
                "iter": k + 1,
                "objective": objective,
                "deploy_rel_error": deploy_rel_error,
                "primal_norm": primal_norm,
                "dual_norm": dual_norm,
                "rho": rho,
                "rho_next": decision.rho,
                "rho_changed": decision.changed,
                "gradient": decision.gradient,
                "loss": decision.loss,
                "decision": decision.reason,
                "status": status,
            }
        )
        if status == "converged":
            break

    weight_num = sum(float(np.sum((block["z"] - block["w_fp"]) ** 2)) for block in blocks)
    weight_den = sum(float(np.sum(block["w_fp"] * block["w_fp"])) for block in blocks)
    metrics = {
        "deploy_rel_error": history[-1]["deploy_rel_error"],
        "weight_rel_error": float(np.sqrt(weight_num / max(weight_den, EPS))),
        "bits": bits,
        "num_blocks": len(blocks),
        "z_mode": z_mode,
        "curvature_weight": curvature_weight,
    }
    return ExperimentResult("tiny_llm_curvature_admm", controller.name, seed, metrics, history)


def admm_configs() -> list[tuple[str, str, object, float, float]]:
    configs = []
    for z_mode in ["uniform", "hessian_diag", "gptq_like", "awq_like"]:
        configs.append(
            (
                f"admm_{z_mode}_fixed10",
                z_mode,
                FixedRho(name=f"admm_{z_mode}_fixed10"),
                10.0,
                1.0,
            )
        )
        configs.append(
            (
                f"admm_{z_mode}_online_aggressive",
                z_mode,
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
                    name=f"admm_{z_mode}_online_aggressive",
                ),
                1.0,
                1.0,
            )
        )
    return configs


def run_proxy_baselines(seed: int, bits: int) -> list[dict]:
    blocks = make_tiny_llm_blocks(seed)
    rows = []
    for method, quantizer in PTQ_BASELINES.items():
        metrics = evaluate_blocks(blocks, quantizer, bits)
        rows.append(
            {
                "problem": "tiny_llm_curvature_admm",
                "method": method,
                "seed": seed,
                "family": "proxy_baseline",
                "deploy_rel_error": metrics["deploy_rel_error"],
                "weight_rel_error": metrics["weight_rel_error"],
                "final_primal": "",
                "final_dual": "",
                "final_rho": "",
                "wall_time_sec": metrics["wall_time_sec"],
                "z_mode": "",
            }
        )
    return rows


def write_result(result: ExperimentResult) -> None:
    write_csv(RESULTS / f"{result.problem}_{result.method}_seed{result.seed}_history.csv", result.history)


def grouped(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    out = []
    for method, items in sorted(groups.items()):
        def mean(key: str) -> float | str:
            values = [float(row[key]) for row in items if row.get(key, "") != ""]
            if not values:
                return ""
            return sum(values) / len(values)

        out.append(
            {
                "method": method,
                "family": items[0]["family"],
                "z_mode": items[0].get("z_mode", ""),
                "deploy_rel_error": mean("deploy_rel_error"),
                "weight_rel_error": mean("weight_rel_error"),
                "final_primal": mean("final_primal"),
                "final_dual": mean("final_dual"),
                "final_rho": mean("final_rho"),
                "wall_time_sec": mean("wall_time_sec"),
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run curvature-aware ADMM quantization experiments.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in range(args.seed, args.seed + args.seeds):
        rows.extend(run_proxy_baselines(seed, args.bits))
        for method, z_mode, controller, rho0, curvature_weight in admm_configs():
            start = time.perf_counter()
            result = run_tiny_llm_curvature_admm(
                seed=seed,
                controller=controller,
                rho0=rho0,
                z_mode=z_mode,
                bits=args.bits,
                max_iter=args.max_iter,
                tol=args.tol,
                curvature_weight=curvature_weight,
            )
            elapsed = time.perf_counter() - start
            summary = summarize_history(result.history)
            row = {
                "problem": result.problem,
                "method": method,
                "seed": seed,
                "family": "curvature_admm",
                "deploy_rel_error": result.metrics["deploy_rel_error"],
                "weight_rel_error": result.metrics["weight_rel_error"],
                "final_primal": summary["final_primal"],
                "final_dual": summary["final_dual"],
                "final_rho": summary["final_rho"],
                "wall_time_sec": elapsed,
                "z_mode": z_mode,
            }
            rows.append(row)
            write_result(result)
            print(
                f"{method:36s} seed={seed} deploy={row['deploy_rel_error']:.4f} "
                f"pr={row['final_primal']:.2e} du={row['final_dual']:.2e} rho={row['final_rho']:.3g}"
            )
    write_csv(RESULTS / "llm_curvature_admm_summary.csv", rows)
    write_csv(RESULTS / "llm_curvature_admm_grouped.csv", grouped(rows))
    print(f"\nWrote curvature-aware ADMM outputs to {RESULTS}")


if __name__ == "__main__":
    main()
