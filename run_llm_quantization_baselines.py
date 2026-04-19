from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np

from online_admm_experiments.controllers import FeasibilityTaskOnlineOGD, FixedRho
from online_admm_experiments.nonconvex_quantization import (
    EPS,
    make_tiny_llm_blocks,
    run_tiny_llm_ptq,
    symmetric_uniform_quantize,
)
from online_admm_experiments.utils import summarize_history, write_csv


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llm_quant_baselines"


def quantize_with_scale(x: np.ndarray, scale: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    safe_scale = np.maximum(scale, EPS)
    codes = np.clip(np.round(x / safe_scale), qmin, qmax)
    return codes * safe_scale


def per_output_scale(w: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    return np.maximum(np.max(np.abs(w), axis=0, keepdims=True) / max(qmax, 1), EPS)


def evaluate_blocks(blocks: list[dict], quantizer: Callable[[dict, int], np.ndarray], bits: int) -> dict:
    deploy_error_num = 0.0
    deploy_error_den = 0.0
    weight_error_num = 0.0
    weight_error_den = 0.0
    rows = []
    start = time.perf_counter()
    for block in blocks:
        w = block["w_fp"]
        z = quantizer(block, bits)
        deploy_err = block["x"] @ z - block["target"]
        deploy_error_num += float(np.sum(deploy_err * deploy_err))
        deploy_error_den += float(np.sum(block["target"] * block["target"]))
        weight_error_num += float(np.sum((z - w) ** 2))
        weight_error_den += float(np.sum(w * w))
        rows.append(
            {
                "block": block["name"],
                "deploy_rel_error": float(
                    np.linalg.norm(deploy_err) / max(np.linalg.norm(block["target"]), EPS)
                ),
                "weight_rel_error": float(np.linalg.norm(z - w) / max(np.linalg.norm(w), EPS)),
            }
        )
    elapsed = time.perf_counter() - start
    return {
        "deploy_rel_error": float(np.sqrt(deploy_error_num / max(deploy_error_den, EPS))),
        "weight_rel_error": float(np.sqrt(weight_error_num / max(weight_error_den, EPS))),
        "wall_time_sec": elapsed,
        "block_rows": rows,
    }


def quantize_fp16(block: dict, bits: int) -> np.ndarray:
    return block["w_fp"].copy()


def quantize_rtn(block: dict, bits: int) -> np.ndarray:
    return symmetric_uniform_quantize(block["w_fp"], bits=bits, axis=0).value


def quantize_hessian_diag(block: dict, bits: int) -> np.ndarray:
    """Activation/Hessian-weighted clipping grid for a simple PTQ baseline."""

    w = block["w_fp"]
    x = block["x"]
    diag = np.mean(x * x, axis=0, keepdims=True).T
    multipliers = [0.55, 0.7, 0.85, 1.0, 1.2, 1.45]
    best_z = None
    best_loss = math.inf
    base = per_output_scale(w, bits)
    for mult in multipliers:
        scale = np.maximum(base * mult, EPS)
        z = quantize_with_scale(w, scale, bits)
        err = z - w
        loss = float(np.sum(diag * err * err))
        if loss < best_loss:
            best_loss = loss
            best_z = z
    assert best_z is not None
    return best_z


def _scaled_quant_grid(
    block: dict,
    bits: int,
    alphas: list[float],
    activation_stat: np.ndarray,
    weight_stat: np.ndarray | None = None,
) -> np.ndarray:
    w = block["w_fp"]
    best_z = None
    best_err = math.inf
    act = np.maximum(activation_stat.reshape(-1, 1), EPS)
    if weight_stat is None:
        weight_stat = np.ones_like(act)
    else:
        weight_stat = np.maximum(weight_stat.reshape(-1, 1), EPS)
    for alpha in alphas:
        scale_in = (act**alpha) / (weight_stat ** max(1.0 - alpha, 0.0))
        scale_in = scale_in / np.exp(np.mean(np.log(np.maximum(scale_in, EPS))))
        w_scaled = scale_in * w
        q_scaled = symmetric_uniform_quantize(w_scaled, bits=bits, axis=0).value
        z = q_scaled / scale_in
        deploy_err = block["x"] @ z - block["target"]
        err = float(np.sum(deploy_err * deploy_err))
        if err < best_err:
            best_err = err
            best_z = z
    assert best_z is not None
    return best_z


def quantize_awq_like(block: dict, bits: int) -> np.ndarray:
    act = np.mean(np.abs(block["x"]), axis=0)
    return _scaled_quant_grid(block, bits, [0.0, 0.25, 0.5, 0.75, 1.0], act)


def quantize_smoothquant_like(block: dict, bits: int) -> np.ndarray:
    act = np.max(np.abs(block["x"]), axis=0)
    weight = np.max(np.abs(block["w_fp"]), axis=1)
    return _scaled_quant_grid(block, bits, [0.25, 0.5, 0.75, 0.9], act, weight)


def quantize_gptq_like(block: dict, bits: int) -> np.ndarray:
    """Small GPTQ-style sequential Hessian compensation baseline.

    This is a compact implementation for the synthetic calibration problem, not
    the official GPTQ kernel. It uses the calibration Gram matrix, fixed
    per-output-channel scales, descending Hessian diagonal order, and the usual
    inverse-Hessian error propagation idea.
    """

    w = block["w_fp"]
    x = block["x"]
    hessian = x.T @ x / x.shape[0]
    damp = 0.01 * float(np.mean(np.diag(hessian)))
    hessian = hessian + (damp + 1e-6) * np.eye(hessian.shape[0])
    order = np.argsort(-np.diag(hessian))
    inv_order = np.argsort(order)
    h_perm = hessian[np.ix_(order, order)]
    w_work = w[order, :].copy()
    h_inv = np.linalg.inv(h_perm)
    scale = per_output_scale(w_work, bits)
    q_perm = np.zeros_like(w_work)
    for i in range(w_work.shape[0]):
        q_i = quantize_with_scale(w_work[i : i + 1, :], scale, bits)[0]
        q_perm[i] = q_i
        err = (w_work[i] - q_i) / max(h_inv[i, i], EPS)
        if i + 1 < w_work.shape[0]:
            w_work[i + 1 :, :] -= h_inv[i + 1 :, i : i + 1] * err.reshape(1, -1)
    return q_perm[inv_order, :]


PTQ_BASELINES: dict[str, Callable[[dict, int], np.ndarray]] = {
    "fp16_reference": quantize_fp16,
    "rtn_per_channel": quantize_rtn,
    "hessian_diag_clip": quantize_hessian_diag,
    "awq_like_scale_grid": quantize_awq_like,
    "smoothquant_like_scale_grid": quantize_smoothquant_like,
    "gptq_like_sequential": quantize_gptq_like,
}


def admm_controllers() -> list[tuple[str, object, float]]:
    return [
        ("admm_fixed_rho_10", FixedRho(name="admm_fixed_rho_10"), 10.0),
        (
            "admm_task_feasibility",
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
                name="admm_task_feasibility",
            ),
            1.0,
        ),
        (
            "admm_task_feas_aggressive",
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
                name="admm_task_feas_aggressive",
            ),
            1.0,
        ),
    ]


def run_ptq_baseline(seed: int, method: str, quantizer: Callable[[dict, int], np.ndarray], bits: int) -> tuple[dict, list[dict]]:
    blocks = make_tiny_llm_blocks(seed)
    metrics = evaluate_blocks(blocks, quantizer, bits)
    row = {
        "problem": "tiny_llm_ptq",
        "method": method,
        "seed": seed,
        "family": "external_proxy",
        "bits": bits,
        "deploy_rel_error": metrics["deploy_rel_error"],
        "weight_rel_error": metrics["weight_rel_error"],
        "wall_time_sec": metrics["wall_time_sec"],
        "final_primal": "",
        "final_dual": "",
        "final_rho": "",
        "rho_changes": "",
    }
    block_rows = [{"method": method, "seed": seed, **block_row} for block_row in metrics["block_rows"]]
    return row, block_rows


def run_admm(seed: int, method: str, controller, rho0: float, bits: int, max_iter: int, tol: float) -> tuple[dict, list[dict]]:
    start = time.perf_counter()
    result = run_tiny_llm_ptq(
        seed=seed,
        controller=controller,
        rho0=rho0,
        bits=bits,
        max_iter=max_iter,
        tol=tol,
    )
    elapsed = time.perf_counter() - start
    summary = summarize_history(result.history)
    row = {
        "problem": result.problem,
        "method": method,
        "seed": seed,
        "family": "admm",
        "bits": bits,
        "deploy_rel_error": result.metrics["deploy_rel_error"],
        "weight_rel_error": result.metrics["weight_rel_error"],
        "wall_time_sec": elapsed,
        "final_primal": summary["final_primal"],
        "final_dual": summary["final_dual"],
        "final_rho": summary["final_rho"],
        "rho_changes": summary["rho_changes"],
    }
    history_path = RESULTS / f"tiny_llm_ptq_{method}_seed{seed}_history.csv"
    write_csv(history_path, result.history)
    return row, []


def grouped(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["method"], []).append(row)
    out = []
    for method, items in sorted(groups.items()):
        def mean(key: str) -> float:
            values = [float(row[key]) for row in items if row[key] != ""]
            return sum(values) / max(len(values), 1)

        out.append(
            {
                "method": method,
                "family": items[0]["family"],
                "deploy_rel_error": mean("deploy_rel_error"),
                "weight_rel_error": mean("weight_rel_error"),
                "wall_time_sec": mean("wall_time_sec"),
                "final_primal": mean("final_primal") if items[0]["family"] == "admm" else "",
                "final_dual": mean("final_dual") if items[0]["family"] == "admm" else "",
                "final_rho": mean("final_rho") if items[0]["family"] == "admm" else "",
            }
        )
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ADMM PTQ against local LLM quantization proxy baselines.")
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
    block_rows: list[dict] = []
    for seed in range(args.seed, args.seed + args.seeds):
        for method, quantizer in PTQ_BASELINES.items():
            row, blocks = run_ptq_baseline(seed, method, quantizer, args.bits)
            rows.append(row)
            block_rows.extend(blocks)
            print(
                f"{method:32s} seed={seed} deploy={row['deploy_rel_error']:.4f} "
                f"weight={row['weight_rel_error']:.4f}"
            )
        for method, controller, rho0 in admm_controllers():
            row, _ = run_admm(seed, method, controller, rho0, args.bits, args.max_iter, args.tol)
            rows.append(row)
            print(
                f"{method:32s} seed={seed} deploy={row['deploy_rel_error']:.4f} "
                f"pr={float(row['final_primal']):.2e} du={float(row['final_dual']):.2e}"
            )
    write_csv(RESULTS / "llm_quant_baseline_summary.csv", rows)
    write_csv(RESULTS / "llm_quant_baseline_grouped.csv", grouped(rows))
    write_csv(RESULTS / "llm_quant_baseline_blocks.csv", block_rows)
    print(f"\nWrote LLM quantization baseline outputs to {RESULTS}")


if __name__ == "__main__":
    main()
