"""Paper-grade real-model PTQ runner.

Quantizes all supported Linear/Conv1D weights in a HuggingFace causal LM using
four methods (rtn, gptq_proxy, awq_proxy, admm_gptq) at user-specified bit
widths, then reports sliding-window perplexity on WikiText-2 test. Hot numerical
paths (ADMM x/z updates, GPTQ sequential update, AWQ grid search) run in torch
on GPU; the online rho controller stays pure Python.

The ADMM x/z/lam updates and residual definitions mirror the canonical form
used in ../online_admm/online_admm.py (unit-ball-constrained least squares)
and online_admm/online_admm_lasso.py (l1). In scaled-dual form u = lam/rho:

    w <- argmin_w 0.5||X(w-w_fp)||^2 + gamma/2||w-w_fp||^2 + rho/2||w-z+u||^2
    z <- curvature-aware quantized projection of (w+u)
    u <- u + (w - z),  with u rescaled (rho_old/rho_new) whenever rho changes
    r_primal = w - z,   s_dual = rho*(z - z_old)

The penalty controller uses the same log-rho OGD / residual-balance / task-aware
families defined in online_admm_experiments/controllers.py, which in turn
follow the `step_loss_from_residuals` / `update_rho_residual_balancing` patterns
from ../online_admm/online_admm.py.

Backward-compatible smoke mode (`--corpus toy --random-tiny`) is preserved for
offline regression testing.
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from online_admm_experiments.controllers import (
    FeasibilityTaskOnlineOGD,
    FixedRho,
    OnlineOGD,
    ResidualBalancing,
    SpectralAADMM,
    TaskNormalizedMagnitudeOGD,
)
from online_admm_experiments.nonconvex_quantization import EPS, combined_norm
from online_admm_experiments.problems import _maybe_rescale_scaled_dual


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "real_model_smoke"
DEFAULT_CACHE_DIR = Path("/data/yutong/hf_cache")

SMOKE_TEXTS = [
    "Online ADMM adapts the penalty parameter by observing primal and dual residuals.",
    "Quantization compresses neural network weights while trying to preserve model predictions.",
    "Large language models contain many linear projections with different activation scales.",
    "A curvature aware update can use calibration activations to protect sensitive directions.",
    "The goal of this smoke test is to check whether the real model pipeline runs on CPU.",
    "A good benchmark should compare reconstruction error, perplexity, runtime, and memory.",
    "Penalty tuning can improve feasibility, but deploy accuracy is the metric that matters.",
    "This small text corpus is only for debugging and should not be used as a final result.",
]


# ---------- small io helpers ----------
def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "__" for ch in label)


# ---------- corpus / batches ----------
def make_toy_batches(tokenizer, block_size: int, max_batches: int, device: torch.device) -> list[torch.Tensor]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer("\n\n".join(SMOKE_TEXTS), return_tensors="pt")
    ids = encoded["input_ids"][0]
    if ids.numel() < block_size + 1:
        repeats = math.ceil((block_size + 1) / max(ids.numel(), 1))
        ids = ids.repeat(repeats)
    batches = []
    stride = max(block_size // 2, 1)
    for start in range(0, max(ids.numel() - block_size, 1), stride):
        batches.append(ids[start : start + block_size].unsqueeze(0).to(device))
        if len(batches) >= max_batches:
            break
    return batches


def make_random_batches(
    vocab_size: int,
    block_size: int,
    max_batches: int,
    device: torch.device,
    seed: int = 0,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return [
        torch.randint(0, vocab_size, (1, block_size), generator=generator).to(device)
        for _ in range(max_batches)
    ]


def load_wikitext2_ids(tokenizer, cache_dir: Path) -> dict[str, torch.Tensor]:
    """Return tokenized concatenated ids for train and test splits of WikiText-2."""
    from datasets import load_dataset

    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    out = {}
    for split in ("train", "test"):
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split, cache_dir=str(cache_dir))
        text = "\n\n".join(row for row in ds["text"] if row.strip())
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0]
        out[split] = ids
    return out


def make_calibration_batches(
    ids: torch.Tensor,
    block_size: int,
    num_seqs: int,
    seed: int,
    device: torch.device,
) -> list[torch.Tensor]:
    rng = np.random.default_rng(seed)
    n = ids.numel()
    if n <= block_size + 1:
        return [ids[:block_size].unsqueeze(0).to(device)]
    starts = rng.integers(0, n - block_size, size=num_seqs)
    return [ids[int(s) : int(s) + block_size].unsqueeze(0).to(device) for s in starts]


def make_eval_batches(
    ids: torch.Tensor,
    block_size: int,
    max_batches: int | None,
    device: torch.device,
) -> list[torch.Tensor]:
    """Non-overlapping blocks covering the test corpus. HF-standard sliding PPL."""
    n = ids.numel()
    batches = []
    for start in range(0, max(n - block_size, 1), block_size):
        batches.append(ids[start : start + block_size].unsqueeze(0).to(device))
        if max_batches is not None and len(batches) >= max_batches:
            break
    return batches


@torch.no_grad()
def evaluate_loss(model, batches: list[torch.Tensor]) -> dict:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    start = time.perf_counter()
    for input_ids in batches:
        out = model(input_ids=input_ids, labels=input_ids)
        tokens = int(input_ids.numel())
        total_loss += float(out.loss.detach().cpu()) * tokens
        total_tokens += tokens
    loss = total_loss / max(total_tokens, 1)
    return {
        "loss": loss,
        "ppl": math.exp(min(loss, 20.0)),
        "eval_time_sec": time.perf_counter() - start,
        "eval_tokens": total_tokens,
    }


# ---------- module discovery / weight IO ----------
def is_supported_module(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return False
    if isinstance(module, nn.Embedding):
        return False
    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"


def module_weight_to_matrix(module: nn.Module) -> np.ndarray:
    """Return weight as (in_features, out_features) float32 numpy."""
    weight = module.weight.detach().cpu().float().numpy()
    if isinstance(module, nn.Linear):
        return weight.T.copy()
    return weight.copy()


def set_module_weight_from_matrix(module: nn.Module, matrix: np.ndarray) -> None:
    if isinstance(module, nn.Linear):
        new_weight = torch.tensor(matrix.T, dtype=module.weight.dtype, device=module.weight.device)
    else:
        new_weight = torch.tensor(matrix, dtype=module.weight.dtype, device=module.weight.device)
    with torch.no_grad():
        module.weight.copy_(new_weight)


# ---------- calibration via running Gram matrix ----------
def collect_module_grams(
    model,
    batches: list[torch.Tensor],
    max_modules: int | None,
    device: torch.device,
) -> dict[str, dict]:
    """Accumulate xtx (d_in, d_in) and abs_x_sum (d_in,) per supported module.

    This is the GPTQ-style calibration pattern: activations are never
    materialized in full, only the second-moment matrix. Memory footprint per
    module is d_in^2 floats (tens of MB max for the models we target).
    """
    modules = [(name, module) for name, module in model.named_modules() if is_supported_module(module)]
    if max_modules is not None:
        modules = modules[:max_modules]

    state: dict[str, dict] = {
        name: {
            "xtx": torch.zeros(module.weight.shape[1 if isinstance(module, nn.Linear) else 0], module.weight.shape[1 if isinstance(module, nn.Linear) else 0], dtype=torch.float32, device=device),
            "abs_x_sum": torch.zeros(module.weight.shape[1 if isinstance(module, nn.Linear) else 0], dtype=torch.float32, device=device),
            "n_samples": 0,
        }
        for name, module in modules
    }
    handles = []

    def make_hook(name: str):
        entry = state[name]

        def hook(module, inputs, output):
            x = inputs[0].detach()
            x = x.reshape(-1, x.shape[-1]).float()
            entry["xtx"] += x.T @ x
            entry["abs_x_sum"] += x.abs().sum(dim=0)
            entry["n_samples"] += int(x.shape[0])

        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))
    try:
        with torch.no_grad():
            for input_ids in batches:
                model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()

    out = {}
    for name, entry in state.items():
        n = max(entry["n_samples"], 1)
        out[name] = {
            "xtx": entry["xtx"] / n,
            "abs_x_mean": entry["abs_x_sum"] / n,
            "n_samples": entry["n_samples"],
        }
    return out


# ---------- torch quantization kernels ----------
def safe_norm_t(x: torch.Tensor, eps_rel: float = 1e-12) -> float:
    """Port of safe_norm from ../online_admm/online_admm.py: sqrt(||x||^2 + eps_rel*(1+||x||^2))."""
    sqnorm = float((x * x).sum())
    eps = eps_rel * (1.0 + sqnorm)
    return math.sqrt(sqnorm + eps)


def _qminmax(bits: int) -> tuple[float, float]:
    qmax = 2 ** (bits - 1) - 1
    return -float(qmax), float(qmax)


def quantize_with_scale_t(x: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    qmin, qmax = _qminmax(bits)
    safe = scale.clamp_min(EPS)
    codes = torch.clamp(torch.round(x / safe), qmin, qmax)
    return codes * safe


def per_output_scale_t(w: torch.Tensor, bits: int) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    return w.abs().amax(dim=0, keepdim=True).clamp_min(EPS) / max(qmax, 1)


def quantize_rtn_t(w: torch.Tensor, bits: int) -> torch.Tensor:
    return quantize_with_scale_t(w, per_output_scale_t(w, bits), bits)


def quantize_gptq_target_t(target: torch.Tensor, q: torch.Tensor, bits: int) -> torch.Tensor:
    """Torch port of `quantize_gptq_target` from run_llm_curvature_admm.py.

    target: (d, d_out)
    q: (d, d) quadratic form for the reconstruction metric.
    """
    d = q.shape[0]
    damp = 0.001 * q.diagonal().mean()
    q = q + (damp + 1e-8) * torch.eye(d, device=q.device, dtype=q.dtype)
    order = torch.argsort(-q.diagonal())
    inv_order = torch.argsort(order)
    q_perm = q[order][:, order]
    w_work = target[order].clone()
    q_inv = torch.linalg.inv(q_perm)
    scale = per_output_scale_t(w_work, bits)
    z_perm = torch.zeros_like(w_work)
    for i in range(d):
        z_i = quantize_with_scale_t(w_work[i : i + 1, :], scale, bits).squeeze(0)
        z_perm[i] = z_i
        err = (w_work[i] - z_i) / q_inv[i, i].clamp_min(EPS)
        if i + 1 < d:
            w_work[i + 1 :, :] -= q_inv[i + 1 :, i : i + 1] * err.unsqueeze(0)
    return z_perm[inv_order]


def quantize_gptq_proxy_t(w: torch.Tensor, xtx: torch.Tensor, bits: int) -> torch.Tensor:
    """Standalone GPTQ-like baseline: sequential OBS update with damping."""
    d = xtx.shape[0]
    hessian = xtx.clone()
    damp = 0.01 * hessian.diagonal().mean()
    hessian = hessian + (damp + 1e-6) * torch.eye(d, device=hessian.device, dtype=hessian.dtype)
    order = torch.argsort(-hessian.diagonal())
    inv_order = torch.argsort(order)
    h_perm = hessian[order][:, order]
    w_work = w[order].clone()
    h_inv = torch.linalg.inv(h_perm)
    scale = per_output_scale_t(w_work, bits)
    q_perm = torch.zeros_like(w_work)
    for i in range(d):
        q_i = quantize_with_scale_t(w_work[i : i + 1, :], scale, bits).squeeze(0)
        q_perm[i] = q_i
        err = (w_work[i] - q_i) / h_inv[i, i].clamp_min(EPS)
        if i + 1 < d:
            w_work[i + 1 :, :] -= h_inv[i + 1 :, i : i + 1] * err.unsqueeze(0)
    return q_perm[inv_order]


def quantize_awq_proxy_t(
    w: torch.Tensor,
    xtx: torch.Tensor,
    abs_x_mean: torch.Tensor,
    bits: int,
    alphas: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
) -> torch.Tensor:
    """AWQ-like: search a per-input-channel scale power and pick by xtx-weighted error."""
    act = abs_x_mean.clamp_min(EPS).reshape(-1, 1)
    best_z = None
    best_loss = math.inf
    for alpha in alphas:
        scale_in = act ** alpha
        # geometric-mean normalize so the scale is multiplicatively unit
        log_mean = torch.log(scale_in.clamp_min(EPS)).mean()
        scale_in = scale_in / torch.exp(log_mean)
        w_scaled = scale_in * w
        q_scaled = quantize_with_scale_t(w_scaled, per_output_scale_t(w_scaled, bits), bits)
        z = q_scaled / scale_in
        err = z - w
        loss = float((err.T @ xtx @ err).diagonal().sum())
        if loss < best_loss:
            best_loss = loss
            best_z = z
    assert best_z is not None
    return best_z


ADMM_CONTROLLERS: dict[str, callable] = {
    # Task-aware + feasibility OGD. The original smoke's choice. Good default for PTQ.
    "task_feasibility": lambda: FeasibilityTaskOnlineOGD(
        eta0=0.18, rho_min=1e-3, rho_max=1e3, grad_clip=2.5, ema=0.2,
        residual_weight=0.55, task_weight=0.0, task_grad_clip=1.0,
        feasibility_weight=0.9, name="admm_task_feasibility",
    ),
    # Plain analytic balance-loss OGD on log(rho). Analogue of the JAX
    # online_type="analytic", loss_type="bal" path in ../online_admm/online_admm.py.
    "bal_ogd": lambda: OnlineOGD(
        eta0=0.25, rho_min=1e-3, rho_max=1e3, grad_clip=2.0, ema=0.2,
        name="admm_bal_ogd",
    ),
    # Boyd residual balancing heuristic — JAX online_type="heuristic".
    "heuristic": lambda: ResidualBalancing(
        mu=10.0, tau=2.0, rho_min=1e-3, rho_max=1e3, name="admm_heuristic",
    ),
    # Xu-Figueiredo-Goldstein spectral AADMM (AISTATS 2017).
    "spectral": lambda: SpectralAADMM(rho_min=1e-3, rho_max=1e3, name="admm_spectral"),
    # Task-normalized magnitude loss: Pareto-facing, useful for residual vs
    # deploy-error tradeoff sweeps.
    "task_norm_magnitude": lambda: TaskNormalizedMagnitudeOGD(
        eta0=0.25, rho_min=1e-3, rho_max=1e3, grad_clip=2.0, ema=0.2,
        name="admm_task_norm_magnitude",
    ),
    "fixed": lambda: FixedRho(name="admm_fixed"),
}


def quantize_admm_gptq_t(
    w_fp: torch.Tensor,
    xtx: torch.Tensor,
    bits: int,
    max_iter: int,
    rho0: float,
    controller_name: str = "task_feasibility",
    gamma: float = 1e-4,
) -> tuple[torch.Tensor, dict]:
    """Online ADMM with curvature-aware GPTQ-like Z update on GPU.

    The x / z / u / lam updates and residual definitions match the canonical
    form in ../online_admm/online_admm.py::admm_one_step, specialized to the
    PTQ objective

        min_w  0.5 ||X(w - w_fp)||^2 + gamma/2 ||w - w_fp||^2 + I_Q(z)   s.t. w = z

    with the z-update replaced by the curvature-aware GPTQ variant from ledger
    section 7. The penalty controller is selected from ADMM_CONTROLLERS.
    """
    device = w_fp.device
    dtype = w_fp.dtype
    d = w_fp.shape[0]
    eye = torch.eye(d, device=device, dtype=dtype)

    target_den = float((w_fp.T @ xtx @ w_fp).diagonal().sum()) + EPS
    z = quantize_rtn_t(w_fp, bits)
    w = w_fp.clone()
    u = torch.zeros_like(w_fp)

    if controller_name not in ADMM_CONTROLLERS:
        raise ValueError(f"Unknown admm controller: {controller_name}. Options: {list(ADMM_CONTROLLERS)}")
    controller = ADMM_CONTROLLERS[controller_name]()
    state = controller.init_state(rho0)
    history = []

    for k in range(max_iter):
        rho = state.rho
        z_old = z.clone()

        # w-update: (xtx + (gamma+rho) I) w = xtx w_fp + gamma w_fp + rho (z - u)
        A = xtx + (gamma + rho) * eye
        B = xtx @ w_fp + gamma * w_fp + rho * (z - u)
        w = torch.linalg.solve(A, B)

        # Curvature-aware GPTQ z-target (unregularized-in-rho metric)
        target = torch.linalg.solve(xtx + rho * eye, xtx @ w_fp + rho * (w + u))
        q_metric = xtx + rho * eye
        z = quantize_gptq_target_t(target, q_metric, bits)
        u = u + w - z

        primal_norm = safe_norm_t(w - z)
        dual_base_norm = safe_norm_t(z - z_old)
        dual_norm = rho * dual_base_norm

        diff = z - w_fp
        err_num = float((diff.T @ xtx @ diff).diagonal().sum())
        deploy_rel_error = math.sqrt(max(err_num, 0.0) / target_den)

        context = {
            "primal_threshold": 1e-4 * (math.sqrt(w.numel()) + max(float(torch.linalg.norm(w)), float(torch.linalg.norm(z)), 1.0)),
            "dual_threshold": 1e-4 * (math.sqrt(w.numel()) + max(dual_norm, 1.0)),
            "task_metric": deploy_rel_error,
            "primal_ref": max(float(torch.linalg.norm(w)), float(torch.linalg.norm(z)), 1.0),
            "dual_ref": max(rho * float(torch.linalg.norm(u)), 1.0),
        }
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        u = _maybe_rescale_scaled_dual_t(u, rho, decision.rho)
        state = decision.state

        history.append({
            "iter": k + 1,
            "primal_norm": primal_norm,
            "dual_norm": dual_norm,
            "rho": rho,
            "rho_next": decision.rho,
            "deploy_rel_error": deploy_rel_error,
        })

    final = history[-1]
    return z, {
        "final_primal": final["primal_norm"],
        "final_dual": final["dual_norm"],
        "final_rho": final["rho"],
        "module_recon_error": final["deploy_rel_error"],
        "rho_trajectory": [row["rho"] for row in history],
        "history": history,
    }


def _maybe_rescale_scaled_dual_t(u: torch.Tensor, old_rho: float, new_rho: float) -> torch.Tensor:
    if old_rho == new_rho:
        return u
    return (old_rho / new_rho) * u


# ---------- per-method module application ----------
def apply_quantization(
    model,
    grams: dict[str, dict],
    method: str,
    bits: int,
    max_admm_iter: int,
    device: torch.device,
    admm_controller: str = "task_feasibility",
) -> dict:
    stats = {
        "num_quantized_modules": 0,
        "recon_errors": [],
        "primals": [],
        "duals": [],
        "rhos": [],
        "per_module": [],
    }
    for name, module in model.named_modules():
        if name not in grams or not is_supported_module(module):
            continue
        w_np = module_weight_to_matrix(module)
        w = torch.from_numpy(w_np).to(device=device, dtype=torch.float32)
        xtx = grams[name]["xtx"]
        abs_x_mean = grams[name]["abs_x_mean"]
        if xtx.shape[0] != w.shape[0]:
            continue

        target_den = float((w.T @ xtx @ w).diagonal().sum()) + EPS

        module_stats = {"name": name, "d_in": w.shape[0], "d_out": w.shape[1]}
        if method == "rtn":
            z = quantize_rtn_t(w, bits)
        elif method == "gptq_proxy":
            z = quantize_gptq_proxy_t(w, xtx, bits)
        elif method == "awq_proxy":
            z = quantize_awq_proxy_t(w, xtx, abs_x_mean, bits)
        elif method == "admm_gptq":
            z, admm_stats = quantize_admm_gptq_t(
                w, xtx, bits, max_admm_iter, rho0=1.0, controller_name=admm_controller,
            )
            stats["primals"].append(admm_stats["final_primal"])
            stats["duals"].append(admm_stats["final_dual"])
            stats["rhos"].append(admm_stats["final_rho"])
            module_stats.update(
                final_primal=admm_stats["final_primal"],
                final_dual=admm_stats["final_dual"],
                final_rho=admm_stats["final_rho"],
            )
        else:
            raise ValueError(f"Unsupported method: {method}")

        diff = z - w
        recon = math.sqrt(max(float((diff.T @ xtx @ diff).diagonal().sum()), 0.0) / target_den)
        stats["recon_errors"].append(recon)
        module_stats["recon_error"] = recon
        stats["per_module"].append(module_stats)

        set_module_weight_from_matrix(module, z.cpu().numpy())
        stats["num_quantized_modules"] += 1

        # free intermediate tensors aggressively
        del w, z, diff
    return stats


# ---------- CLI / orchestration ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paper-grade real-model PTQ with online ADMM.")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--random-tiny", action="store_true", help="Use an offline randomly initialized tiny GPT-2 model.")
    parser.add_argument("--corpus", choices=["toy", "wikitext2"], default="wikitext2")
    parser.add_argument("--bits-list", nargs="+", type=int, default=[4])
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--calib-seqs", type=int, default=128)
    parser.add_argument("--eval-max-batches", type=int, default=None)
    parser.add_argument("--max-modules", type=int, default=None)
    parser.add_argument("--max-admm-iter", type=int, default=30)
    parser.add_argument(
        "--admm-controller",
        choices=list(ADMM_CONTROLLERS),
        default="task_feasibility",
        help="Penalty controller for admm_gptq. See ADMM_CONTROLLERS registry.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["fp32", "rtn", "gptq_proxy", "awq_proxy", "admm_gptq"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Hugging Face cache directory for downloaded model/tokenizer files.",
    )
    parser.add_argument("--device", default=None, help="Override torch device (e.g. 'cuda:0').")
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Subdirectory under results/real_model_smoke/ for this run. Defaults to a timestamp.",
    )
    # legacy flags kept for backward compatibility with the old smoke invocation
    parser.add_argument("--max-batches", type=int, default=None, help="Alias for --calib-seqs.")
    parser.add_argument("--max-samples", type=int, default=None, help="Deprecated, ignored (gram matrix is sample-free).")
    parser.add_argument("--bits", type=int, default=None, help="Legacy single-bits flag; prefer --bits-list.")
    parser.add_argument("--seed", type=int, default=None, help="Legacy single-seed flag; prefer --seeds.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # legacy compatibility
    if args.bits is not None and args.bits_list == [4]:
        args.bits_list = [args.bits]
    if args.seed is not None and args.seeds == [0, 1, 2]:
        args.seeds = [args.seed]
    if args.max_batches is not None:
        args.calib_seqs = args.max_batches

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = args.cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_dir))

    subdir = args.output_subdir or time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = RESULTS / subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build model and corpus once (model is reloaded for each method via deepcopy).
    if args.random_tiny:
        torch.manual_seed(0)
        config = GPT2Config(
            vocab_size=128,
            n_positions=args.block_size,
            n_ctx=args.block_size,
            n_embd=32,
            n_layer=2,
            n_head=4,
            n_inner=64,
        )
        base_model = AutoModelForCausalLM.from_config(config).to(device)
        tokenizer = None
        model_label = "random_tiny_gpt2"
        train_ids = None
        test_ids = None
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache_dir)
        base_model = AutoModelForCausalLM.from_pretrained(args.model, cache_dir=cache_dir).to(device)
        model_label = args.model
        if args.corpus == "wikitext2":
            ids = load_wikitext2_ids(tokenizer, cache_dir)
            train_ids, test_ids = ids["train"], ids["test"]
        else:
            train_ids, test_ids = None, None

    base_model.eval()

    # eval batches (shared across all methods/seeds; same held-out text)
    if args.random_tiny:
        eval_batches = make_random_batches(128, args.block_size, args.eval_max_batches or 4, device, seed=99)
    elif args.corpus == "wikitext2":
        eval_batches = make_eval_batches(test_ids, args.block_size, args.eval_max_batches, device)
    else:
        eval_batches = make_toy_batches(tokenizer, args.block_size, args.eval_max_batches or 4, device)

    fp32_metrics = evaluate_loss(base_model, eval_batches)
    print(f"[{model_label}] fp32 ppl={fp32_metrics['ppl']:.3f} loss={fp32_metrics['loss']:.4f} on {fp32_metrics['eval_tokens']} tokens")

    rows: list[dict] = []
    per_module_rows: list[dict] = []

    for seed in args.seeds:
        # calibration batches for this seed
        if args.random_tiny:
            calib_batches = make_random_batches(128, args.block_size, args.calib_seqs, device, seed=seed)
        elif args.corpus == "wikitext2":
            calib_batches = make_calibration_batches(train_ids, args.block_size, args.calib_seqs, seed, device)
        else:
            calib_batches = make_toy_batches(tokenizer, args.block_size, args.calib_seqs, device)

        # gram matrices are independent of method/bits, so build once per seed
        t0 = time.perf_counter()
        grams = collect_module_grams(base_model, calib_batches, args.max_modules, device)
        calib_time = time.perf_counter() - t0
        print(f"[seed={seed}] calibrated {len(grams)} modules on {sum(g['n_samples'] for g in grams.values())} tokens in {calib_time:.1f}s")

        for bits in args.bits_list:
            for method in args.methods:
                t_start = time.perf_counter()
                if method == "fp32":
                    model = base_model
                    quant_stats = {"num_quantized_modules": 0, "recon_errors": [], "primals": [], "duals": [], "rhos": [], "per_module": []}
                else:
                    model = copy.deepcopy(base_model).to(device)
                    quant_stats = apply_quantization(
                        model, grams, method, bits, args.max_admm_iter, device,
                        admm_controller=args.admm_controller,
                    )
                quant_time = time.perf_counter() - t_start

                metrics = evaluate_loss(model, eval_batches)

                row = {
                    "model": model_label,
                    "corpus": args.corpus,
                    "device": str(device),
                    "seed": seed,
                    "method": method,
                    "bits": bits if method != "fp32" else "",
                    "loss": metrics["loss"],
                    "ppl": metrics["ppl"],
                    "delta_loss": metrics["loss"] - fp32_metrics["loss"],
                    "delta_ppl": metrics["ppl"] - fp32_metrics["ppl"],
                    "eval_tokens": metrics["eval_tokens"],
                    "num_quantized_modules": quant_stats["num_quantized_modules"],
                    "mean_module_recon_error": float(np.mean(quant_stats["recon_errors"])) if quant_stats["recon_errors"] else 0.0,
                    "median_module_recon_error": float(np.median(quant_stats["recon_errors"])) if quant_stats["recon_errors"] else 0.0,
                    "max_module_recon_error": float(np.max(quant_stats["recon_errors"])) if quant_stats["recon_errors"] else 0.0,
                    "mean_final_primal": float(np.mean(quant_stats["primals"])) if quant_stats["primals"] else 0.0,
                    "mean_final_dual": float(np.mean(quant_stats["duals"])) if quant_stats["duals"] else 0.0,
                    "mean_final_rho": float(np.mean(quant_stats["rhos"])) if quant_stats["rhos"] else 0.0,
                    "quantize_wall_time_sec": quant_time,
                    "eval_wall_time_sec": metrics["eval_time_sec"],
                }
                rows.append(row)
                for m in quant_stats["per_module"]:
                    per_module_rows.append({"seed": seed, "method": method, "bits": bits, **m})

                print(
                    f"[seed={seed} bits={bits}] {method:12s} ppl={row['ppl']:.3f} Δ={row['delta_ppl']:+.3f} "
                    f"recon={row['mean_module_recon_error']:.4f} "
                    f"rho_final={row['mean_final_rho']:.3f} "
                    f"t_quant={row['quantize_wall_time_sec']:.1f}s"
                )

                # free GPU memory between methods
                if method != "fp32":
                    del model
                    torch.cuda.empty_cache() if device.type == "cuda" else None

    # write outputs
    model_slug = safe_label(model_label)
    write_csv(out_dir / "summary.csv", rows)
    write_csv(out_dir / f"summary_{model_slug}.csv", rows)
    write_csv(out_dir / "per_module.csv", per_module_rows)

    # aggregate over seeds for quick readability
    agg: dict[tuple, dict] = {}
    for row in rows:
        key = (row["method"], row["bits"])
        bucket = agg.setdefault(key, {"ppl": [], "delta_ppl": [], "recon": [], "rho": []})
        bucket["ppl"].append(row["ppl"])
        bucket["delta_ppl"].append(row["delta_ppl"])
        bucket["recon"].append(row["mean_module_recon_error"])
        bucket["rho"].append(row["mean_final_rho"])

    agg_rows = []
    for (method, bits), bucket in sorted(agg.items()):
        agg_rows.append({
            "model": model_label,
            "corpus": args.corpus,
            "method": method,
            "bits": bits,
            "seeds": len(bucket["ppl"]),
            "ppl_mean": float(np.mean(bucket["ppl"])),
            "ppl_std": float(np.std(bucket["ppl"])) if len(bucket["ppl"]) > 1 else 0.0,
            "delta_ppl_mean": float(np.mean(bucket["delta_ppl"])),
            "recon_mean": float(np.mean(bucket["recon"])),
            "rho_mean": float(np.mean(bucket["rho"])),
        })
    write_csv(out_dir / "aggregate.csv", agg_rows)

    summary_text = (
        f"# Real Model PTQ\n\n"
        f"Model: `{model_label}`  \n"
        f"Corpus: `{args.corpus}`  \n"
        f"Device: `{device}`  \n"
        f"Seeds: `{args.seeds}`  \n"
        f"Bits: `{args.bits_list}`  \n"
        f"Methods: `{args.methods}`  \n"
        f"ADMM iters: `{args.max_admm_iter}`  \n"
        f"fp32 ppl: `{fp32_metrics['ppl']:.3f}` on `{fp32_metrics['eval_tokens']}` eval tokens\n\n"
        "| method | bits | seeds | ppl_mean ± std | Δppl | mean recon |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {r['method']} | {r['bits']} | {r['seeds']} | {r['ppl_mean']:.3f} ± {r['ppl_std']:.3f} | {r['delta_ppl_mean']:+.3f} | {r['recon_mean']:.4f} |"
            for r in agg_rows
        )
        + "\n"
    )
    write_text(out_dir / "findings.md", summary_text)
    write_text(out_dir / f"findings_{model_slug}.md", summary_text)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
