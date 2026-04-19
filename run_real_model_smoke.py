from __future__ import annotations

import argparse
import copy
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

from online_admm_experiments.controllers import FeasibilityTaskOnlineOGD
from online_admm_experiments.nonconvex_quantization import EPS, combined_norm
from online_admm_experiments.problems import _maybe_rescale_scaled_dual
from run_llm_curvature_admm import quantize_gptq_target
from run_llm_quantization_baselines import quantize_with_scale


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "real_model_smoke"


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


def make_batches(tokenizer, block_size: int, max_batches: int, device: torch.device) -> list[torch.Tensor]:
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
        torch.randint(0, vocab_size, (1, block_size), generator=generator, device=device)
        for _ in range(max_batches)
    ]


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
    }


def is_supported_module(module: nn.Module) -> bool:
    weight = getattr(module, "weight", None)
    if weight is None or weight.ndim != 2:
        return False
    if isinstance(module, nn.Embedding):
        return False
    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"


def module_weight_to_matrix(module: nn.Module) -> np.ndarray:
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


def collect_module_inputs(
    model,
    batches: list[torch.Tensor],
    max_modules: int,
    max_samples: int,
) -> dict[str, np.ndarray]:
    modules = [(name, module) for name, module in model.named_modules() if is_supported_module(module)]
    modules = modules[:max_modules]
    captured: dict[str, list[torch.Tensor]] = {name: [] for name, _ in modules}
    handles = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            if len(captured[name]) >= 2:
                return
            x = inputs[0].detach().cpu().float()
            x = x.reshape(-1, x.shape[-1])
            captured[name].append(x)

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
    for name, chunks in captured.items():
        if not chunks:
            continue
        x = torch.cat(chunks, dim=0)
        if x.shape[0] > max_samples:
            x = x[:max_samples]
        out[name] = x.numpy()
    return out


def quantize_rtn_matrix(w: np.ndarray, bits: int) -> np.ndarray:
    qmax = 2 ** (bits - 1) - 1
    scale = np.maximum(np.max(np.abs(w), axis=0, keepdims=True) / max(qmax, 1), EPS)
    return quantize_with_scale(w, scale, bits)


def quantize_admm_gptq_matrix(
    w_fp: np.ndarray,
    x: np.ndarray,
    bits: int,
    max_iter: int,
    rho0: float,
) -> tuple[np.ndarray, dict]:
    hessian = x.T @ x / max(x.shape[0], 1)
    z = quantize_rtn_matrix(w_fp, bits)
    w = w_fp.copy()
    u = np.zeros_like(w_fp)
    controller = FeasibilityTaskOnlineOGD(
        eta0=0.18,
        rho_min=1e-3,
        rho_max=1e3,
        grad_clip=2.5,
        ema=0.2,
        residual_weight=0.55,
        task_weight=0.0,
        task_grad_clip=1.0,
        feasibility_weight=0.9,
        name="real_smoke_admm_gptq",
    )
    state = controller.init_state(rho0)
    target_den = max(float(np.sum((x @ w_fp) ** 2)), EPS)
    history = []
    for k in range(max_iter):
        rho = state.rho
        z_old = z.copy()
        lhs = hessian + rho * np.eye(w_fp.shape[0])
        rhs = hessian @ w_fp + rho * (z - u)
        w = np.linalg.solve(lhs, rhs)
        target = np.linalg.solve(hessian + rho * np.eye(w_fp.shape[0]), hessian @ w_fp + rho * (w + u))
        q_metric = hessian + rho * np.eye(w_fp.shape[0])
        z = quantize_gptq_target(target, q_metric, bits)
        u = u + w - z
        primal_norm = float(np.linalg.norm(w - z))
        dual_base_norm = float(np.linalg.norm(z - z_old))
        dual_norm = rho * dual_base_norm
        deploy_rel_error = float(np.sqrt(np.sum((x @ z - x @ w_fp) ** 2) / target_den))
        context = {
            "primal_threshold": 1e-4 * (math.sqrt(w.size) + max(np.linalg.norm(w), np.linalg.norm(z), 1.0)),
            "dual_threshold": 1e-4 * (math.sqrt(w.size) + max(dual_norm, 1.0)),
            "task_metric": deploy_rel_error,
        }
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        u = _maybe_rescale_scaled_dual(u, rho, decision.rho)
        state = decision.state
        history.append(
            {
                "primal_norm": primal_norm,
                "dual_norm": dual_norm,
                "rho": rho,
                "deploy_rel_error": deploy_rel_error,
            }
        )
    return z, {
        "final_primal": history[-1]["primal_norm"],
        "final_dual": history[-1]["dual_norm"],
        "final_rho": history[-1]["rho"],
        "module_recon_error": history[-1]["deploy_rel_error"],
    }


def apply_quantization(
    model,
    calibration_inputs: dict[str, np.ndarray],
    method: str,
    bits: int,
    max_admm_iter: int,
) -> dict:
    stats = {
        "num_quantized_modules": 0,
        "mean_module_recon_error": 0.0,
        "mean_final_primal": 0.0,
        "mean_final_dual": 0.0,
        "mean_final_rho": 0.0,
    }
    recon_errors = []
    primals = []
    duals = []
    rhos = []
    for name, module in model.named_modules():
        if name not in calibration_inputs or not is_supported_module(module):
            continue
        w = module_weight_to_matrix(module)
        x = calibration_inputs[name]
        if x.shape[1] != w.shape[0]:
            continue
        if method == "rtn":
            z = quantize_rtn_matrix(w, bits)
            target_den = max(float(np.sum((x @ w) ** 2)), EPS)
            recon_errors.append(float(np.sqrt(np.sum((x @ z - x @ w) ** 2) / target_den)))
        elif method == "admm_gptq":
            z, module_stats = quantize_admm_gptq_matrix(w, x, bits, max_admm_iter, rho0=1.0)
            recon_errors.append(module_stats["module_recon_error"])
            primals.append(module_stats["final_primal"])
            duals.append(module_stats["final_dual"])
            rhos.append(module_stats["final_rho"])
        else:
            raise ValueError(f"Unsupported method: {method}")
        set_module_weight_from_matrix(module, z)
        stats["num_quantized_modules"] += 1
    stats["mean_module_recon_error"] = sum(recon_errors) / max(len(recon_errors), 1)
    stats["mean_final_primal"] = sum(primals) / max(len(primals), 1)
    stats["mean_final_dual"] = sum(duals) / max(len(duals), 1)
    stats["mean_final_rho"] = sum(rhos) / max(len(rhos), 1)
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPU real-model smoke test for ADMM LLM quantization.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--random-tiny", action="store_true", help="Use an offline randomly initialized tiny GPT-2 model.")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--max-modules", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-admm-iter", type=int, default=8)
    parser.add_argument("--methods", nargs="+", default=["fp32", "rtn", "admm_gptq"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    if args.random_tiny:
        torch.manual_seed(args.seed)
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
        batches = make_random_batches(config.vocab_size, args.block_size, args.max_batches, device, args.seed)
        model_label = "random_tiny_gpt2"
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        base_model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
        batches = make_batches(tokenizer, args.block_size, args.max_batches, device)
        model_label = args.model
    calibration_inputs = collect_module_inputs(base_model, batches, args.max_modules, args.max_samples)
    base_metrics = evaluate_loss(base_model, batches)
    for method in args.methods:
        start = time.perf_counter()
        if method == "fp32":
            model = base_model
            quant_stats = {
                "num_quantized_modules": 0,
                "mean_module_recon_error": 0.0,
                "mean_final_primal": 0.0,
                "mean_final_dual": 0.0,
                "mean_final_rho": 0.0,
            }
        else:
            model = copy.deepcopy(base_model).to(device)
            quant_stats = apply_quantization(model, calibration_inputs, method, args.bits, args.max_admm_iter)
        metrics = evaluate_loss(model, batches)
        row = {
            "model": model_label,
            "device": str(device),
            "method": method,
            "bits": args.bits if method != "fp32" else "",
            "loss": metrics["loss"],
            "ppl": metrics["ppl"],
            "delta_loss": metrics["loss"] - base_metrics["loss"],
            "num_eval_tokens": sum(int(b.numel()) for b in batches),
            "num_quantized_modules": quant_stats["num_quantized_modules"],
            "mean_module_recon_error": quant_stats["mean_module_recon_error"],
            "mean_final_primal": quant_stats["mean_final_primal"],
            "mean_final_dual": quant_stats["mean_final_dual"],
            "mean_final_rho": quant_stats["mean_final_rho"],
            "wall_time_sec": time.perf_counter() - start,
        }
        rows.append(row)
        print(
            f"{method:10s} loss={row['loss']:.4f} ppl={row['ppl']:.2f} "
            f"delta={row['delta_loss']:.4f} modules={row['num_quantized_modules']} "
            f"time={row['wall_time_sec']:.2f}s"
        )
    write_csv(RESULTS / "real_model_smoke_summary.csv", rows)
    write_text(
        RESULTS / "real_model_smoke_findings.md",
        "# Real Model Smoke Test\n\n"
        f"Model: `{model_label}`\n\n"
        f"Device: `{device}`\n\n"
        "This is a CPU/debug-scale smoke test on a tiny text corpus, not a final perplexity benchmark.\n\n"
        + "\n".join(
            f"- `{row['method']}`: loss `{row['loss']:.4f}`, ppl `{row['ppl']:.2f}`, delta loss `{row['delta_loss']:.4f}`, modules `{row['num_quantized_modules']}`"
            for row in rows
        )
        + "\n",
    )
    print(f"\nWrote real-model smoke outputs to {RESULTS}")


if __name__ == "__main__":
    main()
