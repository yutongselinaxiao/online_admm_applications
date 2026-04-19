from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import PenaltyController
from .problems import ExperimentResult, _maybe_rescale_scaled_dual


EPS = 1e-12


@dataclass(frozen=True)
class QuantizedTensor:
    value: np.ndarray
    scale: np.ndarray
    codes: np.ndarray


def symmetric_uniform_quantize(
    x: np.ndarray,
    bits: int = 4,
    axis: int | None = 0,
) -> QuantizedTensor:
    """Symmetric uniform quantization with per-channel scales.

    For a weight matrix shaped (input_dim, output_dim), axis=0 gives one scale
    per output channel. This is the common PTQ convention for linear layers.
    """

    qmax = 2 ** (bits - 1) - 1
    qmin = -qmax
    if axis is None:
        scale = np.array(np.max(np.abs(x)) / max(qmax, 1))
        scale = np.maximum(scale, EPS)
        codes = np.clip(np.round(x / scale), qmin, qmax)
        return QuantizedTensor(codes * scale, scale, codes)

    reduce_axes = tuple(i for i in range(x.ndim) if i != axis)
    scale = np.max(np.abs(x), axis=reduce_axes, keepdims=True) / max(qmax, 1)
    scale = np.maximum(scale, EPS)
    codes = np.clip(np.round(x / scale), qmin, qmax)
    return QuantizedTensor(codes * scale, scale, codes)


def quantize_weights(weights: list[np.ndarray], bits: int) -> list[np.ndarray]:
    return [symmetric_uniform_quantize(w, bits=bits, axis=0).value for w in weights]


def combined_norm(arrays: list[np.ndarray]) -> float:
    return float(np.sqrt(sum(float(np.sum(a * a)) for a in arrays)))


def relative_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), EPS))


def _thresholds(tol: float, size: int, primal_ref: float, dual_ref: float) -> tuple[float, float]:
    primal_threshold = np.sqrt(size) * tol + tol * max(primal_ref, 1.0)
    dual_threshold = np.sqrt(size) * tol + tol * max(dual_ref, 1.0)
    return float(primal_threshold), float(dual_threshold)


def _controller_context(
    tol: float,
    weights: list[np.ndarray],
    z_weights: list[np.ndarray],
    duals: list[np.ndarray],
    rho: float,
) -> dict:
    size = sum(w.size for w in weights)
    primal_ref = max(combined_norm(weights), combined_norm(z_weights), 1.0)
    dual_ref = max(combined_norm([rho * u for u in duals]), 1.0)
    primal_threshold, dual_threshold = _thresholds(tol, size, primal_ref, dual_ref)
    return {
        "primal_ref": primal_ref,
        "dual_ref": dual_ref,
        "primal_threshold": primal_threshold,
        "dual_threshold": dual_threshold,
    }


def make_tanh_regression(seed: int, n: int = 192, d: int = 12, h: int = 24) -> dict:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d))
    teacher_w1 = rng.normal(scale=0.65, size=(d, h))
    teacher_w2 = rng.normal(scale=0.55, size=(h, 1))
    y = np.tanh(x @ teacher_w1) @ teacher_w2
    y += 0.03 * rng.normal(size=y.shape)
    return {"x": x, "y": y, "teacher_w1": teacher_w1, "teacher_w2": teacher_w2}


def run_tanh_qat(
    seed: int,
    controller: PenaltyController,
    rho0: float,
    bits: int = 4,
    max_iter: int = 120,
    tol: float = 1e-4,
    inner_steps: int = 8,
    lr: float = 0.025,
    weight_decay: float = 1e-4,
) -> ExperimentResult:
    """Nonconvex quantized training toy with inexact ADMM x-steps.

    The continuous block is a one-hidden-layer tanh network. The discrete block
    is a quantized deployable copy of the two weight matrices.
    """

    data = make_tanh_regression(seed)
    x, y = data["x"], data["y"]
    rng = np.random.default_rng(seed + 1000)
    d, h = data["teacher_w1"].shape
    w1 = 0.2 * rng.normal(size=(d, h))
    w2 = 0.2 * rng.normal(size=(h, 1))
    z1, z2 = quantize_weights([w1, w2], bits)
    u1 = np.zeros_like(w1)
    u2 = np.zeros_like(w2)
    state = controller.init_state(rho0)
    history: list[dict] = []

    for k in range(max_iter):
        rho = state.rho
        z_old = [z1.copy(), z2.copy()]
        for _ in range(inner_steps):
            hidden = np.tanh(x @ w1)
            pred = hidden @ w2
            err = pred - y
            grad_w2 = hidden.T @ err / x.shape[0]
            grad_hidden = (err @ w2.T) * (1.0 - hidden * hidden)
            grad_w1 = x.T @ grad_hidden / x.shape[0]
            grad_w1 += weight_decay * w1 + rho * (w1 - z1 + u1)
            grad_w2 += weight_decay * w2 + rho * (w2 - z2 + u2)
            step = lr / (1.0 + 0.01 * k)
            w1 -= step * grad_w1
            w2 -= step * grad_w2

        z1, z2 = quantize_weights([w1 + u1, w2 + u2], bits)
        u1 = u1 + w1 - z1
        u2 = u2 + w2 - z2

        primal_arrays = [w1 - z1, w2 - z2]
        dual_base_arrays = [z1 - z_old[0], z2 - z_old[1]]
        primal_norm = combined_norm(primal_arrays)
        dual_base_norm = combined_norm(dual_base_arrays)
        dual_norm = rho * dual_base_norm

        continuous_pred = np.tanh(x @ w1) @ w2
        deploy_pred = np.tanh(x @ z1) @ z2
        train_loss = float(0.5 * np.mean((continuous_pred - y) ** 2))
        deploy_loss = float(0.5 * np.mean((deploy_pred - y) ** 2))
        objective = train_loss + 0.5 * weight_decay * (np.sum(w1 * w1) + np.sum(w2 * w2))
        status = "converged" if max(primal_norm, dual_norm) < tol else "running"

        context = _controller_context(tol, [w1, w2], [z1, z2], [u1, u2], rho)
        context["task_metric"] = deploy_loss
        context["train_loss"] = train_loss
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        if getattr(controller, "rescale_scaled_dual", True):
            u1 = _maybe_rescale_scaled_dual(u1, rho, decision.rho)
            u2 = _maybe_rescale_scaled_dual(u2, rho, decision.rho)
        state = decision.state

        history.append(
            {
                "iter": k + 1,
                "objective": objective,
                "train_loss": train_loss,
                "deploy_loss": deploy_loss,
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

    metrics = {
        "deploy_loss": history[-1]["deploy_loss"],
        "continuous_loss": history[-1]["train_loss"],
        "teacher_weight_rel_error": 0.5
        * (
            relative_error(z1, data["teacher_w1"])
            + relative_error(z2, data["teacher_w2"])
        ),
    }
    return ExperimentResult("tanh_qat_nonconvex", controller.name, seed, metrics, history)


def _heavy_tailed_matrix(rng: np.random.Generator, shape: tuple[int, int], outlier_prob: float) -> np.ndarray:
    w = rng.normal(scale=0.08, size=shape)
    mask = rng.random(shape) < outlier_prob
    w[mask] += rng.normal(scale=0.75, size=np.count_nonzero(mask))
    return w


def make_tiny_llm_blocks(seed: int, d_model: int = 64, d_ff: int = 128, n_calib: int = 256) -> list[dict]:
    rng = np.random.default_rng(seed)
    specs = [
        ("attn_q_proj", d_model, d_model),
        ("attn_k_proj", d_model, d_model),
        ("attn_v_proj", d_model, d_model),
        ("attn_o_proj", d_model, d_model),
        ("mlp_gate_proj", d_model, d_ff),
        ("mlp_up_proj", d_model, d_ff),
        ("mlp_down_proj", d_ff, d_model),
    ]
    blocks = []
    for name, in_dim, out_dim in specs:
        x = rng.standard_t(df=4, size=(n_calib, in_dim))
        channel_scale = np.exp(rng.normal(scale=0.6, size=(1, in_dim)))
        x = x * channel_scale
        w = _heavy_tailed_matrix(rng, (in_dim, out_dim), outlier_prob=0.015)
        if "down" in name:
            w *= 0.7
        y = x @ w
        blocks.append({"name": name, "x": x, "w_fp": w, "target": y})
    return blocks


def run_tiny_llm_ptq(
    seed: int,
    controller: PenaltyController,
    rho0: float,
    bits: int = 4,
    max_iter: int = 100,
    tol: float = 1e-4,
    gamma: float = 1e-4,
) -> ExperimentResult:
    """Synthetic LLM-style blockwise PTQ with a shared online ADMM penalty."""

    blocks = make_tiny_llm_blocks(seed)
    for block in blocks:
        w = block["w_fp"].copy()
        block["w"] = w
        block["z"] = symmetric_uniform_quantize(w, bits=bits, axis=0).value
        block["u"] = np.zeros_like(w)
        xtx = block["x"].T @ block["x"] / block["x"].shape[0]
        xty = block["x"].T @ block["target"] / block["x"].shape[0]
        block["xtx"] = xtx
        block["xty"] = xty

    state = controller.init_state(rho0)
    history: list[dict] = []

    for k in range(max_iter):
        rho = state.rho
        primal_arrays = []
        dual_base_arrays = []
        objective = 0.0
        deploy_error_num = 0.0
        deploy_error_den = 0.0
        z_old_all = []

        for block in blocks:
            z_old = block["z"].copy()
            z_old_all.append(z_old)
            lhs = block["xtx"] + (gamma + rho) * np.eye(block["w_fp"].shape[0])
            rhs = block["xty"] + gamma * block["w_fp"] + rho * (block["z"] - block["u"])
            block["w"] = np.linalg.solve(lhs, rhs)
            block["z"] = symmetric_uniform_quantize(block["w"] + block["u"], bits=bits, axis=0).value
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

        weights = [block["w"] for block in blocks]
        z_weights = [block["z"] for block in blocks]
        duals = [block["u"] for block in blocks]
        context = _controller_context(tol, weights, z_weights, duals, rho)
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
    }
    return ExperimentResult("tiny_llm_ptq", controller.name, seed, metrics, history)
