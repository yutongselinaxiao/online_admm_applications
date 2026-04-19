from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controllers import PenaltyController
from .utils import relative_error, soft_threshold


@dataclass(frozen=True)
class ExperimentResult:
    problem: str
    method: str
    seed: int
    metrics: dict
    history: list[dict]


def _maybe_rescale_scaled_dual(u: np.ndarray, old_rho: float, new_rho: float) -> np.ndarray:
    if old_rho == new_rho:
        return u
    return (old_rho / new_rho) * u


def _thresholds(
    tol: float,
    residual_size: int,
    dual_size: int,
    primal_ref: float,
    dual_ref: float,
) -> tuple[float, float]:
    primal_threshold = (residual_size ** 0.5) * tol + tol * max(primal_ref, 1.0)
    dual_threshold = (dual_size ** 0.5) * tol + tol * max(dual_ref, 1.0)
    return float(primal_threshold), float(dual_threshold)


def make_graphical_lasso_data(seed: int, p: int = 20, n_samples: int = 120) -> dict:
    rng = np.random.default_rng(seed)
    mask = rng.random((p, p)) < 0.08
    mask = np.triu(mask, 1)
    weights = rng.uniform(0.15, 0.45, size=(p, p)) * rng.choice([-1.0, 1.0], size=(p, p))
    precision = mask * weights
    precision = precision + precision.T
    precision += np.diag(np.sum(np.abs(precision), axis=1) + 0.35)
    covariance = np.linalg.inv(precision)
    samples = rng.multivariate_normal(np.zeros(p), covariance, size=n_samples)
    sample_cov = samples.T @ samples / n_samples
    return {"S": sample_cov, "truth": precision}


def run_graphical_lasso(
    seed: int,
    controller: PenaltyController,
    rho0: float,
    lam: float = 0.08,
    max_iter: int = 150,
    tol: float = 1e-4,
    p: int = 20,
    n_samples: int = 120,
) -> ExperimentResult:
    data = make_graphical_lasso_data(seed, p=p, n_samples=n_samples)
    S, truth = data["S"], data["truth"]
    X = np.eye(p)
    Z = np.eye(p)
    U = np.zeros((p, p))
    offdiag = ~np.eye(p, dtype=bool)
    state = controller.init_state(rho0)
    history: list[dict] = []

    for k in range(max_iter):
        rho = state.rho
        Z_old = Z.copy()
        U_old = U.copy()

        M = rho * (Z - U) - S
        vals, vecs = np.linalg.eigh((M + M.T) / 2.0)
        x_vals = (vals + np.sqrt(vals * vals + 4.0 * rho)) / (2.0 * rho)
        X = (vecs * x_vals) @ vecs.T
        X = (X + X.T) / 2.0

        V = X + U
        Z = V.copy()
        Z[offdiag] = soft_threshold(V[offdiag], lam / rho)
        Z = (Z + Z.T) / 2.0
        U = U + X - Z
        lambda_hat = -(rho * U_old + rho * (X - Z_old))
        lambda_new = -(rho * U)

        primal = X - Z
        dual_base = Z - Z_old
        primal_norm = float(np.linalg.norm(primal, "fro"))
        dual_base_norm = float(np.linalg.norm(dual_base, "fro"))
        dual_norm = rho * dual_base_norm
        sign, logdet = np.linalg.slogdet(X)
        objective = float(
            -logdet + np.trace(S @ X) + lam * np.sum(np.abs(Z[offdiag]))
            if sign > 0
            else np.inf
        )
        status = "converged" if max(primal_norm, dual_norm) < tol else "running"

        primal_ref = max(float(np.linalg.norm(X, "fro")), float(np.linalg.norm(Z, "fro")), 1.0)
        dual_ref = max(float(np.linalg.norm(rho * U, "fro")), float(np.linalg.norm(S, "fro")), 1.0)
        primal_threshold, dual_threshold = _thresholds(
            tol, X.size, X.size, primal_ref, dual_ref
        )
        context = {
            "primal_ref": primal_ref,
            "dual_ref": dual_ref,
            "primal_threshold": primal_threshold,
            "dual_threshold": dual_threshold,
            "lambda_hat": lambda_hat,
            "lambda": lambda_new,
            "h_value": X,
            "g_value": -Z,
        }
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        if getattr(controller, "rescale_scaled_dual", True):
            U = _maybe_rescale_scaled_dual(U, rho, decision.rho)
        state = decision.state

        history.append(
            {
                "iter": k + 1,
                "objective": objective,
                "primal_norm": primal_norm,
                "dual_norm": dual_norm,
                "rho": rho,
                "rho_next": decision.rho,
                "rho_changed": decision.changed,
                "gradient": decision.gradient,
                "loss": decision.loss,
                "decision": decision.reason,
                "primal_threshold": primal_threshold,
                "dual_threshold": dual_threshold,
                "status": status,
            }
        )
        if status == "converged":
            break

    metrics = {
        "precision_relative_error": relative_error(Z, truth),
        "support_f1": _support_f1(Z, truth),
    }
    return ExperimentResult("graphical_lasso", controller.name, seed, metrics, history)


def _support_f1(estimate: np.ndarray, truth: np.ndarray, threshold: float = 1e-4) -> float:
    offdiag = ~np.eye(estimate.shape[0], dtype=bool)
    pred = np.abs(estimate[offdiag]) > threshold
    true = np.abs(truth[offdiag]) > threshold
    tp = np.count_nonzero(pred & true)
    fp = np.count_nonzero(pred & ~true)
    fn = np.count_nonzero(~pred & true)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def make_consensus_lasso_data(
    seed: int,
    workers: int = 6,
    n_per_worker: int = 36,
    d: int = 30,
    sparsity: int = 6,
    noise: float = 0.15,
) -> dict:
    rng = np.random.default_rng(seed)
    beta = np.zeros(d)
    support = rng.choice(d, size=sparsity, replace=False)
    beta[support] = rng.normal(0.0, 1.0, size=sparsity)
    parts = []
    for i in range(workers):
        scales = np.exp(rng.normal(0.0, 0.65, size=d))
        A = rng.normal(size=(n_per_worker, d)) * scales
        A /= np.sqrt(np.sum(A * A, axis=0, keepdims=True) + 1e-12)
        b = A @ beta + noise * rng.normal(size=n_per_worker)
        parts.append((A, b))
    return {"parts": parts, "truth": beta}


def run_consensus_lasso(
    seed: int,
    controller: PenaltyController,
    rho0: float,
    lam: float = 0.06,
    max_iter: int = 150,
    tol: float = 1e-4,
) -> ExperimentResult:
    data = make_consensus_lasso_data(seed)
    parts, truth = data["parts"], data["truth"]
    workers = len(parts)
    d = truth.size
    xs = np.zeros((workers, d))
    z = np.zeros(d)
    us = np.zeros((workers, d))
    state = controller.init_state(rho0)
    history: list[dict] = []

    for k in range(max_iter):
        rho = state.rho
        z_old = z.copy()
        us_old = us.copy()
        for i, (A, b) in enumerate(parts):
            lhs = A.T @ A + rho * np.eye(d)
            rhs = A.T @ b + rho * (z - us[i])
            xs[i] = np.linalg.solve(lhs, rhs)
        z = soft_threshold(np.mean(xs + us, axis=0), lam / (workers * rho))
        us = us + xs - z
        lambda_hat = -(rho * us_old + rho * (xs - z_old))
        lambda_new = -(rho * us)

        primal = xs - z
        primal_norm = float(np.linalg.norm(primal))
        dual_base_norm = float(np.sqrt(workers) * np.linalg.norm(z - z_old))
        dual_norm = rho * dual_base_norm
        objective = float(
            sum(0.5 * np.linalg.norm(A @ z - b) ** 2 for A, b in parts)
            + lam * np.sum(np.abs(z))
        )
        status = "converged" if max(primal_norm, dual_norm) < tol else "running"

        primal_ref = max(float(np.linalg.norm(xs)), float(np.sqrt(workers) * np.linalg.norm(z)), 1.0)
        dual_ref = max(float(np.linalg.norm(rho * us)), 1.0)
        primal_threshold, dual_threshold = _thresholds(
            tol, xs.size, d, primal_ref, dual_ref
        )
        context = {
            "primal_ref": primal_ref,
            "dual_ref": dual_ref,
            "primal_threshold": primal_threshold,
            "dual_threshold": dual_threshold,
            "lambda_hat": lambda_hat,
            "lambda": lambda_new,
            "h_value": xs,
            "g_value": -np.tile(z, (workers, 1)),
        }
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        if getattr(controller, "rescale_scaled_dual", True):
            us = _maybe_rescale_scaled_dual(us, rho, decision.rho)
        state = decision.state

        history.append(
            {
                "iter": k + 1,
                "objective": objective,
                "primal_norm": primal_norm,
                "dual_norm": dual_norm,
                "rho": rho,
                "rho_next": decision.rho,
                "rho_changed": decision.changed,
                "gradient": decision.gradient,
                "loss": decision.loss,
                "decision": decision.reason,
                "primal_threshold": primal_threshold,
                "dual_threshold": dual_threshold,
                "status": status,
            }
        )
        if status == "converged":
            break

    metrics = {
        "beta_relative_error": relative_error(z, truth),
        "support_f1": _support_f1_vector(z, truth),
    }
    return ExperimentResult("consensus_lasso", controller.name, seed, metrics, history)


def _support_f1_vector(estimate: np.ndarray, truth: np.ndarray, threshold: float = 1e-4) -> float:
    pred = np.abs(estimate) > threshold
    true = np.abs(truth) > threshold
    tp = np.count_nonzero(pred & true)
    fp = np.count_nonzero(pred & ~true)
    fn = np.count_nonzero(~pred & true)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def make_tv_data(seed: int, n: int = 160, noise: float = 0.18) -> dict:
    rng = np.random.default_rng(seed)
    levels = rng.normal(0.0, 0.8, size=7)
    cuts = np.linspace(0, n, len(levels) + 1, dtype=int)
    signal = np.zeros(n)
    for j, level in enumerate(levels):
        signal[cuts[j] : cuts[j + 1]] = level
    observed = signal + noise * rng.normal(size=n)
    return {"observed": observed, "truth": signal}


def run_tv_denoising(
    seed: int,
    controller: PenaltyController,
    rho0: float,
    lam: float = 0.35,
    max_iter: int = 180,
    tol: float = 1e-4,
    n: int = 160,
) -> ExperimentResult:
    data = make_tv_data(seed, n=n)
    b, truth = data["observed"], data["truth"]
    x = b.copy()
    z = np.diff(x)
    u = np.zeros(n - 1)
    state = controller.init_state(rho0)
    history: list[dict] = []

    for k in range(max_iter):
        rho = state.rho
        z_old = z.copy()
        u_old = u.copy()
        rhs = b + rho * _dt(z - u, n)
        x = _solve_tv_x(rhs, rho)
        dx = np.diff(x)
        z = soft_threshold(dx + u, lam / rho)
        u = u + dx - z
        lambda_hat = -(rho * u_old + rho * (dx - z_old))
        lambda_new = -(rho * u)

        primal = dx - z
        primal_norm = float(np.linalg.norm(primal))
        dual_base_norm = float(np.linalg.norm(_dt(z - z_old, n)))
        dual_norm = rho * dual_base_norm
        objective = float(0.5 * np.linalg.norm(x - b) ** 2 + lam * np.sum(np.abs(z)))
        status = "converged" if max(primal_norm, dual_norm) < tol else "running"

        primal_ref = max(float(np.linalg.norm(dx)), float(np.linalg.norm(z)), 1.0)
        dual_ref = max(float(np.linalg.norm(rho * u)), float(np.linalg.norm(x)), 1.0)
        primal_threshold, dual_threshold = _thresholds(
            tol, dx.size, n, primal_ref, dual_ref
        )
        context = {
            "primal_ref": primal_ref,
            "dual_ref": dual_ref,
            "primal_threshold": primal_threshold,
            "dual_threshold": dual_threshold,
            "lambda_hat": lambda_hat,
            "lambda": lambda_new,
            "h_value": dx,
            "g_value": -z,
        }
        decision = controller.update(k, state, primal_norm, dual_base_norm, context)
        if getattr(controller, "rescale_scaled_dual", True):
            u = _maybe_rescale_scaled_dual(u, rho, decision.rho)
        state = decision.state

        history.append(
            {
                "iter": k + 1,
                "objective": objective,
                "primal_norm": primal_norm,
                "dual_norm": dual_norm,
                "rho": rho,
                "rho_next": decision.rho,
                "rho_changed": decision.changed,
                "gradient": decision.gradient,
                "loss": decision.loss,
                "decision": decision.reason,
                "primal_threshold": primal_threshold,
                "dual_threshold": dual_threshold,
                "status": status,
            }
        )
        if status == "converged":
            break

    metrics = {"signal_relative_error": relative_error(x, truth)}
    return ExperimentResult("tv_denoising", controller.name, seed, metrics, history)


def _dt(v: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(n)
    out[0] = -v[0]
    out[1:-1] = v[:-1] - v[1:]
    out[-1] = v[-1]
    return out


def _solve_tv_x(rhs: np.ndarray, rho: float) -> np.ndarray:
    n = rhs.size
    diag = np.full(n, 1.0 + 2.0 * rho)
    diag[0] = diag[-1] = 1.0 + rho
    off = np.full(n - 1, -rho)
    matrix = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
    return np.linalg.solve(matrix, rhs)
