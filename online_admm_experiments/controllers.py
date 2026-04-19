from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log, sqrt

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class PenaltyState:
    rho: float
    log_rho: float
    ema_log_primal: float | None = None
    ema_log_dual_base: float | None = None
    previous_log_rho: float | None = None
    previous_gradient: float | None = None
    spectral_lambda_hat: np.ndarray | None = None
    spectral_lambda: np.ndarray | None = None
    spectral_h: np.ndarray | None = None
    spectral_g: np.ndarray | None = None
    previous_task_log: float | None = None
    previous_task_log_rho: float | None = None


@dataclass(frozen=True)
class ControllerDecision:
    rho: float
    changed: bool
    gradient: float
    loss: float
    reason: str
    state: PenaltyState


class PenaltyController:
    """Interface for ADMM penalty controllers."""

    name = "controller"
    rescale_scaled_dual = True

    def init_state(self, rho0: float) -> PenaltyState:
        return PenaltyState(rho=rho0, log_rho=log(rho0))

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        return ControllerDecision(
            rho=state.rho,
            changed=False,
            gradient=0.0,
            loss=0.0,
            reason="fixed",
            state=state,
        )


@dataclass
class FixedRho(PenaltyController):
    name: str = "fixed"


@dataclass
class ResidualBalancing(PenaltyController):
    """Classic heuristic baseline with scaled-dual rescaling handled by solvers."""

    mu: float = 10.0
    tau: float = 2.0
    rho_min: float = 1e-4
    rho_max: float = 1e4
    update_period: int = 1
    name: str = "residual_balance"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "period", state)

        dual_norm = state.rho * dual_base_norm
        rho = state.rho
        reason = "balanced"
        if primal_norm > self.mu * max(dual_norm, EPS):
            rho = min(self.rho_max, state.rho * self.tau)
            reason = "primal_gt_dual"
        elif dual_norm > self.mu * max(primal_norm, EPS):
            rho = max(self.rho_min, state.rho / self.tau)
            reason = "dual_gt_primal"

        new_state = replace(state, rho=rho, log_rho=log(rho))
        return ControllerDecision(rho, rho != state.rho, 0.0, 0.0, reason, new_state)


@dataclass
class NormalizedResidualBalancing(PenaltyController):
    """Residual balancing after dividing by ADMM-style stopping thresholds."""

    mu: float = 10.0
    tau: float = 2.0
    rho_min: float = 1e-4
    rho_max: float = 1e4
    update_period: int = 1
    name: str = "residual_balance_normalized"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "period", state)

        context = context or {}
        dual_norm = state.rho * dual_base_norm
        primal_threshold = max(float(context.get("primal_threshold", 1.0)), EPS)
        dual_threshold = max(float(context.get("dual_threshold", 1.0)), EPS)
        primal_scaled = primal_norm / primal_threshold
        dual_scaled = dual_norm / dual_threshold
        rho = state.rho
        reason = "balanced"
        if primal_scaled > self.mu * max(dual_scaled, EPS):
            rho = min(self.rho_max, state.rho * self.tau)
            reason = "primal_gt_dual_normalized"
        elif dual_scaled > self.mu * max(primal_scaled, EPS):
            rho = max(self.rho_min, state.rho / self.tau)
            reason = "dual_gt_primal_normalized"

        new_state = replace(state, rho=rho, log_rho=log(rho))
        loss = 0.5 * (log(max(dual_scaled, EPS)) - log(max(primal_scaled, EPS))) ** 2
        return ControllerDecision(rho, rho != state.rho, 0.0, loss, reason, new_state)


@dataclass
class RelativeResidualBalancing(PenaltyController):
    """Wohlberg-style scale-aware balancing using relative primal and dual residuals."""

    mu: float = 10.0
    tau: float = 2.0
    rho_min: float = 1e-4
    rho_max: float = 1e4
    update_period: int = 1
    name: str = "residual_balance_relative"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "period", state)

        context = context or {}
        dual_norm = state.rho * dual_base_norm
        primal_ref = max(float(context.get("primal_ref", 1.0)), EPS)
        dual_ref = max(float(context.get("dual_ref", 1.0)), EPS)
        primal_relative = primal_norm / primal_ref
        dual_relative = dual_norm / dual_ref
        rho = state.rho
        reason = "balanced"
        if primal_relative > self.mu * max(dual_relative, EPS):
            rho = min(self.rho_max, state.rho * self.tau)
            reason = "primal_gt_dual_relative"
        elif dual_relative > self.mu * max(primal_relative, EPS):
            rho = max(self.rho_min, state.rho / self.tau)
            reason = "dual_gt_primal_relative"

        new_state = replace(state, rho=rho, log_rho=log(rho))
        loss = 0.5 * (log(max(dual_relative, EPS)) - log(max(primal_relative, EPS))) ** 2
        return ControllerDecision(rho, rho != state.rho, 0.0, loss, reason, new_state)


@dataclass
class OnlineOGD(PenaltyController):
    """Projected OGD on u = log(rho) using residual-balance loss.

    Loss: 0.5 * (log(rho * dual_base_norm) - log(primal_norm)) ** 2.
    The gradient in u is the log residual imbalance.
    """

    eta0: float = 0.35
    rho_min: float = 1e-4
    rho_max: float = 1e4
    grad_clip: float = 2.0
    ema: float = 0.2
    update_period: int = 1
    burn_in: int | None = None
    freeze_after: int | None = None
    min_relative_change: float = 1e-4
    name: str = "online_ogd"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        log_primal = log(max(primal_norm, EPS))
        log_dual_base = log(max(dual_base_norm, EPS))

        if state.ema_log_primal is None:
            ema_log_primal = log_primal
            ema_log_dual_base = log_dual_base
        else:
            assert state.ema_log_dual_base is not None
            ema_log_primal = (
                self.ema * log_primal + (1.0 - self.ema) * state.ema_log_primal
            )
            ema_log_dual_base = (
                self.ema * log_dual_base
                + (1.0 - self.ema) * state.ema_log_dual_base
            )

        gradient = state.log_rho + ema_log_dual_base - ema_log_primal
        gradient = float(np.clip(gradient, -self.grad_clip, self.grad_clip))
        loss = 0.5 * gradient * gradient

        carried = replace(
            state,
            ema_log_primal=ema_log_primal,
            ema_log_dual_base=ema_log_dual_base,
        )

        if self.burn_in is not None and k < self.burn_in:
            return ControllerDecision(state.rho, False, gradient, loss, "burn_in", carried)
        if self.freeze_after is not None and k >= self.freeze_after:
            return ControllerDecision(state.rho, False, gradient, loss, "frozen", carried)
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, gradient, loss, "period", carried)

        eta = self.eta0 / sqrt(k + 1.0)
        lo, hi = log(self.rho_min), log(self.rho_max)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, lo, hi))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        if not changed:
            next_log_rho = state.log_rho
            next_rho = state.rho

        new_state = replace(carried, rho=next_rho, log_rho=next_log_rho)
        return ControllerDecision(next_rho, changed, gradient, loss, "ogd", new_state)


@dataclass
class BBOnlinePenalty(OnlineOGD):
    """Simplified BB step-size variant of online log-rho gradient descent.

    This is not the Xu-Figueiredo-Goldstein AADMM method; it is a small
    ablation that uses a BB secant ratio for the one-dimensional online loss.
    """

    bb_min: float = 0.02
    bb_max: float = 1.5
    fallback_eta: float = 0.25
    name: str = "bb_online_penalty_simplified"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        log_primal = log(max(primal_norm, EPS))
        log_dual_base = log(max(dual_base_norm, EPS))
        gradient = state.log_rho + log_dual_base - log_primal
        gradient = float(np.clip(gradient, -self.grad_clip, self.grad_clip))
        loss = 0.5 * gradient * gradient

        if self.freeze_after is not None and k >= self.freeze_after:
            new_state = replace(
                state,
                previous_log_rho=state.log_rho,
                previous_gradient=gradient,
            )
            return ControllerDecision(state.rho, False, gradient, loss, "frozen", new_state)
        if (k + 1) % self.update_period:
            new_state = replace(
                state,
                previous_log_rho=state.log_rho,
                previous_gradient=gradient,
            )
            return ControllerDecision(state.rho, False, gradient, loss, "period", new_state)

        if state.previous_log_rho is None or state.previous_gradient is None:
            eta = self.fallback_eta / sqrt(k + 1.0)
            reason = "bb_fallback"
        else:
            du = state.log_rho - state.previous_log_rho
            dg = gradient - state.previous_gradient
            eta = abs(du / dg) if abs(dg) > EPS else self.fallback_eta / sqrt(k + 1.0)
            eta = float(np.clip(eta, self.bb_min, self.bb_max))
            reason = "bb"

        lo, hi = log(self.rho_min), log(self.rho_max)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, lo, hi))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        if not changed:
            next_log_rho = state.log_rho
            next_rho = state.rho

        new_state = replace(
            state,
            rho=next_rho,
            log_rho=next_log_rho,
            previous_log_rho=state.log_rho,
            previous_gradient=gradient,
        )
        return ControllerDecision(next_rho, changed, gradient, loss, reason, new_state)


@dataclass
class SpectralAADMM(PenaltyController):
    """Xu-Figueiredo-Goldstein style spectral AADMM penalty baseline.

    The controller expects context entries:
      lambda_hat: intermediate unscaled dual after the x-update,
      lambda: final unscaled dual after the z-update,
      h_value: A x^{k+1},
      g_value: B z^{k+1}.

    It estimates the two dual spectral stepsizes and applies the correlation
    safeguard from Xu, Figueiredo, and Goldstein (AISTATS 2017).
    """

    rho_min: float = 1e-4
    rho_max: float = 1e4
    update_period: int = 2
    correlation_threshold: float = 0.2
    name: str = "spectral_aadmm_xu2017"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        if context is None:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "missing_context", state)

        try:
            lambda_hat = np.asarray(context["lambda_hat"], dtype=float).ravel()
            dual = np.asarray(context["lambda"], dtype=float).ravel()
            h_value = np.asarray(context["h_value"], dtype=float).ravel()
            g_value = np.asarray(context["g_value"], dtype=float).ravel()
        except KeyError:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "missing_context", state)

        carried = replace(
            state,
            spectral_lambda_hat=lambda_hat.copy(),
            spectral_lambda=dual.copy(),
            spectral_h=h_value.copy(),
            spectral_g=g_value.copy(),
        )

        if state.spectral_lambda_hat is None or (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, 0.0, 0.0, "spectral_wait", carried)

        d_lambda_hat = lambda_hat - state.spectral_lambda_hat
        d_lambda = dual - state.spectral_lambda
        d_h = h_value - state.spectral_h
        d_g = g_value - state.spectral_g

        alpha_step, alpha_cor = self._spectral_step(d_h, d_lambda_hat)
        beta_step, beta_cor = self._spectral_step(d_g, d_lambda)

        rho = state.rho
        reason = "spectral_reject"
        if alpha_cor > self.correlation_threshold and beta_cor > self.correlation_threshold:
            rho = float((alpha_step * beta_step) ** 0.5)
            reason = "spectral_geometric"
        elif alpha_cor > self.correlation_threshold:
            rho = alpha_step
            reason = "spectral_alpha"
        elif beta_cor > self.correlation_threshold:
            rho = beta_step
            reason = "spectral_beta"

        rho = float(np.clip(rho, self.rho_min, self.rho_max))
        new_state = replace(carried, rho=rho, log_rho=log(rho))
        changed = abs(rho / state.rho - 1.0) >= 1e-12
        loss = max(alpha_cor, 0.0) * max(beta_cor, 0.0)
        return ControllerDecision(rho, changed, 0.0, loss, reason, new_state)

    @staticmethod
    def _spectral_step(delta_grad: np.ndarray, delta_dual: np.ndarray) -> tuple[float, float]:
        grad_norm = float(np.linalg.norm(delta_grad))
        dual_norm = float(np.linalg.norm(delta_dual))
        if grad_norm <= EPS or dual_norm <= EPS:
            return 1.0, -1.0
        dot = float(np.dot(delta_grad, delta_dual))
        cor = dot / max(grad_norm * dual_norm, EPS)
        if dot <= EPS:
            return 1.0, cor

        step_sd = float(np.dot(delta_dual, delta_dual) / dot)
        step_mg = float(dot / max(np.dot(delta_grad, delta_grad), EPS))
        if step_sd <= EPS or step_mg <= EPS:
            return 1.0, cor
        if 2.0 * step_mg > step_sd:
            step = step_mg
        else:
            step = step_sd - 0.5 * step_mg
        return max(step, EPS), cor


@dataclass
class OnlineOGDNoDualRescale(OnlineOGD):
    """Ablation that intentionally disables scaled-dual rescaling after rho changes."""

    name: str = "online_ogd_no_dual_rescale"
    rescale_scaled_dual: bool = False


@dataclass
class TaskAwareOnlineOGD(OnlineOGD):
    """Online log-rho controller with residual and task-loss terms.

    The residual-balance term is analytic in u = log(rho). The task term uses
    a one-dimensional secant estimate from observed task metrics, making it a
    black-box online loss useful for nonconvex PTQ/QAT experiments.
    """

    residual_weight: float = 1.0
    task_weight: float = 0.5
    task_grad_clip: float = 2.0
    name: str = "online_ogd_task_aware"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        context = context or {}
        task_metric = context.get("task_metric")
        log_primal = log(max(primal_norm, EPS))
        log_dual_base = log(max(dual_base_norm, EPS))
        if state.ema_log_primal is None:
            ema_log_primal = log_primal
            ema_log_dual_base = log_dual_base
        else:
            assert state.ema_log_dual_base is not None
            ema_log_primal = (
                self.ema * log_primal + (1.0 - self.ema) * state.ema_log_primal
            )
            ema_log_dual_base = (
                self.ema * log_dual_base
                + (1.0 - self.ema) * state.ema_log_dual_base
            )
        residual_gradient = state.log_rho + ema_log_dual_base - ema_log_primal

        task_gradient = 0.0
        task_log = None
        if task_metric is not None:
            task_log = log(max(float(task_metric), EPS))
            if (
                state.previous_task_log is not None
                and state.previous_task_log_rho is not None
                and abs(state.log_rho - state.previous_task_log_rho) > EPS
            ):
                task_gradient = (
                    (task_log - state.previous_task_log)
                    / (state.log_rho - state.previous_task_log_rho)
                )
                task_gradient = float(
                    np.clip(task_gradient, -self.task_grad_clip, self.task_grad_clip)
                )

        gradient = self.residual_weight * residual_gradient + self.task_weight * task_gradient
        gradient = float(np.clip(gradient, -self.grad_clip, self.grad_clip))
        loss = 0.5 * residual_gradient * residual_gradient
        if task_log is not None:
            loss += self.task_weight * task_log

        carried = replace(
            state,
            ema_log_primal=ema_log_primal,
            ema_log_dual_base=ema_log_dual_base,
            previous_task_log=task_log if task_log is not None else state.previous_task_log,
            previous_task_log_rho=state.log_rho if task_log is not None else state.previous_task_log_rho,
        )

        if self.freeze_after is not None and k >= self.freeze_after:
            return ControllerDecision(state.rho, False, gradient, loss, "frozen", carried)
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, gradient, loss, "period", carried)

        eta = self.eta0 / sqrt(k + 1.0)
        lo, hi = log(self.rho_min), log(self.rho_max)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, lo, hi))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        if not changed:
            next_log_rho = state.log_rho
            next_rho = state.rho

        new_state = replace(carried, rho=next_rho, log_rho=next_log_rho)
        return ControllerDecision(next_rho, changed, gradient, loss, "task_aware_ogd", new_state)


@dataclass
class FeasibilityTaskOnlineOGD(TaskAwareOnlineOGD):
    """Task-aware controller that also penalizes absolute feasibility."""

    feasibility_weight: float = 0.25
    name: str = "online_ogd_task_feasibility"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        context = context or {}
        decision = super().update(k, state, primal_norm, dual_base_norm, context)
        if decision.reason in {"frozen", "period"}:
            return decision

        # Recompute with a feasibility pressure that increases rho when
        # primal mismatch is large relative to its ADMM threshold.
        primal_threshold = max(float(context.get("primal_threshold", 1.0)), EPS)
        feasibility_signal = log(max(primal_norm / primal_threshold, EPS))
        feasibility_signal = float(np.clip(feasibility_signal, -self.grad_clip, self.grad_clip))
        gradient = decision.gradient - self.feasibility_weight * feasibility_signal
        gradient = float(np.clip(gradient, -self.grad_clip, self.grad_clip))
        eta = self.eta0 / sqrt(k + 1.0)
        lo, hi = log(self.rho_min), log(self.rho_max)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, lo, hi))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        new_state = replace(decision.state, rho=next_rho, log_rho=next_log_rho)
        return ControllerDecision(
            next_rho,
            changed,
            gradient,
            decision.loss,
            "task_feasibility_ogd",
            new_state,
        )


@dataclass
class TaskAcceptRejectOnlineOGD(TaskAwareOnlineOGD):
    """Accept an OGD rho update only if task metric is not getting worse too fast."""

    max_task_log_increase: float = 0.01
    shrink_on_reject: float = 0.85
    name: str = "online_ogd_task_accept"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        context = context or {}
        task_metric = context.get("task_metric")
        if (
            task_metric is not None
            and state.previous_task_log is not None
            and log(max(float(task_metric), EPS)) - state.previous_task_log
            > self.max_task_log_increase
        ):
            next_log_rho = state.log_rho + log(self.shrink_on_reject)
            next_log_rho = float(np.clip(next_log_rho, log(self.rho_min), log(self.rho_max)))
            next_rho = exp(next_log_rho)
            task_log = log(max(float(task_metric), EPS))
            new_state = replace(
                state,
                rho=next_rho,
                log_rho=next_log_rho,
                previous_task_log=task_log,
                previous_task_log_rho=state.log_rho,
            )
            return ControllerDecision(
                next_rho,
                next_rho != state.rho,
                0.0,
                task_log,
                "task_reject_shrink",
                new_state,
            )
        return super().update(k, state, primal_norm, dual_base_norm, context)


@dataclass
class SumLogResidualsOGD(OnlineOGD):
    """Ablation for minimizing the sum of normalized log residual magnitudes.

    With residuals treated as fixed at the current round, the derivative of
    log(rho * dual_base_norm) with respect to log(rho) is one. This makes the
    update shrink rho monotonically, so this controller is mainly a diagnostic
    for the "sum of log residuals" idea.
    """

    name: str = "online_ogd_sum_log_residuals"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        context = context or {}
        dual_norm = state.rho * dual_base_norm
        primal_threshold = max(float(context.get("primal_threshold", 1.0)), EPS)
        dual_threshold = max(float(context.get("dual_threshold", 1.0)), EPS)
        log_primal = log(max(primal_norm / primal_threshold, EPS))
        log_dual = log(max(dual_norm / dual_threshold, EPS))
        gradient = 1.0
        loss = log_primal + log_dual

        if self.freeze_after is not None and k >= self.freeze_after:
            return ControllerDecision(state.rho, False, gradient, loss, "frozen", state)
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, gradient, loss, "period", state)

        eta = self.eta0 / sqrt(k + 1.0)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, log(self.rho_min), log(self.rho_max)))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        new_state = replace(state, rho=next_rho, log_rho=next_log_rho)
        return ControllerDecision(next_rho, changed, gradient, loss, "sum_log_residuals", new_state)


@dataclass
class NormalizedResidualMagnitudeOGD(OnlineOGD):
    """Projected OGD on normalized residual magnitudes.

    Loss:
      0.5 * log(||r|| / eps_pri)^2
      + 0.5 * log(rho ||d|| / eps_dual)^2.

    The immediate derivative in u = log(rho) is the normalized dual log
    residual. Unlike the plain sum-log loss, this can either increase or
    decrease rho depending on whether the dual residual is below or above its
    target threshold.
    """

    name: str = "online_ogd_norm_magnitude"

    def _logs(
        self,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None,
    ) -> tuple[float, float]:
        context = context or {}
        dual_norm = state.rho * dual_base_norm
        primal_threshold = max(float(context.get("primal_threshold", 1.0)), EPS)
        dual_threshold = max(float(context.get("dual_threshold", 1.0)), EPS)
        log_primal = log(max(primal_norm / primal_threshold, EPS))
        log_dual = log(max(dual_norm / dual_threshold, EPS))
        return log_primal, log_dual

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        log_primal, log_dual = self._logs(state, primal_norm, dual_base_norm, context)
        gradient = float(np.clip(log_dual, -self.grad_clip, self.grad_clip))
        loss = 0.5 * log_primal * log_primal + 0.5 * log_dual * log_dual

        if self.freeze_after is not None and k >= self.freeze_after:
            return ControllerDecision(state.rho, False, gradient, loss, "frozen", state)
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, gradient, loss, "period", state)

        eta = self.eta0 / sqrt(k + 1.0)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, log(self.rho_min), log(self.rho_max)))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        if not changed:
            next_log_rho = state.log_rho
            next_rho = state.rho
        new_state = replace(state, rho=next_rho, log_rho=next_log_rho)
        return ControllerDecision(next_rho, changed, gradient, loss, "norm_magnitude_ogd", new_state)


@dataclass
class TaskNormalizedMagnitudeOGD(NormalizedResidualMagnitudeOGD):
    """Task-aware normalized residual magnitude controller for PTQ/QAT."""

    task_weight: float = 0.35
    task_grad_clip: float = 1.25
    feasibility_weight: float = 0.25
    name: str = "online_ogd_task_norm_magnitude"

    def update(
        self,
        k: int,
        state: PenaltyState,
        primal_norm: float,
        dual_base_norm: float,
        context: dict | None = None,
    ) -> ControllerDecision:
        context = context or {}
        log_primal, log_dual = self._logs(state, primal_norm, dual_base_norm, context)
        task_metric = context.get("task_metric")
        task_gradient = 0.0
        task_log = None
        if task_metric is not None:
            task_log = log(max(float(task_metric), EPS))
            if (
                state.previous_task_log is not None
                and state.previous_task_log_rho is not None
                and abs(state.log_rho - state.previous_task_log_rho) > EPS
            ):
                task_gradient = (
                    (task_log - state.previous_task_log)
                    / (state.log_rho - state.previous_task_log_rho)
                )
                task_gradient = float(np.clip(task_gradient, -self.task_grad_clip, self.task_grad_clip))

        gradient = log_dual + self.task_weight * task_gradient
        gradient -= self.feasibility_weight * float(np.clip(log_primal, -self.grad_clip, self.grad_clip))
        gradient = float(np.clip(gradient, -self.grad_clip, self.grad_clip))
        loss = 0.5 * log_primal * log_primal + 0.5 * log_dual * log_dual
        if task_log is not None:
            loss += self.task_weight * task_log

        carried = replace(
            state,
            previous_task_log=task_log if task_log is not None else state.previous_task_log,
            previous_task_log_rho=state.log_rho if task_log is not None else state.previous_task_log_rho,
        )

        if self.freeze_after is not None and k >= self.freeze_after:
            return ControllerDecision(state.rho, False, gradient, loss, "frozen", carried)
        if (k + 1) % self.update_period:
            return ControllerDecision(state.rho, False, gradient, loss, "period", carried)

        eta = self.eta0 / sqrt(k + 1.0)
        next_log_rho = float(np.clip(state.log_rho - eta * gradient, log(self.rho_min), log(self.rho_max)))
        next_rho = exp(next_log_rho)
        changed = abs(next_rho / state.rho - 1.0) >= self.min_relative_change
        if not changed:
            next_log_rho = state.log_rho
            next_rho = state.rho
        new_state = replace(carried, rho=next_rho, log_rho=next_log_rho)
        return ControllerDecision(next_rho, changed, gradient, loss, "task_norm_magnitude_ogd", new_state)
