from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def soft_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - tau, 0.0)


def relative_error(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.linalg.norm(x - y) / max(np.linalg.norm(y), 1e-12))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize_history(history: list[dict], target_key: str = "objective") -> dict:
    final = history[-1]
    best = min(history, key=lambda row: row[target_key])
    return {
        "iterations": len(history),
        "final_objective": final[target_key],
        "best_objective": best[target_key],
        "final_primal": final["primal_norm"],
        "final_dual": final["dual_norm"],
        "final_rho": final["rho"],
        "rho_changes": sum(1 for row in history if row["rho_changed"]),
        "final_status": final["status"],
    }
