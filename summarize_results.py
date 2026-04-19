from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SUMMARY = ROOT / "results" / "summary.csv"


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    if not SUMMARY.exists():
        raise SystemExit(f"Missing {SUMMARY}. Run run_experiments.py first.")

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    with SUMMARY.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            groups[(row["problem"], row["method"])].append(row)

    print(
        f"{'problem':17s} {'method':20s} {'iters':>8s} "
        f"{'primal':>10s} {'dual':>10s} {'rho':>9s} {'changes':>8s}"
    )
    for (problem, method), rows in sorted(groups.items()):
        vals = {
            key: mean([float(row[key]) for row in rows if row.get(key)])
            for key in [
                "iterations",
                "final_primal",
                "final_dual",
                "final_rho",
                "rho_changes",
            ]
        }
        print(
            f"{problem:17s} {method:20s} "
            f"{vals['iterations']:8.2f} {vals['final_primal']:10.2e} "
            f"{vals['final_dual']:10.2e} {vals['final_rho']:9.3g} "
            f"{vals['rho_changes']:8.2f}"
        )


if __name__ == "__main__":
    main()

