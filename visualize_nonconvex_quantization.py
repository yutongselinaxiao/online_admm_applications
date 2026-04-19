from __future__ import annotations

import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path

import visualize_results as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "nonconvex_quantization"
VIZ = RESULTS / "visualizations"


METHODS = [
    "fixed_rho_0p1",
    "fixed_rho_1",
    "fixed_rho_10",
    "residual_balance_raw",
    "residual_balance_normalized",
    "residual_balance_relative",
    "online_ogd",
    "online_ogd_task_aware",
    "online_ogd_task_feasibility",
    "online_ogd_task_accept",
    "online_ogd_epoch5",
    "online_ogd_freeze50",
    "online_ogd_no_dual_rescale",
]

base.PALETTE.update(
    {
        "fixed_rho_0p1": "#4e79a7",
        "fixed_rho_1": "#59a14f",
        "fixed_rho_10": "#e15759",
        "residual_balance_raw": "#f28e2b",
        "residual_balance_normalized": "#edc948",
        "residual_balance_relative": "#ff9da7",
        "online_ogd": "#b07aa1",
        "online_ogd_task_aware": "#2f8f6b",
        "online_ogd_task_feasibility": "#d37295",
        "online_ogd_task_accept": "#6f63b6",
        "online_ogd_epoch5": "#76b7b2",
        "online_ogd_freeze50": "#9c755f",
        "online_ogd_no_dual_rescale": "#111827",
    }
)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return sum(clean) / max(len(clean), 1)


def group_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["method"] in METHODS:
            grouped[(row["problem"], row["method"])].append(row)
    out = []
    for (problem, method), items in sorted(grouped.items()):
        task_metric = "deploy_rel_error" if problem == "tiny_llm_ptq" else "deploy_loss"
        out.append(
            {
                "problem": problem,
                "method": method,
                "iterations": mean([to_float(r["iterations"]) for r in items]),
                "final_primal": mean([to_float(r["final_primal"]) for r in items]),
                "final_dual": mean([to_float(r["final_dual"]) for r in items]),
                "final_rho": mean([to_float(r["final_rho"]) for r in items]),
                "rho_changes": mean([to_float(r["rho_changes"]) for r in items]),
                "task_metric": mean([to_float(r.get(task_metric)) for r in items]),
                "wall_time_sec": mean([to_float(r.get("wall_time_sec")) for r in items]),
            }
        )
    return out


def parse_history_name(path: Path) -> tuple[str, str, int] | None:
    stem = path.name
    for problem in ("tanh_qat_nonconvex", "tiny_llm_ptq"):
        prefix = f"{problem}_"
        if not stem.startswith(prefix):
            continue
        rest = stem[len(prefix) :]
        match = re.match(r"(.+)_seed(\d+)_history\.csv$", rest)
        if match:
            method, seed = match.groups()
            return problem, method, int(seed)
    return None


def load_histories() -> list[dict]:
    histories = []
    for path in sorted(RESULTS.glob("*_history.csv")):
        parsed = parse_history_name(path)
        if parsed is None:
            continue
        problem, method, seed = parsed
        if method not in METHODS:
            continue
        histories.append(
            {
                "problem": problem,
                "method": method,
                "seed": seed,
                "rows": read_csv(path),
            }
        )
    return histories


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def metric_table(rows: list[dict]) -> str:
    headers = [
        "problem",
        "method",
        "iterations",
        "task_metric",
        "final_primal",
        "final_dual",
        "final_rho",
        "rho_changes",
        "wall_time_sec",
    ]
    chunks = ["<table>", "<thead><tr>"]
    chunks.extend(f"<th>{esc(h)}</th>" for h in headers)
    chunks.append("</tr></thead><tbody>")
    for row in rows:
        chunks.append("<tr>")
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                text = f"{value:.3g}" if abs(value) < 0.01 or abs(value) > 999 else f"{value:.2f}"
            else:
                text = str(value)
            chunks.append(f"<td>{esc(text)}</td>")
        chunks.append("</tr>")
    chunks.append("</tbody></table>")
    return "\n".join(chunks)


def dashboard(summary_rows: list[dict], chart_files: list[Path]) -> str:
    cards = "\n".join(
        f'<section><h2>{esc(path.stem.replace("_", " ").title())}</h2><img src="{esc(path.name)}" alt="{esc(path.stem)}"></section>'
        for path in chart_files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Nonconvex Online ADMM Quantization</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 28px 22px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p {{ color: #4b5563; line-height: 1.45; }}
    section {{ margin: 24px 0; padding: 18px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    img {{ width: 100%; height: auto; display: block; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
<main>
  <h1>Nonconvex Online ADMM Quantization</h1>
  <p>Experimental nonconvex and discrete-quantization ADMM examples: tanh-network quantized training and synthetic LLM-style blockwise PTQ.</p>
  {cards}
  <section>
    <h2>Grouped Metrics</h2>
    {metric_table(summary_rows)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    summary_path = RESULTS / "nonconvex_quantization_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run run_nonconvex_quantization.py first.")
    VIZ.mkdir(parents=True, exist_ok=True)
    summary_rows = group_summary(read_csv(summary_path))
    histories = load_histories()
    chart_files: list[Path] = []

    for metric, ylabel, log_scale in [
        ("iterations", "iterations", False),
        ("task_metric", "deploy loss / relative error", True),
        ("final_primal", "final primal residual", True),
        ("final_dual", "final dual residual", True),
        ("rho_changes", "rho changes", False),
    ]:
        filename = f"nonconvex_{metric}_by_problem.svg"
        path = VIZ / filename
        write_text(
            path,
            base.grouped_bar_chart(
                summary_rows,
                metric,
                f"Nonconvex Quantization {ylabel.title()}",
                ylabel,
                width=1420,
                height=560,
                log_scale=log_scale,
            ),
        )
        chart_files.append(path)

    for problem in sorted({h["problem"] for h in histories}):
        selected = [h for h in histories if h["problem"] == problem and h["seed"] == 0]
        metric_specs = [
            ("objective", "objective", True),
            ("primal_norm", "primal residual", True),
            ("dual_norm", "dual residual", True),
            ("rho", "rho", True),
        ]
        if problem == "tiny_llm_ptq":
            metric_specs.insert(1, ("deploy_rel_error", "deploy relative error", True))
        else:
            metric_specs.insert(1, ("deploy_loss", "deploy loss", True))
            metric_specs.insert(2, ("train_loss", "continuous train loss", True))
        for metric, ylabel, log_scale in metric_specs:
            filename = f"nonconvex_{problem}_{metric}_seed0.svg"
            path = VIZ / filename
            write_text(
                path,
                base.line_chart(
                    selected,
                    f"{base.nice_label(problem).title()} {base.nice_label(metric).title()}",
                    ylabel,
                    metric,
                    width=1420,
                    height=560,
                    log_scale=log_scale,
                ),
            )
            chart_files.append(path)

    write_text(VIZ / "index.html", dashboard(summary_rows, chart_files))
    print(f"Wrote nonconvex quantization dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
