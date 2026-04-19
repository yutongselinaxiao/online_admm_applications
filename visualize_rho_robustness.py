from __future__ import annotations

import csv
import html
import math
from collections import defaultdict
from pathlib import Path

import visualize_results as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "rho_robustness"
VIZ = RESULTS / "visualizations"


PALETTE = {
    "online_ogd": "#b07aa1",
    "online_ogd_task_aware": "#2f8f6b",
    "online_ogd_task_feasibility": "#d37295",
    "vector_online_ogd": "#111827",
    "vector_task_aware": "#4e79a7",
    "vector_task_feasibility": "#e15759",
}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def to_float(value: str | None, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return sum(clean) / max(len(clean), 1)


def std(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    mu = mean(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))


def task_metric_name(problem: str) -> str:
    return "deploy_loss" if problem == "tanh_qat_nonconvex" else "deploy_rel_error"


def grouped_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, float], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["problem"], row["base_method"], to_float(row["rho0"]))].append(row)

    out = []
    for (problem, method, rho0), items in sorted(grouped.items()):
        metric = task_metric_name(problem)
        task_values = [to_float(r.get(metric)) for r in items]
        out.append(
            {
                "problem": problem,
                "base_method": method,
                "rho0": rho0,
                "task_metric": mean(task_values),
                "task_metric_std": std(task_values),
                "final_primal": mean([to_float(r["final_primal"]) for r in items]),
                "final_dual": mean([to_float(r["final_dual"]) for r in items]),
                "final_rho": mean([to_float(r["final_rho"]) for r in items]),
                "rho_min": mean([to_float(r.get("rho_min")) for r in items]),
                "rho_max": mean([to_float(r.get("rho_max")) for r in items]),
            }
        )
    return out


def robustness_rows(grouped: list[dict]) -> list[dict]:
    by_method: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in grouped:
        by_method[(row["problem"], row["base_method"])].append(row)
    out = []
    for (problem, method), rows in sorted(by_method.items()):
        values = [to_float(str(r["task_metric"])) for r in rows]
        best = min(values)
        worst = max(values)
        out.append(
            {
                "problem": problem,
                "base_method": method,
                "mean_task_metric": mean(values),
                "best_task_metric": best,
                "worst_task_metric": worst,
                "robustness_range": worst - best,
                "relative_range": (worst - best) / max(best, 1e-12),
                "best_rho0": rows[values.index(best)]["rho0"],
                "worst_rho0": rows[values.index(worst)]["rho0"],
            }
        )
    return out


def rho_line_chart(
    rows: list[dict],
    problem: str,
    metric: str,
    title: str,
    ylabel: str,
    log_y: bool = False,
    width: int = 1180,
    height: int = 520,
) -> str:
    selected = [r for r in rows if r["problem"] == problem]
    methods = sorted({r["base_method"] for r in selected})
    rho0s = sorted({to_float(str(r["rho0"])) for r in selected})
    lookup = {(r["base_method"], to_float(str(r["rho0"]))): r for r in selected}
    left, right, top, bottom = 84, 250, 54, 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = [max(to_float(str(r[metric])), 1e-12) for r in selected if math.isfinite(to_float(str(r[metric])))]
    if log_y:
        transformed = [math.log10(v) for v in values]
        y_min = math.floor(min(transformed))
        y_max = math.ceil(max(transformed))
        transform = lambda v: math.log10(max(v, 1e-12))
        label = lambda v: f"1e{int(v)}"
    else:
        y_min = min(values) * 0.95
        y_max = max(values) * 1.05
        transform = lambda v: v
        label = lambda v: f"{v:.3g}"

    def x_pos(rho0: float) -> float:
        idx = rho0s.index(rho0)
        return left + idx / max(len(rho0s) - 1, 1) * plot_w

    def y_pos(value: float) -> float:
        yv = transform(value)
        return top + plot_h - (yv - y_min) / max(y_max - y_min, 1e-12) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-size="20" font-family="Arial" font-weight="700">{esc(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 14}" text-anchor="middle" font-size="12" font-family="Arial">initial rho</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" font-size="12" font-family="Arial">{esc(ylabel)}</text>',
    ]
    for tick in base.axis_ticks(y_min, y_max):
        y = top + plot_h - (tick - y_min) / max(y_max - y_min, 1e-12) * plot_h
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{esc(label(tick))}</text>')
    for rho0 in rho0s:
        x = x_pos(rho0)
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-size="11" font-family="Arial" fill="#4b5563">{rho0:g}</text>')

    for method in methods:
        points = []
        for rho0 in rho0s:
            row = lookup.get((method, rho0))
            if row is None:
                continue
            points.append(f'{x_pos(rho0):.2f},{y_pos(max(to_float(str(row[metric])), 1e-12)):.2f}')
        color = PALETTE.get(method, "#6b7280")
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2.4" stroke-linejoin="round" stroke-linecap="round"><title>{esc(method)}</title></polyline>'
        )
    legend_x, legend_y = width - right + 22, top + 4
    for i, method in enumerate(methods):
        y = legend_y + i * 22
        color = PALETTE.get(method, "#6b7280")
        body.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{y}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        body.append(f'<text x="{legend_x + 22}" y="{y + 4}" font-size="11" font-family="Arial">{esc(method)}</text>')

    return base.svg_wrap(width, height, "\n".join(body), title)


def table(rows: list[dict]) -> str:
    headers = [
        "problem",
        "base_method",
        "mean_task_metric",
        "best_task_metric",
        "worst_task_metric",
        "relative_range",
        "best_rho0",
        "worst_rho0",
    ]
    chunks = ["<table>", "<thead><tr>"]
    chunks.extend(f"<th>{esc(h)}</th>" for h in headers)
    chunks.append("</tr></thead><tbody>")
    for row in rows:
        chunks.append("<tr>")
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                text = f"{value:.3g}" if h != "relative_range" else f"{100 * value:.1f}%"
            else:
                text = str(value)
            chunks.append(f"<td>{esc(text)}</td>")
        chunks.append("</tr>")
    chunks.append("</tbody></table>")
    return "\n".join(chunks)


def dashboard(robustness: list[dict], chart_files: list[Path]) -> str:
    cards = "\n".join(
        f'<section><h2>{esc(path.stem.replace("_", " ").title())}</h2><img src="{esc(path.name)}" alt="{esc(path.stem)}"></section>'
        for path in chart_files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Rho Robustness and Vector-Rho PTQ</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1240px; margin: 0 auto; padding: 28px 22px 44px; }}
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
  <h1>Rho Robustness and Vector-Rho PTQ</h1>
  <p>Initial-rho sweeps for scalar online penalties and per-block vector-rho variants on synthetic LLM PTQ.</p>
  {cards}
  <section>
    <h2>Robustness Summary</h2>
    {table(robustness)}
  </section>
</main>
</body>
</html>
"""


def markdown_report(robustness: list[dict]) -> str:
    def find(problem: str, method: str) -> dict:
        for row in robustness:
            if row["problem"] == problem and row["base_method"] == method:
                return row
        raise KeyError((problem, method))

    scalar = find("tiny_llm_ptq", "online_ogd_task_feasibility")
    vector = find("tiny_llm_ptq_vector_rho", "vector_task_feasibility")
    tanh = find("tanh_qat_nonconvex", "online_ogd")
    return f"""# Rho Initialization and Vector-Rho Findings

The sweep used initial rho values `0.01, 0.1, 1, 10, 100`, three seeds, 100 ADMM iterations, and 4-bit quantization.

## Takeaways

- Scalar online OGD is fairly stable in final rho: on tanh QAT, `online_ogd` returns near `rho ~= 0.6` across a wide initialization range, but deploy loss still worsens when initialized very high. Its best initial rho was `{tanh['best_rho0']:g}` and worst was `{tanh['worst_rho0']:g}`.
- On tiny LLM PTQ, the scalar feasibility-task method is the best online scalar method in this sweep. Its mean deploy relative error across initializations is `{scalar['mean_task_metric']:.3f}`, with best `{scalar['best_task_metric']:.3f}` at `rho0={scalar['best_rho0']:g}` and worst `{scalar['worst_task_metric']:.3f}` at `rho0={scalar['worst_rho0']:g}`.
- Vector rho helps only modestly in this implementation. The best vector feasibility-task result is `{vector['best_task_metric']:.3f}`, compared with the scalar feasibility-task best of `{scalar['best_task_metric']:.3f}`. The gain is not enough to beat the oracle scalar fixed `rho=10` result from the earlier suite.
- The value of vector rho is more about block heterogeneity and diagnostics than an automatic win. It is worth keeping as a serious extension, but it needs a stronger per-block task signal or blockwise curvature normalization.

## Research Implication

Initialization robustness is a useful selling point for online tuning: a robust online method should recover similar final penalties from poor starts. Vector rho should be presented as a natural generalization for multi-block ADMM, but the current evidence says it is not sufficient by itself; the online loss matters more than scalar versus vector parameterization.
"""


def main() -> None:
    summary_path = RESULTS / "rho_robustness_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run run_rho_robustness_vector.py first.")
    VIZ.mkdir(parents=True, exist_ok=True)
    grouped = grouped_rows(read_csv(summary_path))
    robustness = robustness_rows(grouped)
    write_csv(RESULTS / "rho_robustness_grouped.csv", grouped)
    write_csv(RESULTS / "rho_robustness_method_summary.csv", robustness)

    chart_files: list[Path] = []
    for problem in ["tanh_qat_nonconvex", "tiny_llm_ptq", "tiny_llm_ptq_vector_rho"]:
        for metric, ylabel, log_y in [
            ("task_metric", "deploy loss / relative error", False),
            ("final_rho", "final rho", True),
        ]:
            path = VIZ / f"{problem}_{metric}_vs_rho0.svg"
            write_text(
                path,
                rho_line_chart(
                    grouped,
                    problem,
                    metric,
                    f"{problem.replace('_', ' ').title()} {ylabel.title()}",
                    ylabel,
                    log_y=log_y,
                ),
            )
            chart_files.append(path)

    write_text(RESULTS / "rho_robustness_findings.md", markdown_report(robustness))
    write_text(VIZ / "index.html", dashboard(robustness, chart_files))
    print(f"Wrote rho robustness dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
