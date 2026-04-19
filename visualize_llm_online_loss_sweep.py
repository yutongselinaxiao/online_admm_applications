from __future__ import annotations

import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path

import visualize_results as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llm_online_losses"
VIZ = RESULTS / "visualizations"


PALETTE = {
    "fixed_rho_0p1": "#4e79a7",
    "fixed_rho_1": "#59a14f",
    "fixed_rho_10": "#e15759",
    "online_ogd_balance": "#b07aa1",
    "online_ogd_task_aware": "#2f8f6b",
    "online_ogd_task_feasibility": "#d37295",
    "online_ogd_task_feas_mid": "#6f63b6",
    "online_ogd_task_feas_aggressive": "#8c6d31",
    "online_ogd_sum_log_residuals": "#111827",
    "online_ogd_norm_magnitude": "#76b7b2",
    "online_ogd_task_norm_magnitude": "#f28e2b",
    "online_ogd_task_norm_mag_feas_high": "#9c755f",
}

base.PALETTE.update(PALETTE)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def to_float(value: str | None, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def mean(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return sum(clean) / max(len(clean), 1)


def std(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    mu = mean(clean)
    return math.sqrt(sum((v - mu) ** 2 for v in clean) / (len(clean) - 1))


def group_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    out = []
    for method, items in sorted(grouped.items()):
        out.append(
            {
                "problem": "tiny_llm_ptq",
                "method": method,
                "deploy_rel_error": mean([to_float(r["deploy_rel_error"]) for r in items]),
                "deploy_rel_error_std": std([to_float(r["deploy_rel_error"]) for r in items]),
                "best_deploy_rel_error": mean([to_float(r["best_deploy_rel_error"]) for r in items]),
                "final_primal": mean([to_float(r["final_primal"]) for r in items]),
                "final_dual": mean([to_float(r["final_dual"]) for r in items]),
                "final_residual_max": mean([to_float(r["final_residual_max"]) for r in items]),
                "log_residual_auc": mean([to_float(r["log_residual_auc"]) for r in items]),
                "log_primal_auc": mean([to_float(r["log_primal_auc"]) for r in items]),
                "final_rho": mean([to_float(r["final_rho"]) for r in items]),
                "rho_changes": mean([to_float(r["rho_changes"]) for r in items]),
            }
        )
    return out


def parse_history_name(path: Path) -> tuple[str, int] | None:
    match = re.match(r"tiny_llm_ptq_(.+)_seed(\d+)_history\.csv$", path.name)
    if not match:
        return None
    method, seed = match.groups()
    return method, int(seed)


def load_histories() -> list[dict]:
    histories = []
    for path in sorted(RESULTS.glob("*_history.csv")):
        parsed = parse_history_name(path)
        if parsed is None:
            continue
        method, seed = parsed
        histories.append({"problem": "tiny_llm_ptq", "method": method, "seed": seed, "rows": read_csv(path)})
    return histories


def scatter_chart(
    rows: list[dict],
    x_metric: str,
    y_metric: str,
    title: str,
    xlabel: str,
    ylabel: str,
    width: int = 1120,
    height: int = 560,
    log_x: bool = True,
    log_y: bool = False,
) -> str:
    left, right, top, bottom = 96, 300, 56, 82
    plot_w = width - left - right
    plot_h = height - top - bottom
    xs = [max(to_float(str(r[x_metric])), 1e-12) for r in rows]
    ys = [max(to_float(str(r[y_metric])), 1e-12) for r in rows]

    def scale(values: list[float], log_scale: bool):
        if log_scale:
            vals = [math.log10(v) for v in values]
            lo, hi = math.floor(min(vals)), math.ceil(max(vals))
            return lo, hi, lambda v: math.log10(max(v, 1e-12)), lambda v: f"1e{int(v)}"
        lo, hi = min(values) * 0.96, max(values) * 1.04
        return lo, hi, lambda v: v, lambda v: f"{v:.3g}"

    x_min, x_max, x_tf, x_label = scale(xs, log_x)
    y_min, y_max, y_tf, y_label = scale(ys, log_y)

    def x_pos(value: float) -> float:
        return left + (x_tf(value) - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def y_pos(value: float) -> float:
        return top + plot_h - (y_tf(value) - y_min) / max(y_max - y_min, 1e-12) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="30" font-size="20" font-family="Arial" font-weight="700">{esc(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 16}" text-anchor="middle" font-size="12" font-family="Arial">{esc(xlabel)}</text>',
        f'<text x="20" y="{top + plot_h / 2}" transform="rotate(-90 20 {top + plot_h / 2})" font-size="12" font-family="Arial">{esc(ylabel)}</text>',
    ]
    for tick in base.axis_ticks(x_min, x_max):
        x = left + (tick - x_min) / max(x_max - x_min, 1e-12) * plot_w
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 20}" text-anchor="middle" font-size="11" font-family="Arial" fill="#4b5563">{esc(x_label(tick))}</text>')
    for tick in base.axis_ticks(y_min, y_max):
        y = top + plot_h - (tick - y_min) / max(y_max - y_min, 1e-12) * plot_h
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{esc(y_label(tick))}</text>')

    for row in rows:
        method = row["method"]
        color = PALETTE.get(method, "#6b7280")
        x = x_pos(max(to_float(str(row[x_metric])), 1e-12))
        y = y_pos(max(to_float(str(row[y_metric])), 1e-12))
        body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="6" fill="{color}" opacity="0.9"><title>{esc(method)}</title></circle>')
        body.append(f'<text x="{x + 8:.2f}" y="{y - 8:.2f}" font-size="10" font-family="Arial" fill="#111827">{esc(method.replace("online_ogd_", ""))}</text>')

    legend_x, legend_y = width - right + 24, top + 8
    for i, row in enumerate(rows):
        method = row["method"]
        y = legend_y + i * 21
        body.append(f'<circle cx="{legend_x}" cy="{y}" r="5" fill="{PALETTE.get(method, "#6b7280")}"/>')
        body.append(f'<text x="{legend_x + 12}" y="{y + 4}" font-size="10" font-family="Arial">{esc(method)}</text>')

    return base.svg_wrap(width, height, "\n".join(body), title)


def metric_table(rows: list[dict]) -> str:
    headers = [
        "method",
        "deploy_rel_error",
        "final_primal",
        "final_dual",
        "final_residual_max",
        "log_residual_auc",
        "final_rho",
    ]
    chunks = ["<table>", "<thead><tr>"]
    chunks.extend(f"<th>{esc(h)}</th>" for h in headers)
    chunks.append("</tr></thead><tbody>")
    for row in sorted(rows, key=lambda r: r["deploy_rel_error"]):
        chunks.append("<tr>")
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                text = f"{value:.3g}" if abs(value) < 0.01 or abs(value) > 999 else f"{value:.3f}"
            else:
                text = str(value)
            chunks.append(f"<td>{esc(text)}</td>")
        chunks.append("</tr>")
    chunks.append("</tbody></table>")
    return "\n".join(chunks)


def markdown_report(rows: list[dict]) -> str:
    ranked = sorted(rows, key=lambda r: r["deploy_rel_error"])
    by_method = {r["method"]: r for r in rows}
    best = ranked[0]
    sum_log = by_method["online_ogd_sum_log_residuals"]
    task_feas = by_method["online_ogd_task_feasibility"]
    task_feas_mid = by_method["online_ogd_task_feas_mid"]
    task_feas_aggressive = by_method["online_ogd_task_feas_aggressive"]
    norm_mag = by_method["online_ogd_task_norm_mag_feas_high"]
    fixed10 = by_method["fixed_rho_10"]
    return f"""# LLM Online Loss Sweep Findings

Run configuration: synthetic LLM PTQ, three seeds, 100 ADMM iterations, 4-bit symmetric per-channel quantization.

## Main Results

- Best deploy relative error: `{best['method']}` at `{best['deploy_rel_error']:.3f}`.
- The aggressive feasibility-task online method beats fixed `rho=10` on deploy error: `{task_feas_aggressive['deploy_rel_error']:.3f}` versus `{fixed10['deploy_rel_error']:.3f}`. The cost is a larger average final dual residual: `{task_feas_aggressive['final_dual']:.2f}` versus `{fixed10['final_dual']:.2f}`.
- The milder feasibility-task variants form a useful ladder: base feasibility `{task_feas['deploy_rel_error']:.3f}`, mid feasibility `{task_feas_mid['deploy_rel_error']:.3f}`, aggressive feasibility `{task_feas_aggressive['deploy_rel_error']:.3f}`.
- The normalized-magnitude high-feasibility variant reaches `{norm_mag['deploy_rel_error']:.3f}`, so this residual-size objective is not competitive here.
- The plain sum-log residual loss is not good here: it reaches `{sum_log['deploy_rel_error']:.3f}` and drives the final rho down to `{sum_log['final_rho']:.4f}`. This supports the analytic concern that minimizing `log ||r|| + log ||s||` has a one-sided immediate gradient in `log(rho)`.

## Tradeoff

The scatter plots show a real tradeoff: penalties that improve deploy accuracy often do not minimize the combined ADMM residual score. In PTQ, stronger coupling can lower quantization error while leaving larger dual movement. This suggests that faster ADMM residual convergence is not automatically worth it if the objective is deploy accuracy.
"""


def dashboard(rows: list[dict], chart_files: list[Path]) -> str:
    cards = "\n".join(
        f'<section><h2>{esc(path.stem.replace("_", " ").title())}</h2><img src="{esc(path.name)}" alt="{esc(path.stem)}"></section>'
        for path in chart_files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>LLM Online Loss Sweep</title>
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
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
<main>
  <h1>LLM Online Loss Sweep</h1>
  <p>Deploy accuracy and ADMM residual tradeoffs for alternative online penalty losses.</p>
  {cards}
  <section>
    <h2>Mean Metrics</h2>
    {metric_table(rows)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    summary_path = RESULTS / "llm_online_loss_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run run_llm_online_loss_sweep.py first.")
    VIZ.mkdir(parents=True, exist_ok=True)
    summary_rows = group_summary(read_csv(summary_path))
    write_csv(RESULTS / "llm_online_loss_grouped.csv", summary_rows)
    histories = load_histories()
    chart_files: list[Path] = []

    for metric, ylabel, log_scale in [
        ("deploy_rel_error", "deploy relative error", False),
        ("final_primal", "final primal residual", True),
        ("final_dual", "final dual residual", True),
        ("final_residual_max", "max residual", True),
        ("log_residual_auc", "log residual auc", False),
        ("final_rho", "final rho", True),
    ]:
        path = VIZ / f"llm_loss_{metric}_bar.svg"
        write_text(
            path,
            base.grouped_bar_chart(
                summary_rows,
                metric,
                f"LLM PTQ {ylabel.title()}",
                ylabel,
                width=1320,
                height=560,
                log_scale=log_scale,
            ),
        )
        chart_files.append(path)

    for x_metric, x_label, log_x in [
        ("final_residual_max", "final max ADMM residual", True),
        ("final_primal", "final primal residual", True),
        ("log_residual_auc", "mean log10 max residual", False),
    ]:
        path = VIZ / f"llm_loss_tradeoff_{x_metric}_vs_deploy.svg"
        write_text(
            path,
            scatter_chart(
                summary_rows,
                x_metric,
                "deploy_rel_error",
                f"Deploy Accuracy vs {x_label.title()}",
                x_label,
                "deploy relative error",
                log_x=log_x,
                log_y=False,
            ),
        )
        chart_files.append(path)

    selected = [h for h in histories if h["seed"] == 0]
    for metric, ylabel, log_scale in [
        ("deploy_rel_error", "deploy relative error", True),
        ("primal_norm", "primal residual", True),
        ("dual_norm", "dual residual", True),
        ("rho", "rho", True),
    ]:
        path = VIZ / f"llm_loss_{metric}_seed0.svg"
        write_text(
            path,
            base.line_chart(
                selected,
                f"LLM PTQ {ylabel.title()}",
                ylabel,
                metric,
                width=1420,
                height=600,
                log_scale=log_scale,
            ),
        )
        chart_files.append(path)

    write_text(RESULTS / "llm_online_loss_findings.md", markdown_report(summary_rows))
    write_text(VIZ / "index.html", dashboard(summary_rows, chart_files))
    print(f"Wrote LLM online-loss dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
