from __future__ import annotations

import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VIZ = RESULTS / "visualizations"

PALETTE = {
    "fixed_rho_0.1": "#4e79a7",
    "fixed_rho_1.0": "#59a14f",
    "fixed_rho_10.0": "#e15759",
    "residual_balance": "#f28e2b",
    "online_ogd": "#b07aa1",
    "online_ogd_epoch3": "#76b7b2",
}


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
        grouped[(row["problem"], row["method"])].append(row)

    out = []
    for (problem, method), items in sorted(grouped.items()):
        out.append(
            {
                "problem": problem,
                "method": method,
                "iterations": mean([to_float(r["iterations"]) for r in items]),
                "final_primal": mean([to_float(r["final_primal"]) for r in items]),
                "final_dual": mean([to_float(r["final_dual"]) for r in items]),
                "final_rho": mean([to_float(r["final_rho"]) for r in items]),
                "rho_changes": mean([to_float(r["rho_changes"]) for r in items]),
            }
        )
    return out


def parse_history_name(path: Path) -> tuple[str, str, int] | None:
    stem = path.name
    for problem in ("graphical_lasso", "consensus_lasso", "tv_denoising"):
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
        rows = read_csv(path)
        histories.append(
            {
                "problem": problem,
                "method": method,
                "seed": seed,
                "rows": rows,
                "path": path,
            }
        )
    return histories


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def svg_wrap(width: int, height: int, body: str, title: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{esc(title)}">\n'
        f"<title>{esc(title)}</title>\n"
        f"{body}\n</svg>\n"
    )


def nice_label(text: str) -> str:
    return text.replace("_", " ")


def axis_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmin == vmax:
        return [vmin]
    return [vmin + i * (vmax - vmin) / (count - 1) for i in range(count)]


def grouped_bar_chart(
    rows: list[dict],
    metric: str,
    title: str,
    ylabel: str,
    width: int = 980,
    height: int = 470,
    log_scale: bool = False,
) -> str:
    problems = sorted({row["problem"] for row in rows})
    methods = sorted({row["method"] for row in rows})
    lookup = {(row["problem"], row["method"]): row for row in rows}
    left, right, top, bottom = 82, 24, 56, 100
    plot_w = width - left - right
    plot_h = height - top - bottom

    values = [max(to_float(str(row[metric])), 1e-12) for row in rows]
    if log_scale:
        yvals = [math.log10(v) for v in values]
        y_min = math.floor(min(yvals))
        y_max = math.ceil(max(yvals))
        transform = lambda v: math.log10(max(v, 1e-12))
        label = lambda v: f"1e{int(v)}"
    else:
        y_min = 0.0
        y_max = max(values) * 1.1 if values else 1.0
        transform = lambda v: v
        label = lambda v: f"{v:.0f}" if v >= 10 else f"{v:.2g}"

    def y_pos(value: float) -> float:
        y_value = transform(value)
        return top + plot_h - (y_value - y_min) / max(y_max - y_min, 1e-12) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-size="20" font-family="Arial" font-weight="700">{esc(title)}</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" font-size="12" font-family="Arial">{esc(ylabel)}</text>',
    ]

    for tick in axis_ticks(y_min, y_max):
        y = top + plot_h - (tick - y_min) / max(y_max - y_min, 1e-12) * plot_h
        body.append(f'<line x1="{left}" x2="{width - right}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{esc(label(tick))}</text>')

    group_w = plot_w / max(len(problems), 1)
    gap = 8
    bar_w = max((group_w - 2 * gap) / max(len(methods), 1), 4)
    for pi, problem in enumerate(problems):
        base_x = left + pi * group_w + gap
        body.append(f'<text x="{left + pi * group_w + group_w / 2:.2f}" y="{height - 54}" text-anchor="middle" font-size="12" font-family="Arial">{esc(nice_label(problem))}</text>')
        for mi, method in enumerate(methods):
            row = lookup.get((problem, method))
            if row is None:
                continue
            value = max(to_float(str(row[metric])), 1e-12)
            x = base_x + mi * bar_w
            y = y_pos(value)
            h = top + plot_h - y
            body.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_w - 2, 1):.2f}" height="{h:.2f}" '
                f'fill="{PALETTE.get(method, "#6b7280")}"><title>{esc(problem)} / {esc(method)}: {value:.4g}</title></rect>'
            )

    legend_x, legend_y = left, height - 30
    for i, method in enumerate(methods):
        x = legend_x + i * 145
        body.append(f'<rect x="{x}" y="{legend_y - 10}" width="10" height="10" fill="{PALETTE.get(method, "#6b7280")}"/>')
        body.append(f'<text x="{x + 15}" y="{legend_y}" font-size="11" font-family="Arial">{esc(method)}</text>')

    return svg_wrap(width, height, "\n".join(body), title)


def line_chart(
    series: list[dict],
    title: str,
    ylabel: str,
    metric: str,
    width: int = 980,
    height: int = 470,
    log_scale: bool = False,
) -> str:
    left, right, top, bottom = 78, 152, 56, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_iter = max(
        int(to_float(row["iter"], 0))
        for item in series
        for row in item["rows"]
    )
    values = [
        max(to_float(row[metric]), 1e-12)
        for item in series
        for row in item["rows"]
    ]
    if log_scale:
        transformed = [math.log10(v) for v in values]
        y_min = math.floor(min(transformed))
        y_max = math.ceil(max(transformed))
        transform = lambda v: math.log10(max(v, 1e-12))
        label = lambda v: f"1e{int(v)}"
    else:
        y_min = min(values) * 0.95 if values else 0.0
        y_max = max(values) * 1.05 if values else 1.0
        if abs(y_max - y_min) < 1e-12:
            y_min -= 1.0
            y_max += 1.0
        transform = lambda v: v
        label = lambda v: f"{v:.2g}"

    def x_pos(iteration: float) -> float:
        return left + (iteration - 1.0) / max(max_iter - 1.0, 1.0) * plot_w

    def y_pos(value: float) -> float:
        y_value = transform(value)
        return top + plot_h - (y_value - y_min) / max(y_max - y_min, 1e-12) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-size="20" font-family="Arial" font-weight="700">{esc(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="Arial">ADMM iteration</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" font-size="12" font-family="Arial">{esc(ylabel)}</text>',
    ]

    for tick in axis_ticks(y_min, y_max):
        y = top + plot_h - (tick - y_min) / max(y_max - y_min, 1e-12) * plot_h
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{esc(label(tick))}</text>')

    for tick in axis_ticks(1, max_iter):
        x = x_pos(tick)
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" font-family="Arial" fill="#4b5563">{int(tick)}</text>')

    for item in series:
        method = item["method"]
        rows = item["rows"]
        points = [
            f'{x_pos(to_float(row["iter"])):.2f},{y_pos(max(to_float(row[metric]), 1e-12)):.2f}'
            for row in rows
        ]
        color = PALETTE.get(method, "#6b7280")
        body.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"><title>{esc(method)}</title></polyline>'
        )

    legend_x, legend_y = width - right + 18, top + 4
    for i, item in enumerate(series):
        y = legend_y + i * 20
        method = item["method"]
        body.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{y}" y2="{y}" stroke="{PALETTE.get(method, "#6b7280")}" stroke-width="3"/>')
        body.append(f'<text x="{legend_x + 22}" y="{y + 4}" font-size="11" font-family="Arial">{esc(method)}</text>')

    return svg_wrap(width, height, "\n".join(body), title)


def metric_table(rows: list[dict]) -> str:
    headers = ["problem", "method", "iterations", "final_primal", "final_dual", "final_rho", "rho_changes"]
    body = ["<table>", "<thead><tr>"]
    body.extend(f"<th>{esc(h)}</th>" for h in headers)
    body.append("</tr></thead><tbody>")
    for row in rows:
        body.append("<tr>")
        for h in headers:
            value = row[h]
            if isinstance(value, float):
                text = f"{value:.3g}" if abs(value) < 0.01 or abs(value) > 999 else f"{value:.2f}"
            else:
                text = str(value)
            body.append(f"<td>{esc(text)}</td>")
        body.append("</tr>")
    body.append("</tbody></table>")
    return "\n".join(body)


def build_dashboard(summary_rows: list[dict], chart_files: list[Path]) -> str:
    cards = "\n".join(
        f'<section><h2>{esc(path.stem.replace("_", " ").title())}</h2><img src="{esc(path.name)}" alt="{esc(path.stem)}"></section>'
        for path in chart_files
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Online ADMM Experiment Visualizations</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 28px 22px 44px; }}
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
  <h1>Online ADMM Experiment Visualizations</h1>
  <p>Generated from <code>results/summary.csv</code> and per-iteration history CSV files. Lower iteration counts and residuals are better; rho trajectory charts show how each controller moves the penalty during ADMM.</p>
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
    summary_path = RESULTS / "summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run run_experiments.py first.")

    VIZ.mkdir(parents=True, exist_ok=True)
    summary_rows = group_summary(read_csv(summary_path))
    histories = load_histories()
    chart_files: list[Path] = []

    chart_specs = [
        ("iterations_by_problem.svg", grouped_bar_chart(summary_rows, "iterations", "Average Iterations To Tolerance", "iterations")),
        ("final_primal_by_problem.svg", grouped_bar_chart(summary_rows, "final_primal", "Average Final Primal Residual", "primal residual", log_scale=True)),
        ("final_dual_by_problem.svg", grouped_bar_chart(summary_rows, "final_dual", "Average Final Dual Residual", "dual residual", log_scale=True)),
        ("rho_changes_by_problem.svg", grouped_bar_chart(summary_rows, "rho_changes", "Average Penalty Changes", "rho changes")),
    ]
    for filename, svg in chart_specs:
        path = VIZ / filename
        write_text(path, svg)
        chart_files.append(path)

    for problem in sorted({h["problem"] for h in histories}):
        selected = [
            h
            for h in histories
            if h["problem"] == problem and h["seed"] == 0
        ]
        if not selected:
            continue
        for metric, ylabel, log_scale in [
            ("objective", "objective", False),
            ("primal_norm", "primal residual", True),
            ("dual_norm", "dual residual", True),
            ("rho", "rho", True),
        ]:
            filename = f"{problem}_{metric}_seed0.svg"
            title = f"{nice_label(problem).title()} {nice_label(metric).title()} Trajectories, Seed 0"
            path = VIZ / filename
            write_text(path, line_chart(selected, title, ylabel, metric, log_scale=log_scale))
            chart_files.append(path)

    write_text(VIZ / "index.html", build_dashboard(summary_rows, chart_files))
    print(f"Wrote visualization dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
