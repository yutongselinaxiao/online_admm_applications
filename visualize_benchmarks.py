from __future__ import annotations

import csv
import html
import math
import re
from collections import defaultdict
from pathlib import Path

import visualize_results as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "benchmarks"
VIZ = RESULTS / "visualizations"

PAPER_METHODS = [
    "fixed_rho_1",
    "oracle_fixed_grid",
    "residual_balance_raw",
    "residual_balance_normalized",
    "residual_balance_relative",
    "spectral_aadmm_xu2017",
    "bb_online_penalty_simplified",
    "online_ogd",
    "online_ogd_epoch5",
    "online_ogd_freeze50",
    "online_ogd_no_dual_rescale",
]

ADAPTIVE_RHO_MARKERS = [
    "online_ogd",
    "spectral_aadmm_xu2017",
    "residual_balance_relative",
    "bb_online_penalty_simplified",
]

base.PALETTE.update(
    {
        "fixed_rho_1": "#59a14f",
        "oracle_fixed_grid": "#111827",
        "residual_balance_raw": "#f28e2b",
        "residual_balance_normalized": "#edc948",
        "residual_balance_relative": "#ff9da7",
        "spectral_aadmm_xu2017": "#9c755f",
        "bb_online_penalty_simplified": "#bab0ac",
        "online_ogd": "#b07aa1",
        "online_ogd_epoch5": "#76b7b2",
        "online_ogd_freeze50": "#4e79a7",
        "online_ogd_no_dual_rescale": "#e15759",
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
        if row["method"] in PAPER_METHODS:
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
                "wall_time_sec": mean([to_float(r.get("wall_time_sec")) for r in items]),
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
        if method not in PAPER_METHODS:
            continue
        histories.append(
            {
                "problem": problem,
                "method": method,
                "seed": seed,
                "rows": read_csv(path),
                "path": path,
            }
        )
    return histories


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def fixed_rho_value(method: str) -> float | None:
    prefix = "fixed_rho_"
    if not method.startswith(prefix):
        return None
    value = method[len(prefix) :].replace("p", ".").replace("m", "-")
    try:
        return float(value)
    except ValueError:
        return None


def sensitivity_rows(rows: list[dict], problem: str) -> list[dict]:
    grouped: dict[float, list[dict]] = defaultdict(list)
    for row in rows:
        if row["problem"] != problem:
            continue
        rho = fixed_rho_value(row["method"])
        if rho is not None:
            grouped[rho].append(row)

    out = []
    for rho, items in sorted(grouped.items()):
        residuals = [
            max(to_float(item["final_primal"]), to_float(item["final_dual"]))
            for item in items
        ]
        iterations = [to_float(item["iterations"]) for item in items]
        out.append(
            {
                "rho": rho,
                "iterations_mean": mean(iterations),
                "iterations_min": min(iterations),
                "iterations_max": max(iterations),
                "residual_mean": mean(residuals),
                "residual_min": min(residuals),
                "residual_max": max(residuals),
            }
        )
    return out


def adaptive_marker_rows(rows: list[dict], problem: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["problem"] == problem and row["method"] in ADAPTIVE_RHO_MARKERS:
            grouped[row["method"]].append(row)
    return [
        {
            "method": method,
            "rho": mean([to_float(row["final_rho"]) for row in items]),
            "iterations": mean([to_float(row["iterations"]) for row in items]),
        }
        for method, items in sorted(grouped.items())
    ]


def oracle_rho(rows: list[dict], problem: str) -> float | None:
    values = [
        to_float(row.get("oracle_rho"))
        for row in rows
        if row["problem"] == problem
        and row["method"] == "oracle_fixed_grid"
        and row.get("oracle_rho")
    ]
    if not values:
        return None
    return mean(values)


def sensitivity_chart(
    fixed_rows: list[dict],
    markers: list[dict],
    problem: str,
    metric: str,
    title: str,
    ylabel: str,
    oracle: float | None,
    width: int = 1420,
    height: int = 560,
    log_y: bool = False,
) -> str:
    left, right, top, bottom = 86, 260, 56, 72
    plot_w = width - left - right
    plot_h = height - top - bottom
    rhos = [row["rho"] for row in fixed_rows]
    x_min, x_max = math.log10(min(rhos)), math.log10(max(rhos))
    values = [max(row[f"{metric}_mean"], 1e-12) for row in fixed_rows]
    values += [max(row[f"{metric}_min"], 1e-12) for row in fixed_rows]
    values += [max(row[f"{metric}_max"], 1e-12) for row in fixed_rows]
    if log_y:
        y_values = [math.log10(v) for v in values]
        y_min, y_max = math.floor(min(y_values)), math.ceil(max(y_values))
        y_transform = lambda value: math.log10(max(value, 1e-12))
        y_label = lambda value: f"1e{int(value)}"
    else:
        y_min = 0.0
        y_max = max(values) * 1.1
        y_transform = lambda value: value
        y_label = lambda value: f"{value:.0f}" if value >= 10 else f"{value:.2g}"

    def x_pos(rho: float) -> float:
        return left + (math.log10(rho) - x_min) / max(x_max - x_min, 1e-12) * plot_w

    def y_pos(value: float) -> float:
        y_value = y_transform(value)
        return top + plot_h - (y_value - y_min) / max(y_max - y_min, 1e-12) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-size="20" font-family="Arial" font-weight="700">{esc(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 18}" text-anchor="middle" font-size="12" font-family="Arial">fixed rho grid, log scale</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" font-size="12" font-family="Arial">{esc(ylabel)}</text>',
    ]

    for tick in base.axis_ticks(y_min, y_max):
        y = top + plot_h - (tick - y_min) / max(y_max - y_min, 1e-12) * plot_h
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">{esc(y_label(tick))}</text>')

    for rho in rhos:
        x = x_pos(rho)
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" font-family="Arial" fill="#4b5563">{rho:g}</text>')

    band_upper = " ".join(
        f'{x_pos(row["rho"]):.2f},{y_pos(row[f"{metric}_max"]):.2f}'
        for row in fixed_rows
    )
    band_lower = " ".join(
        f'{x_pos(row["rho"]):.2f},{y_pos(row[f"{metric}_min"]):.2f}'
        for row in reversed(fixed_rows)
    )
    mean_points = " ".join(
        f'{x_pos(row["rho"]):.2f},{y_pos(row[f"{metric}_mean"]):.2f}'
        for row in fixed_rows
    )
    body.append(f'<polygon points="{band_upper} {band_lower}" fill="#d1d5db" opacity="0.45"><title>seed min-max band</title></polygon>')
    body.append(f'<polyline points="{mean_points}" fill="none" stroke="#111827" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"><title>fixed rho mean</title></polyline>')
    for row in fixed_rows:
        body.append(f'<circle cx="{x_pos(row["rho"]):.2f}" cy="{y_pos(row[f"{metric}_mean"]):.2f}" r="4" fill="#111827"><title>rho={row["rho"]:g}, mean={row[f"{metric}_mean"]:.4g}</title></circle>')

    if oracle is not None:
        x = x_pos(oracle)
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#111827" stroke-width="2" stroke-dasharray="6 5"/>')
        body.append(f'<text x="{x + 5:.2f}" y="{top + 16}" font-size="11" font-family="Arial" fill="#111827">oracle rho {oracle:.3g}</text>')

    legend_x = left + plot_w + 24
    legend_y = top + 8
    body.append(f'<line x1="{legend_x}" x2="{legend_x + 24}" y1="{legend_y}" y2="{legend_y}" stroke="#111827" stroke-width="3"/>')
    body.append(f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-size="11" font-family="Arial">fixed rho mean</text>')
    body.append(f'<rect x="{legend_x}" y="{legend_y + 14}" width="24" height="10" fill="#d1d5db" opacity="0.45"/>')
    body.append(f'<text x="{legend_x + 32}" y="{legend_y + 24}" font-size="11" font-family="Arial">seed min-max band</text>')

    for i, marker in enumerate(markers):
        color = base.PALETTE.get(marker["method"], "#6b7280")
        x = x_pos(max(min(marker["rho"], max(rhos)), min(rhos)))
        y0 = top + plot_h + 28 + 13 * (i % 5)
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="{color}" stroke-width="2" stroke-dasharray="3 4" opacity="0.9"><title>{esc(marker["method"])} final rho {marker["rho"]:.4g}</title></line>')
        lx = legend_x
        ly = legend_y + 46 + i * 19
        body.append(f'<line x1="{lx}" x2="{lx + 24}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="2" stroke-dasharray="3 4"/>')
        body.append(f'<text x="{lx + 32}" y="{ly + 4}" font-size="11" font-family="Arial">{esc(marker["method"])} rho {marker["rho"]:.3g}</text>')

    return base.svg_wrap(width, height, "\n".join(body), title)


def rho_trajectory_with_oracle(
    series: list[dict],
    title: str,
    oracle: float | None,
    width: int = 1420,
    height: int = 560,
) -> str:
    left, right, top, bottom = 78, 190, 56, 62
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_iter = max(int(to_float(row["iter"], 0)) for item in series for row in item["rows"])
    values = [max(to_float(row["rho"]), 1e-12) for item in series for row in item["rows"]]
    if oracle is not None:
        values.append(max(oracle, 1e-12))
    y_min = math.floor(math.log10(min(values)))
    y_max = math.ceil(math.log10(max(values)))

    def x_pos(iteration: float) -> float:
        return left + (iteration - 1.0) / max(max_iter - 1.0, 1.0) * plot_w

    def y_pos(value: float) -> float:
        y_value = math.log10(max(value, 1e-12))
        return top + plot_h - (y_value - y_min) / max(y_max - y_min, 1e-12) * plot_h

    body = [
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="28" font-size="20" font-family="Arial" font-weight="700">{esc(title)}</text>',
        f'<text x="{left + plot_w / 2}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="Arial">ADMM iteration</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" font-size="12" font-family="Arial">rho, log scale</text>',
    ]

    for tick in base.axis_ticks(y_min, y_max):
        y = top + plot_h - (tick - y_min) / max(y_max - y_min, 1e-12) * plot_h
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        body.append(f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="Arial" fill="#4b5563">1e{int(tick)}</text>')

    for tick in base.axis_ticks(1, max_iter):
        x = x_pos(tick)
        body.append(f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        body.append(f'<text x="{x:.2f}" y="{top + plot_h + 18}" text-anchor="middle" font-size="11" font-family="Arial" fill="#4b5563">{int(tick)}</text>')

    if oracle is not None:
        y = y_pos(oracle)
        body.append(f'<line x1="{left}" x2="{left + plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#111827" stroke-width="2" stroke-dasharray="7 5"/>')
        body.append(f'<text x="{left + 8}" y="{y - 6:.2f}" font-size="11" font-family="Arial" fill="#111827">oracle fixed rho {oracle:.3g}</text>')

    for item in series:
        method = item["method"]
        rows = item["rows"]
        points = " ".join(
            f'{x_pos(to_float(row["iter"])):.2f},{y_pos(max(to_float(row["rho"]), 1e-12)):.2f}'
            for row in rows
        )
        color = base.PALETTE.get(method, "#6b7280")
        body.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"><title>{esc(method)}</title></polyline>')

    legend_x, legend_y = width - right + 18, top + 4
    if oracle is not None:
        body.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{legend_y}" y2="{legend_y}" stroke="#111827" stroke-width="2" stroke-dasharray="7 5"/>')
        body.append(f'<text x="{legend_x + 22}" y="{legend_y + 4}" font-size="11" font-family="Arial">oracle rho</text>')
        legend_y += 22
    for i, item in enumerate(series):
        y = legend_y + i * 20
        method = item["method"]
        body.append(f'<line x1="{legend_x}" x2="{legend_x + 16}" y1="{y}" y2="{y}" stroke="{base.PALETTE.get(method, "#6b7280")}" stroke-width="3"/>')
        body.append(f'<text x="{legend_x + 22}" y="{y + 4}" font-size="11" font-family="Arial">{esc(method)}</text>')

    return base.svg_wrap(width, height, "\n".join(body), title)


def metric_table(rows: list[dict]) -> str:
    headers = [
        "problem",
        "method",
        "iterations",
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
  <title>Expanded Online ADMM Benchmarks</title>
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
  <h1>Expanded Online ADMM Benchmarks</h1>
  <p>Includes oracle fixed-rho grid, raw/normalized/relative residual balancing, BB-style online penalty adaptation, OGD variants, freeze ablation, and no-dual-rescale failure-mode ablation.</p>
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
    summary_path = RESULTS / "benchmark_summary.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing {summary_path}. Run run_benchmark_suite.py first.")

    VIZ.mkdir(parents=True, exist_ok=True)
    raw_summary_rows = read_csv(summary_path)
    summary_rows = group_summary(raw_summary_rows)
    histories = load_histories()
    chart_files: list[Path] = []

    chart_specs = [
        (
            "benchmark_iterations_by_problem.svg",
            base.grouped_bar_chart(
                summary_rows,
                "iterations",
                "Benchmark Average Iterations To Tolerance",
                "iterations",
                width=1420,
                height=560,
            ),
        ),
        (
            "benchmark_wall_time_by_problem.svg",
            base.grouped_bar_chart(
                summary_rows,
                "wall_time_sec",
                "Benchmark Average Wall Time",
                "seconds",
                width=1420,
                height=560,
            ),
        ),
        (
            "benchmark_final_primal_by_problem.svg",
            base.grouped_bar_chart(
                summary_rows,
                "final_primal",
                "Benchmark Average Final Primal Residual",
                "primal residual",
                width=1420,
                height=560,
                log_scale=True,
            ),
        ),
        (
            "benchmark_final_dual_by_problem.svg",
            base.grouped_bar_chart(
                summary_rows,
                "final_dual",
                "Benchmark Average Final Dual Residual",
                "dual residual",
                width=1420,
                height=560,
                log_scale=True,
            ),
        ),
        (
            "benchmark_rho_changes_by_problem.svg",
            base.grouped_bar_chart(
                summary_rows,
                "rho_changes",
                "Benchmark Average Penalty Changes",
                "rho changes",
                width=1420,
                height=560,
            ),
        ),
    ]

    for filename, svg in chart_specs:
        path = VIZ / filename
        write_text(path, svg)
        chart_files.append(path)

    for problem in sorted({row["problem"] for row in raw_summary_rows}):
        fixed_rows = sensitivity_rows(raw_summary_rows, problem)
        markers = adaptive_marker_rows(raw_summary_rows, problem)
        oracle = oracle_rho(raw_summary_rows, problem)
        if not fixed_rows:
            continue
        for metric, ylabel, log_y in [
            ("iterations", "iterations to tolerance", False),
            ("residual", "max final residual", True),
        ]:
            filename = f"sensitivity_{problem}_{metric}.svg"
            title = f"{base.nice_label(problem).title()} Fixed-Rho Sensitivity: {ylabel.title()}"
            path = VIZ / filename
            write_text(
                path,
                sensitivity_chart(
                    fixed_rows,
                    markers,
                    problem,
                    metric,
                    title,
                    ylabel,
                    oracle,
                    log_y=log_y,
                ),
            )
            chart_files.append(path)

    for problem in sorted({h["problem"] for h in histories}):
        selected = [
            h
            for h in histories
            if h["problem"] == problem and h["seed"] == 0
        ]
        for metric, ylabel, log_scale in [
            ("objective", "objective", False),
            ("primal_norm", "primal residual", True),
            ("dual_norm", "dual residual", True),
            ("rho", "rho", True),
        ]:
            filename = f"benchmark_{problem}_{metric}_seed0.svg"
            title = f"{base.nice_label(problem).title()} {base.nice_label(metric).title()} Benchmark Trajectories, Seed 0"
            path = VIZ / filename
            if metric == "rho":
                svg = rho_trajectory_with_oracle(
                    selected,
                    title,
                    oracle_rho(raw_summary_rows, problem),
                    width=1420,
                    height=560,
                )
            else:
                svg = base.line_chart(
                    selected,
                    title,
                    ylabel,
                    metric,
                    width=1420,
                    height=560,
                    log_scale=log_scale,
                )
            write_text(path, svg)
            chart_files.append(path)

    write_text(VIZ / "index.html", dashboard(summary_rows, chart_files))
    print(f"Wrote benchmark visualization dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
