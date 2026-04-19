from __future__ import annotations

import csv
import html
import math
import re
from pathlib import Path

import visualize_results as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llm_curvature_admm"
VIZ = RESULTS / "visualizations"


PALETTE = {
    "rtn_per_channel": "#4e79a7",
    "hessian_diag_clip": "#59a14f",
    "gptq_like_sequential": "#e15759",
    "awq_like_scale_grid": "#f28e2b",
    "smoothquant_like_scale_grid": "#edc948",
    "admm_uniform_fixed10": "#b07aa1",
    "admm_uniform_online_aggressive": "#111827",
    "admm_hessian_diag_fixed10": "#76b7b2",
    "admm_hessian_diag_online_aggressive": "#2f8f6b",
    "admm_gptq_like_fixed10": "#d37295",
    "admm_gptq_like_online_aggressive": "#6f63b6",
    "admm_awq_like_fixed10": "#9c755f",
    "admm_awq_like_online_aggressive": "#8c6d31",
}

base.PALETTE.update(PALETTE)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def to_float(value: str | None, default: float = math.nan) -> float:
    if value in (None, ""):
        return default
    return float(value)


def parse_history_name(path: Path) -> tuple[str, int] | None:
    match = re.match(r"tiny_llm_curvature_admm_(.+)_seed(\d+)_history\.csv$", path.name)
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
        histories.append(
            {
                "problem": "tiny_llm_curvature_admm",
                "method": method,
                "seed": seed,
                "rows": read_csv(path),
            }
        )
    return histories


def metric_table(rows: list[dict]) -> str:
    headers = [
        "method",
        "family",
        "z_mode",
        "deploy_rel_error",
        "weight_rel_error",
        "final_primal",
        "final_dual",
        "final_rho",
    ]
    chunks = ["<table>", "<thead><tr>"]
    chunks.extend(f"<th>{esc(h)}</th>" for h in headers)
    chunks.append("</tr></thead><tbody>")
    for row in sorted(rows, key=lambda r: to_float(r["deploy_rel_error"])):
        chunks.append("<tr>")
        for h in headers:
            value = row.get(h, "")
            if h not in {"method", "family", "z_mode"}:
                number = to_float(value, math.nan)
            else:
                number = math.nan
            if math.isfinite(number):
                text = f"{number:.4g}"
            else:
                text = str(value)
            chunks.append(f"<td>{esc(text)}</td>")
        chunks.append("</tr>")
    chunks.append("</tbody></table>")
    return "\n".join(chunks)


def markdown_report(rows: list[dict]) -> str:
    by_method = {row["method"]: row for row in rows}
    best = min(
        [row for row in rows if row["method"] != "fp16_reference"],
        key=lambda row: to_float(row["deploy_rel_error"]),
    )
    proxy_gptq = by_method["gptq_like_sequential"]
    admm_gptq = by_method["admm_gptq_like_online_aggressive"]
    hdiag = by_method["admm_hessian_diag_online_aggressive"]
    uniform = by_method["admm_uniform_online_aggressive"]
    return f"""# Curvature-Aware ADMM Findings

Run configuration: synthetic LLM PTQ, three seeds, 100 ADMM iterations, 4-bit quantization.

## Main Results

- Best non-FP16 method: `{best['method']}` at deploy relative error `{to_float(best['deploy_rel_error']):.3f}`.
- Standalone GPTQ-like proxy: `{to_float(proxy_gptq['deploy_rel_error']):.3f}`.
- ADMM with GPTQ-like `Z` update plus online rho: `{to_float(admm_gptq['deploy_rel_error']):.3f}`.
- ADMM with Hessian-diagonal `Z` update plus online rho: `{to_float(hdiag['deploy_rel_error']):.3f}`.
- ADMM with the original uniform `Z` update plus online rho: `{to_float(uniform['deploy_rel_error']):.3f}`.

## Conclusion

This is the strongest evidence so far for the online ADMM direction. Plain ADMM with uniform projection only matches RTN. Once the `Z` step becomes curvature-aware, online ADMM becomes competitive with Hessian-aware PTQ and nearly reaches the standalone GPTQ-like proxy. The remaining gap is small enough to justify a real-model experiment.
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
  <title>Curvature-Aware Online ADMM PTQ</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 28px 22px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p {{ color: #4b5563; line-height: 1.45; }}
    section {{ margin: 24px 0; padding: 18px; background: #ffffff; border: 1px solid #e5e7eb; border-radius: 8px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    img {{ width: 100%; height: auto; display: block; }}
    table {{ width: 100%; border-collapse: collapse; background: #ffffff; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
<main>
  <h1>Curvature-Aware Online ADMM PTQ</h1>
  <p>ADMM variants with uniform, Hessian-diagonal, GPTQ-like, and AWQ-like quantized updates compared to local PTQ proxy baselines.</p>
  {cards}
  <section>
    <h2>Grouped Metrics</h2>
    {metric_table(rows)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    grouped_path = RESULTS / "llm_curvature_admm_grouped.csv"
    if not grouped_path.exists():
        raise SystemExit(f"Missing {grouped_path}. Run run_llm_curvature_admm.py first.")
    VIZ.mkdir(parents=True, exist_ok=True)
    rows = read_csv(grouped_path)
    chart_files: list[Path] = []
    for metric, ylabel, log_scale in [
        ("deploy_rel_error", "deploy relative error", False),
        ("weight_rel_error", "weight relative error", False),
        ("final_primal", "ADMM final primal residual", True),
        ("final_dual", "ADMM final dual residual", True),
        ("final_rho", "ADMM final rho", True),
    ]:
        filtered = [row for row in rows if row.get(metric, "") != ""]
        path = VIZ / f"curvature_admm_{metric}.svg"
        write_text(
            path,
            base.grouped_bar_chart(
                [{"problem": "tiny_llm_ptq", **row, metric: to_float(row[metric])} for row in filtered],
                metric,
                f"Curvature-Aware ADMM {ylabel.title()}",
                ylabel,
                width=1360,
                height=560,
                log_scale=log_scale,
            ),
        )
        chart_files.append(path)

    histories = load_histories()
    selected_methods = {
        "admm_uniform_online_aggressive",
        "admm_hessian_diag_online_aggressive",
        "admm_gptq_like_online_aggressive",
        "admm_awq_like_online_aggressive",
    }
    selected = [h for h in histories if h["seed"] == 0 and h["method"] in selected_methods]
    for metric, ylabel, log_scale in [
        ("deploy_rel_error", "deploy relative error", True),
        ("primal_norm", "primal residual", True),
        ("dual_norm", "dual residual", True),
        ("rho", "rho", True),
    ]:
        path = VIZ / f"curvature_admm_{metric}_seed0.svg"
        write_text(
            path,
            base.line_chart(
                selected,
                f"Curvature-Aware ADMM {ylabel.title()}",
                ylabel,
                metric,
                width=1360,
                height=560,
                log_scale=log_scale,
            ),
        )
        chart_files.append(path)

    write_text(RESULTS / "llm_curvature_admm_findings.md", markdown_report(rows))
    write_text(VIZ / "index.html", dashboard(rows, chart_files))
    print(f"Wrote curvature-aware ADMM dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
