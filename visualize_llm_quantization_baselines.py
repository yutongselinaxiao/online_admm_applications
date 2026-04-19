from __future__ import annotations

import csv
import html
import math
from pathlib import Path

import visualize_results as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "llm_quant_baselines"
VIZ = RESULTS / "visualizations"


PALETTE = {
    "fp16_reference": "#9ca3af",
    "rtn_per_channel": "#4e79a7",
    "hessian_diag_clip": "#59a14f",
    "awq_like_scale_grid": "#f28e2b",
    "smoothquant_like_scale_grid": "#edc948",
    "gptq_like_sequential": "#e15759",
    "admm_fixed_rho_10": "#b07aa1",
    "admm_task_feasibility": "#d37295",
    "admm_task_feas_aggressive": "#111827",
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


def method_table(rows: list[dict]) -> str:
    headers = [
        "method",
        "family",
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
            if h in {"deploy_rel_error", "weight_rel_error", "final_primal", "final_dual", "final_rho"}:
                number = to_float(value, math.nan)
                text = "" if not math.isfinite(number) else f"{number:.4g}"
            else:
                text = str(value)
            chunks.append(f"<td>{esc(text)}</td>")
        chunks.append("</tr>")
    chunks.append("</tbody></table>")
    return "\n".join(chunks)


def markdown_report(rows: list[dict]) -> str:
    by_method = {row["method"]: row for row in rows}
    gptq = by_method["gptq_like_sequential"]
    hessian = by_method["hessian_diag_clip"]
    rtn = by_method["rtn_per_channel"]
    admm = by_method["admm_task_feas_aggressive"]
    fixed = by_method["admm_fixed_rho_10"]
    return f"""# LLM Quantization Baseline Findings

These are local proxy implementations on the synthetic LLM PTQ calibration problem, not official GPTQ/AWQ/SmoothQuant library runs.

## Mean Deploy Relative Error

- GPTQ-like sequential Hessian compensation: `{to_float(gptq['deploy_rel_error']):.3f}`
- Hessian diagonal clipping: `{to_float(hessian['deploy_rel_error']):.3f}`
- RTN per-channel: `{to_float(rtn['deploy_rel_error']):.3f}`
- ADMM aggressive task-feasibility: `{to_float(admm['deploy_rel_error']):.3f}`
- ADMM fixed rho=10: `{to_float(fixed['deploy_rel_error']):.3f}`

## Conclusion

The current ADMM quantizer does not beat Hessian-aware PTQ. It is roughly competitive with RTN only after aggressive feasibility tuning, but GPTQ-like error compensation is clearly stronger on this calibration objective.

This does not kill the online-rho idea. It says the current projection step is too weak. The strongest next version should combine online ADMM penalty tuning with a Hessian-aware or activation-aware quantized `Z` update, rather than comparing pure uniform projection ADMM against GPTQ/AWQ directly.
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
  <title>LLM Quantization Baselines</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #111827; background: #f8fafc; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px 22px 44px; }}
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
  <h1>LLM Quantization Baselines</h1>
  <p>Local proxy baselines for synthetic transformer-block PTQ: RTN, Hessian-aware clipping, GPTQ-like compensation, AWQ-like scaling, SmoothQuant-like scaling, and ADMM variants.</p>
  {cards}
  <section>
    <h2>Grouped Metrics</h2>
    {method_table(rows)}
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    grouped_path = RESULTS / "llm_quant_baseline_grouped.csv"
    if not grouped_path.exists():
        raise SystemExit(f"Missing {grouped_path}. Run run_llm_quantization_baselines.py first.")
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
        path = VIZ / f"llm_quant_baseline_{metric}.svg"
        write_text(
            path,
            base.grouped_bar_chart(
                [{"problem": "tiny_llm_ptq", **row, metric: to_float(row[metric])} for row in filtered],
                metric,
                f"LLM PTQ Baseline {ylabel.title()}",
                ylabel,
                width=1180,
                height=520,
                log_scale=log_scale,
            ),
        )
        chart_files.append(path)
    write_text(RESULTS / "llm_quant_baseline_findings.md", markdown_report(rows))
    write_text(VIZ / "index.html", dashboard(rows, chart_files))
    print(f"Wrote LLM quantization baseline dashboard to {VIZ / 'index.html'}")


if __name__ == "__main__":
    main()
