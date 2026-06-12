#!/usr/bin/env python3
"""Build a quality-vs-cost tradeoff table and SVG plot from full resultsets."""

from __future__ import annotations

import csv
import html
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.comparison import all_class_macro_f1, end_to_end_accuracy, stable_macro_f1
from eval_harness.dataset import load_dataset
from eval_harness.scoring import certified_issues, load_resultset, model_scored_metrics


FULL_ISSUE_COUNT = 530


def full_resultsets(runs_dir: Path) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for path in sorted(runs_dir.glob("*/results/*.json")):
        resultset = load_resultset(path)
        result_count = resultset.get("result_count") or len(resultset.get("results", []))
        if result_count < FULL_ISSUE_COUNT:
            continue
        model_id = resultset["model_id"]
        current = selected.get(model_id)
        if current is None or resultset_rank(resultset) > resultset_rank(current):
            selected[model_id] = resultset
    return selected


def resultset_rank(resultset: dict[str, Any]) -> tuple[int, int, float]:
    operational = resultset.get("operational_summary", {})
    calls_ok = int(operational.get("calls_ok") or 0)
    calls_error = int(operational.get("calls_error") or 0)
    wall_clock_ms = float(operational.get("throughput", {}).get("wall_clock_ms") or 10**12)
    return (calls_ok, -calls_error, -wall_clock_ms)


def build_rows(dataset_path: Path, runs_dir: Path) -> list[dict[str, Any]]:
    dataset = load_dataset(dataset_path)
    scored = certified_issues(dataset)
    rows: list[dict[str, Any]] = []
    for model_id, resultset in sorted(full_resultsets(runs_dir).items()):
        metrics = model_scored_metrics(resultset, scored)
        operational = resultset.get("operational_summary", {})
        cost = operational.get("cost", {})
        latency = operational.get("latency_ms", {})
        throughput = operational.get("throughput", {})
        retries = operational.get("retries", {})
        rows.append(
            {
                "model_id": model_id,
                "resultset_path": resultset.get("_path"),
                "calls_total": operational.get("calls_total"),
                "calls_ok": operational.get("calls_ok"),
                "calls_error": operational.get("calls_error"),
                "error_rate": operational.get("error_rate"),
                "evaluated_count": metrics.get("evaluated_count"),
                "correct_count": metrics.get("correct_count"),
                "evaluated_accuracy": metrics.get("accuracy"),
                "end_to_end_scored_accuracy": end_to_end_accuracy(metrics),
                "stable_macro_f1": stable_macro_f1(metrics),
                "all_class_macro_f1": all_class_macro_f1(metrics),
                "bug_recall": (metrics.get("per_class", {}).get("bug") or {}).get("recall"),
                "security_recall": (metrics.get("per_class", {}).get("security") or {}).get("recall"),
                "total_success_cost_usd": cost.get("total_cost_usd"),
                "avg_cost_per_ok_call_usd": cost.get("avg_cost_per_ok_call_usd"),
                "cost_per_correct_usd": metrics.get("cost", {}).get("cost_per_correct_classification_usd"),
                "p50_latency_ms": latency.get("p50"),
                "p95_latency_ms": latency.get("p95"),
                "requests_per_second": throughput.get("requests_per_second"),
                "rate_limit_retry_events": retries.get("rate_limit_retry_events"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "model_id",
        "stable_macro_f1",
        "all_class_macro_f1",
        "evaluated_accuracy",
        "end_to_end_scored_accuracy",
        "bug_recall",
        "security_recall",
        "total_success_cost_usd",
        "avg_cost_per_ok_call_usd",
        "cost_per_correct_usd",
        "p95_latency_ms",
        "requests_per_second",
        "calls_total",
        "calls_ok",
        "calls_error",
        "error_rate",
        "rate_limit_retry_events",
        "evaluated_count",
        "correct_count",
        "resultset_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def write_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    width, height = 1280, 760
    left, right, top, bottom = 90, 260, 70, 100
    plot_w = width - left - right
    plot_h = height - top - bottom
    known = [row for row in rows if is_number(row.get("avg_cost_per_ok_call_usd")) and is_number(row.get("stable_macro_f1"))]
    unknown = [row for row in rows if row not in known and is_number(row.get("stable_macro_f1"))]
    max_cost = max([float(row["avg_cost_per_ok_call_usd"]) for row in known] or [1.0])
    max_cost *= 1.1

    def x(cost: float) -> float:
        return left + (cost / max_cost) * plot_w

    def y(f1: float) -> float:
        return top + (1.0 - f1) * plot_h

    def color(row: dict[str, Any]) -> str:
        err = float(row.get("error_rate") or 0)
        p95 = row.get("p95_latency_ms")
        if err > 0.05:
            return "#dc2626"
        if p95 is not None and float(p95) > 5000:
            return "#f59e0b"
        if str(row["model_id"]).startswith("router:"):
            return "#7c3aed"
        return "#0073ea"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f7fb"/>',
        '<text x="40" y="38" font-family="Arial" font-size="24" font-weight="700" fill="#182033">Full-run model tradeoff: quality vs average classification cost</text>',
        '<text x="40" y="60" font-family="Arial" font-size="13" fill="#667085">Y = stable macro F1 on scored set. X = average cost per successful classification. Color flags latency/error risk; routers without pricing are shown separately.</text>',
    ]
    # axes
    lines += [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155" stroke-width="1"/>',
    ]
    for tick in range(0, 6):
        value = tick / 5
        yy = y(value)
        lines.append(f'<line x1="{left-5}" y1="{yy:.1f}" x2="{left + plot_w}" y2="{yy:.1f}" stroke="#d9e0ea" stroke-width="1"/>')
        lines.append(f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" font-family="Arial" font-size="12" fill="#475467">{value:.1f}</text>')
    for tick in range(0, 6):
        cost = max_cost * tick / 5
        xx = x(cost)
        lines.append(f'<line x1="{xx:.1f}" y1="{top + plot_h}" x2="{xx:.1f}" y2="{top + plot_h + 5}" stroke="#334155"/>')
        lines.append(f'<text x="{xx:.1f}" y="{top + plot_h + 24}" text-anchor="middle" font-family="Arial" font-size="12" fill="#475467">${cost:.6f}</text>')
    lines.append(f'<text x="{left + plot_w/2}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#182033">Avg cost per classification, USD</text>')
    lines.append(f'<text x="24" y="{top + plot_h/2}" transform="rotate(-90 24 {top + plot_h/2})" text-anchor="middle" font-family="Arial" font-size="14" font-weight="700" fill="#182033">Stable macro F1</text>')

    for row in known:
        xx = x(float(row["avg_cost_per_ok_call_usd"]))
        yy = y(float(row["stable_macro_f1"]))
        radius = 6 + min(float(row.get("p95_latency_ms") or 0) / 3000, 8)
        label = html.escape(str(row["model_id"]))
        lines.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="{radius:.1f}" fill="{color(row)}" fill-opacity="0.82" stroke="#111827" stroke-width="0.6"><title>{label}\\nF1={row["stable_macro_f1"]:.3f}\\nAvg cost=${row["avg_cost_per_ok_call_usd"]:.6f}\\np95={row.get("p95_latency_ms")}</title></circle>')
        lines.append(f'<text x="{xx+9:.1f}" y="{yy-8:.1f}" font-family="Arial" font-size="11" fill="#111827">{label}</text>')

    # cost-unavailable lane
    ux = left + plot_w + 70
    lines.append(f'<line x1="{left + plot_w + 35}" y1="{top}" x2="{left + plot_w + 35}" y2="{top + plot_h}" stroke="#94a3b8" stroke-dasharray="4 4"/>')
    lines.append(f'<text x="{ux}" y="{top - 8}" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="#475467">Cost unavailable</text>')
    for index, row in enumerate(sorted(unknown, key=lambda r: float(r["stable_macro_f1"]), reverse=True)):
        yy = y(float(row["stable_macro_f1"]))
        label = html.escape(str(row["model_id"]))
        jitter = (index % 3 - 1) * 18
        lines.append(f'<circle cx="{ux+jitter:.1f}" cy="{yy:.1f}" r="8" fill="{color(row)}" fill-opacity="0.82" stroke="#111827" stroke-width="0.6"><title>{label}\\nF1={row["stable_macro_f1"]:.3f}\\nCost unavailable\\np95={row.get("p95_latency_ms")}</title></circle>')
        lines.append(f'<text x="{ux+20+jitter:.1f}" y="{yy+4:.1f}" font-family="Arial" font-size="11" fill="#111827">{label}</text>')

    legend_y = height - 72
    for i, (c, text) in enumerate([
        ("#0073ea", "normal"),
        ("#f59e0b", "p95 > 5s"),
        ("#dc2626", "error rate > 5%"),
        ("#7c3aed", "router"),
    ]):
        lx = left + i * 150
        lines.append(f'<circle cx="{lx}" cy="{legend_y}" r="6" fill="{c}"/>')
        lines.append(f'<text x="{lx+12}" y="{legend_y+4}" font-family="Arial" font-size="12" fill="#475467">{text}</text>')

    lines.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and value == value


def main() -> int:
    dataset_path = Path("data/labels/classification_corpus.jsonl")
    out_dir = Path("analysis/model_tradeoff")
    rows = build_rows(dataset_path, Path("runs"))
    rows.sort(
        key=lambda row: (
            row.get("stable_macro_f1") is not None,
            row.get("stable_macro_f1") or -1,
            -(row.get("total_success_cost_usd") or 999),
        ),
        reverse=True,
    )
    write_csv(out_dir / "full_run_model_tradeoff.csv", rows)
    write_svg(out_dir / "quality_vs_cost.svg", rows)
    print(f"Full-run models: {len(rows)}")
    print(f"Wrote {out_dir / 'full_run_model_tradeoff.csv'}")
    print(f"Wrote {out_dir / 'quality_vs_cost.svg'}")
    for row in rows:
        print(
            f"{row['model_id']}: stable_f1={row['stable_macro_f1']} "
            f"avg_cost={row['avg_cost_per_ok_call_usd']} p95={row['p95_latency_ms']} "
            f"ok={row['calls_ok']}/{row['calls_total']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
