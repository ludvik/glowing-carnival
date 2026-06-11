#!/usr/bin/env python3
"""Summarize screening resultsets into screening decisions."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.dataset import load_dataset
from eval_harness.prompt import LABELS
from eval_harness.scoring import certified_issues, load_resultset, model_scored_metrics


SUMMARY_COLUMNS = [
    "model_id",
    "result_run_id",
    "resultset_path",
    "pool_tier",
    "expected_role",
    "calls_total",
    "calls_ok",
    "calls_error",
    "success_rate",
    "valid_output_rate",
    "parsed_valid_output_rate",
    "strict_valid_output_rate",
    "invalid_label_count",
    "parse_error_count",
    "auth_error_count",
    "timeout_count",
    "rate_limit_count",
    "other_error_count",
    "dominant_error_type",
    "dominant_error_message",
    "p50_latency_ms",
    "p95_latency_ms",
    "avg_latency_ms",
    "requests_per_second",
    "total_cost_usd",
    "avg_cost_per_ok_call_usd",
    "prompt_tokens_total",
    "completion_tokens_total",
    "sanity_correct_count",
    "sanity_accuracy",
    "sample_ok_raw_output",
    "decision",
    "decision_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize screening resultsets.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset", default="data/labels/screening_corpus.jsonl")
    parser.add_argument("--screening-pool", default="config/screening_pool.csv")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--baseline-run-dir",
        action="append",
        default=[],
        help="Optional prior run-dir used to fill models missing from --run-dir.",
    )
    parser.add_argument(
        "--prefer-run-dir",
        default=None,
        help="Run-dir to prefer when duplicate model resultsets exist. Defaults to --run-dir.",
    )
    return parser.parse_args()


def load_screening_pool(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["model_id"]: row for row in csv.DictReader(handle)}


def strict_json_contract_ok(raw_output: str | None) -> bool:
    if raw_output is None:
        return False
    text = raw_output.strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and set(payload.keys()) <= {"label", "rationale"}
        and payload.get("label") in LABELS
        and isinstance(payload.get("rationale", ""), str)
    )


def error_counts(resultset: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for result in resultset.get("results", []):
        if result.get("status") == "error" and result.get("error"):
            counts[result["error"].get("type") or "other"] += 1
    return counts


def error_messages(resultset: dict[str, Any]) -> dict[str, str]:
    messages: dict[str, str] = {}
    for result in resultset.get("results", []):
        if result.get("status") == "error" and result.get("error"):
            error = result["error"]
            error_type = error.get("type") or "other"
            message = str(error.get("message") or "").replace("\n", " ").strip()
            messages.setdefault(error_type, message[:240])
    return messages


def usage_totals(resultset: dict[str, Any]) -> tuple[int | None, int | None]:
    input_total = 0
    output_total = 0
    observed = False
    for result in resultset.get("results", []):
        usage = result.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        if isinstance(input_tokens, int):
            input_total += input_tokens
            observed = True
        if isinstance(output_tokens, int):
            output_total += output_tokens
            observed = True
    if not observed:
        return None, None
    return input_total, output_total


def sample_ok_raw_output(resultset: dict[str, Any]) -> str:
    for result in resultset.get("results", []):
        if result.get("status") == "ok":
            raw_output = result.get("response", {}).get("raw_output")
            if isinstance(raw_output, str) and raw_output.strip():
                return raw_output.strip()[:500]
    return ""


def make_row(
    resultset: dict[str, Any],
    dataset: dict[str, Any],
    pool: dict[str, dict[str, str]],
    median_p95: float | None,
) -> dict[str, Any]:
    model_id = resultset["model_id"]
    operational = resultset.get("operational_summary", {})
    calls_total = operational.get("calls_total", len(resultset.get("results", [])))
    calls_ok = operational.get("calls_ok", 0)
    calls_error = operational.get("calls_error", calls_total - calls_ok)
    success_rate = calls_ok / calls_total if calls_total else None
    strict_ok = sum(
        1
        for result in resultset.get("results", [])
        if result.get("status") == "ok" and strict_json_contract_ok(result.get("response", {}).get("raw_output"))
    )
    strict_valid_output_rate = strict_ok / calls_total if calls_total else None
    errors = error_counts(resultset)
    messages = error_messages(resultset)
    dominant_error_type = ""
    dominant_error_message = ""
    if errors:
        dominant_error_type, _ = errors.most_common(1)[0]
        dominant_error_message = messages.get(dominant_error_type, "")
    scored_metrics = model_scored_metrics(resultset, certified_issues(dataset))
    prompt_tokens_total, completion_tokens_total = usage_totals(resultset)
    pool_row = pool.get(model_id, {})

    latency = operational.get("latency_ms", {})
    throughput = operational.get("throughput", {})
    cost = operational.get("cost", {})
    sanity_accuracy = scored_metrics.get("accuracy")
    decision, reason = decide(
        success_rate,
        strict_valid_output_rate,
        errors,
        latency.get("p95"),
        median_p95,
        sanity_accuracy,
    )
    return {
        "model_id": model_id,
        "result_run_id": resultset.get("run_id", ""),
        "resultset_path": resultset.get("_path", ""),
        "pool_tier": pool_row.get("pool_tier", ""),
        "expected_role": pool_row.get("expected_role", ""),
        "calls_total": calls_total,
        "calls_ok": calls_ok,
        "calls_error": calls_error,
        "success_rate": round(success_rate, 4) if success_rate is not None else None,
        "valid_output_rate": round(success_rate, 4) if success_rate is not None else None,
        "parsed_valid_output_rate": round(success_rate, 4) if success_rate is not None else None,
        "strict_valid_output_rate": round(strict_valid_output_rate, 4) if strict_valid_output_rate is not None else None,
        "invalid_label_count": errors.get("invalid_label", 0),
        "parse_error_count": errors.get("parse_error", 0),
        "auth_error_count": errors.get("auth_error", 0),
        "timeout_count": errors.get("timeout", 0),
        "rate_limit_count": errors.get("rate_limit", 0),
        "other_error_count": sum(
            count
            for error_type, count in errors.items()
            if error_type not in {"invalid_label", "parse_error", "auth_error", "timeout", "rate_limit"}
        ),
        "dominant_error_type": dominant_error_type,
        "dominant_error_message": dominant_error_message,
        "p50_latency_ms": latency.get("p50"),
        "p95_latency_ms": latency.get("p95"),
        "avg_latency_ms": latency.get("avg"),
        "requests_per_second": throughput.get("requests_per_second"),
        "total_cost_usd": cost.get("total_cost_usd"),
        "avg_cost_per_ok_call_usd": cost.get("avg_cost_per_ok_call_usd"),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "sanity_correct_count": scored_metrics.get("correct_count"),
        "sanity_accuracy": round(sanity_accuracy, 4) if sanity_accuracy is not None else None,
        "sample_ok_raw_output": sample_ok_raw_output(resultset),
        "decision": decision,
        "decision_reason": reason,
    }


def decide(
    success_rate: float | None,
    valid_output_rate: float | None,
    errors: Counter[str],
    p95_latency_ms: float | None,
    median_p95_latency_ms: float | None,
    sanity_accuracy: float | None,
) -> tuple[str, str]:
    if success_rate is None or valid_output_rate is None:
        return "needs_review", "No calls found in resultset."
    if errors.get("permission") or errors.get("unsupported_model") or errors.get("auth_error"):
        return "reject", "Provider/model permission or support errors observed."
    if errors.get("parse_error") and valid_output_rate < 0.80:
        return "reject", "Systematic output-contract parse failures observed."
    if errors.get("invalid_label") and valid_output_rate < 0.80:
        return "reject", "Systematic invalid-label failures observed."
    if success_rate < 0.80 or valid_output_rate < 0.80:
        return "reject", "Reliability or strict output-contract rate is below 0.80."
    if valid_output_rate < 0.90:
        return "pass_to_pilot_with_caution", "Strict output-contract rate is below 0.90."
    if p95_latency_ms is not None and p95_latency_ms > 30000:
        return "needs_review", "p95 latency exceeds 30000ms."
    if (
        p95_latency_ms is not None
        and median_p95_latency_ms is not None
        and median_p95_latency_ms > 0
        and p95_latency_ms > 3 * median_p95_latency_ms
    ):
        return "needs_review", "p95 latency is more than 3x the median successful-model p95."
    if success_rate >= 0.90 and valid_output_rate >= 0.90 and not errors:
        if sanity_accuracy is not None and sanity_accuracy < 0.50:
            return "pass_to_pilot_with_caution", "Reliable output contract, but low screening sanity accuracy."
        return "pass_to_pilot", "Reliable calls and parsed output contract."
    return "pass_to_pilot_with_caution", "Meets minimum screening reliability with minor errors or latency concerns."


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in SUMMARY_COLUMNS})


def load_resultsets_by_model(run_dir: Path) -> dict[str, dict[str, Any]]:
    result_dir = run_dir / "results"
    if not result_dir.exists():
        raise SystemExit(f"Result directory not found: {result_dir}")
    resultsets = {}
    for path in sorted(result_dir.glob("*.json")):
        resultset = load_resultset(path)
        resultsets[resultset["model_id"]] = resultset
    return resultsets


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    prefer_run_dir = Path(args.prefer_run_dir) if args.prefer_run_dir else run_dir
    output_dir = Path(args.output_dir) if args.output_dir else run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(Path(args.dataset))
    pool = load_screening_pool(Path(args.screening_pool))

    primary = load_resultsets_by_model(run_dir)
    if not primary:
        raise SystemExit(f"No resultsets found under {run_dir / 'results'}")
    combined = dict(primary)
    baseline_count = 0
    for baseline_run_dir in args.baseline_run_dir:
        baseline_path = Path(baseline_run_dir)
        baseline = load_resultsets_by_model(baseline_path)
        if prefer_run_dir == baseline_path:
            before = len(combined)
            combined = {**combined, **baseline}
            baseline_count += max(0, len(combined) - before)
        else:
            for model_id, resultset in baseline.items():
                if model_id not in combined:
                    combined[model_id] = resultset
                    baseline_count += 1
    resultsets = list(combined.values())
    p95_values = [
        resultset.get("operational_summary", {}).get("latency_ms", {}).get("p95")
        for resultset in resultsets
        if resultset.get("operational_summary", {}).get("calls_ok", 0) > 0
        and resultset.get("operational_summary", {}).get("latency_ms", {}).get("p95") is not None
    ]
    median_p95 = median(p95_values) if p95_values else None
    rows = [make_row(resultset, dataset, pool, median_p95) for resultset in resultsets]
    rows.sort(key=lambda row: (row["decision"], row["model_id"]))

    write_csv(output_dir / "screening_summary.csv", rows)
    write_csv(output_dir / "screening_passed_models.csv", [r for r in rows if r["decision"] in {"pass_to_pilot", "pass_to_pilot_with_caution"}])
    write_csv(output_dir / "screening_rejected_models.csv", [r for r in rows if r["decision"] == "reject"])
    write_csv(output_dir / "screening_needs_review.csv", [r for r in rows if r["decision"] == "needs_review"])

    distribution = Counter(
        issue["ground_truth"]["label"]
        for issue in certified_issues(dataset)
        if issue.get("ground_truth", {}).get("label")
    )
    payload = {
        "run_id": run_dir.name,
        "baseline_run_dirs": args.baseline_run_dir,
        "prefer_run_dir": str(prefer_run_dir),
        "dataset_path": args.dataset,
        "issue_count": len(dataset["issues"]),
        "screening_issue_distribution": dict(sorted(distribution.items())),
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "primary_resultset_count": len(primary),
        "baseline_fill_count": baseline_count,
        "models": rows,
    }
    (output_dir / "screening_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote screening summaries to {output_dir}")
    print(f"Models summarized: {len(rows)}")
    if args.baseline_run_dir:
        print(f"Baseline fill count: {baseline_count}")
    print("Decisions:")
    for decision, count in Counter(row["decision"] for row in rows).most_common():
        print(f"  {decision}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
