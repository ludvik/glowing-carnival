from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from eval_harness.prompt import LABELS
from eval_harness.scoring import (
    certified_issues,
    issue_map,
    model_scored_metrics,
    result_map,
    unscored_analysis,
    uncertified_issues,
)

CRITICAL_LABELS = {"bug", "security"}
SECURITY_KEYWORDS = ("cve", "vulnerability", "credential", "token", "secret", "auth", "plaintext", "leak")


def discover_runs(root: Path = Path("runs")) -> list[Path]:
    return sorted(root.glob("*/run.json"), key=lambda path: path.stat().st_mtime, reverse=True)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resultset_paths(run_dir: Path) -> list[Path]:
    return sorted((run_dir / "results").glob("*.json"))


def stable_macro_f1(metrics: dict[str, Any], min_support: int = 10) -> float | None:
    values = [
        row.get("f1")
        for row in metrics.get("per_class", {}).values()
        if row.get("support", 0) >= min_support and row.get("f1") is not None
    ]
    return sum(values) / len(values) if values else None


def all_class_macro_f1(metrics: dict[str, Any]) -> float | None:
    values = [
        row.get("f1")
        for row in metrics.get("per_class", {}).values()
        if row.get("f1") is not None
    ]
    return sum(values) / len(values) if values else None


def end_to_end_accuracy(metrics: dict[str, Any]) -> float | None:
    scored_count = metrics.get("scored_issue_count") or 0
    return metrics.get("correct_count", 0) / scored_count if scored_count else None


def strict_output_contract_summary(resultset: dict[str, Any]) -> dict[str, Any]:
    total = len(resultset.get("results", []))
    parsed_ok = sum(1 for result in resultset.get("results", []) if result.get("status") == "ok")
    strict_ok = 0
    for result in resultset.get("results", []):
        raw = result.get("response", {}).get("raw_output")
        if result.get("status") == "ok" and isinstance(raw, str) and is_strict_json_output(raw):
            strict_ok += 1
    return {
        "calls_total": total,
        "parsed_valid_count": parsed_ok,
        "parsed_valid_rate": parsed_ok / total if total else None,
        "strict_valid_count": strict_ok,
        "strict_valid_rate": strict_ok / total if total else None,
    }


def is_strict_json_output(raw: str) -> bool:
    text = raw.strip()
    if not (text.startswith("{") and text.endswith("}")):
        return False
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(payload, dict)
        and payload.get("label") in LABELS
        and set(payload.keys()) <= {"label", "rationale"}
    )


def compare_two_models(dataset: dict[str, Any], result_a: dict[str, Any], result_b: dict[str, Any]) -> dict[str, Any]:
    scored = certified_issues(dataset)
    unscored = uncertified_issues(dataset)
    metrics_a = model_scored_metrics(result_a, scored)
    metrics_b = model_scored_metrics(result_b, scored)
    return {
        "scored": scored,
        "unscored": unscored,
        "metrics": {result_a["model_id"]: metrics_a, result_b["model_id"]: metrics_b},
        "unscored_analysis": unscored_analysis([result_a, result_b], unscored),
        "strict_contract": {
            result_a["model_id"]: strict_output_contract_summary(result_a),
            result_b["model_id"]: strict_output_contract_summary(result_b),
        },
    }


def executive_comparison_table(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    metrics_by_model: dict[str, dict[str, Any]],
    strict_by_model: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = []
    for resultset in (result_a, result_b):
        model_id = resultset["model_id"]
        metrics = metrics_by_model[model_id]
        operational = resultset.get("operational_summary", {})
        latency = operational.get("latency_ms", {})
        cost = operational.get("cost", {})
        throughput = operational.get("throughput", {})
        per_class = metrics.get("per_class", {})
        rows.append(
            {
                "model_id": model_id,
                "scored_issue_count": metrics.get("scored_issue_count"),
                "evaluated_count": metrics.get("evaluated_count"),
                "correct_count": metrics.get("correct_count"),
                "evaluated_accuracy": metrics.get("accuracy"),
                "end_to_end_scored_accuracy": end_to_end_accuracy(metrics),
                "stable_macro_f1": stable_macro_f1(metrics),
                "all_class_macro_f1": all_class_macro_f1(metrics),
                "bug_recall": (per_class.get("bug") or {}).get("recall"),
                "security_recall": (per_class.get("security") or {}).get("recall"),
                "strict_output_valid_rate": strict_by_model[model_id].get("strict_valid_rate"),
                "parsed_output_valid_rate": strict_by_model[model_id].get("parsed_valid_rate"),
                "total_scored_cost_usd": metrics.get("cost", {}).get("total_scored_eval_cost_usd"),
                "cost_per_correct_usd": metrics.get("cost", {}).get("cost_per_correct_classification_usd"),
                "total_run_cost_usd": cost.get("total_cost_usd"),
                "avg_cost_per_ok_call_usd": cost.get("avg_cost_per_ok_call_usd"),
                "p50_latency_ms": latency.get("p50"),
                "p95_latency_ms": latency.get("p95"),
                "error_rate": operational.get("error_rate"),
                "requests_per_second": throughput.get("requests_per_second"),
            }
        )
    return pd.DataFrame(rows)


def side_by_side_per_class_table(
    model_a: str,
    model_b: str,
    metrics_a: dict[str, Any],
    metrics_b: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    for label in LABELS:
        row_a = metrics_a.get("per_class", {}).get(label, {})
        row_b = metrics_b.get("per_class", {}).get(label, {})
        rows.append(
            {
                "label": label,
                "support": row_a.get("support", row_b.get("support")),
                f"{model_a} precision": row_a.get("precision"),
                f"{model_a} recall": row_a.get("recall"),
                f"{model_a} f1": row_a.get("f1"),
                f"{model_b} precision": row_b.get("precision"),
                f"{model_b} recall": row_b.get("recall"),
                f"{model_b} f1": row_b.get("f1"),
                "delta recall": none_sub(row_b.get("recall"), row_a.get("recall")),
                "delta f1": none_sub(row_b.get("f1"), row_a.get("f1")),
            }
        )
    return pd.DataFrame(rows)


def none_sub(b: float | None, a: float | None) -> float | None:
    return None if a is None or b is None else b - a


def confusion_matrix_dataframe(metrics: dict[str, Any], normalized: bool = False) -> pd.DataFrame:
    matrix = metrics.get("confusion_matrix", {})
    frame = pd.DataFrame(matrix).T.reindex(index=LABELS, columns=LABELS).fillna(0)
    if normalized:
        frame = frame.div(frame.sum(axis=1).replace(0, pd.NA), axis=0).fillna(0)
    return frame


def scored_case_table(dataset: dict[str, Any], result_a: dict[str, Any], result_b: dict[str, Any]) -> pd.DataFrame:
    return case_table(certified_issues(dataset), result_a, result_b, include_truth=True)


def unscored_case_table(dataset: dict[str, Any], result_a: dict[str, Any], result_b: dict[str, Any]) -> pd.DataFrame:
    return case_table(uncertified_issues(dataset), result_a, result_b, include_truth=False)


def case_table(issues: list[dict[str, Any]], result_a: dict[str, Any], result_b: dict[str, Any], include_truth: bool) -> pd.DataFrame:
    map_a = result_map(result_a)
    map_b = result_map(result_b)
    columns = [
        "issue_number",
        "title",
        "html_url",
        "ground_truth",
        f"{result_a['model_id']} label",
        f"{result_b['model_id']} label",
        "A correct",
        "B correct",
        "agree",
        "models_disagree",
        "critical_truth",
        "security_keyword",
        "A status",
        "B status",
        "A latency_ms",
        "B latency_ms",
        "A cost_usd",
        "B cost_usd",
        "A raw_output",
        "B raw_output",
        "A rationale",
        "B rationale",
        "A error",
        "B error",
        "body_excerpt",
    ]
    rows = []
    for issue in issues:
        issue_number = int(issue["issue_number"])
        row_a = map_a.get(issue_number)
        row_b = map_b.get(issue_number)
        pred_a = prediction_label(row_a)
        pred_b = prediction_label(row_b)
        truth = issue.get("ground_truth", {}).get("label") if include_truth else None
        rows.append(
            {
                "issue_number": issue_number,
                "title": issue.get("title"),
                "html_url": issue.get("html_url"),
                "ground_truth": truth,
                f"{result_a['model_id']} label": pred_a,
                f"{result_b['model_id']} label": pred_b,
                "A correct": pred_a == truth if include_truth and row_a else None,
                "B correct": pred_b == truth if include_truth and row_b else None,
                "agree": pred_a == pred_b and pred_a is not None,
                "models_disagree": pred_a != pred_b,
                "critical_truth": truth in CRITICAL_LABELS if include_truth else False,
                "security_keyword": security_keyword_flag(issue),
                "A status": (row_a or {}).get("status", "missing"),
                "B status": (row_b or {}).get("status", "missing"),
                "A latency_ms": (row_a or {}).get("timing", {}).get("total_latency_ms"),
                "B latency_ms": (row_b or {}).get("timing", {}).get("total_latency_ms"),
                "A cost_usd": (row_a or {}).get("cost", {}).get("total_cost_usd"),
                "B cost_usd": (row_b or {}).get("cost", {}).get("total_cost_usd"),
                "A raw_output": (row_a or {}).get("response", {}).get("raw_output"),
                "B raw_output": (row_b or {}).get("response", {}).get("raw_output"),
                "A rationale": (row_a or {}).get("response", {}).get("rationale"),
                "B rationale": (row_b or {}).get("response", {}).get("rationale"),
                "A error": json.dumps((row_a or {}).get("error"), ensure_ascii=False),
                "B error": json.dumps((row_b or {}).get("error"), ensure_ascii=False),
                "body_excerpt": (issue.get("body") or "")[:1200],
            }
        )
    return pd.DataFrame(rows, columns=columns)


def prediction_label(result: dict[str, Any] | None) -> str | None:
    if result is None:
        return "missing"
    if result.get("status") != "ok":
        return "error"
    return result.get("response", {}).get("parsed_label") or "error"


def security_keyword_flag(issue: dict[str, Any]) -> bool:
    text = f"{issue.get('title') or ''}\n{issue.get('body') or ''}".lower()
    return any(keyword in text for keyword in SECURITY_KEYWORDS)


def prediction_distribution_table(dataset: dict[str, Any], result_a: dict[str, Any], result_b: dict[str, Any]) -> pd.DataFrame:
    analysis = unscored_analysis([result_a, result_b], uncertified_issues(dataset))
    rows = []
    for label in (*LABELS, "error", "missing"):
        row = {"label": label}
        for model_id, distribution in analysis.get("prediction_distribution", {}).items():
            row[model_id] = distribution.get(label, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def operational_summary_table(resultsets: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for resultset in resultsets:
        summary = resultset.get("operational_summary", {})
        latency = summary.get("latency_ms", {})
        throughput = summary.get("throughput", {})
        cost = summary.get("cost", {})
        retries = summary.get("retries", {})
        rows.append(
            {
                "model_id": resultset["model_id"],
                "calls_total": summary.get("calls_total"),
                "calls_ok": summary.get("calls_ok"),
                "calls_error": summary.get("calls_error"),
                "error_rate": summary.get("error_rate"),
                "errors_by_type": json.dumps(summary.get("errors_by_type", {}), sort_keys=True),
                "retry_events_total": retries.get("total_retry_events"),
                "rate_limit_retry_events": retries.get("rate_limit_retry_events"),
                "retry_events_by_type": json.dumps(retries.get("retry_events_by_type", {}), sort_keys=True),
                "p50_latency_ms": latency.get("p50"),
                "p95_latency_ms": latency.get("p95"),
                "avg_latency_ms": latency.get("avg"),
                "wall_clock_ms": throughput.get("wall_clock_ms"),
                "concurrency": throughput.get("concurrency"),
                "requests_per_second": throughput.get("requests_per_second"),
                "total_cost_usd": cost.get("total_cost_usd"),
                "avg_cost_per_ok_call_usd": cost.get("avg_cost_per_ok_call_usd"),
            }
        )
    return pd.DataFrame(rows)


def token_cost_trace(resultsets: list[dict[str, Any]], catalog: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for resultset in resultsets:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        missing_usage = 0
        for result in resultset.get("results", []):
            usage = result.get("usage")
            if not usage:
                missing_usage += 1
                continue
            prompt_tokens += usage.get("input_tokens") or 0
            completion_tokens += usage.get("output_tokens") or 0
            total_tokens += usage.get("total_tokens") or 0
        model = catalog.get("models", {}).get(resultset["model_id"], {})
        input_price = model.get("input_price_per_1m_tokens")
        output_price = model.get("output_price_per_1m_tokens")
        calculated = None
        if input_price is not None and output_price is not None:
            calculated = prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price
        rows.append(
            {
                "model_id": resultset["model_id"],
                "input_price_per_1m": input_price,
                "output_price_per_1m": output_price,
                "prompt_tokens_total": prompt_tokens,
                "completion_tokens_total": completion_tokens,
                "total_tokens": total_tokens,
                "missing_usage_count": missing_usage,
                "calculated_total_cost_usd": calculated,
                "formula": "prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price",
            }
        )
    return pd.DataFrame(rows)


def scored_label_distribution(dataset: dict[str, Any]) -> pd.DataFrame:
    counts = Counter(issue["ground_truth"]["label"] for issue in certified_issues(dataset))
    return pd.DataFrame([{"label": label, "count": counts.get(label, 0)} for label in LABELS])


def validate_same_issue_set(result_a: dict[str, Any], result_b: dict[str, Any]) -> dict[str, Any]:
    a = {int(row["issue_number"]) for row in result_a.get("results", [])}
    b = {int(row["issue_number"]) for row in result_b.get("results", [])}
    return {
        "a_count": len(a),
        "b_count": len(b),
        "shared_count": len(a & b),
        "a_missing_from_b": sorted(a - b),
        "b_missing_from_a": sorted(b - a),
    }
