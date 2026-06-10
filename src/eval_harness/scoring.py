from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from eval_harness.prompt import LABELS


def load_resultset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["_path"] = str(path)
    return payload


def result_map(resultset: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(result["issue_number"]): result for result in resultset.get("results", [])}


def issue_map(dataset: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(issue["issue_number"]): issue for issue in dataset["issues"]}


def certified_issues(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        issue
        for issue in dataset["issues"]
        if issue.get("ground_truth", {}).get("status") == "certified"
        and issue.get("ground_truth", {}).get("label") in LABELS
    ]


def uncertified_issues(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        issue
        for issue in dataset["issues"]
        if issue.get("ground_truth", {}).get("status") != "certified"
        or issue.get("ground_truth", {}).get("label") not in LABELS
    ]


def model_scored_metrics(
    resultset: dict[str, Any],
    scored_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    results = result_map(resultset)
    confusion = empty_confusion_matrix()
    per_class_counts = {
        label: {"tp": 0, "fp": 0, "fn": 0}
        for label in LABELS
    }
    evaluated = 0
    correct = 0
    missing = 0
    errored = 0
    invalid = 0
    evaluated_costs: list[float] = []
    errors: list[dict[str, Any]] = []

    for issue in scored_issues:
        issue_number = int(issue["issue_number"])
        truth = issue["ground_truth"]["label"]
        result = results.get(issue_number)
        if result is None:
            missing += 1
            continue
        if result.get("status") != "ok":
            errored += 1
            errors.append(issue_error_row(issue, result))
            continue

        predicted = result.get("response", {}).get("parsed_label")
        if predicted not in LABELS:
            invalid += 1
            errors.append(issue_error_row(issue, result))
            continue

        evaluated += 1
        confusion[truth][predicted] += 1
        cost = result.get("cost", {}).get("total_cost_usd")
        if isinstance(cost, int | float):
            evaluated_costs.append(float(cost))
        if predicted == truth:
            correct += 1

    for label in LABELS:
        per_class_counts[label]["tp"] = confusion[label][label]
        per_class_counts[label]["fp"] = sum(
            confusion[actual][label] for actual in LABELS if actual != label
        )
        per_class_counts[label]["fn"] = sum(
            confusion[label][predicted] for predicted in LABELS if predicted != label
        )

    per_class = {
        label: precision_recall_f1(counts)
        | {
            "support": sum(confusion[label].values()),
            "true_positive": counts["tp"],
            "false_positive": counts["fp"],
            "false_negative": counts["fn"],
        }
        for label, counts in per_class_counts.items()
    }

    total_cost = sum(evaluated_costs) if evaluated_costs else None
    return {
        "model_id": resultset["model_id"],
        "resultset_path": resultset.get("_path"),
        "scored_issue_count": len(scored_issues),
        "evaluated_count": evaluated,
        "correct_count": correct,
        "missing_count": missing,
        "errored_count": errored,
        "invalid_count": invalid,
        "accuracy": correct / evaluated if evaluated else None,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "cost": {
            "total_scored_eval_cost_usd": total_cost,
            "avg_scored_eval_cost_per_call_usd": total_cost / evaluated
            if total_cost is not None and evaluated
            else None,
            "cost_per_correct_classification_usd": total_cost / correct
            if total_cost is not None and correct
            else None,
        },
        "errors": errors,
    }


def scored_disagreements(
    resultsets: list[dict[str, Any]],
    scored_issues: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(resultsets) < 2:
        return []
    maps = {resultset["model_id"]: result_map(resultset) for resultset in resultsets}
    rows = []
    for issue in scored_issues:
        predictions = {
            model_id: prediction_payload(results.get(int(issue["issue_number"])))
            for model_id, results in maps.items()
        }
        labels = {
            payload["parsed_label"]
            for payload in predictions.values()
            if payload["parsed_label"] is not None
        }
        if len(labels) > 1:
            rows.append(issue_comparison_row(issue, predictions))
    return rows


def unscored_analysis(
    resultsets: list[dict[str, Any]],
    unscored_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    maps = {resultset["model_id"]: result_map(resultset) for resultset in resultsets}
    distributions = {
        resultset["model_id"]: prediction_distribution(resultset, unscored_issues)
        for resultset in resultsets
    }
    rows = []
    agree = 0
    comparable = 0
    for issue in unscored_issues:
        predictions = {
            model_id: prediction_payload(results.get(int(issue["issue_number"])))
            for model_id, results in maps.items()
        }
        parsed_labels = [
            payload["parsed_label"]
            for payload in predictions.values()
            if payload["status"] == "ok" and payload["parsed_label"] is not None
        ]
        if len(parsed_labels) == len(resultsets) and len(parsed_labels) > 1:
            comparable += 1
            if len(set(parsed_labels)) == 1:
                agree += 1
            else:
                rows.append(issue_comparison_row(issue, predictions))
    return {
        "unscored_issue_count": len(unscored_issues),
        "comparable_issue_count": comparable,
        "agreement_count": agree,
        "agreement_rate": agree / comparable if comparable else None,
        "prediction_distribution": distributions,
        "disagreements": rows,
    }


def operational_metrics(resultsets: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        resultset["model_id"]: {
            "resultset_path": resultset.get("_path"),
            "provider_model": resultset.get("provider_model"),
            "created_at": resultset.get("created_at"),
            "completed_at": resultset.get("completed_at"),
            "wall_clock_ms": resultset.get("wall_clock_ms"),
            "result_count": resultset.get("result_count"),
            "operational_summary": resultset.get("operational_summary"),
        }
        for resultset in resultsets
    }


def prediction_distribution(
    resultset: dict[str, Any],
    issues: list[dict[str, Any]],
) -> dict[str, int]:
    results = result_map(resultset)
    counts: Counter[str] = Counter()
    for issue in issues:
        result = results.get(int(issue["issue_number"]))
        label = (result or {}).get("response", {}).get("parsed_label")
        if label in LABELS:
            counts[label] += 1
        elif result is None:
            counts["missing"] += 1
        else:
            counts["error"] += 1
    return {label: counts.get(label, 0) for label in (*LABELS, "error", "missing")}


def empty_confusion_matrix() -> dict[str, dict[str, int]]:
    return {actual: {predicted: 0 for predicted in LABELS} for actual in LABELS}


def precision_recall_f1(counts: dict[str, int]) -> dict[str, float | None]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def prediction_payload(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {
            "status": "missing",
            "parsed_label": None,
            "rationale": None,
            "raw_output": None,
            "error": None,
            "cost_usd": None,
            "latency_ms": None,
        }
    return {
        "status": result.get("status"),
        "parsed_label": result.get("response", {}).get("parsed_label"),
        "rationale": result.get("response", {}).get("rationale"),
        "raw_output": result.get("response", {}).get("raw_output"),
        "error": result.get("error"),
        "cost_usd": result.get("cost", {}).get("total_cost_usd"),
        "latency_ms": result.get("timing", {}).get("total_latency_ms"),
    }


def issue_comparison_row(
    issue: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "issue_number": issue["issue_number"],
        "title": issue["title"],
        "html_url": issue.get("html_url"),
        "ground_truth": issue.get("ground_truth"),
        "predictions": predictions,
    }


def issue_error_row(issue: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return {
        "issue_number": issue["issue_number"],
        "title": issue["title"],
        "ground_truth": issue.get("ground_truth"),
        "status": result.get("status"),
        "error": result.get("error"),
        "raw_output": result.get("response", {}).get("raw_output"),
    }
