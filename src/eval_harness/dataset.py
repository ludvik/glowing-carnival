from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_dataset(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues = normalize_issues(payload)
    dataset_id = payload.get("dataset_id") or payload.get("source", {}).get("repo", "dataset")
    return {
        "dataset_id": dataset_id,
        "source": payload.get("source", {}),
        "path": str(path),
        "issues": issues,
    }


def normalize_issues(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if "issues" in payload:
        return [normalize_issue(issue, None) for issue in payload["issues"]]

    issues: list[dict[str, Any]] = []
    for issue in payload.get("scored", []):
        issues.append(normalize_issue(issue, issue.get("ground_truth_label")))
    for issue in payload.get("unscored", []):
        issues.append(normalize_issue(issue, None))
    for issue in payload.get("needs_review", []):
        normalized = normalize_issue(issue, None)
        normalized["ground_truth"]["status"] = "needs_adjudication"
        issues.append(normalized)

    issues.sort(key=lambda issue: issue["issue_number"])
    return issues


def normalize_issue(issue: dict[str, Any], label: str | None) -> dict[str, Any]:
    ground_truth = issue.get("ground_truth")
    if not ground_truth:
        ground_truth = {
            "status": "certified" if label else "uncertified",
            "label": label,
            "method": issue.get("ground_truth_source"),
            "confidence": "medium" if label else None,
            "rationale": issue.get("ground_truth_rationale"),
        }

    return {
        "issue_number": issue.get("issue_number", issue.get("number")),
        "title": issue.get("title") or "",
        "body": issue.get("body") or "",
        "state": issue.get("state"),
        "html_url": issue.get("html_url"),
        "ground_truth": ground_truth,
    }
