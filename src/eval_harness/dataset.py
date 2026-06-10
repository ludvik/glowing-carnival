from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LABELS = {"bug", "enhancement", "question", "documentation", "security", "other"}


def load_dataset(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return load_classification_corpus(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    issues = normalize_issues(payload)
    dataset_id = payload.get("dataset_id") or payload.get("source", {}).get("repo", "dataset")
    return {
        "dataset_id": dataset_id,
        "source": payload.get("source", {}),
        "path": str(path),
        "issues": issues,
    }


def load_classification_corpus(path: Path) -> dict[str, Any]:
    """Load the scored-set builder's JSONL corpus.

    `classification_corpus.jsonl` is the durable model-classification input:
    every source issue appears exactly once, while only `split=scored` rows carry
    certified ground truth. Review and unscored rows are still classified by
    models, but excluded from accuracy/F1 scoring.
    """

    issues = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        issues.append(normalize_corpus_record(record))

    issues.sort(key=lambda issue: issue["issue_number"])
    return {
        "dataset_id": path.stem,
        "source": {"format": "classification_corpus_jsonl"},
        "path": str(path),
        "issues": issues,
    }


def normalize_corpus_record(record: dict[str, Any]) -> dict[str, Any]:
    split = record.get("split")
    label = record.get("ground_truth") if split == "scored" else None
    if label and label not in LABELS:
        raise ValueError(f"Invalid ground_truth label for issue {record.get('issue_number')}: {label}")

    if split == "scored":
        status = "certified"
        method = "scored_set"
    elif split == "review":
        status = "needs_adjudication"
        method = None
    else:
        status = "uncertified"
        method = None

    confidence = record.get("confidence")
    return {
        "issue_number": int(record["issue_number"]),
        "title": record.get("title") or "",
        "body": record.get("body") or "",
        "state": record.get("state"),
        "html_url": record.get("html_url"),
        "ground_truth": {
            "status": status,
            "label": label,
            "method": method,
            "confidence": confidence if label else None,
            "rationale": record.get("rationale") if label else None,
        },
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
