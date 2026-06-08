#!/usr/bin/env python3
"""Build the scored and unscored issue dataset used by the eval harness."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


TARGET_LABELS = ("bug", "enhancement", "question", "documentation", "security", "other")

# Maintainer labels are treated as weak ground truth only when they map cleanly
# to exactly one target class. Labels not listed here are metadata or workflow
# labels and do not contribute to the ground-truth class.
LABEL_MAP = {
    "bug": "bug",
    "suggestion": "enhancement",
    "enhancement": "enhancement",
    "api-parity": "enhancement",
    "question": "question",
    "troubleshooting": "question",
    "docs": "documentation",
    "security vulnerability": "security",
    "duplicate": "other",
}

IGNORED_LABELS = {
    "app-platform",
    "blocked",
    "do-api",
    "good first issue",
    "hacktoberfest",
    "help wanted",
    "needs investigation",
    "packaging",
    "snap",
    "version 2.x",
    "waiting-response",
    "windows",
    "wip",
    "work-around-available",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a stable golden dataset from the fetched GitHub issues corpus."
    )
    parser.add_argument(
        "--input",
        default="data/doctl_issues.json",
        help="Fetched GitHub issues corpus.",
    )
    parser.add_argument(
        "--output",
        default="data/golden_dataset.json",
        help="Path to write the generated dataset.",
    )
    parser.add_argument(
        "--overrides",
        default="data/golden_overrides.json",
        help="Optional manual adjudications for conflicting maintainer labels.",
    )
    return parser.parse_args()


def compact_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "number": issue["number"],
        "title": issue["title"],
        "body": issue["body"],
        "state": issue["state"],
        "labels": [label["name"] for label in issue.get("labels", [])],
        "created_at": issue["created_at"],
        "updated_at": issue["updated_at"],
        "html_url": issue["html_url"],
    }


def classify_ground_truth(issue: dict[str, Any]) -> tuple[str | None, str]:
    label_names = [
        str(label.get("name", "")).strip()
        for label in issue.get("labels", [])
        if str(label.get("name", "")).strip()
    ]
    mapped = {
        LABEL_MAP[label.lower()]
        for label in label_names
        if label.lower() in LABEL_MAP
    }

    if len(mapped) == 1:
        return next(iter(mapped)), "single_mapped_maintainer_label"
    if len(mapped) > 1:
        return None, "conflicting_mapped_maintainer_labels"
    if label_names:
        return None, "no_target_schema_label"
    return None, "unlabeled"


def validate_override(issue_number: int, override: dict[str, Any]) -> None:
    label = override.get("ground_truth_label")
    if label not in TARGET_LABELS:
        raise ValueError(f"Override for issue {issue_number} has invalid label: {label!r}")
    if not override.get("rationale"):
        raise ValueError(f"Override for issue {issue_number} is missing a rationale")


def build_dataset(
    corpus: dict[str, Any],
    overrides: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    scored: list[dict[str, Any]] = []
    unscored: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    override_count = 0

    for issue in corpus["issues"]:
        ground_truth_label, reason = classify_ground_truth(issue)
        row = compact_issue(issue)
        override = overrides.get(issue["number"])

        if override is not None:
            validate_override(issue["number"], override)
            row["ground_truth_label"] = override["ground_truth_label"]
            row["ground_truth_source"] = "manual_adjudication"
            row["ground_truth_rationale"] = override["rationale"]
            row["automated_ground_truth_reason"] = reason
            scored.append(row)
            override_count += 1
            continue

        if ground_truth_label is not None:
            row["ground_truth_label"] = ground_truth_label
            row["ground_truth_source"] = reason
            scored.append(row)
            continue

        row["exclusion_reason"] = reason
        exclusion_reasons[reason] += 1
        if reason == "conflicting_mapped_maintainer_labels":
            review.append(row)
        else:
            unscored.append(row)

    scored.sort(key=lambda issue: issue["number"])
    unscored.sort(key=lambda issue: issue["number"])
    review.sort(key=lambda issue: issue["number"])

    scored_counts = Counter(issue["ground_truth_label"] for issue in scored)
    payload = {
        "schema": {
            "target_labels": list(TARGET_LABELS),
            "label_map": LABEL_MAP,
            "ignored_labels": sorted(IGNORED_LABELS),
            "methodology": (
                "Maintainer labels are used as weak ground truth only when exactly one "
                "mapped target class is present. Issues with no mapped target label remain "
                "unscored. Issues with multiple mapped target classes are separated for "
                "manual review instead of scored automatically."
            ),
        },
        "source": corpus["source"],
        "counts": {
            "total_issues": len(corpus["issues"]),
            "scored": len(scored),
            "unscored": len(unscored),
            "needs_review": len(review),
            "manual_overrides": override_count,
            "scored_by_label": dict(sorted(scored_counts.items())),
            "excluded_by_reason": dict(sorted(exclusion_reasons.items())),
        },
        "scored": scored,
        "unscored": unscored,
        "needs_review": review,
    }
    return payload


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    overrides_path = Path(args.overrides)

    corpus = json.loads(input_path.read_text(encoding="utf-8"))
    raw_overrides = {}
    if overrides_path.exists():
        raw_overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
    overrides = {int(issue_number): override for issue_number, override in raw_overrides.items()}

    payload = build_dataset(corpus, overrides)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    counts = payload["counts"]
    print(
        f"Wrote {counts['scored']} scored, {counts['unscored']} unscored, "
        f"and {counts['needs_review']} review issues to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
