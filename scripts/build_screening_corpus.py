#!/usr/bin/env python3
"""Build a small stratified screening corpus from the classification corpus."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LABELS = {"bug", "enhancement", "question", "documentation", "security", "other"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build screening_corpus.jsonl from selected issue IDs.")
    parser.add_argument("--dataset", default="data/labels/classification_corpus.jsonl")
    parser.add_argument("--screening-issues", default="config/screening_issues.csv")
    parser.add_argument("--output", default="data/labels/screening_corpus.jsonl")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def load_screening_issues(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"issue_number", "expected_label", "why_included"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"{path} missing columns: {', '.join(sorted(missing))}")

    seen: set[int] = set()
    for row in rows:
        try:
            issue_number = int(row["issue_number"])
        except ValueError as exc:
            raise SystemExit(f"Invalid issue_number: {row['issue_number']}") from exc
        if issue_number in seen:
            raise SystemExit(f"Duplicate issue_number in {path}: {issue_number}")
        seen.add(issue_number)
        if row["expected_label"] not in LABELS:
            raise SystemExit(f"Invalid expected_label for issue {issue_number}: {row['expected_label']}")
        if not row["why_included"].strip():
            raise SystemExit(f"why_included is required for issue {issue_number}")
    return rows


def main() -> int:
    args = parse_args()
    dataset_rows = load_jsonl(Path(args.dataset))
    by_issue = {int(row["issue_number"]): row for row in dataset_rows}
    screening_rows = load_screening_issues(Path(args.screening_issues))

    output_rows = []
    for screening in screening_rows:
        issue_number = int(screening["issue_number"])
        source = by_issue.get(issue_number)
        if source is None:
            raise SystemExit(f"Screening issue {issue_number} not found in {args.dataset}")
        expected_label = screening["expected_label"]
        if source.get("split") == "scored" and source.get("ground_truth") != expected_label:
            raise SystemExit(
                f"Screening issue {issue_number} expected_label={expected_label} "
                f"does not match certified ground_truth={source.get('ground_truth')}"
            )
        row = dict(source)
        row["screening_sample"] = True
        row["screening_expected_label"] = expected_label
        row["screening_why_included"] = screening["why_included"]
        output_rows.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")

    distribution = Counter(row["screening_expected_label"] for row in output_rows)
    print(f"Screening issue count: {len(output_rows)}")
    print("Distribution by expected_label:")
    for label in sorted(LABELS):
        print(f"  {label}: {distribution.get(label, 0)}")
    print(f"Output path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
