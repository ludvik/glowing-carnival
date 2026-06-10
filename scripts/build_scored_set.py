#!/usr/bin/env python3
"""Build a deterministic high-confidence scored set for issue classification.

The goal of this script is not to relabel every doctl issue. It creates a
smaller, defensible scored subset by combining weak maintainer-label evidence
with transparent text heuristics, while routing ambiguous cases to review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


TARGET_LABELS = ("bug", "enhancement", "question", "documentation", "security", "other")
LABEL_SOURCE_PRIORITY = {
    "manual_override": 0,
    "maintainer_label+keyword": 1,
    "maintainer_label": 2,
    "keyword": 3,
}

PRIMARY_LABEL_MAP = {
    "bug": "bug",
    "question": "question",
    "docs": "documentation",
    "security vulnerability": "security",
    "enhancement": "enhancement",
    "suggestion": "enhancement",
    "api-parity": "enhancement",
}

SECONDARY_LABELS = {
    "packaging",
    "snap",
    "windows",
    "wip",
    "waiting-response",
    "good first issue",
    "help wanted",
    "needs investigation",
    "do-api",
    "troubleshooting",
    "blocked",
    "version 2.x",
}

PATTERNS = {
    "bug": [
        r"\bcrash\b",
        r"\bpanic\b",
        r"fatal error",
        r"segmentation",
        r"stack trace",
        r"does not work",
        r"doesn't work",
        r"\bfails?\b",
        r"\bfailed\b",
        r"\bfailure\b",
        r"error:",
        r"returns 404",
        r"returns 500",
        r"401 unable to authenticate",
        r"permission denied",
        r"invalid json",
        r"invalid output",
        r"\bignored\b",
        r"not respected",
        r"\bwrong\b",
        r"\bincorrect\b",
        r"\bbroken\b",
        r"\bregression\b",
        r"cannot execute",
        r"cannot detach",
        r"cannot create",
        r"empty result",
        r"no results",
        r"\bhangs\b",
        r"\bfreezes\b",
    ],
    "enhancement": [
        r"feature request",
        r"add support",
        r"add ability",
        r"add command",
        r"\bimplement\b",
        r"support for",
        r"support matching",
        r"\ballow\b",
        r"ability to",
        r"would be nice",
        r"would be useful",
        r"could you add",
        r"\bproposal\b",
        r"\bsuggested\b",
        r"\bimprove\b",
        r"\bexpose\b",
        r"make doctl available",
        r"package manager",
        r"autocomplete",
        r"completion",
        r"api parity",
    ],
    "question": [
        r"how do i",
        r"how to",
        r"is there a way",
        r"is it possible",
        r"\bcan i\b",
        r"\bcan we\b",
        r"what is",
        r"what does",
        r"why is",
        r"am i doing something wrong",
        r"any chance",
        r"any plans",
        r"does doctl support",
        r"how can i",
        r"please help",
        r"thanks in advance",
        r"workaround",
    ],
    "documentation": [
        r"documentation",
        r"\bdocs\b",
        r"readme",
        r"tutorial",
        r"\bguide\b",
        r"help text",
        r"help menu",
        r"\busage\b",
        r"\bexamples?\b",
        r"\bunclear\b",
        r"\bclarify\b",
        r"missing docs",
        r"document how",
        r"\bmismatch\b",
        r"install instructions",
        r"\binstructions\b",
        r"\btypo\b",
    ],
    "security": [
        r"cve-",
        r"vulnerability",
        r"security vulnerability",
        r"\bcvss\b",
        r"\bcredential\b",
        r"\bcredentials\b",
        r"\bsecret\b",
        r"token exposed",
        r"token leakage",
        r"plaintext token",
        r"auth bypass",
        r"unsafe default",
        r"whitesource",
        r"mend-bolt",
        r"vulnerable library",
    ],
    "other": [
        r"\bspam\b",
        r"off-topic",
        r"duplicate only",
        r"not enough information",
    ],
}

COMPILED_PATTERNS = {
    label: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
    for label, patterns in PATTERNS.items()
}


@dataclass
class ClassifiedIssue:
    issue: dict[str, Any]
    decision: str
    candidate_label: str | None
    confidence: float
    label_source: str
    source_signals: list[str]
    rationale: str
    reason: str
    override: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a deterministic high-confidence scored set from doctl issues."
    )
    parser.add_argument("--input", default="data/doctl_issues.json")
    parser.add_argument("--output-dir", default="data/labels")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overrides", default="data/labels/manual_overrides.csv")
    parser.add_argument("--include-keyword-candidates", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=35)
    return parser.parse_args()


def load_issues(path: Path) -> list[dict[str, Any]]:
    """Load and validate the stable GitHub issue snapshot."""
    if not path.exists():
        raise SystemExit(f"Input file does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Input JSON is invalid: {exc}") from exc
    issues = payload.get("issues")
    if not isinstance(issues, list):
        raise SystemExit("Input JSON must contain an 'issues' list.")
    for issue in issues:
        for field in ("number", "title", "html_url"):
            if field not in issue:
                raise SystemExit(f"Issue is missing required field {field!r}: {issue}")
    return issues


def normalize_labels(issue: dict[str, Any]) -> list[str]:
    """Return normalized maintainer label names."""
    labels = []
    for label in issue.get("labels", []):
        name = str(label.get("name", "")).strip()
        if name:
            labels.append(name)
    return labels


def normalize_label_key(label: str) -> str:
    return label.strip().lower()


def normalize_text(title: str, body: str) -> tuple[str, str, str]:
    """Normalize title/body while stripping markdown link/image syntax for matching."""
    title_text = title or ""
    body_text = body or ""
    combined = f"{title_text}\n{body_text}"
    combined = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", combined)
    combined = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", combined)
    combined = re.sub(r"`([^`]+)`", r"\1", combined)
    combined = re.sub(r"\s+", " ", combined).strip().lower()
    normalized_title = re.sub(r"\s+", " ", title_text).strip().lower()
    normalized_body = re.sub(r"\s+", " ", body_text).strip().lower()
    return normalized_title, normalized_body, combined


def compute_text_signals(title: str, body: str) -> dict[str, dict[str, Any]]:
    """Find deterministic keyword signals for each target label."""
    normalized_title, normalized_body, combined = normalize_text(title, body)
    signals: dict[str, dict[str, Any]] = {}
    for label, patterns in COMPILED_PATTERNS.items():
        matches = []
        title_matches = []
        for pattern in patterns:
            if pattern.search(combined):
                matches.append(pattern.pattern)
            if pattern.search(normalized_title):
                title_matches.append(pattern.pattern)
        signals[label] = {
            "matches": sorted(set(matches)),
            "title_matches": sorted(set(title_matches)),
            "count": len(set(matches)),
            "title_count": len(set(title_matches)),
        }

    if is_test_issue(normalized_title, normalized_body):
        signals["other"]["matches"].append("exact_test_issue")
        signals["other"]["title_matches"].append("exact_test_issue")
        signals["other"]["count"] += 1
        signals["other"]["title_count"] += 1
    if is_low_information(normalized_title, normalized_body):
        signals["other"]["matches"].append("low_information")
        signals["other"]["count"] += 1
    return signals


def map_maintainer_labels(labels: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Map maintainer labels into primary weak labels and secondary context."""
    mapped = []
    primary_names = []
    secondary = []
    for label in labels:
        key = normalize_label_key(label)
        if key in PRIMARY_LABEL_MAP:
            mapped.append(PRIMARY_LABEL_MAP[key])
            primary_names.append(label)
        elif key in SECONDARY_LABELS:
            secondary.append(label)
    return sorted(set(mapped)), primary_names, secondary


def score_candidate(
    candidate_label: str | None,
    primary_labels: list[str],
    primary_names: list[str],
    text_signals: dict[str, dict[str, Any]],
    title: str,
    body: str,
    conflicting_primary: bool,
) -> tuple[float, list[str]]:
    """Compute a transparent confidence score for the candidate label."""
    if candidate_label is None:
        return 0.0, []

    score = 0.0
    signals = []
    if candidate_label in primary_labels:
        score += 0.55
        signals.append(f"primary_maintainer_label:{','.join(primary_names)}")

    same = text_signals[candidate_label]
    if same["count"] > 0:
        score += 0.30
        signals.append(f"text_signal:{candidate_label}:{','.join(same['matches'][:5])}")
    if same["count"] >= 2:
        score += 0.10
    if same["count"] >= 3:
        score += 0.10
    if same["title_count"] > 0:
        score += 0.15
        signals.append(f"title_signal:{candidate_label}:{','.join(same['title_matches'][:5])}")
    if is_high_specific_signal(candidate_label, same, title):
        score += 0.25
        signals.append(f"high_specific_text:{candidate_label}")
    if normalize_label_key("security vulnerability") in [normalize_label_key(name) for name in primary_names]:
        score += 0.10
    if normalize_label_key("docs") in [normalize_label_key(name) for name in primary_names]:
        score += 0.10

    competing = strong_competing_labels(candidate_label, text_signals)
    if candidate_label in primary_labels and not competing and not is_short_low_information(title, body):
        score += 0.20
        signals.append("clean_single_primary_label")
    if competing:
        score -= 0.25
        signals.append(f"competing_text_signal:{','.join(competing)}")
    if is_short_low_information(title, body) and not same["title_count"]:
        score -= 0.20
        signals.append("low_information_body")
    if conflicting_primary:
        score -= 0.30
        signals.append("conflicting_primary_maintainer_labels")

    return clamp(score), signals


def classify_issue(
    issue: dict[str, Any],
    include_keyword_candidates: bool,
    min_confidence: float,
) -> ClassifiedIssue:
    """Classify one issue into scored/review/unscored using inspectable rules."""
    labels = normalize_labels(issue)
    primary_labels, primary_names, secondary = map_maintainer_labels(labels)
    text_signals = compute_text_signals(issue.get("title", ""), issue.get("body", ""))
    candidate_label = choose_candidate_label(primary_labels, text_signals, include_keyword_candidates)
    conflicting_primary = len(primary_labels) > 1

    confidence, source_signals = score_candidate(
        candidate_label,
        primary_labels,
        primary_names,
        text_signals,
        issue.get("title", ""),
        issue.get("body", ""),
        conflicting_primary,
    )
    label_source = infer_label_source(candidate_label, primary_labels, text_signals)
    rationale = build_rationale(candidate_label, confidence, label_source, source_signals)

    if conflicting_primary:
        return ClassifiedIssue(
            issue,
            "review",
            None,
            confidence,
            label_source,
            source_signals,
            rationale,
            "conflicting_primary_labels",
        )

    ambiguity = ambiguity_reason(candidate_label, primary_labels, text_signals)
    if ambiguity:
        return ClassifiedIssue(
            issue,
            "review",
            candidate_label,
            confidence,
            label_source,
            source_signals,
            rationale,
            ambiguity,
        )

    if candidate_label is None:
        reason = "only_secondary_labels" if secondary else "no_high_confidence_signal"
        return ClassifiedIssue(
            issue,
            "unscored",
            None,
            confidence,
            "none",
            source_signals,
            "No high-confidence ground-truth signal.",
            reason,
        )

    if confidence >= min_confidence:
        return ClassifiedIssue(
            issue,
            "scored",
            candidate_label,
            confidence,
            label_source,
            source_signals,
            rationale,
            "eligible_high_confidence",
        )

    return ClassifiedIssue(
        issue,
        "unscored",
        candidate_label,
        confidence,
        label_source,
        source_signals,
        rationale,
        "below_min_confidence",
    )


def apply_overrides(
    classified: dict[int, ClassifiedIssue],
    overrides_path: Path | None,
) -> None:
    """Apply manual decisions before stratified selection."""
    if overrides_path is None or not overrides_path.exists():
        return
    with overrides_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"issue_number", "decision", "ground_truth", "rationale"}
        if set(reader.fieldnames or []) < required:
            raise SystemExit(f"Override CSV must contain columns: {sorted(required)}")
        for row in reader:
            issue_number = int(row["issue_number"])
            if issue_number not in classified:
                raise SystemExit(f"Override references unknown issue_number: {issue_number}")
            decision = row["decision"].strip().lower()
            label = row["ground_truth"].strip().lower()
            rationale = row["rationale"].strip()
            if decision not in {"scored", "unscored", "review"}:
                raise SystemExit(f"Invalid override decision for issue {issue_number}: {decision}")
            if decision == "scored" and label not in TARGET_LABELS:
                raise SystemExit(f"Invalid override ground_truth for issue {issue_number}: {label}")
            if decision == "scored" and not rationale:
                raise SystemExit(f"Manual scored override missing rationale: {issue_number}")

            current = classified[issue_number]
            classified[issue_number] = ClassifiedIssue(
                current.issue,
                decision,
                label if decision == "scored" else (label if label in TARGET_LABELS else None),
                1.0 if decision == "scored" else current.confidence,
                "manual_override",
                ["manual_override"],
                rationale or "Manual override.",
                f"manual_override_{decision}",
                override=True,
            )


def select_stratified_scored_set(
    rows: list[ClassifiedIssue],
    max_per_class: int,
) -> list[ClassifiedIssue]:
    """Select high-confidence scored rows with a hard per-class cap."""
    by_label: dict[str, list[ClassifiedIssue]] = defaultdict(list)
    for row in rows:
        if row.decision == "scored" and row.candidate_label in TARGET_LABELS:
            by_label[row.candidate_label].append(row)

    selected = []
    for label in TARGET_LABELS:
        candidates = sorted(by_label[label], key=scored_sort_key)
        manual = [row for row in candidates if row.label_source == "manual_override"]
        automatic = [row for row in candidates if row.label_source != "manual_override"]
        remaining_slots = max(max_per_class - len(manual), 0)
        selected.extend([*manual, *automatic[:remaining_slots]])
    return sorted(selected, key=lambda row: row.issue["number"])


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def write_summary(
    path: Path,
    issues: list[dict[str, Any]],
    selected: list[ClassifiedIssue],
    unscored: list[ClassifiedIssue],
    review: list[ClassifiedIssue],
    manual_review: list[dict[str, Any]],
    all_rows: list[ClassifiedIssue],
    args: argparse.Namespace,
    input_sha256: str,
) -> None:
    scored_support = {label: 0 for label in TARGET_LABELS}
    scored_support.update(Counter(row.candidate_label for row in selected))
    review_distribution = {label: 0 for label in TARGET_LABELS}
    review_distribution.update(
        Counter(row["candidate_label"] for row in manual_review if row.get("candidate_label"))
    )
    thin_labels = [label for label, count in scored_support.items() if count < 10]
    summary = {
        "total_issues": len(issues),
        "scored_count": len(selected),
        "unscored_count": len(unscored),
        "review_count": len(review),
        "scored_distribution_by_label": dict(sorted(Counter(row.candidate_label for row in selected).items())),
        "scored_support_by_label": dict(sorted(scored_support.items())),
        "review_candidate_distribution_by_label": dict(sorted(review_distribution.items())),
        "thin_labels": thin_labels,
        "candidate_distribution_by_label": dict(
            sorted(Counter(row.candidate_label for row in all_rows if row.candidate_label).items())
        ),
        "maintainer_label_distribution": maintainer_label_distribution(issues),
        "exclusion_reason_counts": dict(
            sorted(Counter(row.reason for row in [*unscored, *review]).items())
        ),
        "min_confidence": args.min_confidence,
        "max_per_class": args.max_per_class,
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "input_sha256": input_sha256,
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def choose_candidate_label(
    primary_labels: list[str],
    text_signals: dict[str, dict[str, Any]],
    include_keyword_candidates: bool,
) -> str | None:
    if len(primary_labels) == 1:
        return primary_labels[0]
    if len(primary_labels) > 1:
        return None

    ranked = text_rank(text_signals)
    if not ranked:
        return None
    label, score = ranked[0]
    if label in {"security", "documentation", "question", "other"} and score >= 2:
        return label
    if include_keyword_candidates and score >= 3:
        return label
    return None


def text_rank(text_signals: dict[str, dict[str, Any]]) -> list[tuple[str, int]]:
    scores = []
    for label, data in text_signals.items():
        score = data["count"] + data["title_count"]
        if label == "other" and "exact_test_issue" in data["matches"]:
            score += 5
        scores.append((label, score))
    return [(label, score) for label, score in sorted(scores, key=lambda item: (-item[1], item[0])) if score > 0]


def ambiguity_reason(
    candidate_label: str | None,
    primary_labels: list[str],
    text_signals: dict[str, dict[str, Any]],
) -> str | None:
    if candidate_label is None:
        return None
    competing = strong_competing_labels(candidate_label, text_signals)
    if not competing:
        return None
    if {candidate_label, *competing} >= {"bug", "question"}:
        if candidate_label in primary_labels:
            return None
        return "ambiguous_bug_question"
    if {candidate_label, *competing} >= {"documentation", "enhancement"}:
        if candidate_label in primary_labels and text_signals[candidate_label]["title_count"] > 0:
            return None
        return "ambiguous_documentation_enhancement"
    if candidate_label == "security":
        return None
    if candidate_label in primary_labels and text_signals[candidate_label]["count"] > 0:
        return None
    return "strong_competing_text_signal"


def strong_competing_labels(candidate_label: str, text_signals: dict[str, dict[str, Any]]) -> list[str]:
    candidate_strength = signal_strength(text_signals[candidate_label])
    competing = []
    for label in TARGET_LABELS:
        if label == candidate_label:
            continue
        strength = signal_strength(text_signals[label])
        if strength >= 3 and strength >= candidate_strength:
            competing.append(label)
    return competing


def signal_strength(data: dict[str, Any]) -> int:
    return int(data["count"]) + int(data["title_count"])


def infer_label_source(
    candidate_label: str | None,
    primary_labels: list[str],
    text_signals: dict[str, dict[str, Any]],
) -> str:
    if candidate_label is None:
        return "none"
    has_primary = candidate_label in primary_labels
    has_keyword = text_signals[candidate_label]["count"] > 0
    if has_primary and has_keyword:
        return "maintainer_label+keyword"
    if has_primary:
        return "maintainer_label"
    if has_keyword:
        return "keyword"
    return "none"


def build_rationale(
    candidate_label: str | None,
    confidence: float,
    label_source: str,
    source_signals: list[str],
) -> str:
    if candidate_label is None:
        return "No single high-confidence target label could be certified."
    signal_text = "; ".join(source_signals[:4]) or "no explicit signal"
    return (
        f"Assigned {candidate_label} from {label_source} with confidence "
        f"{confidence:.2f}. Signals: {signal_text}."
    )


def scored_sort_key(row: ClassifiedIssue) -> tuple[float, int, int, int]:
    comments = int(row.issue.get("comments") or 0)
    return (
        -row.confidence,
        LABEL_SOURCE_PRIORITY.get(row.label_source, 99),
        -comments,
        int(row.issue["number"]),
    )


def scored_record(row: ClassifiedIssue) -> dict[str, Any]:
    issue = row.issue
    return {
        "issue_number": issue["number"],
        "issue_id": issue.get("id"),
        "title": issue.get("title", ""),
        "body_excerpt": excerpt(issue.get("body", "")),
        "body": issue.get("body", ""),
        "html_url": issue.get("html_url", ""),
        "state": issue.get("state", ""),
        "created_at": issue.get("created_at", ""),
        "closed_at": issue.get("closed_at", ""),
        "maintainer_labels": ";".join(normalize_labels(issue)),
        "ground_truth": row.candidate_label,
        "confidence": f"{row.confidence:.3f}",
        "label_source": row.label_source,
        "source_signals": json.dumps(row.source_signals, ensure_ascii=False),
        "rationale": row.rationale,
    }


def classification_corpus_record(row: ClassifiedIssue, split: str) -> dict[str, Any]:
    issue = row.issue
    return {
        "issue_number": issue["number"],
        "issue_id": issue.get("id"),
        "title": issue.get("title", ""),
        "body": issue.get("body", ""),
        "html_url": issue.get("html_url", ""),
        "state": issue.get("state", ""),
        "maintainer_labels": normalize_labels(issue),
        "split": split,
        "ground_truth": row.candidate_label if split == "scored" else None,
        "candidate_label": row.candidate_label,
        "confidence": round(row.confidence, 3),
        "reason": row.reason,
        "rationale": row.rationale,
    }


def unscored_record(row: ClassifiedIssue) -> dict[str, Any]:
    issue = row.issue
    return {
        "issue_number": issue["number"],
        "title": issue.get("title", ""),
        "body_excerpt": excerpt(issue.get("body", "")),
        "html_url": issue.get("html_url", ""),
        "state": issue.get("state", ""),
        "maintainer_labels": ";".join(normalize_labels(issue)),
        "predicted_candidate_label": row.candidate_label or "",
        "confidence": f"{row.confidence:.3f}",
        "reason": row.reason,
    }


def manual_review_record(row: ClassifiedIssue, split: str) -> dict[str, Any]:
    label = row.candidate_label or ""
    return {
        "issue_number": row.issue["number"],
        "title": row.issue.get("title", ""),
        "body_excerpt": excerpt(row.issue.get("body", "")),
        "html_url": row.issue.get("html_url", ""),
        "state": row.issue.get("state", ""),
        "maintainer_labels": ";".join(normalize_labels(row.issue)),
        "split": split,
        "candidate_label": label,
        "ground_truth": row.candidate_label if split == "scored" else "",
        "confidence": f"{row.confidence:.3f}",
        "reason": row.reason,
        "label_source": row.label_source,
        "review_priority": "high"
        if label in {"documentation", "question", "other", "security"}
        else "medium",
        "rationale": row.rationale,
    }


def review_record(row: ClassifiedIssue) -> dict[str, Any]:
    issue = row.issue
    return {
        "issue_number": issue["number"],
        "title": issue.get("title", ""),
        "body_excerpt": excerpt(issue.get("body", "")),
        "html_url": issue.get("html_url", ""),
        "state": issue.get("state", ""),
        "maintainer_labels": ";".join(normalize_labels(issue)),
        "candidate_label": row.candidate_label or "",
        "confidence": f"{row.confidence:.3f}",
        "review_reason": row.reason,
        "source_signals": json.dumps(row.source_signals, ensure_ascii=False),
        "suggested_rationale": row.rationale,
    }


def validate_outputs(
    issues: list[dict[str, Any]],
    scored: list[ClassifiedIssue],
    unscored: list[ClassifiedIssue],
    review: list[ClassifiedIssue],
) -> None:
    scored_numbers = [row.issue["number"] for row in scored]
    if len(scored_numbers) != len(set(scored_numbers)):
        raise SystemExit("Duplicate issue_number appears in scored_set.")
    overlap = set(scored_numbers) & {row.issue["number"] for row in unscored}
    if overlap:
        raise SystemExit(f"Issue appears in both scored_set and unscored_set: {sorted(overlap)}")
    for row in scored:
        if row.candidate_label not in TARGET_LABELS:
            raise SystemExit(f"Invalid scored ground_truth for issue {row.issue['number']}")
        if not row.rationale.strip():
            raise SystemExit(f"Scored row has empty rationale for issue {row.issue['number']}")
    split_numbers = [row.issue["number"] for row in [*scored, *unscored, *review]]
    if len(split_numbers) != len(set(split_numbers)):
        raise SystemExit("An issue appears in multiple output splits.")
    if set(split_numbers) != {issue["number"] for issue in issues}:
        raise SystemExit("Output splits do not contain all input issues exactly once.")
    for row in [*scored, *unscored, *review]:
        if not 0.0 <= row.confidence <= 1.0:
            raise SystemExit(f"Confidence outside [0, 1] for issue {row.issue['number']}")


def maintainer_label_distribution(issues: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for issue in issues:
        counter.update(normalize_labels(issue))
    return dict(sorted(counter.items()))


def is_test_issue(title: str, body: str) -> bool:
    body_clean = body.strip().lower()
    title_clean = title.strip().lower()
    return title_clean == "test" and body_clean in {"", "test"}


def is_low_information(title: str, body: str) -> bool:
    text = f"{title} {body}".strip()
    return len(text) < 20


def is_short_low_information(title: str, body: str) -> bool:
    return len((body or "").strip()) < 40 and len((title or "").strip()) < 80


def is_high_specific_signal(label: str, signal: dict[str, Any], title: str) -> bool:
    title_lower = title.lower()
    if label == "security":
        return bool(signal["matches"])
    if label == "documentation":
        return bool(signal["title_matches"]) or "document" in title_lower
    if label == "question":
        return title_lower.startswith(("how ", "how to", "is there", "is it", "can i", "why "))
    if label == "other":
        return "exact_test_issue" in signal["matches"] or "low_information" in signal["matches"]
    return False


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def excerpt(body: str, limit: int = 240) -> str:
    text = re.sub(r"\s+", " ", body or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def input_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_summary(summary_path: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    print(f"Total issues: {summary['total_issues']}")
    print(f"Scored count: {summary['scored_count']}")
    print(f"Review count: {summary['review_count']}")
    print(f"Unscored count: {summary['unscored_count']}")
    print("Scored distribution by label:")
    for label, count in summary["scored_distribution_by_label"].items():
        print(f"  {label}: {count}")
    print("Top exclusion reasons:")
    for reason, count in Counter(summary["exclusion_reason_counts"]).most_common(8):
        print(f"  {reason}: {count}")
    if summary["thin_labels"]:
        print("WARNING: Fewer than 10 scored examples for:")
        for label in summary["thin_labels"]:
            print(f"  {label}: {summary['scored_support_by_label'][label]}")
    print(f"Outputs written under: {summary_path.parent}")


def build_manual_review_candidates(
    selected: list[ClassifiedIssue],
    unscored: list[ClassifiedIssue],
    review: list[ClassifiedIssue],
) -> list[dict[str, Any]]:
    """Build a review helper file without changing scoring splits."""
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add(row: ClassifiedIssue, split: str) -> None:
        issue_number = int(row.issue["number"])
        if issue_number in seen:
            return
        seen.add(issue_number)
        rows.append(manual_review_record(row, split))

    for row in sorted(review, key=lambda item: item.issue["number"]):
        add(row, "review")
    for row in sorted(unscored, key=lambda item: item.issue["number"]):
        if row.candidate_label in {"documentation", "question", "other", "security"} and row.confidence >= 0.40:
            add(row, "unscored")
    for row in sorted(selected, key=lambda item: item.issue["number"]):
        if row.label_source == "maintainer_label" and row.confidence <= 0.80:
            add(row, "scored")

    rows.sort(
        key=lambda row: (
            0 if row["review_priority"] == "high" else 1,
            row["split"],
            int(row["issue_number"]),
        )
    )
    return rows


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if args.max_per_class < 1:
        raise SystemExit("--max-per-class must be >= 1")
    if not 0.0 <= args.min_confidence <= 1.0:
        raise SystemExit("--min-confidence must be between 0 and 1")

    issues = load_issues(input_path)
    classified = {
        int(issue["number"]): classify_issue(
            issue,
            include_keyword_candidates=args.include_keyword_candidates,
            min_confidence=args.min_confidence,
        )
        for issue in issues
    }
    apply_overrides(classified, Path(args.overrides) if args.overrides else None)

    all_rows = list(classified.values())
    selected = select_stratified_scored_set(all_rows, args.max_per_class)
    selected_numbers = {row.issue["number"] for row in selected}
    review = [
        row
        for row in all_rows
        if row.decision == "review" and row.issue["number"] not in selected_numbers
    ]
    unscored = []
    for row in all_rows:
        if row.issue["number"] in selected_numbers or row.decision == "review":
            continue
        if row.decision == "scored":
            row = ClassifiedIssue(
                row.issue,
                "unscored",
                row.candidate_label,
                row.confidence,
                row.label_source,
                row.source_signals,
                row.rationale,
                "over_max_per_class",
                row.override,
            )
        unscored.append(row)

    validate_outputs(issues, selected, unscored, review)
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_rows = [scored_record(row) for row in selected]
    jsonl_rows = [dict(record) for record in scored_rows]
    split_by_issue = {
        **{row.issue["number"]: "scored" for row in selected},
        **{row.issue["number"]: "unscored" for row in unscored},
        **{row.issue["number"]: "review" for row in review},
    }
    row_by_issue = {row.issue["number"]: row for row in [*selected, *unscored, *review]}
    corpus_rows = [
        classification_corpus_record(row_by_issue[issue["number"]], split_by_issue[issue["number"]])
        for issue in sorted(issues, key=lambda item: item["number"])
    ]
    manual_review_rows = build_manual_review_candidates(selected, unscored, review)
    write_csv(
        output_dir / "scored_set.csv",
        scored_rows,
        [
            "issue_number",
            "issue_id",
            "title",
            "body_excerpt",
            "html_url",
            "state",
            "created_at",
            "closed_at",
            "maintainer_labels",
            "ground_truth",
            "confidence",
            "label_source",
            "source_signals",
            "rationale",
        ],
    )
    write_jsonl(output_dir / "scored_set.jsonl", jsonl_rows)
    write_jsonl(output_dir / "classification_corpus.jsonl", corpus_rows)
    write_csv(
        output_dir / "unscored_set.csv",
        [unscored_record(row) for row in sorted(unscored, key=lambda row: row.issue["number"])],
        [
            "issue_number",
            "title",
            "body_excerpt",
            "html_url",
            "state",
            "maintainer_labels",
            "predicted_candidate_label",
            "confidence",
            "reason",
        ],
    )
    write_csv(
        output_dir / "review_queue.csv",
        [review_record(row) for row in sorted(review, key=lambda row: row.issue["number"])],
        [
            "issue_number",
            "title",
            "body_excerpt",
            "html_url",
            "state",
            "maintainer_labels",
            "candidate_label",
            "confidence",
            "review_reason",
            "source_signals",
            "suggested_rationale",
        ],
    )
    write_csv(
        output_dir / "manual_review_candidates.csv",
        manual_review_rows,
        [
            "issue_number",
            "title",
            "body_excerpt",
            "html_url",
            "state",
            "maintainer_labels",
            "split",
            "candidate_label",
            "ground_truth",
            "confidence",
            "reason",
            "label_source",
            "review_priority",
            "rationale",
        ],
    )
    summary_path = output_dir / "labeling_summary.json"
    write_summary(
        summary_path,
        issues,
        selected,
        unscored,
        review,
        manual_review_rows,
        all_rows,
        args,
        input_sha256(input_path),
    )
    print_summary(summary_path)
    print("Output paths:")
    for name in (
        "scored_set.csv",
        "scored_set.jsonl",
        "classification_corpus.jsonl",
        "unscored_set.csv",
        "review_queue.csv",
        "manual_review_candidates.csv",
        "labeling_summary.json",
    ):
        print(f"  {output_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
