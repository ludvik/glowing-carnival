#!/usr/bin/env python3
"""Generate scored, unscored, and operational summaries from eval resultsets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.dataset import load_dataset
from eval_harness.resultset import write_json_atomic
from eval_harness.scoring import (
    certified_issues,
    load_resultset,
    model_scored_metrics,
    operational_metrics,
    scored_disagreements,
    unscored_analysis,
    uncertified_issues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score eval resultsets against a dataset.")
    parser.add_argument("--dataset", default="data/golden_dataset.json")
    parser.add_argument(
        "--resultsets",
        required=True,
        help="Comma-separated paths to model resultset JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for summary JSON. Defaults to <first resultset run>/summaries.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_dataset(Path(args.dataset))
    resultset_paths = [Path(path.strip()) for path in args.resultsets.split(",") if path.strip()]
    resultsets = [load_resultset(path) for path in resultset_paths]
    if not resultsets:
        raise SystemExit("--resultsets must include at least one path")

    scored = certified_issues(dataset)
    unscored = uncertified_issues(dataset)
    output_dir = Path(args.output_dir) if args.output_dir else resultset_paths[0].parents[1] / "summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_payload = {
        "dataset_id": dataset["dataset_id"],
        "dataset_path": args.dataset,
        "scored_issue_count": len(scored),
        "models": {
            resultset["model_id"]: model_scored_metrics(resultset, scored)
            for resultset in resultsets
        },
        "disagreements": scored_disagreements(resultsets, scored),
    }
    unscored_payload = {
        "dataset_id": dataset["dataset_id"],
        "dataset_path": args.dataset,
        **unscored_analysis(resultsets, unscored),
    }
    operational_payload = {
        "dataset_id": dataset["dataset_id"],
        "dataset_path": args.dataset,
        "models": operational_metrics(resultsets),
    }

    write_json_atomic(output_dir / "scored_metrics.json", scored_payload)
    write_json_atomic(output_dir / "unscored_analysis.json", unscored_payload)
    write_json_atomic(output_dir / "operational_metrics.json", operational_payload)

    print(f"Wrote summaries to {output_dir}")
    for model_id, metrics in scored_payload["models"].items():
        print(
            f"{model_id}: accuracy={metrics['accuracy']} "
            f"evaluated={metrics['evaluated_count']}/{metrics['scored_issue_count']}"
        )
    if len(resultsets) > 1:
        print(
            "unscored agreement="
            f"{unscored_payload['agreement_rate']} "
            f"({unscored_payload['agreement_count']}/{unscored_payload['comparable_issue_count']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
