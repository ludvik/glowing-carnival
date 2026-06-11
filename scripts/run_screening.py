#!/usr/bin/env python3
"""Thin wrapper around run_eval.py for the curated screening pool."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_CATALOG = ROOT / "config/model_catalog.json"
SCREENING_CORPUS = ROOT / "data/labels/screening_corpus.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the curated model screening run.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--models", default=None, help="Optional comma-separated screening-pool subset.")
    return parser.parse_args()


def screening_pool_model_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        return [row["model_id"] for row in csv.DictReader(handle)]


def ensure_catalog() -> None:
    if MODEL_CATALOG.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_runner_model_catalog.py"),
            "--metadata",
            "config/model_metadata.json",
            "--screening-pool",
            "config/screening_pool.csv",
            "--output",
            "config/model_catalog.json",
        ],
        cwd=ROOT,
        check=True,
    )


def ensure_screening_corpus() -> None:
    if SCREENING_CORPUS.exists():
        return
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/build_screening_corpus.py"),
            "--dataset",
            "data/labels/classification_corpus.jsonl",
            "--screening-issues",
            "config/screening_issues.csv",
            "--output",
            "data/labels/screening_corpus.jsonl",
        ],
        cwd=ROOT,
        check=True,
    )


def main() -> int:
    args = parse_args()
    ensure_catalog()
    ensure_screening_corpus()

    model_ids = screening_pool_model_ids(ROOT / "config/screening_pool.csv")
    if args.models:
        requested = [model_id.strip() for model_id in args.models.split(",") if model_id.strip()]
        missing = sorted(set(requested) - set(model_ids))
        if missing:
            raise SystemExit(f"Requested models are not in config/screening_pool.csv: {', '.join(missing)}")
        model_ids = requested

    run_id = args.run_id or f"screening-v1-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_eval.py"),
        "--dataset",
        "data/labels/screening_corpus.jsonl",
        "--model-catalog",
        "config/model_catalog.json",
        "--prompt",
        "config/prompts/classification_template.txt",
        "--models",
        ",".join(model_ids),
        "--output-dir",
        "runs",
        "--run-id",
        run_id,
        "--concurrency",
        str(args.concurrency),
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--max-retries",
        str(args.max_retries),
        "--temperature",
        "0",
        "--max-output-tokens",
        str(args.max_output_tokens),
        "--all",
    ]
    print("Running screening run command:")
    print(" ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
