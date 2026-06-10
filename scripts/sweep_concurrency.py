#!/usr/bin/env python3
"""Run a small concurrency sweep against the real eval engine."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep eval concurrency values.")
    parser.add_argument("--model", required=True, help="Single model id from catalog.")
    parser.add_argument("--limit", type=int, required=True, help="Issue count per run.")
    parser.add_argument(
        "--concurrency-values",
        default="1,2,4,8",
        help="Comma-separated concurrency values.",
    )
    parser.add_argument("--dataset", default="data/labels/classification_corpus.jsonl")
    parser.add_argument("--prompt", default="config/prompts/classification_template.txt")
    parser.add_argument("--output-dir", default="runs")
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    values = [int(value.strip()) for value in args.concurrency_values.split(",") if value.strip()]
    if not values or any(value < 1 for value in values):
        raise SystemExit("--concurrency-values must contain positive integers")

    sweep_id = f"sweep-{args.model}-limit{args.limit}"
    sweep_dir = Path(args.output_dir) / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for concurrency in values:
        run_id = f"{sweep_id}-c{concurrency}"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_eval.py"),
            "--dataset",
            args.dataset,
            "--prompt",
            args.prompt,
            "--models",
            args.model,
            "--limit",
            str(args.limit),
            "--concurrency",
            str(concurrency),
            "--timeout-seconds",
            str(args.timeout_seconds),
            "--max-retries",
            str(args.max_retries),
            "--max-output-tokens",
            str(args.max_output_tokens),
            "--output-dir",
            args.output_dir,
            "--run-id",
            run_id,
        ]
        print(f"Running concurrency={concurrency}: {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

        result_path = Path(args.output_dir) / run_id / "results" / f"{args.model}.json"
        resultset = json.loads(result_path.read_text(encoding="utf-8"))
        summary = resultset["operational_summary"]
        rows.append(
            {
                "run_id": run_id,
                "model_id": args.model,
                "limit": args.limit,
                "concurrency": concurrency,
                "calls_ok": summary["calls_ok"],
                "calls_error": summary["calls_error"],
                "error_rate": summary["error_rate"],
                "p50_latency_ms": summary["latency_ms"]["p50"],
                "p95_latency_ms": summary["latency_ms"]["p95"],
                "wall_clock_ms": summary["throughput"]["wall_clock_ms"],
                "requests_per_second": summary["throughput"]["requests_per_second"],
                "total_cost_usd": summary["cost"]["total_cost_usd"],
                "errors_by_type": summary["errors_by_type"],
            }
        )

    summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps({"sweep_id": sweep_id, "runs": rows}, indent=2) + "\n")
    print(f"Wrote sweep summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
