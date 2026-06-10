#!/usr/bin/env python3
"""Run real DigitalOcean Serverless Inference classification evals."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from eval_harness.client import DigitalOceanSIClient
from eval_harness.dataset import load_dataset
from eval_harness.models import load_model_catalog, resolve_models
from eval_harness.prompt import LABELS, load_prompt
from eval_harness.resultset import isoformat, make_run_id, utc_now, write_json_atomic
from eval_harness.runner import RunConfig, run_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the issue classification eval engine.")
    parser.add_argument("--dataset", default="data/golden_dataset.json")
    parser.add_argument("--model-catalog", default="config/model_catalog.json")
    parser.add_argument("--prompt", default="config/prompts/classification_template.txt")
    parser.add_argument("--models", default=None, help="Comma-separated model ids from catalog.")
    parser.add_argument("--output-dir", default=os.environ.get("EVAL_OUTPUT_DIR", "runs"))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("EVAL_CONCURRENCY", "4")))
    parser.add_argument("--timeout-seconds", type=float, default=45)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--progress-interval", type=int, default=25)
    parser.add_argument("--limit", type=int, default=None, help="Limit issue count for cost-controlled runs.")
    parser.add_argument("--all", action="store_true", help="Run all issues in the dataset.")
    parser.add_argument(
        "--retry-failed",
        default=None,
        help="Path to a prior model resultset; retry only retryable failed calls.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print prompt and run details.")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DIGITALOCEAN_SI_BASE_URL", "https://inference.do-ai.run"),
    )
    return parser.parse_args()


async def main_async() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if not args.all and args.limit is None and not args.retry_failed:
        raise SystemExit("Pass --limit N for a cost-controlled run, or --all for the full corpus.")

    retry_issue_numbers: set[int] | None = None
    retry_source = None
    if args.retry_failed:
        retry_source = json.loads(Path(args.retry_failed).read_text(encoding="utf-8"))
        retry_issue_numbers = {
            int(result["issue_number"])
            for result in retry_source.get("results", [])
            if result.get("status") == "error" and result.get("retryable")
        }
        if not retry_issue_numbers:
            raise SystemExit(f"No retryable failed calls found in {args.retry_failed}")
        if args.models:
            model_ids = [model_id.strip() for model_id in args.models.split(",") if model_id.strip()]
        else:
            model_ids = [retry_source["model_id"]]
    else:
        if not args.models:
            raise SystemExit("--models is required unless --retry-failed is used.")
        model_ids = [model_id.strip() for model_id in args.models.split(",") if model_id.strip()]

    dataset = load_dataset(Path(args.dataset))
    issues = dataset["issues"]
    if retry_issue_numbers is not None:
        issues = [issue for issue in issues if issue["issue_number"] in retry_issue_numbers]
        if not issues:
            raise SystemExit("Retry resultset issues were not found in the dataset.")
    if args.limit is not None:
        issues = issues[: args.limit]

    api_key = os.environ.get("DIGITALOCEAN_SI_API_KEY") or os.environ.get("DIGITALOCEAN_TOKEN")
    if not api_key:
        raise SystemExit("Set DIGITALOCEAN_SI_API_KEY or DIGITALOCEAN_TOKEN to call DigitalOcean SI.")

    catalog = load_model_catalog(Path(args.model_catalog))
    models = resolve_models(catalog, model_ids)
    system_prompt = load_prompt(Path(args.prompt))
    run_id = args.run_id or make_run_id()
    output_root = Path(args.output_dir) / run_id
    results_dir = output_root / "results"
    run_started_perf = time.perf_counter()

    config = RunConfig(
        run_id=run_id,
        concurrency=args.concurrency,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        prompt_source=args.prompt,
        verbose=args.verbose,
        progress_dir=output_root / "progress",
        progress_interval=args.progress_interval,
    )
    run_payload = {
        "run_id": run_id,
        "status": "running",
        "created_at": isoformat(utc_now()),
        "completed_at": None,
        "wall_clock_ms": None,
        "dataset_id": dataset["dataset_id"],
        "dataset_path": args.dataset,
        "issue_count": len(issues),
        "model_ids": model_ids,
        "prompt_source": args.prompt,
        "model_catalog_path": args.model_catalog,
        "base_url_host": args.base_url.split("//")[-1].split("/")[0],
        "concurrency": args.concurrency,
        "timeout_seconds": args.timeout_seconds,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "max_output_tokens": args.max_output_tokens,
        "streaming": False,
        "label_schema": list(LABELS),
        "retry_failed_source": args.retry_failed,
        "retry_issue_numbers": sorted(retry_issue_numbers) if retry_issue_numbers else None,
    }
    write_json_atomic(output_root / "run.json", run_payload)

    client = DigitalOceanSIClient(api_key=api_key, base_url=args.base_url)
    if args.verbose:
        print("Eval run configuration:")
        print(f"  run_id: {run_id}")
        print(f"  dataset: {args.dataset}")
        print(f"  issue_count: {len(issues)}")
        print(f"  models: {', '.join(model_ids)}")
        print(f"  prompt: {args.prompt}")
        print(f"  base_url: {args.base_url}")
        print(f"  output_dir: {output_root}")
        print(f"  concurrency: {args.concurrency}")
        print(f"  timeout_seconds: {args.timeout_seconds}")
        print(f"  max_retries: {args.max_retries}")
        print(f"  temperature: {args.temperature}")
        print(f"  max_output_tokens: {args.max_output_tokens}")
        print(f"  progress_interval: {args.progress_interval}")

    try:
        for model_id, model in models.items():
            print(f"Running {model_id} on {len(issues)} issues with concurrency={args.concurrency}")
            resultset = await run_model(client, model_id, model, issues, system_prompt, config)
            write_json_atomic(results_dir / f"{model_id}.json", resultset)
            summary = resultset["operational_summary"]
            print(
                f"Finished {model_id}: ok={summary['calls_ok']} "
                f"errors={summary['calls_error']} p95={summary['latency_ms']['p95']}ms "
                f"rps={summary['throughput']['requests_per_second']}"
            )
        run_payload["status"] = "completed"
        run_payload["completed_at"] = isoformat(utc_now())
        run_payload["wall_clock_ms"] = round((time.perf_counter() - run_started_perf) * 1000, 3)
        write_json_atomic(output_root / "run.json", run_payload)
    except Exception:
        run_payload["status"] = "failed"
        run_payload["completed_at"] = isoformat(utc_now())
        run_payload["wall_clock_ms"] = round((time.perf_counter() - run_started_perf) * 1000, 3)
        write_json_atomic(output_root / "run.json", run_payload)
        raise

    print(f"Wrote run artifacts to {output_root}")
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
