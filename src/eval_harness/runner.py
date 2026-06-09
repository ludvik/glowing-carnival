from __future__ import annotations

import asyncio
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any

from eval_harness.client import DigitalOceanSIClient, extract_content, extract_usage
from eval_harness.cost import calculate_cost
from eval_harness.errors import InferenceError
from eval_harness.prompt import build_messages, parse_model_output
from eval_harness.resultset import isoformat, utc_now


@dataclass(frozen=True)
class RunConfig:
    run_id: str
    concurrency: int
    timeout_seconds: float
    max_retries: int
    temperature: float
    max_output_tokens: int
    prompt_source: str
    streaming: bool = False
    verbose: bool = False


async def run_model(
    client: DigitalOceanSIClient,
    model_id: str,
    model: dict[str, Any],
    issues: list[dict[str, Any]],
    system_prompt: str,
    config: RunConfig,
) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(config.concurrency)
    started = utc_now()
    wall_start = time.perf_counter()

    tasks = [
        run_call(client, semaphore, model_id, model, issue, system_prompt, config)
        for issue in issues
    ]
    results = await asyncio.gather(*tasks)
    completed = utc_now()
    wall_clock_ms = round((time.perf_counter() - wall_start) * 1000, 3)

    return {
        "run_id": config.run_id,
        "model_id": model_id,
        "provider_model": model["provider_model"],
        "created_at": isoformat(started),
        "completed_at": isoformat(completed),
        "wall_clock_ms": wall_clock_ms,
        "result_count": len(results),
        "results": results,
        "operational_summary": summarize_results(results, wall_clock_ms, config.concurrency),
    }


async def run_call(
    client: DigitalOceanSIClient,
    semaphore: asyncio.Semaphore,
    model_id: str,
    model: dict[str, Any],
    issue: dict[str, Any],
    system_prompt: str,
    config: RunConfig,
) -> dict[str, Any]:
    queued_at = utc_now()
    queued_perf = time.perf_counter()

    async with semaphore:
        started_at = utc_now()
        started_perf = time.perf_counter()
        queue_wait_ms = (started_perf - queued_perf) * 1000
        attempts = 0
        last_error: InferenceError | None = None
        messages = build_messages(system_prompt, issue)
        if config.verbose:
            print(
                f"\n[{model_id}] issue #{issue['issue_number']} prompt messages:\n"
                f"{messages}\n",
                flush=True,
            )

        while attempts <= config.max_retries:
            attempts += 1
            request_sent_at = utc_now()
            request_perf = time.perf_counter()
            try:
                response = await asyncio.to_thread(
                    client.chat_completion,
                    model["provider_model"],
                    messages,
                    config.temperature,
                    config.max_output_tokens,
                    config.timeout_seconds,
                )
                response_completed_at = utc_now()
                response_perf = time.perf_counter()
                raw_output = extract_content(response)
                usage = extract_usage(response)
                parsed_label, rationale, parse_error = parse_model_output(raw_output)
                status = "ok"
                error = None
                retryable = False
                if parse_error:
                    status = "error"
                    error_type = "invalid_label" if "Invalid label" in parse_error else "parse_error"
                    error = {"type": error_type, "message": parse_error, "http_status": None}

                ended_at = utc_now()
                ended_perf = time.perf_counter()
                return {
                    "call_id": f"{config.run_id}:{model_id}:{issue['issue_number']}",
                    "run_id": config.run_id,
                    "model_id": model_id,
                    "provider_model": model["provider_model"],
                    "issue_number": issue["issue_number"],
                    "status": status,
                    "attempts": attempts,
                    "retryable": retryable,
                    "request": {
                        "temperature": config.temperature,
                        "max_output_tokens": config.max_output_tokens,
                        "timeout_seconds": config.timeout_seconds,
                        "streaming": config.streaming,
                        "prompt_source": config.prompt_source,
                    },
                    "response": {
                        "raw_output": raw_output,
                        "parsed_label": parsed_label,
                        "rationale": rationale,
                        "parse_error": parse_error,
                    },
                    "usage": usage,
                    "cost": calculate_cost(usage, model),
                    "timing": {
                        "queued_at": isoformat(queued_at),
                        "started_at": isoformat(started_at),
                        "request_sent_at": isoformat(request_sent_at),
                        "first_token_at": None,
                        "response_completed_at": isoformat(response_completed_at),
                        "ended_at": isoformat(ended_at),
                        "queue_wait_ms": round(queue_wait_ms, 3),
                        "time_to_first_token_ms": None,
                        "generation_ms": None,
                        "total_latency_ms": round((response_perf - request_perf) * 1000, 3),
                        "wall_time_ms": round((ended_perf - queued_perf) * 1000, 3),
                        "measurement_source": "client_non_streaming",
                    },
                    "error": error,
                }
            except InferenceError as exc:
                last_error = exc
                if not exc.retryable or attempts > config.max_retries:
                    break
                await asyncio.sleep(min(2**attempts, 8))

        ended_at = utc_now()
        ended_perf = time.perf_counter()
        error = last_error or InferenceError("other", "Unknown inference error")
        return {
            "call_id": f"{config.run_id}:{model_id}:{issue['issue_number']}",
            "run_id": config.run_id,
            "model_id": model_id,
            "provider_model": model["provider_model"],
            "issue_number": issue["issue_number"],
            "status": "error",
            "attempts": attempts,
            "retryable": error.retryable,
            "request": {
                "temperature": config.temperature,
                "max_output_tokens": config.max_output_tokens,
                "timeout_seconds": config.timeout_seconds,
                "streaming": config.streaming,
                "prompt_source": config.prompt_source,
            },
            "response": {
                "raw_output": None,
                "parsed_label": None,
                "rationale": None,
                "parse_error": None,
            },
            "usage": None,
            "cost": calculate_cost(None, model),
            "timing": {
                "queued_at": isoformat(queued_at),
                "started_at": isoformat(started_at),
                "request_sent_at": None,
                "first_token_at": None,
                "response_completed_at": None,
                "ended_at": isoformat(ended_at),
                "queue_wait_ms": round(queue_wait_ms, 3),
                "time_to_first_token_ms": None,
                "generation_ms": None,
                "total_latency_ms": None,
                "wall_time_ms": round((ended_perf - queued_perf) * 1000, 3),
                "measurement_source": "client_non_streaming",
            },
            "error": {
                "type": error.error_type,
                "message": error.message,
                "http_status": error.http_status,
            },
        }


def summarize_results(
    results: list[dict[str, Any]],
    wall_clock_ms: float | None = None,
    concurrency: int | None = None,
) -> dict[str, Any]:
    ok_results = [result for result in results if result["status"] == "ok"]
    latency_values = [
        result["timing"]["total_latency_ms"]
        for result in ok_results
        if result["timing"].get("total_latency_ms") is not None
    ]
    total_costs = [
        result["cost"]["total_cost_usd"]
        for result in ok_results
        if result.get("cost", {}).get("total_cost_usd") is not None
    ]
    errors = Counter(
        result["error"]["type"]
        for result in results
        if result["status"] == "error" and result.get("error")
    )
    return {
        "calls_total": len(results),
        "calls_ok": len(ok_results),
        "calls_error": len(results) - len(ok_results),
        "error_rate": (len(results) - len(ok_results)) / len(results) if results else 0,
        "errors_by_type": dict(sorted(errors.items())),
        "latency_ms": {
            "p50": percentile(latency_values, 50),
            "p95": percentile(latency_values, 95),
            "avg": round(sum(latency_values) / len(latency_values), 3)
            if latency_values
            else None,
        },
        "cost": {
            "total_cost_usd": sum(total_costs) if total_costs else None,
            "avg_cost_per_ok_call_usd": (
                sum(total_costs) / len(total_costs) if total_costs else None
            ),
        },
        "throughput": {
            "wall_clock_ms": wall_clock_ms,
            "concurrency": concurrency,
            "requests_per_second": round(len(results) / (wall_clock_ms / 1000), 3)
            if wall_clock_ms and wall_clock_ms > 0
            else None,
        },
    }


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 3)
