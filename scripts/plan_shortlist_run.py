#!/usr/bin/env python3
"""Plan which shortlist models still need a full-corpus run.

This script is intentionally read-only. It scans existing resultsets, screening
summaries, and the current model catalog, then prints a concise plan that avoids
rerunning models with full 530-issue resultsets and skips models whose previous
screening runs indicate they are too slow or systematically failing.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


FULL_ISSUE_COUNT = 530
MAX_PROJECTED_SECONDS = 600


def load_catalog() -> list[str]:
    payload = json.loads(Path("config/model_catalog.json").read_text(encoding="utf-8"))
    return list(payload["models"].keys())


def load_full_resultsets() -> dict[str, dict[str, Any]]:
    full: dict[str, dict[str, Any]] = {}
    for path in Path("runs").glob("*/results/*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        model_id = payload.get("model_id") or path.stem
        result_count = payload.get("result_count") or len(payload.get("results", []))
        if result_count < FULL_ISSUE_COUNT:
            continue
        full[model_id] = {
            "path": str(path),
            "result_count": result_count,
            "calls_ok": payload.get("operational_summary", {}).get("calls_ok"),
            "calls_error": payload.get("operational_summary", {}).get("calls_error"),
            "p95_latency_ms": payload.get("operational_summary", {}).get("latency_ms", {}).get("p95"),
            "requests_per_second": payload.get("operational_summary", {}).get("throughput", {}).get("requests_per_second"),
        }
    return full


def load_best_screening() -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for path in Path("runs").glob("*/smoke_summary.csv"):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                model_id = row["model_id"]
                record = {
                    "path": str(path),
                    "calls_ok": int(float(row.get("calls_ok") or 0)),
                    "calls_total": int(float(row.get("calls_total") or 0)),
                    "success_rate": number(row.get("success_rate")),
                    "strict_valid_output_rate": number(row.get("strict_valid_output_rate")),
                    "p95_latency_ms": number(row.get("p95_latency_ms")),
                    "requests_per_second": number(row.get("requests_per_second")),
                    "decision": row.get("decision") or "",
                    "dominant_error_type": row.get("dominant_error_type") or "",
                }
                old = best.get(model_id)
                if old is None or sort_key(record) > sort_key(old):
                    best[model_id] = record
    return best


def number(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def sort_key(record: dict[str, Any]) -> tuple[int, float, float]:
    p95 = record.get("p95_latency_ms")
    return (
        int(record.get("calls_ok") or 0),
        float(record.get("strict_valid_output_rate") or 0),
        -float(p95 if p95 is not None else 999999),
    )


def projected_seconds(record: dict[str, Any] | None, model_id: str) -> float | None:
    if record and record.get("requests_per_second"):
        return FULL_ISSUE_COUNT / float(record["requests_per_second"])
    if model_id.startswith("router:"):
        # Router models have little or no screening history. Evaluate them as a
        # separate router-policy group instead of excluding them for missing data.
        return None
    return None


def main() -> int:
    catalog = load_catalog()
    full = load_full_resultsets()
    screening = load_best_screening()
    selected: list[str] = []

    print("model_id,status,reason,full_resultset,screening_source,projected_seconds")
    for model_id in catalog:
        if model_id in full:
            print(f"{model_id},skip,already_has_full_result,{full[model_id]['path']},,")
            continue
        record = screening.get(model_id)
        projected = projected_seconds(record, model_id)
        if record and record.get("dominant_error_type") == "auth_error":
            print(f"{model_id},skip,auth_error_in_screening,,{record['path']},{projected or ''}")
            continue
        if record and (record.get("calls_ok") or 0) == 0 and not model_id.startswith("router:"):
            print(f"{model_id},skip,no_successful_screening_calls,,{record['path']},{projected or ''}")
            continue
        if projected is not None and projected > MAX_PROJECTED_SECONDS:
            print(f"{model_id},skip,projected_over_10_minutes,,{record['path']},{round(projected, 1)}")
            continue
        if model_id.startswith("router:"):
            selected.append(model_id)
            print(f"{model_id},run,router_policy_candidate_without_full_result,,{record['path'] if record else ''},{projected or ''}")
            continue
        selected.append(model_id)
        print(f"{model_id},run,missing_full_result_and_within_time_budget,,{record['path'] if record else ''},{round(projected, 1) if projected else ''}")

    print()
    print("RUN_MODELS=" + ",".join(selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
