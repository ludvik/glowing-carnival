#!/usr/bin/env python3
"""Validate the curated model smoke-pool configuration.

This is intentionally lightweight: it checks that the curated smoke pool is
present in the latest DigitalOcean Serverless Inference model snapshot and that
each selected model has enough local metadata for smoke-test planning and later
cost accounting.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


SMOKE_POOL_PATH = Path("config/smoke_pool.csv")
METADATA_PATH = Path("config/model_metadata.json")
RAW_MODELS_PATH = Path("data/models/raw_models_response.json")


def load_smoke_pool(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    missing_ids = [index + 2 for index, row in enumerate(rows) if not row.get("model_id")]
    if missing_ids:
        raise SystemExit(f"{path} has rows without model_id at CSV lines: {missing_ids}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path} must contain a JSON object.")
    return payload


def extract_raw_model_ids(payload: dict[str, Any]) -> set[str]:
    response = payload.get("response", payload)
    if isinstance(response, dict) and isinstance(response.get("data"), list):
        raw_models = response["data"]
    elif isinstance(response, dict) and isinstance(response.get("models"), list):
        raw_models = response["models"]
    elif isinstance(response, list):
        raw_models = response
    else:
        raise SystemExit(f"{RAW_MODELS_PATH} does not contain a recognizable model list.")

    model_ids = {
        str(model.get("id") or model.get("model")).strip()
        for model in raw_models
        if isinstance(model, dict) and (model.get("id") or model.get("model"))
    }
    if not model_ids:
        raise SystemExit(f"{RAW_MODELS_PATH} has no model IDs.")
    return model_ids


def require_numeric_price(model_id: str, record: dict[str, Any], key: str) -> None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{model_id}.{key} must be numeric.")
    if value < 0:
        raise SystemExit(f"{model_id}.{key} must not be negative.")


def main() -> int:
    smoke_pool = load_smoke_pool(SMOKE_POOL_PATH)
    metadata = load_json(METADATA_PATH)
    raw_model_ids = extract_raw_model_ids(load_json(RAW_MODELS_PATH))
    metadata_models = metadata.get("models", {})
    if not isinstance(metadata_models, dict):
        raise SystemExit("config/model_metadata.json field 'models' must be an object.")

    smoke_model_ids = [row["model_id"] for row in smoke_pool]
    duplicate_ids = sorted({model_id for model_id in smoke_model_ids if smoke_model_ids.count(model_id) > 1})
    if duplicate_ids:
        raise SystemExit(f"Duplicate smoke-pool model IDs: {', '.join(duplicate_ids)}")

    router_ids = [model_id for model_id in smoke_model_ids if model_id.startswith("router:")]
    if router_ids:
        raise SystemExit(f"Router models must not be in smoke_pool.csv: {', '.join(router_ids)}")

    missing_from_snapshot = sorted(set(smoke_model_ids) - raw_model_ids)
    if missing_from_snapshot:
        raise SystemExit(f"Smoke-pool models missing from raw model snapshot: {', '.join(missing_from_snapshot)}")

    missing_metadata = sorted(model_id for model_id in smoke_model_ids if model_id not in metadata_models)
    if missing_metadata:
        raise SystemExit(f"Smoke-pool models missing metadata: {', '.join(missing_metadata)}")

    for model_id in smoke_model_ids:
        record = metadata_models[model_id]
        require_numeric_price(model_id, record, "input_price_per_1m")
        require_numeric_price(model_id, record, "output_price_per_1m")
        if record.get("include_for_smoke_test") is not True:
            raise SystemExit(f"{model_id}.include_for_smoke_test must be true.")
        if record.get("chat_completion_supported") is not True:
            raise SystemExit(f"{model_id}.chat_completion_supported must be true.")

    router_metadata = [
        model_id
        for model_id in raw_model_ids
        if model_id.startswith("router:") and metadata_models.get(model_id, {}).get("exclude_reason") == "router_not_single_model"
    ]

    print(f"Smoke-pool models: {len(smoke_model_ids)}")
    print(f"Smoke-pool models present in raw snapshot: {len(smoke_model_ids) - len(missing_from_snapshot)}")
    print(f"Smoke-pool models with pricing metadata: {len(smoke_model_ids)}")
    print(f"Router exclusions configured: {len(router_metadata)}")
    print("Model config validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
