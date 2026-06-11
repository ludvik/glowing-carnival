#!/usr/bin/env python3
"""Build the eval runner model catalog from curated screening-pool metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build config/model_catalog.json for run_eval.py.")
    parser.add_argument("--metadata", default="config/model_metadata.json")
    parser.add_argument("--screening-pool", default="config/screening_pool.csv")
    parser.add_argument("--output", default="config/model_catalog.json")
    return parser.parse_args()


def load_screening_pool(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"{path} is empty.")
    seen: set[str] = set()
    for row in rows:
        model_id = row.get("model_id", "").strip()
        if not model_id:
            raise SystemExit(f"{path} contains an empty model_id.")
        if model_id in seen:
            raise SystemExit(f"Duplicate model_id in {path}: {model_id}")
        if model_id.startswith("router:"):
            raise SystemExit(f"Router model must not be in screening pool: {model_id}")
        seen.add(model_id)
    return rows


def numeric_price(model_id: str, record: dict[str, Any], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SystemExit(f"{model_id}.{key} must be numeric.")
    if value < 0:
        raise SystemExit(f"{model_id}.{key} must not be negative.")
    return float(value)


def main() -> int:
    args = parse_args()
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    metadata_models = metadata.get("models", {})
    screening_pool = load_screening_pool(Path(args.screening_pool))

    catalog_models: dict[str, dict[str, Any]] = {}
    for row in screening_pool:
        model_id = row["model_id"]
        model = metadata_models.get(model_id)
        if not model:
            raise SystemExit(f"Screening-pool model missing metadata: {model_id}")
        if model.get("include_for_screening") is not True:
            raise SystemExit(f"{model_id}.include_for_screening must be true.")
        if model.get("chat_completion_supported") is not True:
            raise SystemExit(f"{model_id}.chat_completion_supported must be true.")

        input_price = numeric_price(model_id, model, "input_price_per_1m")
        output_price = numeric_price(model_id, model, "output_price_per_1m")
        catalog_models[model_id] = {
            "display_name": model.get("display_name") or model_id,
            "provider": "digitalocean_si",
            "provider_name": model.get("provider"),
            "family": model.get("family"),
            "provider_model": model_id,
            "input_price_per_1m_tokens": input_price,
            "output_price_per_1m_tokens": output_price,
            "pricing_source": model.get("pricing_source", "config/model_metadata.json"),
            "pricing_captured_at": model.get("pricing_captured_at"),
            "pool_tier": row.get("pool_tier"),
            "expected_role": row.get("expected_role"),
            "why_included": row.get("why_included"),
            "context_window": model.get("context_window"),
            "max_output_tokens": model.get("max_output_tokens"),
        }

    output = {
        "pricing_source": "config/model_metadata.json",
        "models": catalog_models,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Catalog model count: {len(catalog_models)}")
    print("Model ids:")
    for model_id in catalog_models:
        print(f"  {model_id}")
    print(f"Output path: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
