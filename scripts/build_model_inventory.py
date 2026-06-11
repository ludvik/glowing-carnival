#!/usr/bin/env python3
"""Capture a reproducible DigitalOcean Serverless Inference model inventory.

This script is Step 2 of the evaluation funnel. It does not classify issues and
does not call chat completions. It captures the models visible to the configured
DigitalOcean Serverless Inference key, normalizes model metadata, merges local
pricing/capability annotations, and writes explainable eligible/excluded lists
for later model screening.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


DEFAULT_BASE_URL = "https://inference.do-ai.run"
SOURCE = "digitalocean_serverless_inference"
TEXT_MODALITIES = {"text", "multimodal"}
CSV_COLUMNS = [
    "model_id",
    "display_name",
    "provider",
    "family",
    "object_type",
    "owned_by",
    "created",
    "modality",
    "supported_endpoints",
    "serverless_visible",
    "chat_completion_supported",
    "responses_supported",
    "embedding_supported",
    "image_supported",
    "audio_supported",
    "video_supported",
    "context_window",
    "max_output_tokens",
    "input_price_per_1m",
    "output_price_per_1m",
    "pricing_source",
    "pricing_captured_at",
    "pricing_missing",
    "cost_ready",
    "eligibility_status",
    "exclusion_reason",
    "notes",
    "fetched_at_utc",
]
REVIEW_COLUMNS = CSV_COLUMNS + ["suggested_review_reason"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a DigitalOcean Serverless Inference model inventory snapshot."
    )
    parser.add_argument("--output-dir", default="data/models")
    parser.add_argument("--metadata", default="config/model_metadata.json")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key-env", default="DIGITALOCEAN_SI_API_KEY")
    parser.add_argument("--offline-fixture", default=None)
    parser.add_argument(
        "--allow-missing-pricing",
        action="store_true",
        help=(
            "Backward-compatible no-op for screening inventory. Missing pricing is "
            "always allowed for model screening, but cost/pilot/final evaluation "
            "must validate pricing metadata."
        ),
    )
    parser.add_argument("--include-unknown-chat-models", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=30)
    return parser.parse_args()


def normalize_base_url(value: str) -> str:
    """Return a base URL without a trailing `/v1`.

    DigitalOcean accepts an OpenAI-compatible base URL. Users often configure
    either the service root or the `/v1` root, so normalize before appending
    `/v1/models` to avoid calling `/v1/v1/models`.
    """

    base_url = (value or DEFAULT_BASE_URL).strip().rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[:-3].rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise SystemExit(f"Invalid base URL: {value}")
    return base_url


def fetch_models(base_url: str, api_key: str, timeout_seconds: float) -> dict[str, Any]:
    """Fetch the raw model response without logging or persisting credentials."""

    endpoint = f"{base_url}/v1/models"
    request = urllib.request.Request(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        message = exc.read().decode("utf-8", errors="replace")
        if exc.code in {401, 403}:
            raise SystemExit(
                f"GET {endpoint} failed with HTTP {exc.code}. "
                "The model access key may be missing, invalid, or scoped to no models."
            ) from exc
        if exc.code == 429:
            raise SystemExit(
                f"GET {endpoint} failed with HTTP 429. The models endpoint was rate limited."
            ) from exc
        raise SystemExit(f"GET {endpoint} failed with HTTP {exc.code}: {message}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"GET {endpoint} failed: {exc.reason}") from exc

    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Models endpoint returned invalid JSON: {exc}") from exc


def load_offline_fixture(path: Path) -> dict[str, Any]:
    """Load either a raw `/v1/models` response or this script's wrapped output."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Offline fixture is invalid JSON: {path}: {exc}") from exc
    if isinstance(payload, dict) and "response" in payload and "source" in payload:
        return payload["response"]
    return payload


def extract_model_list(response: Any) -> list[dict[str, Any]]:
    """Extract a recognizable model list from common OpenAI-compatible shapes."""

    if isinstance(response, list):
        models = response
    elif isinstance(response, dict) and isinstance(response.get("data"), list):
        models = response["data"]
    elif isinstance(response, dict) and isinstance(response.get("models"), list):
        models = response["models"]
    else:
        raise SystemExit("API response does not contain a recognizable model list.")

    normalized = []
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            raise SystemExit(f"Model entry at index {index} is not an object.")
        normalized.append(model)
    return normalized


def ensure_example_metadata_config(path: Path) -> None:
    """Create the example metadata config if the repository does not have one."""

    example_path = path.parent / "model_metadata.example.json"
    if example_path.exists():
        return
    example_path.write_text(
        json.dumps(
            {
                "models": {
                    "example-model-id": {
                        "display_name": "Example Model",
                        "provider": "Example Provider",
                        "family": "Example Family",
                        "modality": "text",
                        "supported_endpoints": ["chat_completions"],
                        "chat_completion_supported": True,
                        "responses_supported": False,
                        "embedding_supported": False,
                        "image_supported": False,
                        "audio_supported": False,
                        "video_supported": False,
                        "context_window": 131072,
                        "max_output_tokens": 4096,
                        "input_price_per_1m": 0.25,
                        "output_price_per_1m": 0.75,
                        "pricing_source": "DigitalOcean pricing page, captured manually",
                        "pricing_captured_at": "2026-06-10",
                        "include_for_screening": True,
                        "exclude_reason": "",
                        "notes": "Cost-efficient candidate",
                    }
                },
                "provider_aliases": {
                    "openai": "OpenAI",
                    "anthropic": "Anthropic",
                    "meta": "Meta",
                    "mistral": "Mistral",
                    "deepseek": "DeepSeek",
                    "qwen": "Alibaba/Qwen",
                },
                "exclude_id_patterns": [
                    "embedding",
                    "embed",
                    "rerank",
                    "image",
                    "audio",
                    "speech",
                    "tts",
                    "video",
                    "fal",
                    "whisper",
                    "all-mini-lm",
                    "bge-",
                    "e5-",
                    "gte-",
                    "multi-qa",
                    "mpnet",
                    "sentence-transformer",
                    "^router:",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def load_metadata_config(path: Path) -> tuple[dict[str, Any], bool]:
    """Load local metadata overlay, or proceed with deterministic heuristics."""

    ensure_example_metadata_config(path)
    if not path.exists():
        print(
            f"WARNING: {path} does not exist. Proceeding with API metadata and heuristics only; "
            "create config/model_metadata.json before final eval.",
            file=sys.stderr,
        )
        return {
            "models": {},
            "provider_aliases": default_provider_aliases(),
            "exclude_id_patterns": default_exclude_patterns(),
        }, False

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Metadata config is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Metadata config must be a JSON object: {path}")
    payload.setdefault("models", {})
    payload.setdefault("provider_aliases", default_provider_aliases())
    payload.setdefault("exclude_id_patterns", default_exclude_patterns())
    validate_metadata_pricing(payload)
    return payload, True


def default_provider_aliases() -> dict[str, str]:
    return {
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "meta": "Meta",
        "mistral": "Mistral",
        "deepseek": "DeepSeek",
        "qwen": "Alibaba/Qwen",
        "google": "Google",
        "cohere": "Cohere",
        "stability": "Stability AI",
    }


def default_exclude_patterns() -> list[str]:
    return [
        "embedding",
        "embed",
        "rerank",
        "image",
        "audio",
        "speech",
        "tts",
        "video",
        "fal",
        "whisper",
        "all-mini-lm",
        "bge-",
        "e5-",
        "gte-",
        "multi-qa",
        "mpnet",
        "sentence-transformer",
        "^router:",
    ]


def validate_metadata_pricing(metadata: dict[str, Any]) -> None:
    for model_id, record in metadata.get("models", {}).items():
        for key in ("input_price_per_1m", "output_price_per_1m"):
            if key not in record or record[key] in ("", None):
                continue
            value = coerce_number(record[key], f"{model_id}.{key}")
            if value < 0:
                raise SystemExit(f"{model_id}.{key} must not be negative.")


def normalize_model_record(
    raw_model: dict[str, Any],
    metadata: dict[str, Any],
    fetched_at_utc: str,
    allow_missing_pricing: bool,
    include_unknown_chat_models: bool,
) -> dict[str, Any]:
    """Normalize API and local metadata into one explainable inventory row."""

    model_id = str(raw_model.get("id") or raw_model.get("model") or "").strip()
    if not model_id:
        raise SystemExit(f"Model entry is missing an id: {raw_model}")

    local = metadata.get("models", {}).get(model_id, {})
    provider = infer_provider(model_id, raw_model, local, metadata)
    family = infer_family(model_id, raw_model, local, provider)
    modality = infer_modality(model_id, raw_model, local)
    endpoints = infer_supported_endpoints(model_id, raw_model, local, modality)

    input_price = normalize_optional_number(local.get("input_price_per_1m"), f"{model_id}.input_price_per_1m")
    output_price = normalize_optional_number(local.get("output_price_per_1m"), f"{model_id}.output_price_per_1m")
    pricing_missing = input_price is None or output_price is None

    row = {
        "model_id": model_id,
        "display_name": local.get("display_name") or raw_model.get("display_name") or raw_model.get("name") or model_id,
        "provider": provider,
        "family": family,
        "object_type": raw_model.get("object") or "",
        "owned_by": raw_model.get("owned_by") or raw_model.get("owner") or "",
        "created": raw_model.get("created"),
        "modality": modality,
        "supported_endpoints": endpoints,
        "serverless_visible": True,
        "chat_completion_supported": bool(local.get("chat_completion_supported", "chat_completions" in endpoints)),
        "responses_supported": bool(local.get("responses_supported", "responses" in endpoints)),
        "embedding_supported": bool(local.get("embedding_supported", "embeddings" in endpoints)),
        "image_supported": bool(local.get("image_supported", modality == "image")),
        "audio_supported": bool(local.get("audio_supported", modality == "audio")),
        "video_supported": bool(local.get("video_supported", modality == "video")),
        "context_window": local.get("context_window") or raw_model.get("context_window") or raw_model.get("context_length"),
        "max_output_tokens": local.get("max_output_tokens") or raw_model.get("max_output_tokens"),
        "input_price_per_1m": input_price,
        "output_price_per_1m": output_price,
        "pricing_source": local.get("pricing_source") or "",
        "pricing_captured_at": local.get("pricing_captured_at") or "",
        "pricing_missing": pricing_missing,
        "cost_ready": not pricing_missing,
        "eligibility_status": "",
        "exclusion_reason": "",
        "notes": local.get("notes") or "",
        "fetched_at_utc": fetched_at_utc,
        "raw_model": raw_model,
        "_local_metadata": local,
    }
    decision = decide_eligibility(row, metadata, allow_missing_pricing, include_unknown_chat_models)
    row.update(decision)
    row.pop("_local_metadata", None)
    return row


def infer_provider(
    model_id: str,
    raw_model: dict[str, Any],
    local: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    if local.get("provider"):
        return str(local["provider"])
    owned_by = str(raw_model.get("owned_by") or raw_model.get("owner") or "").strip()
    aliases = metadata.get("provider_aliases", {})
    if owned_by:
        return aliases.get(owned_by.lower(), owned_by)

    lowered = model_id.lower()
    for token, provider in aliases.items():
        if token.lower() in lowered:
            return provider
    if "llama" in lowered:
        return "Meta"
    if "mistral" in lowered or "ministral" in lowered or "mixtral" in lowered:
        return "Mistral"
    if "gpt-oss" in lowered or "openai" in lowered:
        return "OpenAI"
    if "deepseek" in lowered:
        return "DeepSeek"
    if "qwen" in lowered:
        return "Alibaba/Qwen"
    if "gemma" in lowered:
        return "Google"
    return "unknown"


def infer_family(model_id: str, raw_model: dict[str, Any], local: dict[str, Any], provider: str) -> str:
    if local.get("family"):
        return str(local["family"])
    lowered = model_id.lower()
    known_tokens = [
        ("gpt-oss", "gpt-oss"),
        ("llama", "llama"),
        ("mistral", "mistral"),
        ("ministral", "ministral"),
        ("mixtral", "mixtral"),
        ("deepseek", "deepseek"),
        ("qwen", "qwen"),
        ("gemma", "gemma"),
        ("glm", "glm"),
        ("kimi", "kimi"),
        ("minimax", "minimax"),
        ("nemotron", "nemotron"),
        ("trinity", "trinity"),
        ("claude", "claude"),
        ("gemini", "gemini"),
        ("embedding", "embedding"),
        ("rerank", "rerank"),
        ("whisper", "whisper"),
    ]
    for token, family in known_tokens:
        if token in lowered:
            return family
    if raw_model.get("family"):
        return str(raw_model["family"])
    return provider.lower() if provider and provider != "unknown" else "unknown"


def infer_modality(model_id: str, raw_model: dict[str, Any], local: dict[str, Any]) -> str:
    if local.get("modality"):
        return str(local["modality"])
    if raw_model.get("modality"):
        return str(raw_model["modality"])

    lowered = model_id.lower()
    if "bge-reranker" in lowered or "rerank" in lowered:
        return "rerank"
    if re.search(r"embed|embedding|text-embedding|all-mini-lm|bge-|e5-|gte-|multi-qa|mpnet|sentence-transformer", lowered):
        return "embedding"
    if re.search(r"openai-gpt-image-|image|vision|sdxl|stable-diffusion", lowered):
        return "image"
    if re.search(r"qwen3-tts-|audio|speech|tts|whisper", lowered):
        return "audio"
    if re.search(r"video|wan.*t2v", lowered):
        return "video"
    if "fal" in lowered:
        return "async"
    return "text"


def infer_supported_endpoints(model_id: str, raw_model: dict[str, Any], local: dict[str, Any], modality: str) -> list[str]:
    if isinstance(local.get("supported_endpoints"), list):
        return [str(endpoint) for endpoint in local["supported_endpoints"]]
    raw_endpoints = raw_model.get("supported_endpoints") or raw_model.get("endpoints")
    if isinstance(raw_endpoints, list):
        return [str(endpoint) for endpoint in raw_endpoints]

    if modality == "embedding":
        return ["embeddings"]
    if modality == "rerank":
        return ["rerank"]
    if modality in {"image", "audio", "video", "async"}:
        return [modality]

    lowered = model_id.lower()
    generative_tokens = (
        "gpt",
        "claude",
        "llama",
        "mistral",
        "ministral",
        "deepseek",
        "qwen",
        "gemma",
        "glm",
        "kimi",
        "minimax",
        "nemotron",
        "trinity",
        "opus",
        "sonnet",
        "haiku",
        "o1",
        "o3",
        "instruct",
        "thinking",
        "coder",
    )
    if any(token in lowered for token in generative_tokens):
        return ["chat_completions"]
    return []


def decide_eligibility(
    row: dict[str, Any],
    metadata: dict[str, Any],
    allow_missing_pricing: bool,
    include_unknown_chat_models: bool,
) -> dict[str, Any]:
    """Place every visible model into eligible, excluded, or review."""

    model_id = row["model_id"]
    local = row.get("_local_metadata", {})
    lowered = model_id.lower()
    notes = row.get("notes") or ""
    explicitly_included = local.get("include_for_screening") is True

    if lowered.startswith("router:") and not explicitly_included:
        return {
            "eligibility_status": "needs_metadata_review",
            "exclusion_reason": "",
            "suggested_review_reason": "router_not_single_model",
            "notes": append_note(notes, "Router model is not a single final model candidate by default."),
        }

    explicit_exclude = str(local.get("exclude_reason") or "").strip()
    if explicit_exclude:
        return {
            "eligibility_status": "excluded",
            "exclusion_reason": explicit_exclude,
            "suggested_review_reason": "",
            "notes": notes,
        }
    if local.get("include_for_screening") is False:
        return {
            "eligibility_status": "excluded",
            "exclusion_reason": "metadata_include_false",
            "suggested_review_reason": "",
            "notes": notes,
        }
    if local.get("deprecated") or local.get("unavailable"):
        return {
            "eligibility_status": "excluded",
            "exclusion_reason": "deprecated_or_unavailable",
            "suggested_review_reason": "",
            "notes": notes,
        }

    for pattern in metadata.get("exclude_id_patterns", default_exclude_patterns()):
        if re.search(str(pattern), lowered):
            return {
                "eligibility_status": "excluded",
                "exclusion_reason": f"excluded_id_pattern:{pattern}",
                "suggested_review_reason": "",
                "notes": notes,
            }

    non_text_flags = [
        row["embedding_supported"],
        row["image_supported"],
        row["audio_supported"],
        row["video_supported"],
        row["modality"] in {"embedding", "rerank", "image", "audio", "video", "async"},
    ]
    if any(non_text_flags):
        return {
            "eligibility_status": "excluded",
            "exclusion_reason": f"non_text_modality:{row['modality']}",
            "suggested_review_reason": "",
            "notes": notes,
        }

    chat_capable = row["chat_completion_supported"] or "chat_completions" in row["supported_endpoints"]
    if local.get("chat_completion_supported") is False:
        return {
            "eligibility_status": "excluded",
            "exclusion_reason": "chat_completions_not_supported",
            "suggested_review_reason": "",
            "notes": notes,
        }

    if not chat_capable:
        if include_unknown_chat_models:
            status = "screening_eligible_needs_pricing" if row["pricing_missing"] else "screening_eligible"
            notes = append_note(notes, "Endpoint support inferred as uncertain; included by flag.")
        else:
            return {
                "eligibility_status": "needs_metadata_review",
                "exclusion_reason": "",
                "suggested_review_reason": "chat_capability_uncertain",
                "notes": append_note(notes, "Need to confirm chat completions support."),
            }
    else:
        status = "screening_eligible"

    if row["pricing_missing"]:
        status = "screening_eligible_needs_pricing"
        notes = append_note(
            notes,
            "Pricing missing; allowed for screening run but must be filled before cost/pilot/final evaluation.",
        )
    elif allow_missing_pricing:
        notes = append_note(
            notes,
            "--allow-missing-pricing is no longer required for screening eligibility; pricing is present for this model.",
        )

    return {
        "eligibility_status": status,
        "exclusion_reason": "",
        "suggested_review_reason": "",
        "notes": notes,
    }


def append_note(existing: str, addition: str) -> str:
    return f"{existing} {addition}".strip() if existing else addition


def normalize_optional_number(value: Any, field_name: str) -> float | None:
    if value in ("", None):
        return None
    number = coerce_number(value, field_name)
    if number < 0:
        raise SystemExit(f"{field_name} must not be negative.")
    return number


def coerce_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise SystemExit(f"{field_name} must be numeric, not boolean.")
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError as exc:
        raise SystemExit(f"{field_name} must be numeric: {value}") from exc


def csv_row(record: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    row = {}
    for column in columns:
        value = record.get(column, "")
        if isinstance(value, list):
            value = ";".join(value)
        elif isinstance(value, bool):
            value = str(value).lower()
        elif value is None:
            value = ""
        row[column] = value
    return row


def write_csv(path: Path, records: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow(csv_row(record, columns))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_outputs(
    all_records: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    review: list[dict[str, Any]],
) -> None:
    ids = [record["model_id"] for record in all_records]
    if any(not model_id for model_id in ids):
        raise SystemExit("A CSV row has an empty model_id.")
    duplicates = sorted(model_id for model_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise SystemExit(f"Duplicate model_id values after normalization: {', '.join(duplicates)}")

    eligible_ids = {record["model_id"] for record in eligible}
    excluded_ids = {record["model_id"] for record in excluded}
    review_ids = {record["model_id"] for record in review}
    overlap = eligible_ids & excluded_ids
    if overlap:
        raise SystemExit(f"Model appears in both eligible and excluded outputs: {sorted(overlap)}")
    if eligible_ids & review_ids or excluded_ids & review_ids:
        raise SystemExit("A model appears in multiple final routing outputs.")

    routed_ids = eligible_ids | excluded_ids | review_ids
    if set(ids) != routed_ids:
        missing = sorted(set(ids) - routed_ids)
        extra = sorted(routed_ids - set(ids))
        raise SystemExit(f"Routing mismatch. Missing={missing}; extra={extra}")

    for record in all_records:
        for key in ("input_price_per_1m", "output_price_per_1m"):
            value = record.get(key)
            if value is not None and value < 0:
                raise SystemExit(f"{record['model_id']} has negative {key}.")


def print_summary(summary: dict[str, Any], output_paths: dict[str, Path]) -> None:
    print(f"Total visible models: {summary['total_visible_models']}")
    print(f"Screening eligible count: {summary['screening_eligible_count']}")
    print(f"Screening eligible needs pricing count: {summary['screening_eligible_needs_pricing_count']}")
    print(f"Pricing needed count: {summary['pricing_needed_count']}")
    print(f"Cost ready count: {summary['cost_ready_count']}")
    print(f"Excluded count: {summary['excluded_count']}")
    print(f"Needs metadata review count: {summary['needs_metadata_review_count']}")
    print(f"Missing pricing count: {summary['missing_pricing_count']}")
    if summary["pricing_needed_count"]:
        print(
            "WARNING: "
            f"{summary['pricing_needed_count']} screening-eligible models are missing pricing metadata. "
            "They can be screening-tested, but cost/pilot/final eval requires pricing enrichment."
        )
    print("Provider counts:")
    for provider, count in summary["provider_counts"].items():
        print(f"  {provider}: {count}")
    print("Top exclusion reasons:")
    for reason, count in list(summary["exclusion_reason_counts"].items())[:10]:
        print(f"  {reason}: {count}")
    print("Output paths:")
    for path in output_paths.values():
        print(f"  {path}")


def main() -> int:
    args = parse_args()
    env_base_url = os.environ.get("DO_INFERENCE_BASE_URL")
    base_url = normalize_base_url(env_base_url or args.base_url)
    endpoint = f"{base_url}/v1/models"
    fetched_at_utc = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    output_dir = Path(args.output_dir)
    metadata_path = Path(args.metadata)

    metadata, metadata_exists = load_metadata_config(metadata_path)
    if args.offline_fixture:
        response = load_offline_fixture(Path(args.offline_fixture))
    else:
        api_key = os.environ.get(args.api_key_env)
        if not api_key:
            raise SystemExit(
                f"Missing {args.api_key_env}. Provide a DigitalOcean model access key "
                "or pass --offline-fixture."
            )
        response = fetch_models(base_url, api_key, args.timeout_seconds)

    raw_wrapper = {
        "fetched_at_utc": fetched_at_utc,
        "base_url": base_url,
        "endpoint": endpoint,
        "source": SOURCE,
        "response": response,
    }
    raw_sha256 = sha256_json(response)
    raw_models = extract_model_list(response)
    records = [
        normalize_model_record(
            raw_model,
            metadata,
            fetched_at_utc,
            args.allow_missing_pricing,
            args.include_unknown_chat_models,
        )
        for raw_model in raw_models
    ]
    records.sort(key=lambda record: record["model_id"])

    eligible = [
        record
        for record in records
        if record["eligibility_status"] in {"screening_eligible", "screening_eligible_needs_pricing"}
    ]
    cost_ready = [record for record in eligible if record["cost_ready"]]
    pricing_needed = [record for record in eligible if record["pricing_missing"]]
    excluded = [record for record in records if record["eligibility_status"] == "excluded"]
    review = [record for record in records if record["eligibility_status"] == "needs_metadata_review"]
    validate_outputs(records, eligible, excluded, review)

    provider_counts = Counter(record["provider"] for record in records)
    family_counts = Counter(record["family"] for record in records)
    modality_counts = Counter(record["modality"] for record in records)
    exclusion_reason_counts = Counter(record["exclusion_reason"] for record in excluded)
    missing_pricing_count = sum(1 for record in records if record["pricing_missing"])
    cost_ready_count = len(cost_ready)
    screening_eligible_count = len(eligible)
    screening_eligible_needs_pricing_count = len(pricing_needed)
    pricing_needed_count = len(pricing_needed)
    notes = [
        "Snapshot reflects models visible to the configured DigitalOcean Serverless Inference key at fetch time.",
        "API key and Authorization header are never persisted.",
        "Pricing comes from local metadata; missing pricing is allowed for model screening but must be filled before cost/pilot/final evaluation.",
    ]
    if not metadata_exists:
        notes.append("Metadata config was not found; deterministic API/ID heuristics were used.")
    if args.offline_fixture:
        notes.append(f"Snapshot was generated from offline fixture: {args.offline_fixture}.")

    summary = {
        "fetched_at_utc": fetched_at_utc,
        "base_url": base_url,
        "endpoint": endpoint,
        "total_visible_models": len(records),
        "eligible_count": len(eligible),
        "screening_eligible_count": screening_eligible_count,
        "screening_eligible_needs_pricing_count": screening_eligible_needs_pricing_count,
        "excluded_count": len(excluded),
        "needs_metadata_review_count": len(review),
        "provider_counts": dict(sorted(provider_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "modality_counts": dict(sorted(modality_counts.items())),
        "exclusion_reason_counts": dict(exclusion_reason_counts.most_common()),
        "missing_pricing_count": missing_pricing_count,
        "cost_ready_count": cost_ready_count,
        "pricing_needed_count": pricing_needed_count,
        "metadata_file": str(metadata_path),
        "raw_response_sha256": raw_sha256,
        "notes": notes,
    }

    snapshot_records = [
        {key: value for key, value in record.items() if key != "suggested_review_reason"}
        for record in records
    ]
    output_paths = {
        "raw_models_response": output_dir / "raw_models_response.json",
        "models_snapshot": output_dir / "models_snapshot.json",
        "models_inventory": output_dir / "models_inventory.csv",
        "eligible_models": output_dir / "eligible_models.csv",
        "cost_ready_models": output_dir / "cost_ready_models.csv",
        "pricing_needed": output_dir / "pricing_needed.csv",
        "excluded_models": output_dir / "excluded_models.csv",
        "needs_metadata_review": output_dir / "needs_metadata_review.csv",
        "model_inventory_summary": output_dir / "model_inventory_summary.json",
    }

    write_json(output_paths["raw_models_response"], raw_wrapper)
    write_json(output_paths["models_snapshot"], {"models": snapshot_records})
    write_csv(output_paths["models_inventory"], records, CSV_COLUMNS)
    write_csv(output_paths["eligible_models"], eligible, CSV_COLUMNS)
    write_csv(output_paths["cost_ready_models"], cost_ready, CSV_COLUMNS)
    write_csv(output_paths["pricing_needed"], pricing_needed, CSV_COLUMNS)
    write_csv(output_paths["excluded_models"], excluded, CSV_COLUMNS)
    write_csv(output_paths["needs_metadata_review"], review, REVIEW_COLUMNS)
    write_json(output_paths["model_inventory_summary"], summary)

    print_summary(summary, output_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
