from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_model_catalog(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["models"]


def resolve_models(catalog: dict[str, Any], model_ids: list[str]) -> dict[str, dict[str, Any]]:
    missing = [model_id for model_id in model_ids if model_id not in catalog]
    if missing:
        available = ", ".join(sorted(catalog))
        raise ValueError(f"Unknown model id(s): {', '.join(missing)}. Available: {available}")
    return {model_id: catalog[model_id] for model_id in model_ids}
