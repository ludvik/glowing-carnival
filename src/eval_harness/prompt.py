from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LABELS = ("bug", "enhancement", "question", "documentation", "security", "other")


def load_prompt(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_messages(system_prompt: str, issue: dict[str, Any]) -> list[dict[str, str]]:
    body = issue["body"].strip()
    if len(body) > 12000:
        body = body[:12000] + "\n\n[Body truncated for evaluation prompt.]"

    user_payload = {
        "issue_number": issue["issue_number"],
        "title": issue["title"],
        "body": body,
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]


def parse_model_output(raw_output: str) -> tuple[str | None, str | None, str | None]:
    text = raw_output.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, None, "No JSON object found in model output."

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, None, f"Invalid JSON: {exc}"

    label = payload.get("label")
    rationale = payload.get("rationale")
    if label not in LABELS:
        return None, rationale, f"Invalid label: {label!r}"
    if rationale is not None and not isinstance(rationale, str):
        rationale = str(rationale)
    return label, rationale, None
