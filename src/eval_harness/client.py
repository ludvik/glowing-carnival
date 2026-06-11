from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from eval_harness.errors import InferenceError, classify_http_error
from eval_harness.errors import extract_rate_limit_headers


class DigitalOceanSIClient:
    def __init__(self, api_key: str, base_url: str = "https://inference.do-ai.run") -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def chat_completion(
        self,
        provider_model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/v1/chat/completions"
        body = json.dumps(
            {
                "model": provider_model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_output_tokens,
                "stream": False,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "glowing-carnival-eval-engine",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
                return {
                    "body": body,
                    "headers": extract_rate_limit_headers(response.headers),
                }
        except urllib.error.HTTPError as exc:
            raise classify_http_error(exc) from exc
        except urllib.error.URLError as exc:
            message = str(exc.reason)
            if isinstance(exc.reason, socket.timeout):
                raise InferenceError("timeout", message, retryable=True) from exc
            raise InferenceError("network_error", message, retryable=True) from exc
        except TimeoutError as exc:
            raise InferenceError("timeout", str(exc), retryable=True) from exc


def extract_content(response: dict[str, Any]) -> str:
    response = response.get("body", response)
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def extract_usage(response: dict[str, Any]) -> dict[str, Any] | None:
    response = response.get("body", response)
    usage = response.get("usage")
    if not usage:
        return None
    input_tokens = usage.get("prompt_tokens", usage.get("input_tokens"))
    output_tokens = usage.get("completion_tokens", usage.get("output_tokens"))
    total_tokens = usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_source": "provider",
    }


def extract_response_headers(response: dict[str, Any]) -> dict[str, Any]:
    return response.get("headers", {})


def extract_response_debug(response: dict[str, Any]) -> dict[str, Any]:
    """Return a sanitized response-body shape for debugging parse failures.

    This intentionally excludes request headers and credentials. It is only
    meant to help distinguish true empty model output from response-shape
    mismatches in OpenAI-compatible providers.
    """

    body = response.get("body", response)
    choices = body.get("choices") if isinstance(body, dict) else None
    debug: dict[str, Any] = {
        "body_keys": sorted(body.keys()) if isinstance(body, dict) else [],
        "choices_count": len(choices) if isinstance(choices, list) else None,
    }
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message") if isinstance(first.get("message"), dict) else {}
        content = message.get("content")
        reasoning_content = message.get("reasoning_content")
        debug["first_choice_keys"] = sorted(first.keys())
        debug["finish_reason"] = first.get("finish_reason")
        debug["message_keys"] = sorted(message.keys())
        debug["message_content_type"] = type(content).__name__
        debug["message_content_length"] = len(content) if isinstance(content, str) else None
        debug["message_content_preview"] = content[:500] if isinstance(content, str) else None
        debug["reasoning_content_type"] = type(reasoning_content).__name__
        debug["reasoning_content_length"] = (
            len(reasoning_content) if isinstance(reasoning_content, str) else None
        )
        debug["reasoning_content_preview"] = (
            reasoning_content[:500] if isinstance(reasoning_content, str) else None
        )
        debug["message_role"] = message.get("role")
    return debug
