from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from typing import Any

from eval_harness.errors import InferenceError, classify_http_error


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
                return json.loads(response.read().decode("utf-8"))
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
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") or ""
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content)


def extract_usage(response: dict[str, Any]) -> dict[str, Any] | None:
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
