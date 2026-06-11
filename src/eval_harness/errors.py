from __future__ import annotations

import urllib.error
from typing import Any


class InferenceError(Exception):
    def __init__(
        self,
        error_type: str,
        message: str,
        http_status: int | None = None,
        retryable: bool = False,
        headers: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.http_status = http_status
        self.retryable = retryable
        self.headers = headers or {}


def classify_http_error(exc: urllib.error.HTTPError) -> InferenceError:
    try:
        detail = exc.read().decode("utf-8", errors="replace")
    except Exception:
        detail = str(exc)

    headers = extract_rate_limit_headers(exc.headers)
    if exc.code == 429:
        return InferenceError("rate_limit", detail, exc.code, retryable=True, headers=headers)
    if exc.code in {401, 403}:
        return InferenceError("auth_error", detail, exc.code, retryable=False, headers=headers)
    if 500 <= exc.code < 600:
        return InferenceError("api_error", detail, exc.code, retryable=True, headers=headers)
    return InferenceError("client_error", detail, exc.code, retryable=False, headers=headers)


def extract_rate_limit_headers(headers: Any) -> dict[str, int | str]:
    values: dict[str, int | str] = {}
    names = (
        "ratelimit-limit",
        "ratelimit-remaining",
        "ratelimit-reset",
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-reset-requests",
        "x-ratelimit-limit-tokens-per-minute",
        "x-ratelimit-remaining-tokens-per-minute",
        "x-ratelimit-reset-tokens-per-minute",
        "x-ratelimit-limit-tokens-per-day",
        "x-ratelimit-remaining-tokens-per-day",
        "x-ratelimit-reset-tokens-per-day",
        "x-request-id",
        "request-id",
        "do-upstream-service-time",
    )
    for name in names:
        value = headers.get(name) if headers else None
        if value is None:
            continue
        try:
            values[name] = int(value)
        except (TypeError, ValueError):
            values[name] = str(value)
    return values
