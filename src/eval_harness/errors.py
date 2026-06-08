from __future__ import annotations

import urllib.error


class InferenceError(Exception):
    def __init__(
        self,
        error_type: str,
        message: str,
        http_status: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.http_status = http_status
        self.retryable = retryable


def classify_http_error(exc: urllib.error.HTTPError) -> InferenceError:
    try:
        detail = exc.read().decode("utf-8", errors="replace")
    except Exception:
        detail = str(exc)

    if exc.code == 429:
        return InferenceError("rate_limit", detail, exc.code, retryable=True)
    if exc.code in {401, 403}:
        return InferenceError("auth_error", detail, exc.code, retryable=False)
    if 500 <= exc.code < 600:
        return InferenceError("api_error", detail, exc.code, retryable=True)
    return InferenceError("client_error", detail, exc.code, retryable=False)
