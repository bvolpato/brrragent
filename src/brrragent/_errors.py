import re

_RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
_RATE_LIMIT_CODES = {
    "INSUFFICIENT_QUOTA",
    "QUOTA_EXCEEDED",
    "RATE_LIMIT_EXCEEDED",
    "RESOURCE_EXHAUSTED",
    "TOO_MANY_REQUESTS",
}
_TRANSIENT_CODES = _RATE_LIMIT_CODES | {
    "INTERNAL",
    "SERVICE_UNAVAILABLE",
    "UNAVAILABLE",
}
_RATE_LIMIT_TEXT = re.compile(
    r"\b(?:429|rate[\s_-]+limit(?:ed|ing)?|too many requests|"
    r"resource_exhausted|quota|insufficient_quota)\b",
    re.IGNORECASE,
)
_TRANSIENT_TEXT = re.compile(
    r"\b(?:500|502|503|504|unavailable|overloaded|internal)\b", re.IGNORECASE
)


def _provider_status(error: BaseException) -> int | None:
    for value in (getattr(error, "status_code", None), getattr(error, "code", None)):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _provider_code(error: BaseException) -> str | None:
    code = getattr(error, "code", None)
    if not isinstance(code, str):
        code = getattr(error, "status", None)
    if not isinstance(code, str):
        return None
    return code.upper().replace("-", "_").replace(" ", "_")


def is_rate_limit_error(error: BaseException) -> bool:
    status = _provider_status(error)
    if status is not None:
        return status == 429

    code = _provider_code(error)
    if code is not None:
        return code in _RATE_LIMIT_CODES

    message = str(error)
    return bool(
        _RATE_LIMIT_TEXT.search(message)
        or any(code.lower() in message.lower() for code in _RATE_LIMIT_CODES)
    )


def is_transient_error(error: BaseException) -> bool:
    status = _provider_status(error)
    if status is not None:
        return status in _RETRYABLE_STATUSES

    code = _provider_code(error)
    if code is not None:
        return code in _TRANSIENT_CODES

    message = str(error)
    return is_rate_limit_error(error) or bool(_TRANSIENT_TEXT.search(message))


def safe_error_summary(error: BaseException) -> str:
    """Describe an error without copying provider-controlled response text."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return f"{name} (HTTP {status})"
    return name
