def safe_error_summary(error: BaseException) -> str:
    """Describe an error without copying provider-controlled response text."""
    name = type(error).__name__
    status = getattr(error, "status_code", None)
    if isinstance(status, int):
        return f"{name} (HTTP {status})"
    return name
