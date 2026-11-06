import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from brrragent import gemini, openai_direct, openrouter
from brrragent._errors import (
    is_rate_limit_error,
    is_transient_error,
    safe_error_summary,
)
from brrragent.keys import KeyPool


class ProviderError(Exception):
    def __init__(self, message, status_code=500):
        self.status_code = status_code
        super().__init__(message)


def test_safe_error_summary_excludes_provider_message():
    error = ProviderError("password=secret-value")

    assert safe_error_summary(error) == "ProviderError (HTTP 500)"


def test_provider_retry_does_not_expose_error_message(caplog):
    secret = "500 password=secret-value"

    class Completions:
        def create(self, **_kwargs):
            raise ProviderError(secret)

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))

    with caplog.at_level(logging.WARNING), pytest.raises(ValueError) as error:
        openrouter._call_with_retry(
            client=client,
            model="test-model",
            messages=[],
            tools=[],
            temperature=0,
            max_tokens=1,
            max_retries=1,
            response_format=None,
            key_pool=None,
            current_key="",
            base_url="https://example.com/v1",
        )

    assert secret not in caplog.text
    assert secret not in str(error.value)
    assert error.value.__suppress_context__ is True
    assert "ProviderError (HTTP 500)" in caplog.text


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_provider_error_classifiers_use_structured_http_status(status_code):
    error = ProviderError("provider omitted the status from its message", status_code)

    assert is_rate_limit_error(error) is (status_code == 429)
    assert is_transient_error(error)


@pytest.mark.parametrize("message", ["rate_limit_exceeded", "quota_exceeded"])
def test_provider_error_classifiers_preserve_legacy_quota_codes(message):
    error = RuntimeError(message)

    assert is_rate_limit_error(error)
    assert is_transient_error(error)


@pytest.mark.parametrize(
    "backend", ["gemini", "openrouter", "openai_chat", "openai_responses"]
)
@pytest.mark.parametrize("error_kind", ["http_status", "legacy_text"])
def test_generate_content_404_is_not_retried_or_evicted(backend, error_kind):
    message = "models/example is not supported for generateContent"
    if error_kind == "http_status":
        error = ProviderError(message, status_code=404)
    elif backend == "gemini":
        genai_errors = pytest.importorskip("google.genai.errors")
        error = genai_errors.ClientError(
            404, {"error": {"code": 404, "message": message}}
        )
    else:
        error = RuntimeError(message)
    create = Mock(side_effect=error)
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        responses=SimpleNamespace(create=create),
    )
    pool = KeyPool(["test-key"], name="test")
    common = {
        "client": client,
        "model": "test-model",
        "max_retries": 3,
        "key_pool": pool,
        "current_key": "test-key",
    }

    with pytest.raises(type(error)):
        if backend == "gemini":
            gemini._call_with_retry(contents=[], config=None, **common)
        elif backend == "openrouter":
            openrouter._call_with_retry(
                messages=[],
                tools=[],
                temperature=0,
                max_tokens=1,
                base_url="https://example.test/v1",
                **common,
            )
        elif backend == "openai_chat":
            openai_direct._call_with_retry(
                messages=[], tools=[], temperature=0, max_tokens=1, **common
            )
        else:
            openai_direct._call_responses_with_retry(
                input_items=[],
                instructions=None,
                tools=[],
                temperature=0,
                max_tokens=1,
                **common,
            )

    assert create.call_count == 1
    assert pool.active_count == 1
