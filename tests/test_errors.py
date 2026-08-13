import logging
from types import SimpleNamespace

import pytest

from brrragent import openrouter
from brrragent._errors import safe_error_summary


class ProviderError(Exception):
    status_code = 500


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
