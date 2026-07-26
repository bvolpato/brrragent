from types import SimpleNamespace

from brrragent.prompt_cache import gemini_usage, openai_usage


def test_openai_usage_extracts_cached_tokens():
    usage = SimpleNamespace(
        input_tokens=1200,
        output_tokens=80,
        total_tokens=1280,
        input_tokens_details=SimpleNamespace(cached_tokens=1024),
    )

    result = openai_usage(usage)

    assert result.input_tokens == 1200
    assert result.cache_read_input_tokens == 1024
    assert result.output_tokens == 80


def test_gemini_usage_extracts_cached_tokens():
    usage = SimpleNamespace(
        prompt_token_count=4096,
        candidates_token_count=90,
        total_token_count=4186,
        cached_content_token_count=3072,
    )

    result = gemini_usage(usage)

    assert result.input_tokens == 4096
    assert result.cache_read_input_tokens == 3072
    assert result.output_tokens == 90
