from brrragent import PromptCacheConfig, openrouter


class DummyCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return object()


class DummyChat:
    def __init__(self):
        self.completions = DummyCompletions()


class DummyClient:
    def __init__(self):
        self.chat = DummyChat()


def test_openrouter_call_uses_reasoning_kwargs_when_requested():
    client = DummyClient()

    response = openrouter._call_with_retry(
        client=client,
        model="gpt-5.4",
        messages=[],
        tools=[],
        temperature=0.7,
        max_tokens=1234,
        max_retries=1,
        response_format=None,
        key_pool=None,
        current_key="",
        base_url="https://yunwu.ai/v1",
        reasoning_effort="high",
    )

    assert response is not None
    call = client.chat.completions.calls[0]
    assert call["model"] == "gpt-5.4"
    assert call["max_completion_tokens"] == 1234
    assert call["reasoning_effort"] == "high"
    assert "max_tokens" not in call
    assert "temperature" not in call


def test_openrouter_call_sets_stable_cache_key():
    client = DummyClient()

    openrouter._call_with_retry(
        client=client,
        model="anthropic/claude-sonnet-4",
        messages=[],
        tools=[],
        temperature=0.7,
        max_tokens=1234,
        max_retries=1,
        response_format=None,
        key_pool=None,
        current_key="",
        base_url="https://openrouter.ai/api/v1",
        prompt_cache=PromptCacheConfig("workflow:v2"),
    )

    assert client.chat.completions.calls[0]["extra_body"] == {
        "prompt_cache_key": "workflow:v2",
        "session_id": "workflow:v2",
    }


def test_anthropic_system_block_marks_explicit_cache_breakpoint():
    content = openrouter._system_content(
        "stable system",
        "anthropic/claude-sonnet-4",
        PromptCacheConfig("workflow:v2"),
    )

    assert content[0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert (
        openrouter._system_content(
            "stable system",
            "google/gemini-3-flash-preview",
            PromptCacheConfig("workflow:v2"),
        )
        == "stable system"
    )
