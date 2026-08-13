import inspect
import sys
from copy import deepcopy
from types import SimpleNamespace

import pytest

from brrragent import ImageInput, PromptCacheConfig
from brrragent.openai_direct import (
    _build_chat_kwargs,
    _build_responses_kwargs,
    _should_use_responses,
    _to_responses_tool,
    parse_openai_model,
    run_openai_agent,
)
from brrragent.prompt_cache import AgentUsage


class FakeChatMessage:
    def __init__(self, *, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, *, exclude_none):
        message = {"role": "assistant"}
        if self.content is not None:
            message["content"] = self.content
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return message


class FakeMcp:
    def __init__(self):
        self.calls = []

    def get_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up a value",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            }
        ]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"result":"tool evidence"}'


class FakeCreate:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def fake_client(*, chat_results=(), response_results=()):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=FakeCreate(chat_results))
        ),
        responses=SimpleNamespace(create=FakeCreate(response_results)),
    )


def install_openai(monkeypatch, client_factory):
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=client_factory))


def run_with_defaults(*, mcp, **overrides):
    kwargs = {
        "system_prompt": "Use tools when useful",
        "user_prompt": "Find the answer",
        "model": "openai/gpt-4o",
        "api_key": "test-key",
        "key_pool": None,
        "mcp": mcp,
        "extra_tools": None,
        "max_turns": 2,
        "temperature": 0.2,
        "max_tokens": 256,
        "max_retries": 2,
        "on_tool_call": None,
        "response_schema": None,
    }
    kwargs.update(overrides)
    return run_openai_agent(**kwargs)


def test_openai_sdk_signatures_accept_adapter_kwargs():
    from openai import OpenAI

    prompt_cache = PromptCacheConfig("workflow:v2")
    chat_kwargs = _build_chat_kwargs(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        temperature=0.4,
        max_tokens=4096,
        response_format=None,
        prompt_cache=prompt_cache,
    )
    responses_kwargs = _build_responses_kwargs(
        model="openai/gpt-5.6-luna:medium",
        input_items=[{"role": "user", "content": "dynamic"}],
        instructions="stable system",
        tools=[],
        temperature=0.4,
        max_tokens=1000,
        response_schema={"type": "object", "properties": {}},
        prompt_cache=prompt_cache,
    )

    with OpenAI(api_key="not-a-real-key") as client:
        inspect.signature(client.chat.completions.create).bind(**chat_kwargs)
        inspect.signature(client.responses.create).bind(**responses_kwargs)


def test_parse_openai_model_strips_prefix_and_reasoning_suffix():
    assert parse_openai_model("openai/gpt-5.5:xhigh") == ("gpt-5.5", "xhigh")
    assert parse_openai_model("gpt-4o") == ("gpt-4o", None)


def test_build_chat_kwargs_uses_reasoning_fields_for_xhigh():
    kwargs = _build_chat_kwargs(
        model="openai/gpt-5.5:xhigh",
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        temperature=0.9,
        max_tokens=32000,
        response_format=None,
    )

    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["max_completion_tokens"] == 32000
    assert kwargs["reasoning_effort"] == "xhigh"
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    assert "tools" not in kwargs


def test_build_chat_kwargs_uses_classic_fields_for_non_reasoning_model():
    kwargs = _build_chat_kwargs(
        model="openai/gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "search"}}],
        temperature=0.4,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    assert kwargs["model"] == "gpt-4o"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["temperature"] == 0.4
    assert kwargs["tools"]
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "max_completion_tokens" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_reasoning_model_with_tools_uses_responses_path():
    assert _should_use_responses(
        "openai/gpt-5.5:xhigh",
        [{"type": "function", "function": {"name": "search"}}],
        None,
    )
    assert not _should_use_responses("openai/gpt-5.5:xhigh", [], None)


def test_to_responses_tool_flattens_openai_function_tool():
    tool = _to_responses_tool(
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }
    )

    assert tool == {
        "type": "function",
        "name": "search",
        "description": "Search",
        "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
        "strict": False,
    }


def test_build_responses_kwargs_uses_reasoning_and_text_schema():
    kwargs = _build_responses_kwargs(
        model="openai/gpt-5.5:xhigh",
        input_items=[{"role": "user", "content": "hi"}],
        instructions="system",
        tools=[{"type": "function", "function": {"name": "search"}}],
        temperature=0.4,
        max_tokens=32000,
        response_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    )

    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["instructions"] == "system"
    assert kwargs["max_output_tokens"] == 32000
    assert kwargs["store"] is False
    assert kwargs["reasoning"] == {"effort": "xhigh"}
    assert kwargs["tools"][0]["name"] == "search"
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert "temperature" not in kwargs


def test_gpt_56_responses_marks_stable_instructions_for_explicit_cache():
    kwargs = _build_responses_kwargs(
        model="openai/gpt-5.6-luna:medium",
        input_items=[{"role": "user", "content": "dynamic"}],
        instructions="stable system",
        tools=[],
        temperature=0.4,
        max_tokens=1000,
        response_schema={"type": "object", "properties": {}},
        prompt_cache=PromptCacheConfig("workflow:v2"),
    )

    assert kwargs["prompt_cache_key"] == "workflow:v2"
    assert kwargs["prompt_cache_options"] == {"mode": "explicit"}
    assert "instructions" not in kwargs
    assert kwargs["input"][0]["content"][0]["prompt_cache_breakpoint"] == {
        "mode": "explicit"
    }


def test_chat_agent_runs_tool_round_trip_and_reports_each_usage(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(name="lookup", arguments='{"query":"weather"}'),
    )
    responses = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=FakeChatMessage(tool_calls=[tool_call]),
                    finish_reason="tool_calls",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=2,
                total_tokens=12,
                prompt_tokens_details=SimpleNamespace(cached_tokens=4),
            ),
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=FakeChatMessage(content="Final from evidence"),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=15,
                completion_tokens=3,
                total_tokens=18,
            ),
        ),
    ]
    client = fake_client(chat_results=responses)
    constructor_calls = []

    def openai_factory(**kwargs):
        constructor_calls.append(kwargs)
        return client

    install_openai(monkeypatch, openai_factory)
    mcp = FakeMcp()
    tool_events = []
    usage_events = []

    result = run_with_defaults(
        mcp=mcp,
        on_tool_call=lambda name, args: tool_events.append((name, args)),
        on_usage=usage_events.append,
    )

    assert result == "Final from evidence"
    assert constructor_calls == [
        {
            "api_key": "test-key",
            "timeout": 120.0,
            "max_retries": 0,
            "default_headers": {"X-Title": "brrragent"},
        }
    ]
    assert mcp.calls == [("lookup", {"query": "weather"})]
    assert tool_events == mcp.calls
    assert usage_events == [
        AgentUsage(
            input_tokens=10,
            output_tokens=2,
            total_tokens=12,
            cache_read_input_tokens=4,
        ),
        AgentUsage(input_tokens=15, output_tokens=3, total_tokens=18),
    ]

    calls = client.chat.completions.create.calls
    assert len(calls) == 2
    assert calls[0]["messages"] == [
        {"role": "system", "content": "Use tools when useful"},
        {"role": "user", "content": "Find the answer"},
    ]
    assert calls[1]["messages"][-2:] == [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query":"weather"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"result":"tool evidence"}',
        },
    ]


def test_chat_agent_sends_image_content(monkeypatch):
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeChatMessage(content="mushroom"), finish_reason="stop"
            )
        ],
        usage=None,
    )
    client = fake_client(chat_results=[response])
    install_openai(monkeypatch, lambda **kwargs: client)

    result = run_with_defaults(
        mcp=FakeMcp(),
        images=(ImageInput("https://example.test/image.jpg", detail="low"),),
    )

    assert result == "mushroom"
    assert client.chat.completions.create.calls[0]["messages"][1]["content"] == [
        {"type": "text", "text": "Find the answer"},
        {
            "type": "image_url",
            "image_url": {
                "url": "https://example.test/image.jpg",
                "detail": "low",
            },
        },
    ]


def test_responses_agent_sends_image_content(monkeypatch):
    response = SimpleNamespace(
        id="resp_image",
        output=[],
        output_text="mushroom",
        usage=None,
    )
    client = fake_client(response_results=[response])
    install_openai(monkeypatch, lambda **kwargs: client)

    result = run_with_defaults(
        mcp=FakeMcp(),
        model="openai/gpt-5.6-luna:medium",
        response_schema={"type": "object", "properties": {}},
        images=(ImageInput.from_bytes(b"png", media_type="image/png"),),
    )

    assert result == "mushroom"
    content = client.responses.create.calls[0]["input"][0]["content"]
    assert content[0] == {"type": "input_text", "text": "Find the answer"}
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_responses_agent_synthesizes_after_max_turns_and_handles_invalid_json(
    monkeypatch,
):
    function_call = SimpleNamespace(
        type="function_call",
        name="lookup",
        arguments="{not-json",
        call_id="call_bad_json",
    )
    responses = [
        SimpleNamespace(
            id="resp_tool",
            output=[function_call],
            output_text="",
            usage=SimpleNamespace(input_tokens=5, output_tokens=1, total_tokens=6),
        ),
        SimpleNamespace(
            id="resp_final",
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(
                            type="output_text", text="Synthesized without more tools"
                        )
                    ],
                )
            ],
            output_text=None,
            usage=SimpleNamespace(input_tokens=8, output_tokens=4, total_tokens=12),
        ),
    ]
    client = fake_client(response_results=responses)
    install_openai(monkeypatch, lambda **kwargs: client)
    mcp = FakeMcp()
    tool_events = []
    usage_events = []

    result = run_with_defaults(
        mcp=mcp,
        model="openai/gpt-5.5:high",
        max_turns=1,
        on_tool_call=lambda name, args: tool_events.append((name, args)),
        on_usage=usage_events.append,
    )

    assert result == "Synthesized without more tools"
    assert mcp.calls == [("lookup", {})]
    assert tool_events == mcp.calls
    assert usage_events == [
        AgentUsage(input_tokens=5, output_tokens=1, total_tokens=6),
        AgentUsage(input_tokens=8, output_tokens=4, total_tokens=12),
    ]

    calls = client.responses.create.calls
    assert len(calls) == 2
    assert calls[0]["instructions"] == "Use tools when useful"
    assert calls[0]["store"] is False
    assert calls[0]["tools"][0]["name"] == "lookup"
    assert calls[1]["previous_response_id"] == "resp_tool"
    assert calls[1].get("tools", []) == []
    assert calls[1]["input"] == [
        {
            "type": "function_call_output",
            "call_id": "call_bad_json",
            "output": '{"result":"tool evidence"}',
        },
        {
            "role": "user",
            "content": "Stop calling tools. Provide the final answer using the evidence already gathered.",
        },
    ]


def test_chat_agent_rotates_key_after_429(monkeypatch):
    rate_limited_client = fake_client(
        chat_results=[RuntimeError("429 rate limit exceeded")]
    )
    successful_client = fake_client(
        chat_results=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=FakeChatMessage(content="Recovered"),
                        finish_reason="stop",
                    )
                ],
                usage=None,
            )
        ]
    )
    constructor_calls = []

    def openai_factory(**kwargs):
        constructor_calls.append(kwargs)
        if kwargs["api_key"] == "key-one":
            return rate_limited_client
        return successful_client

    install_openai(monkeypatch, openai_factory)

    class FakeKeyPool:
        def __init__(self):
            self.keys = iter(["key-one", "key-two"])
            self.rate_limits = []

        def acquire(self):
            return next(self.keys)

        def report_rate_limit(self, key):
            self.rate_limits.append(key)

    key_pool = FakeKeyPool()

    result = run_with_defaults(
        mcp=SimpleNamespace(get_openai_tools=lambda: []),
        api_key=None,
        key_pool=key_pool,
        max_turns=1,
    )

    assert result == "Recovered"
    assert key_pool.rate_limits == ["key-one"]
    assert constructor_calls[0]["api_key"] == "key-one"
    assert {call["api_key"] for call in constructor_calls[1:]} == {"key-two"}
    assert all(call["max_retries"] == 0 for call in constructor_calls)
    assert len(rate_limited_client.chat.completions.create.calls) == 1
    assert len(successful_client.chat.completions.create.calls) == 1


def test_responses_agent_rotates_key_without_sdk_retries(monkeypatch):
    rate_limited_client = fake_client(
        response_results=[RuntimeError("429 rate limit exceeded")]
    )
    successful_client = fake_client(
        response_results=[
            SimpleNamespace(
                id="resp_final",
                output=[],
                output_text="Recovered",
                usage=None,
            )
        ]
    )
    constructor_calls = []

    def openai_factory(**kwargs):
        constructor_calls.append(kwargs)
        if kwargs["api_key"] == "key-one":
            return rate_limited_client
        return successful_client

    install_openai(monkeypatch, openai_factory)

    class FakeKeyPool:
        def __init__(self):
            self.keys = iter(["key-one", "key-two"])
            self.rate_limits = []

        def acquire(self):
            return next(self.keys)

        def report_rate_limit(self, key):
            self.rate_limits.append(key)

    key_pool = FakeKeyPool()

    result = run_with_defaults(
        mcp=FakeMcp(),
        model="openai/gpt-5.5:high",
        api_key=None,
        key_pool=key_pool,
        max_turns=1,
    )

    assert result == "Recovered"
    assert key_pool.rate_limits == ["key-one"]
    assert constructor_calls[0]["api_key"] == "key-one"
    assert {call["api_key"] for call in constructor_calls[1:]} == {"key-two"}
    assert all(call["max_retries"] == 0 for call in constructor_calls)
    assert len(rate_limited_client.responses.create.calls) == 1
    assert len(successful_client.responses.create.calls) == 1


def test_chat_agent_propagates_non_transient_error_without_retry(monkeypatch):
    client = fake_client(chat_results=[RuntimeError("invalid API key")])
    install_openai(monkeypatch, lambda **kwargs: client)

    with pytest.raises(RuntimeError, match="invalid API key"):
        run_with_defaults(
            mcp=SimpleNamespace(get_openai_tools=lambda: []),
            max_retries=3,
        )

    assert len(client.chat.completions.create.calls) == 1
