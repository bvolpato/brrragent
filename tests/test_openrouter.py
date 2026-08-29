import copy
from types import SimpleNamespace

from brrragent import ImageInput, PromptCacheConfig, openrouter
from brrragent.prompt_cache import AgentUsage


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


class FakeMessage:
    def __init__(self, *, content, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self):
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ],
        }


class FakeMcp:
    def __init__(self):
        self.tools = [{"type": "function", "function": {"name": "lookup"}}]
        self.calls = []

    def get_openai_tools(self):
        return list(self.tools)

    def call_tool(self, name, args):
        self.calls.append((name, args))
        return "tool result"


class RecordingKeyPool:
    def __init__(self, keys):
        self.keys = iter(keys)
        self.acquired = []
        self.rate_limited = []

    def acquire(self):
        key = next(self.keys)
        self.acquired.append(key)
        return key

    def report_rate_limit(self, key):
        self.rate_limited.append(key)


class FakeOpenAIFactory:
    def __init__(self, tool_response, final_response):
        self.tool_response = tool_response
        self.final_response = final_response
        self.client_keys = []
        self.calls = []
        self.second_key_clients = 0

    def __call__(self, *, api_key, **client_kwargs):
        self.client_keys.append(api_key)
        if api_key == "second-fake-key":
            self.second_key_clients += 1
            response = (
                self.tool_response
                if self.second_key_clients == 1
                else self.final_response
            )
        else:
            response = RuntimeError("429 rate limit")
        factory = self

        class Completions:
            def create(self, **kwargs):
                factory.calls.append(
                    {
                        "api_key": api_key,
                        "client_kwargs": client_kwargs,
                        **copy.deepcopy(kwargs),
                    }
                )
                if isinstance(response, Exception):
                    raise response
                return response

        return SimpleNamespace(
            chat=SimpleNamespace(completions=Completions()),
        )


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
        base_url="https://api.openlux.ai/v1",
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


def test_openrouter_agent_rotates_key_and_completes_tool_loop(monkeypatch):
    tool_call = SimpleNamespace(
        id="tool-call-1",
        function=SimpleNamespace(name="lookup", arguments='{"query":"brrragent"}'),
    )
    tool_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeMessage(content=None, tool_calls=[tool_call]),
                finish_reason="tool_calls",
            )
        ],
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeMessage(content='{"answer":"done"}'),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": 15,
            "completion_tokens": 4,
            "total_tokens": 19,
        },
    )
    factory = FakeOpenAIFactory(tool_response, final_response)
    monkeypatch.setattr("openai.OpenAI", factory)
    monkeypatch.setattr(
        openrouter.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("rotation must not sleep")
        ),
    )
    pool = RecordingKeyPool(["first-fake-key", "second-fake-key"])
    mcp = FakeMcp()
    tool_events = []
    usages = []
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    extra_tool = {"type": "function", "function": {"name": "extra"}}

    result = openrouter.run_openrouter_agent(
        system_prompt="system",
        user_prompt="question",
        model="provider/test-model",
        key_pool=pool,
        base_url="https://provider.example/v1",
        mcp=mcp,
        extra_tools=[extra_tool],
        max_turns=3,
        temperature=0.2,
        max_tokens=321,
        max_retries=2,
        on_tool_call=lambda name, args: tool_events.append((name, args)),
        response_schema=schema,
        on_usage=usages.append,
    )

    assert result == '{"answer":"done"}'
    assert pool.acquired == ["first-fake-key", "second-fake-key"]
    assert pool.rate_limited == ["first-fake-key"]
    assert factory.client_keys[0] == "first-fake-key"
    assert set(factory.client_keys[1:]) == {"second-fake-key"}
    assert all(call["client_kwargs"]["max_retries"] == 0 for call in factory.calls)
    assert mcp.calls == [("lookup", {"query": "brrragent"})]
    assert tool_events == mcp.calls
    assert usages == [
        AgentUsage(10, 2, 12, cache_read_input_tokens=3),
        AgentUsage(15, 4, 19),
    ]

    assert [call["api_key"] for call in factory.calls] == [
        "first-fake-key",
        "second-fake-key",
        "second-fake-key",
    ]
    successful_tool_call = factory.calls[1]
    assert successful_tool_call["model"] == "provider/test-model"
    assert successful_tool_call["tools"] == [*mcp.tools, extra_tool]
    assert successful_tool_call["temperature"] == 0.2
    assert successful_tool_call["max_tokens"] == 321
    assert successful_tool_call["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "agent_response",
            "strict": True,
            "schema": schema,
        },
    }

    final_call = factory.calls[2]
    assert [message["role"] for message in final_call["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert final_call["messages"][-1] == {
        "role": "tool",
        "tool_call_id": "tool-call-1",
        "content": "tool result",
    }


def test_openrouter_agent_sends_image_content(monkeypatch):
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=FakeMessage(content="mushroom"), finish_reason="stop"
            )
        ],
        usage=None,
    )
    factory = FakeOpenAIFactory(final_response, final_response)
    monkeypatch.setattr("openai.OpenAI", factory)

    result = openrouter.run_openrouter_agent(
        system_prompt="system",
        user_prompt="question",
        model="provider/vision-model",
        api_key="second-fake-key",
        base_url="https://provider.example/v1",
        mcp=FakeMcp(),
        extra_tools=None,
        max_turns=1,
        temperature=0.2,
        max_tokens=321,
        max_retries=1,
        on_tool_call=None,
        response_schema=None,
        images=(ImageInput("https://example.test/image.png", detail="high"),),
    )

    assert result == "mushroom"
    assert factory.calls[0]["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "https://example.test/image.png",
            "detail": "high",
        },
    }
