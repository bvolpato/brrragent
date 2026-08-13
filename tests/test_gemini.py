import sys
from types import ModuleType, SimpleNamespace

from brrragent import ImageInput, gemini
from brrragent.prompt_cache import AgentUsage


class FakePart:
    def __init__(self, *, text=None, function_call=None, function_response=None):
        self.text = text
        self.function_call = function_call
        self.function_response = function_response

    @classmethod
    def from_bytes(cls, *, data, mime_type):
        return SimpleNamespace(source="bytes", data=data, mime_type=mime_type)

    @classmethod
    def from_uri(cls, *, file_uri, mime_type=None):
        return SimpleNamespace(source="uri", file_uri=file_uri, mime_type=mime_type)


class FakeContent:
    def __init__(self, *, role, parts):
        self.role = role
        self.parts = parts


class FakeFunctionResponse:
    def __init__(self, *, name, response):
        self.name = name
        self.response = response


class FakeGenerateContentConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class FakeMcp:
    def __init__(self):
        self.tools = [{"function_declarations": [{"name": "lookup"}]}]
        self.calls = []

    def get_genai_tools(self):
        return self.tools

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


def _response(*parts, usage=None):
    content = SimpleNamespace(parts=list(parts))
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


def _install_fake_google_sdk(monkeypatch, generate):
    client_keys = []
    calls = []

    class FakeClient:
        def __init__(self, *, api_key):
            client_keys.append(api_key)
            self.api_key = api_key
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def generate_content(self, **kwargs):
            calls.append(
                {
                    **kwargs,
                    "contents": list(kwargs["contents"]),
                    "api_key": self.api_key,
                }
            )
            return generate(self.api_key, len(calls), kwargs)

    types_module = ModuleType("google.genai.types")
    types_module.Content = FakeContent
    types_module.Part = FakePart
    types_module.FunctionResponse = FakeFunctionResponse
    types_module.GenerateContentConfig = FakeGenerateContentConfig

    genai_module = ModuleType("google.genai")
    genai_module.Client = FakeClient
    genai_module.types = types_module

    google_module = ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)
    return client_keys, calls


def test_google_genai_sdk_exposes_adapter_types():
    from google.genai import types as gt

    config = gt.GenerateContentConfig(system_instruction="system", tools=[])
    content = gt.Content(role="user", parts=[gt.Part(text="question")])
    function_response = gt.Part(
        function_response=gt.FunctionResponse(
            name="lookup", response={"result": "answer"}
        )
    )

    assert config.system_instruction == "system"
    assert content.parts[0].text == "question"
    assert function_response.function_response.name == "lookup"


def test_gemini_agent_executes_tool_then_returns_schema_response(monkeypatch):
    responses = iter(
        [
            _response(
                FakePart(
                    function_call=SimpleNamespace(
                        name="lookup", args={"query": "brrragent"}
                    )
                ),
                usage={
                    "prompt_token_count": 10,
                    "candidates_token_count": 2,
                    "total_token_count": 12,
                    "cached_content_token_count": 3,
                },
            ),
            _response(
                FakePart(text='{"answer":"done"}'),
                usage={
                    "prompt_token_count": 15,
                    "candidates_token_count": 4,
                    "total_token_count": 19,
                },
            ),
        ]
    )
    client_keys, calls = _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: next(responses)
    )
    mcp = FakeMcp()
    tool_events = []
    usages = []
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        api_key="fake-key",
        mcp=mcp,
        max_turns=3,
        temperature=0.2,
        max_tokens=321,
        max_retries=1,
        on_tool_call=lambda name, args: tool_events.append((name, args)),
        response_schema=schema,
        on_usage=usages.append,
    )

    assert result == '{"answer":"done"}'
    assert client_keys == ["fake-key"]
    assert mcp.calls == [("lookup", {"query": "brrragent"})]
    assert tool_events == mcp.calls
    assert usages == [
        AgentUsage(10, 2, 12, cache_read_input_tokens=3),
        AgentUsage(15, 4, 19),
    ]

    assert len(calls) == 2
    assert calls[0]["model"] == "gemini-test"
    assert [content.role for content in calls[0]["contents"]] == ["user"]
    assert [content.role for content in calls[1]["contents"]] == [
        "user",
        "model",
        "user",
    ]
    tool_response = calls[1]["contents"][-1].parts[0].function_response
    assert tool_response.name == "lookup"
    assert tool_response.response == {"result": "tool result"}

    config = calls[0]["config"]
    assert config.system_instruction == "system"
    assert config.tools == mcp.tools
    assert config.max_output_tokens == 321
    assert config.response_mime_type == "application/json"
    assert config.response_schema is schema


def test_gemini_agent_sends_data_and_url_images(monkeypatch):
    response = _response(FakePart(text="mushroom"))
    _, calls = _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: response
    )

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        api_key="fake-key",
        mcp=FakeMcp(),
        max_turns=1,
        temperature=0.2,
        max_tokens=321,
        max_retries=1,
        on_tool_call=None,
        response_schema=None,
        images=(
            ImageInput.from_bytes(b"png", media_type="image/png"),
            ImageInput("https://example.test/image.jpg", media_type="image/jpeg"),
        ),
    )

    assert result == "mushroom"
    parts = calls[0]["contents"][0].parts
    assert parts[0].text == "question"
    assert (parts[1].source, parts[1].data, parts[1].mime_type) == (
        "bytes",
        b"png",
        "image/png",
    )
    assert (parts[2].source, parts[2].file_uri, parts[2].mime_type) == (
        "uri",
        "https://example.test/image.jpg",
        "image/jpeg",
    )


def test_gemini_agent_rotates_key_and_retries_429_without_sleep(monkeypatch):
    final_response = _response(FakePart(text="rotated response"))

    def generate(key, _call_number, _kwargs):
        if key == "first-fake-key":
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return final_response

    _, calls = _install_fake_google_sdk(monkeypatch, generate)
    monkeypatch.setattr(
        gemini.time,
        "sleep",
        lambda _seconds: (_ for _ in ()).throw(
            AssertionError("rotation must not sleep")
        ),
    )
    pool = RecordingKeyPool(["first-fake-key", "second-fake-key"])

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        key_pool=pool,
        mcp=FakeMcp(),
        max_turns=1,
        temperature=0,
        max_tokens=100,
        max_retries=2,
        on_tool_call=None,
        response_schema=None,
    )

    assert result == "rotated response"
    assert pool.acquired == ["first-fake-key", "second-fake-key"]
    assert pool.rate_limited == ["first-fake-key"]
    assert [call["api_key"] for call in calls] == [
        "first-fake-key",
        "second-fake-key",
    ]
