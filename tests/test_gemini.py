import json
import sys
from importlib.metadata import version
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
        def __init__(self, *, api_key, http_options=None):
            assert http_options.api_version == "v1beta"
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
    types_module.HttpOptions = SimpleNamespace

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


def test_gemini_sdk_preserves_beta_model_tool_roundtrip(monkeypatch):
    from google import genai
    from google.genai import types as gt

    if int(version("google-genai").split(".")[0]) >= 3:
        import httpx2 as transport
    else:
        import httpx as transport

    requests = []
    responses = iter(
        [
            {"functionCall": {"name": "lookup", "args": {"query": "example"}}},
            {"text": '{"answer":"done"}'},
        ]
    )

    def handle(request):
        requests.append(request)
        return transport.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [next(responses)]}}
                ]
            },
        )

    original_client = genai.Client
    clients = []

    if "client_args" not in gt.HttpOptions.model_fields:
        import requests as requests_transport

        def legacy_request(_session, *, method, url, data, **_kwargs):
            response = handle(transport.Request(method, url, content=data))
            legacy_response = requests_transport.Response()
            legacy_response.status_code = response.status_code
            legacy_response._content = response.content
            return legacy_response

        monkeypatch.setattr(requests_transport.Session, "request", legacy_request)

    def client_factory(*, api_key, http_options=None):
        options = http_options.model_dump(exclude_none=True) if http_options else {}
        if "client_args" in gt.HttpOptions.model_fields:
            options["client_args"] = {"transport": transport.MockTransport(handle)}
        client = original_client(
            api_key=api_key, http_options=gt.HttpOptions(**options)
        )
        if "client_args" in gt.HttpOptions.model_fields:
            clients.append(client)
        return client

    monkeypatch.setattr(genai, "Client", client_factory)
    monkeypatch.delenv("GOOGLE_GENAI_API_VERSION", raising=False)
    monkeypatch.delenv("GOOGLE_GENAI_HTTP_CLIENT", raising=False)
    mcp = FakeMcp()
    try:
        result = gemini.run_gemini_agent(
            system_prompt="system",
            user_prompt="question",
            model="gemini-flash-latest",
            api_key="synthetic-key",
            mcp=mcp,
            max_turns=3,
            temperature=0.2,
            max_tokens=100,
            max_retries=1,
            on_tool_call=None,
            response_schema=None,
        )
    finally:
        for client in clients:
            client.close()

    assert result == '{"answer":"done"}'
    assert mcp.calls == [("lookup", {"query": "example"})]
    assert len(requests) == 2
    assert all(
        request.url.path == "/v1beta/models/gemini-flash-latest:generateContent"
        for request in requests
    )
    body = json.loads(requests[1].content)
    assert body["contents"][-1]["parts"][0]["functionResponse"] == {
        "name": "lookup",
        "response": {"result": "tool result"},
    }


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


def test_gemini_agent_synthesizes_after_last_tool_turn_without_tools(monkeypatch):
    responses = iter(
        [
            _response(
                FakePart(
                    function_call=SimpleNamespace(
                        name="lookup", args={"query": "brrragent"}
                    )
                ),
                usage={"prompt_token_count": 10, "total_token_count": 10},
            ),
            _response(
                FakePart(text='{"answer":"synthesized"}'),
                usage={"prompt_token_count": 20, "total_token_count": 20},
            ),
        ]
    )
    _, calls = _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: next(responses)
    )
    mcp = FakeMcp()
    usages = []
    schema = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
    }
    image = ImageInput.from_bytes(b"png", media_type="image/png")

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        api_key="fake-key",
        mcp=mcp,
        max_turns=1,
        temperature=0.2,
        max_tokens=321,
        max_retries=1,
        on_tool_call=None,
        response_schema=schema,
        on_usage=usages.append,
        images=(image,),
    )

    assert result == '{"answer":"synthesized"}'
    assert mcp.calls == [("lookup", {"query": "brrragent"})]
    assert usages == [AgentUsage(10, 0, 10), AgentUsage(20, 0, 20)]
    assert len(calls) == 2
    assert calls[0]["config"].tools == mcp.tools
    assert calls[1]["config"].tools == []
    final_config = calls[1]["config"]
    assert final_config.system_instruction == "system"
    assert final_config.response_mime_type == "application/json"
    assert final_config.response_schema is schema
    final_contents = calls[1]["contents"]
    assert [content.role for content in final_contents] == [
        "user",
        "model",
        "user",
        "user",
    ]
    assert final_contents[0].parts[1].source == "bytes"
    assert final_contents[0].parts[1].data == b"png"
    assert final_contents[2].parts[0].function_response.response == {
        "result": "tool result"
    }


def test_gemini_agent_handles_contentless_regular_candidate(monkeypatch):
    response = SimpleNamespace(
        candidates=[SimpleNamespace(content=None)], usage_metadata=None
    )
    _install_fake_google_sdk(monkeypatch, lambda _key, _call_number, _kwargs: response)

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
    )

    assert result == "[No final response after max tool turns]"


def test_gemini_agent_handles_contentless_final_candidate(monkeypatch):
    responses = iter(
        [
            _response(FakePart(function_call=SimpleNamespace(name="lookup", args={}))),
            SimpleNamespace(
                candidates=[SimpleNamespace(content=None)], usage_metadata=None
            ),
        ]
    )
    _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: next(responses)
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
    )

    assert result == "[No final response after max tool turns]"


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


def test_gemini_agent_merges_extra_tools_with_mcp_declarations(monkeypatch):
    response = _response(FakePart(text="done"))
    _, calls = _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: response
    )
    sys.modules["google.genai.types"].FunctionDeclaration = lambda **kwargs: (
        SimpleNamespace(**kwargs)
    )
    sys.modules["google.genai.types"].Tool = lambda *, function_declarations: (
        SimpleNamespace(function_declarations=function_declarations)
    )

    mcp = FakeMcp()
    mcp.tools = [
        SimpleNamespace(function_declarations=[SimpleNamespace(name="mcp_lookup")])
    ]
    extra_tools = [
        {
            "type": "function",
            "function": {
                "name": "local_tool",
                "description": "Local helper",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cursor": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                        }
                    },
                },
            },
        }
    ]

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        api_key="fake-key",
        mcp=mcp,
        extra_tools=extra_tools,
        max_turns=1,
        temperature=0,
        max_tokens=100,
        max_retries=1,
        on_tool_call=None,
        response_schema=None,
    )

    assert result == "done"
    configured_tools = calls[0]["config"].tools
    assert len(configured_tools) == 1
    decls = configured_tools[0].function_declarations
    assert [decl.name for decl in decls] == ["mcp_lookup", "local_tool"]
    assert decls[1].parameters == {
        "type": "object",
        "properties": {"cursor": {"type": "string"}},
    }


def test_gemini_agent_rejects_non_mapping_tool_arguments(monkeypatch):
    from brrragent._tool_calls import INVALID_TOOL_ARGUMENTS

    responses = iter(
        [
            _response(
                FakePart(
                    function_call=SimpleNamespace(name="lookup", args=["not-a-dict"])
                )
            ),
            _response(FakePart(text="recovered")),
        ]
    )
    _, calls = _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: next(responses)
    )
    mcp = FakeMcp()
    events = []

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        api_key="fake-key",
        mcp=mcp,
        max_turns=2,
        temperature=0,
        max_tokens=100,
        max_retries=1,
        on_tool_call=lambda name, args: events.append((name, args)),
        response_schema=None,
    )

    assert result == "recovered"
    assert mcp.calls == []
    assert events == []
    assert calls[1]["contents"][-1].parts[0].function_response.response == {
        "result": INVALID_TOOL_ARGUMENTS
    }


def test_gemini_agent_excludes_thought_parts_from_final_text(monkeypatch):
    responses = iter(
        [
            _response(
                SimpleNamespace(
                    text="Internal reasoning summary",
                    thought=True,
                    function_call=None,
                ),
                SimpleNamespace(
                    text='{"answer":"clean"}',
                    thought=False,
                    function_call=None,
                ),
            )
        ]
    )
    _install_fake_google_sdk(
        monkeypatch, lambda _key, _call_number, _kwargs: next(responses)
    )

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-3.1-pro-preview",
        api_key="fake-key",
        mcp=FakeMcp(),
        max_turns=1,
        temperature=0,
        max_tokens=100,
        max_retries=1,
        on_tool_call=None,
        response_schema=None,
    )

    assert result == '{"answer":"clean"}'


def test_gemini_agent_closes_clients_and_reuses_rotated_client(monkeypatch):
    closed = []
    created = []
    responses = iter(
        [
            RuntimeError("429 RESOURCE_EXHAUSTED"),
            _response(FakePart(function_call=SimpleNamespace(name="lookup", args={}))),
            _response(FakePart(text="done")),
        ]
    )

    class ClosingFakeClient:
        def __init__(self, *, api_key, http_options=None):
            created.append(api_key)
            self.api_key = api_key
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def generate_content(self, **_kwargs):
            nxt = next(responses)
            if isinstance(nxt, Exception):
                raise nxt
            return nxt

        def close(self):
            closed.append(self.api_key)

    _install_fake_google_sdk(monkeypatch, lambda *_a: None)
    sys.modules["google.genai"].Client = ClosingFakeClient
    pool = RecordingKeyPool(["first-key", "second-key"])

    result = gemini.run_gemini_agent(
        system_prompt="system",
        user_prompt="question",
        model="gemini-test",
        key_pool=pool,
        mcp=FakeMcp(),
        max_turns=2,
        temperature=0,
        max_tokens=100,
        max_retries=2,
        on_tool_call=None,
        response_schema=None,
    )

    assert result == "done"
    assert created == ["first-key", "second-key"]
    assert closed == ["first-key", "second-key"]
