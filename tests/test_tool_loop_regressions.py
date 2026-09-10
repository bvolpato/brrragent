import json
from contextlib import ExitStack

import httpx
import pytest
from openai import OpenAI

from brrragent import PromptCacheConfig, openai_direct, openrouter
from brrragent.prompt_cache import AgentUsage


class RecordingMcp:
    def __init__(self):
        self.calls = []

    def get_openai_tools(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return "tool evidence"


@pytest.mark.parametrize("max_turns", [1, 3])
@pytest.mark.parametrize("explicit_cache", [False, True])
def test_responses_replays_stateless_history(monkeypatch, max_turns, explicit_cache):
    requests = []
    output = [
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "opaque-reasoning",
        },
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "phase": "commentary",
            "content": [
                {"type": "output_text", "text": "Checking.", "annotations": []}
            ],
        },
        {
            "id": "fc_1",
            "type": "function_call",
            "call_id": "call_1",
            "name": "lookup",
            "arguments": "{}",
            "status": "completed",
        },
    ]
    final_output = [
        {
            "id": "msg_final",
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": [
                {"type": "output_text", "text": "Final answer", "annotations": []}
            ],
        }
    ]

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(requests)}",
                "object": "response",
                "created_at": 0,
                "model": "gpt-5.6-luna",
                "output": output if len(requests) == 1 else final_output,
            },
        )

    mcp = RecordingMcp()
    with OpenAI(
        api_key="fake-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        monkeypatch.setattr(openai_direct, "_new_openai_client", lambda _key: client)
        result = openai_direct.run_openai_agent(
            system_prompt="Keep following these instructions.",
            user_prompt="Find the answer.",
            model="openai/gpt-5.6-luna:medium",
            api_key="fake-key",
            mcp=mcp,
            extra_tools=None,
            max_turns=max_turns,
            temperature=0.2,
            max_tokens=256,
            max_retries=1,
            on_tool_call=None,
            response_schema=None,
            prompt_cache=PromptCacheConfig("test:v1") if explicit_cache else None,
        )

    assert result == "Final answer"
    assert mcp.calls == [("lookup", {})]
    assert len(requests) == 2
    first, second = requests
    for request in requests:
        assert request["store"] is False
        assert "previous_response_id" not in request
        assert "reasoning.encrypted_content" in request["include"]
        if explicit_cache:
            assert request["input"][0]["role"] == "developer"
            assert request["input"][0]["content"][0]["text"] == (
                "Keep following these instructions."
            )
            assert request["prompt_cache_options"] == {"mode": "explicit"}
        else:
            assert request["instructions"] == "Keep following these instructions."
    expected = [
        *first["input"],
        *output,
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "tool evidence",
        },
    ]
    if max_turns == 1:
        assert not second.get("tools")
        assert second["input"][:-1] == expected
        assert "Stop calling tools" in second["input"][-1]["content"]
    else:
        assert second["tools"] == first["tools"]
        assert second["input"] == expected


@pytest.mark.parametrize("provider", [openai_direct, openrouter])
@pytest.mark.parametrize("final_text", ['{"answer":"from evidence"}', ""])
def test_chat_limit_synthesizes_with_retry_and_preserves_options(
    monkeypatch, provider, final_text
):
    requests = []
    evicted = []
    keys = iter(["first-fake-key", "second-fake-key"])

    class Pool:
        def acquire(self):
            return next(keys)

        def report_rate_limit(self, key):
            evicted.append(key)

    def respond(request):
        requests.append(json.loads(request.content))
        if len(requests) == 2:
            return httpx.Response(
                429,
                json={"error": {"message": "rate limit", "type": "rate_limit_error"}},
            )
        message = {"role": "assistant", "content": final_text}
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl_{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if len(requests) == 1 else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "total_tokens": 12,
                },
            },
        )

    usage = []
    mcp = RecordingMcp()
    with ExitStack() as stack:

        def make_client(**kwargs):
            assert kwargs["max_retries"] == 0
            return stack.enter_context(
                OpenAI(
                    **kwargs,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond)),
                )
            )

        monkeypatch.setattr("openai.OpenAI", make_client)
        kwargs = {}
        if provider is openrouter:
            kwargs = {
                "base_url": "https://openrouter.ai/api/v1",
                "reasoning_effort": "high",
            }
        run = (
            provider.run_openai_agent
            if provider is openai_direct
            else provider.run_openrouter_agent
        )
        result = run(
            system_prompt="Use evidence.",
            user_prompt="Find the answer.",
            model="gpt-4o",
            key_pool=Pool(),
            mcp=mcp,
            extra_tools=None,
            max_turns=1,
            temperature=0.2,
            max_tokens=256,
            max_retries=2,
            on_tool_call=None,
            response_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
            prompt_cache=PromptCacheConfig("test:v1"),
            on_usage=usage.append,
            **kwargs,
        )

    assert result == (final_text or "[No final response after max tool turns]")
    assert mcp.calls == [("lookup", {})]
    assert evicted == ["first-fake-key"]
    assert len(requests) == 3
    first, synthesis, retry = requests
    assert synthesis == retry
    assert not synthesis.get("tools")
    assert synthesis["messages"][:2] == first["messages"]
    assert synthesis["messages"][-2] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "tool evidence",
    }
    assert "Stop calling tools" in synthesis["messages"][-1]["content"]
    for option in (
        "response_format",
        "prompt_cache_key",
        "extra_body",
        "reasoning_effort",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
    ):
        assert synthesis.get(option) == first.get(option)
    assert usage == [AgentUsage(input_tokens=10, output_tokens=2, total_tokens=12)] * 2
