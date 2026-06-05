from brrragent.openai_direct import (
    _build_chat_kwargs,
    _build_responses_kwargs,
    _should_use_responses,
    _to_responses_tool,
    parse_openai_model,
)


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
                "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
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
        response_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )

    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["instructions"] == "system"
    assert kwargs["max_output_tokens"] == 32000
    assert kwargs["reasoning"] == {"effort": "xhigh"}
    assert kwargs["tools"][0]["name"] == "search"
    assert kwargs["text"]["format"]["type"] == "json_schema"
    assert "temperature" not in kwargs
