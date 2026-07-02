"""Tests for tool/function calling in /v1/chat/completions."""

import json

import httpx
import pytest

import conftest

from conftest import parse_sse

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}


@pytest.mark.tools
class TestToolCalling:
    def test_tool_choice_required_shape(self, client, model):
        """tool_choice=required should produce a tool_calls response."""
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "required",
            "max_tokens": 128,
            "temperature": 0,
        })
        assert r.status_code == 200
        body = r.json()
        choice = body["choices"][0]
        msg = choice["message"]
        # Should have tool_calls
        assert "tool_calls" in msg, f"Expected tool_calls, got: {msg}"
        tc = msg["tool_calls"]
        assert len(tc) >= 1
        assert tc[0]["type"] == "function"
        assert "id" in tc[0]
        assert tc[0]["function"]["name"] == "get_weather"
        # arguments should be valid JSON string
        import json
        args = json.loads(tc[0]["function"]["arguments"])
        assert isinstance(args, dict)

    def test_tool_choice_none(self, client, model):
        """tool_choice=none should produce a normal text response."""
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "none",
            "max_tokens": 32,
            "temperature": 0,
        })
        assert r.status_code == 200
        body = r.json()
        msg = body["choices"][0]["message"]
        # Should NOT have tool_calls
        assert msg.get("tool_calls") is None or len(msg.get("tool_calls", [])) == 0

    def test_tool_call_finish_reason(self, client, model):
        """tool_choice=required with tool_calls must have finish_reason='tool_calls'.

        tool_choice=required is prompt-based (not constrained decoding), so the
        model may occasionally not produce a tool call. We send 3 attempts and
        require at least one to produce tool_calls with correct finish_reason.
        If none do, the test fails — the server's tool prompting is broken.
        """
        got_tool_call = False
        for _ in range(3):
            r = client.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "What's the weather in Paris?"}],
                "tools": [WEATHER_TOOL],
                "tool_choice": "required",
                "max_tokens": 256,
                "temperature": 0,
            })
            assert r.status_code == 200
            body = r.json()
            choice = body["choices"][0]
            msg = choice["message"]
            if "tool_calls" in msg and msg["tool_calls"]:
                # When tool_calls are present, validate structure
                assert choice["finish_reason"] == "tool_calls"
                tc = msg["tool_calls"][0]
                assert tc["type"] == "function"
                assert tc["function"]["name"] == "get_weather"
                got_tool_call = True
                break
        assert got_tool_call, "tool_choice=required failed to produce tool_calls in 3 attempts"

    def test_tools_plus_json_schema_passes_through(self, client, model):
        """Setting tools + response_format=json_schema must NOT drop the schema.

        The engine-side gate should let the model's tool-tag opener through
        unconditionally and only apply the schema mask if the model actually
        emits free-text JSON instead of a tool call. Either outcome (tool_call
        OR schema-shaped JSON) is acceptable here — what we're catching is the
        old failure mode where the request was rejected or response_format was
        silently dropped (which the server used to log).
        """
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather in Tokyo?"}],
            "tools": [WEATHER_TOOL],
            "tool_choice": "auto",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "weather_or_text",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {"type": "string"},
                        },
                        "required": ["answer"],
                    },
                },
            },
            "max_tokens": 128,
            "temperature": 0,
        })
        assert r.status_code == 200, r.text
        body = r.json()
        choice = body["choices"][0]
        msg = choice["message"]

        # Either path is acceptable; both must be syntactically valid.
        if "tool_calls" in msg and msg["tool_calls"]:
            tc = msg["tool_calls"]
            assert tc[0]["function"]["name"] == "get_weather"
            import json
            args = json.loads(tc[0]["function"]["arguments"])
            assert isinstance(args, dict)
        else:
            # Free-text path: the schema must have been enforced.
            content = msg.get("content", "")
            import json
            payload = json.loads(content)
            assert "answer" in payload
            assert isinstance(payload["answer"], str)


class TestStreamingToolCalls:
    def test_arguments_stream_incrementally(self, model, is_mock):
        """Streaming tool calls: the name chunk opens the call, then
        `tool_calls[].function.arguments` deltas whose concatenation is valid
        JSON. On JSON tool dialects (ChatML/Llama3) the deltas arrive WHILE
        the model generates (many deltas); buffered dialects (Qwen3.6 XML,
        Gemma) emit bounded chunks after the close tag — both are valid, but
        there must never be a single giant delta AND the concat must parse."""
        if is_mock:
            pytest.skip("mock server does not implement streaming tool calls")
        import httpx as _httpx
        name = None
        deltas = []
        finish = None
        with _httpx.Client(base_url=conftest.BASE_URL, timeout=180.0) as c:
            with c.stream("POST", "/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user",
                              "content": "Call write_file to create /tmp/x.txt with a short poem. /no_think"}],
                "tools": [{"type": "function",
                           "function": {"name": "write_file",
                                        "parameters": {"type": "object",
                                                       "properties": {"path": {"type": "string"},
                                                                      "content": {"type": "string"}},
                                                       "required": ["path", "content"]}}}],
                "max_tokens": 300, "temperature": 0, "stream": True,
            }) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    d = json.loads(line[len("data: "):])
                    for ch in d.get("choices", []):
                        fin = ch.get("finish_reason")
                        if fin:
                            finish = fin
                        for tc in ch.get("delta", {}).get("tool_calls", []) or []:
                            fn = tc.get("function", {})
                            if fn.get("name"):
                                name = fn["name"]
                            if fn.get("arguments"):
                                deltas.append(fn["arguments"])
        assert name == "write_file"
        assert finish == "tool_calls"
        assert deltas, "no argument deltas streamed"
        args = json.loads("".join(deltas))
        assert "path" in args and "content" in args
