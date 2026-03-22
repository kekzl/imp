"""Tests for tool/function calling in /v1/chat/completions."""

import httpx
import pytest

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
