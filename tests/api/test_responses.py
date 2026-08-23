"""
Tests for the /v1/responses endpoint (OpenAI Responses API — the Agents SDK /
Codex dialect).

What it tests:   Request/response shapes, output items, streaming event
                 sequence, stateful-field rejection.
What it does NOT test: model quality; transform internals (covered by
                 tests/test_responses_transform.cpp).
External state:  Running imp-server (skipped against the mock server, which
                 does not implement /v1/responses — same policy as
                 /v1/messages).
"""

import json

import httpx
import pytest

import conftest


@pytest.fixture(autouse=True)
def _skip_on_mock(is_mock):
    if is_mock:
        pytest.skip("mock server does not implement /v1/responses")


class TestResponsesNonStream:
    def test_string_input_text_output(self, model):
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            r = c.post("/v1/responses", json={
                "model": model,
                "input": "What is the capital of France? One word. /no_think",
                "max_output_tokens": 30,
                "temperature": 0,
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "response"
        assert body["id"].startswith("resp_")
        assert body["status"] in ("completed", "incomplete")
        msgs = [it for it in body["output"] if it["type"] == "message"]
        assert msgs, body["output"]
        text = "".join(p["text"] for p in msgs[0]["content"] if p["type"] == "output_text")
        assert "Paris" in text
        assert body["usage"]["input_tokens"] > 0
        assert body["usage"]["output_tokens"] > 0

    def test_function_call_item(self, model):
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            r = c.post("/v1/responses", json={
                "model": model,
                "input": "What is the weather in Paris? Use the tool. /no_think",
                "tools": [{"type": "function", "name": "get_weather",
                           "description": "Get weather for a city",
                           "parameters": {"type": "object",
                                          "properties": {"city": {"type": "string"}},
                                          "required": ["city"]}}],
                "max_output_tokens": 200,
                "temperature": 0,
            })
        assert r.status_code == 200, r.text
        body = r.json()
        calls = [it for it in body["output"] if it["type"] == "function_call"]
        assert calls, body["output"]
        fc = calls[0]
        assert fc["name"] == "get_weather"
        assert fc["call_id"]
        args = json.loads(fc["arguments"])
        assert isinstance(args, dict)

    # Validation runs before model resolution, so this holds model-less (#1600).
    @pytest.mark.nomodel
    def test_stateful_fields_rejected(self, model):
        with httpx.Client(base_url=conftest.BASE_URL, timeout=30.0) as c:
            r = c.post("/v1/responses", json={
                "model": model, "input": "x", "previous_response_id": "resp_123",
            })
            assert r.status_code == 400
            r = c.post("/v1/responses", json={
                "model": model, "input": "x", "store": True,
            })
            assert r.status_code == 400


class TestResponsesStream:
    def test_event_sequence(self, model):
        events = []
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            with c.stream("POST", "/v1/responses", json={
                "model": model,
                "input": "Say hello in five words. /no_think",
                "max_output_tokens": 40,
                "temperature": 0,
                "stream": True,
            }) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if line.startswith("event: "):
                        events.append(line[len("event: "):])
        assert events[0] == "response.created"
        assert events[1] == "response.in_progress"
        assert events[-1] in ("response.completed", "response.incomplete")
        assert "response.output_item.added" in events
        assert "response.output_text.delta" in events
        assert "response.output_text.done" in events
        assert "response.output_item.done" in events

    def test_sequence_numbers_monotonic(self, model):
        seqs = []
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            with c.stream("POST", "/v1/responses", json={
                "model": model, "input": "Count to three. /no_think",
                "max_output_tokens": 30, "temperature": 0, "stream": True,
            }) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        seqs.append(json.loads(line[6:])["sequence_number"])
        # Without this the test passes on an EMPTY stream: [] == sorted([]) and
        # 0 == 0. That is how it "passed" against a model-less server, and it
        # would pass the same way here if the stream ever came back empty
        # (#1600).
        assert seqs, "no SSE data frames arrived — nothing to check monotonicity on"
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)

    def test_streaming_function_call_arguments(self, model):
        """function_call items stream: item.added carries the name, then
        function_call_arguments.delta events whose concatenation equals the
        arguments in ..._arguments.done (incremental for JSON tool dialects,
        single-delta for buffered dialects — both shapes are valid)."""
        added_names, deltas, done_args = [], [], None
        with httpx.Client(base_url=conftest.BASE_URL, timeout=180.0) as c:
            with c.stream("POST", "/v1/responses", json={
                "model": model,
                "input": "Call write_file to create /tmp/x.txt with a short poem. /no_think",
                "tools": [{"type": "function", "name": "write_file",
                           "parameters": {"type": "object",
                                          "properties": {"path": {"type": "string"},
                                                         "content": {"type": "string"}},
                                          "required": ["path", "content"]}}],
                "max_output_tokens": 300, "temperature": 0, "stream": True,
            }) as r:
                for line in r.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    d = json.loads(line[6:])
                    t = d.get("type")
                    if t == "response.output_item.added" and d["item"]["type"] == "function_call":
                        added_names.append(d["item"]["name"])
                    elif t == "response.function_call_arguments.delta":
                        deltas.append(d["delta"])
                    elif t == "response.function_call_arguments.done":
                        done_args = d["arguments"]
        assert added_names == ["write_file"]
        assert done_args is not None
        if deltas:  # incremental dialect
            assert "".join(deltas) == done_args
        args = json.loads(done_args)
        assert "path" in args and "content" in args
