"""
Tests for OpenAI API contract compliance.

Tests HTTP status codes, JSON response schema, and endpoint behavior
against both mock and real server. Every test validates structure, not
model output quality.

What it tests:   HTTP contract, response schema, Content-Type headers.
What it does NOT test: Model correctness, token quality, numerical precision.
External state:  Running imp-server or mock server.
"""

import json
import pytest


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_status_ok(self, client):
        r = client.get("/health")
        body = r.json()
        assert body["status"] == "ok"

    def test_health_has_model_loaded(self, client):
        r = client.get("/health")
        body = r.json()
        assert "model_loaded" in body
        assert isinstance(body["model_loaded"], bool)

    def test_health_has_queue_depth(self, client):
        r = client.get("/health")
        body = r.json()
        assert "queue_depth" in body
        assert isinstance(body["queue_depth"], int)


class TestModelsEndpoint:
    def test_models_returns_200(self, client):
        r = client.get("/v1/models")
        assert r.status_code == 200

    def test_models_has_list_object(self, client):
        r = client.get("/v1/models")
        body = r.json()
        assert body["object"] == "list"

    def test_models_data_is_array(self, client):
        r = client.get("/v1/models")
        body = r.json()
        assert isinstance(body["data"], list)

    def test_models_entries_have_required_fields(self, client):
        r = client.get("/v1/models")
        body = r.json()
        if body["data"]:
            entry = body["data"][0]
            assert "id" in entry
            assert "object" in entry
            assert entry["object"] == "model"


class TestChatCompletionsSchema:
    """Verify response JSON matches the OpenAI chat completions schema."""

    def test_non_stream_full_schema(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8,
            "temperature": 0,
        })
        assert r.status_code == 200
        body = r.json()

        # Top-level fields
        assert isinstance(body["id"], str)
        assert body["object"] == "chat.completion"
        assert isinstance(body["created"], int)
        assert body["model"] == model

        # Choices
        choices = body["choices"]
        assert isinstance(choices, list)
        assert len(choices) == 1
        c = choices[0]
        assert c["index"] == 0
        assert c["message"]["role"] == "assistant"
        assert isinstance(c["message"]["content"], str)
        assert c["finish_reason"] in ("stop", "length")

        # Usage
        usage = body["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert usage["prompt_tokens"] > 0
        assert isinstance(usage["completion_tokens"], int)
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_content_type_json(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 1,
        })
        assert "application/json" in r.headers.get("content-type", "")

    def test_content_type_sse_when_streaming(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": True,
        })
        assert r.status_code == 200
        assert "text/event-stream" in r.headers.get("content-type", "")


class TestStreamingSchema:
    """Verify SSE streaming chunks match the OpenAI streaming schema."""

    def _get_events(self, client, model, max_tokens=8):
        from conftest import parse_sse
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
        })
        assert r.status_code == 200
        return parse_sse(r.text), r.text

    def test_chunks_are_valid_json(self, client, model):
        _, raw = self._get_events(client, model)
        for line in raw.splitlines():
            if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                json.loads(line[6:])  # must not raise

    def test_chunk_object_type(self, client, model):
        events, _ = self._get_events(client, model)
        for ev in events:
            assert ev["object"] == "chat.completion.chunk"

    def test_chunk_has_id_and_created(self, client, model):
        events, _ = self._get_events(client, model)
        for ev in events:
            assert "id" in ev
            assert "created" in ev

    def test_done_sentinel_present(self, client, model):
        _, raw = self._get_events(client, model)
        assert "data: [DONE]" in raw

    def test_last_chunk_has_finish_reason(self, client, model):
        events, _ = self._get_events(client, model)
        last = events[-1]
        assert last["choices"][0]["finish_reason"] in ("stop", "length")

    def test_non_last_chunks_have_null_finish_reason(self, client, model):
        events, _ = self._get_events(client, model)
        for ev in events[:-1]:
            if ev["choices"]:  # skip usage-only chunks
                assert ev["choices"][0]["finish_reason"] is None

    def test_stream_usage_when_requested(self, client, model):
        from conftest import parse_sse
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        events = parse_sse(r.text)
        usage_events = [e for e in events if e.get("usage")]
        assert len(usage_events) >= 1
        u = usage_events[-1]["usage"]
        assert u["prompt_tokens"] > 0
        assert u["completion_tokens"] > 0
        assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]


class TestMetricsEndpoint:
    def test_metrics_returns_200(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200

    def test_metrics_content_type(self, client):
        r = client.get("/metrics")
        ct = r.headers.get("content-type", "")
        assert "text/plain" in ct

    def test_metrics_has_required_counters(self, client):
        r = client.get("/metrics")
        text = r.text
        assert "imp_requests_total" in text
        assert "imp_requests_failed_total" in text

    def test_metrics_has_model_loaded(self, client):
        r = client.get("/metrics")
        assert "imp_model_loaded" in r.text
