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


@pytest.mark.nomodel
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


@pytest.mark.nomodel
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

    def test_models_exposes_context_length(self, client):
        # Context window is auto-detected by clients via vLLM's max_model_len
        # and llama.cpp's meta.n_ctx_train on the model object.
        r = client.get("/v1/models")
        body = r.json()
        if body["data"]:
            entry = body["data"][0]
            assert entry["max_model_len"] > 0
            assert entry["meta"]["n_ctx_train"] == entry["max_model_len"]


class TestContextProbes:
    """Context-window auto-detection endpoints for OpenAI-compatible clients."""

    def test_props_returns_n_ctx(self, client):
        # llama.cpp shape: /props with n_ctx (top-level + generation settings).
        r = client.get("/props")
        assert r.status_code == 200
        body = r.json()
        assert body["n_ctx"] > 0
        assert body["default_generation_settings"]["n_ctx"] == body["n_ctx"]

    def test_info_returns_total_tokens(self, client):
        # TGI shape: /info with max_total_tokens >= max_input_tokens.
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert body["max_total_tokens"] > 0
        assert body["max_input_tokens"] < body["max_total_tokens"]

    def test_probes_agree_on_context_length(self, client):
        # All three conventions must report the same window.
        models = client.get("/v1/models").json()["data"]
        if not models:
            # `return` here made this a silent pass on a model-less server -
            # indistinguishable from three probes that agreed (#1600).
            pytest.skip("no model loaded: nothing to compare the three probes against")
        max_model_len = models[0]["max_model_len"]
        assert client.get("/props").json()["n_ctx"] == max_model_len
        assert client.get("/info").json()["max_total_tokens"] == max_model_len


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

    # Holds model-less too: the error envelope is JSON by invariant (#1600).
    @pytest.mark.nomodel
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


@pytest.mark.nomodel
class TestRequestIdEcho:
    """X-Request-Id propagation: a client-sent id comes back on every
    response (post-routing echo), including refusals, sanitized against
    header injection."""

    def test_echoed_on_health(self, client):
        r = client.get("/health", headers={"X-Request-Id": "trace-abc-123"})
        assert r.headers.get("x-request-id") == "trace-abc-123"

    def test_echoed_on_unknown_endpoint(self, client):
        r = client.get("/nonexistent", headers={"X-Request-Id": "trace-404"})
        assert r.status_code == 404
        assert r.headers.get("x-request-id") == "trace-404"

    def test_echoed_on_chat_route(self, client):
        # Model-less real server refuses this request, the mock answers it;
        # the id comes back either way - that is the contract under test.
        r = client.post(
            "/v1/chat/completions",
            json={"model": "none", "messages": [{"role": "user", "content": "x"}]},
            headers={"X-Request-Id": "trace-chat-1"},
        )
        assert r.headers.get("x-request-id") == "trace-chat-1"

    def test_absent_header_adds_nothing_on_health(self, client):
        r = client.get("/health")
        assert "x-request-id" not in r.headers

    def test_long_id_is_truncated(self, client):
        r = client.get("/health", headers={"X-Request-Id": "a" * 300})
        echoed = r.headers.get("x-request-id", "")
        # sanitize_for_echo caps at 128 chars + "..." marker
        assert echoed == "a" * 128 + "..."


@pytest.mark.nomodel
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
