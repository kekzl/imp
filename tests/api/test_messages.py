"""
Tests for the /v1/messages endpoint (Anthropic Messages API).

What it tests:   The Anthropic dialect on the wire — status codes, the error
                 envelope (top-level `type: error`, not the OpenAI shape), the
                 response shape and the SSE event order.
What it does NOT test: transform internals (tests/test_anthropic_transform.cpp
                 covers `anthropic_to_openai_body()` as a unit) or model quality.
External state:  Running imp-server. The validation half is marked `nomodel`
                 and runs in CI against the shipping binary started without a
                 model (#1302); it never reaches generation, because the server
                 validates before it looks for weights. Skipped against the
                 Python mock, which does not implement this endpoint at all —
                 that gap is #1329.
"""

import json

import httpx
import pytest

import conftest


@pytest.fixture(autouse=True)
def _skip_on_mock(is_mock):
    if is_mock:
        pytest.skip("mock server does not implement /v1/messages (#1329)")


def _msg(**kw):
    body = {"max_tokens": 4, "messages": [{"role": "user", "content": "Hi"}]}
    body.update(kw)
    return body


def _assert_anthropic_error(r, status, err_type="invalid_request_error"):
    """Every error on this endpoint must arrive in the Anthropic envelope.

    An Anthropic SDK reads the top-level `type` first; the OpenAI shape used two
    files away in handlers.cpp has no such key, so a client would see an
    unrecognised body instead of a reason.
    """
    assert r.status_code == status, r.text
    body = r.json()
    assert body.get("type") == "error", body
    assert body["error"]["type"] == err_type, body
    assert body["error"]["message"], body


@pytest.mark.nomodel
class TestMessagesValidation:
    """Rejections. Reached before the server resolves a model, hence `nomodel`."""

    @pytest.mark.parametrize("max_tokens", [0, -1])
    def test_max_tokens_below_one(self, client, model, max_tokens):
        r = client.post("/v1/messages", json=_msg(model=model, max_tokens=max_tokens))
        _assert_anthropic_error(r, 400)
        assert "max_tokens" in r.json()["error"]["message"]

    def test_missing_messages(self, client, model):
        r = client.post("/v1/messages", json={"model": model, "max_tokens": 4})
        _assert_anthropic_error(r, 400)

    def test_empty_messages(self, client, model):
        r = client.post("/v1/messages", json=_msg(model=model, messages=[]))
        _assert_anthropic_error(r, 400)

    def test_messages_not_an_array(self, client, model):
        r = client.post("/v1/messages", json=_msg(model=model, messages="nope"))
        _assert_anthropic_error(r, 400)

    def test_missing_model(self, client):
        r = client.post("/v1/messages", json=_msg())
        _assert_anthropic_error(r, 400)
        assert "model" in r.json()["error"]["message"]

    @pytest.mark.parametrize("temperature", [-0.1, 2.5])
    def test_temperature_out_of_range(self, client, model, temperature):
        """imp validates [0,2] here, not Anthropic's [0,1] — the OpenAI bound is
        shared by both endpoints. Pinned so the leniency is a decision, not an
        accident (#1329)."""
        r = client.post("/v1/messages", json=_msg(model=model, temperature=temperature))
        _assert_anthropic_error(r, 400)
        assert "temperature" in r.json()["error"]["message"]

    def test_top_p_out_of_range(self, client, model):
        r = client.post("/v1/messages", json=_msg(model=model, top_p=1.5))
        _assert_anthropic_error(r, 400)
        assert "top_p" in r.json()["error"]["message"]

    @pytest.mark.parametrize("raw", ["not json{{{", "", "[1,2,3]", "42"])
    def test_unparseable_body_keeps_the_anthropic_envelope(self, client, raw):
        """The parse-error path is the one most likely to fall back to the
        OpenAI shape: it runs before any Anthropic-specific code."""
        r = client.post(
            "/v1/messages",
            content=raw,
            headers={"content-type": "application/json"},
        )
        _assert_anthropic_error(r, 400)

    def test_unknown_subpath_is_a_not_found_error(self, client):
        r = client.post("/v1/messages/nope", json={})
        _assert_anthropic_error(r, 404, err_type="not_found_error")


@pytest.mark.nomodel
class TestMessagesLeniency:
    """Places imp deliberately accepts what the upstream API rejects.

    These are not 400s, so they are pinned by what they are NOT: a client that
    omits `max_tokens` must not be rejected — imp supplies the server default
    (handlers_messages.cpp). Written as "not a 400" so the assertion holds both
    model-less (503) and with weights (200).
    """

    def test_max_tokens_may_be_omitted(self, client, model):
        r = client.post("/v1/messages", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert r.status_code != 400, r.text

    def test_max_tokens_null_is_treated_as_absent(self, client, model):
        r = client.post("/v1/messages", json=_msg(model=model, max_tokens=None))
        assert r.status_code != 400, r.text


class TestMessagesGeneration:
    """Needs weights: not `nomodel`, so it stays out of the CPU-only CI lane."""

    def test_non_stream_response_shape(self, model):
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            r = c.post("/v1/messages", json={
                "model": model,
                "max_tokens": 24,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Say hello. /no_think"}],
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert body["model"]
        assert body["id"]
        assert isinstance(body["content"], list) and body["content"]
        assert body["content"][0]["type"] == "text"
        assert body["content"][0]["text"]
        assert body["stop_reason"] in ("end_turn", "max_tokens", "stop_sequence")
        assert body["usage"]["input_tokens"] > 0
        assert body["usage"]["output_tokens"] > 0

    def test_max_tokens_is_honoured_and_reported(self, model):
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            r = c.post("/v1/messages", json={
                "model": model,
                "max_tokens": 3,
                "temperature": 0,
                "messages": [{"role": "user", "content": "Count from one to fifty."}],
            })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["stop_reason"] == "max_tokens"
        assert body["usage"]["output_tokens"] <= 3

    def test_stream_event_order(self, model):
        """The documented Anthropic SSE sequence, in order, with a real
        per-token stream (#754) — not a replay of a finished completion."""
        events = []
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            with c.stream("POST", "/v1/messages", json={
                "model": model,
                "max_tokens": 24,
                "temperature": 0,
                "stream": True,
                "messages": [{"role": "user", "content": "Say hello. /no_think"}],
            }) as r:
                assert r.status_code == 200
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))

        types = [e["type"] for e in events]
        assert types[0] == "message_start", types[:5]
        assert types[-1] == "message_stop", types[-5:]
        for expected in ("content_block_start", "content_block_delta",
                         "content_block_stop", "message_delta"):
            assert expected in types, (expected, types)
        assert types.index("content_block_start") < types.index("content_block_delta")
        assert types.index("content_block_delta") < types.index("content_block_stop")
        assert types.index("content_block_stop") < types.index("message_delta")

        start = events[0]["message"]
        assert start["type"] == "message"
        assert start["role"] == "assistant"
        text = "".join(e["delta"]["text"] for e in events
                       if e["type"] == "content_block_delta" and "text" in e.get("delta", {}))
        assert text.strip()

    def test_count_tokens(self, model):
        with httpx.Client(base_url=conftest.BASE_URL, timeout=120.0) as c:
            short = c.post("/v1/messages/count_tokens", json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
            })
            long = c.post("/v1/messages/count_tokens", json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi " * 200}],
            })
        assert short.status_code == 200, short.text
        assert long.status_code == 200, long.text
        n_short = short.json()["input_tokens"]
        n_long = long.json()["input_tokens"]
        assert n_short > 0
        # Compared against each other, not a magic number: the absolute count is
        # tokenizer- and template-specific, but 200 repetitions of a word have to
        # add at least ~150 tokens on any tokenizer. A ratio would break on a
        # model whose chat template inflates the short baseline.
        assert n_long > n_short + 150, (n_short, n_long)
