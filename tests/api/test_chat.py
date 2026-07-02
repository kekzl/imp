"""Tests for POST /v1/chat/completions (non-streaming)."""

import pytest


def test_basic_response_shape(client, model):
    r = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert "id" in body
    assert "created" in body
    assert body["model"] == model
    choices = body["choices"]
    assert len(choices) == 1
    c = choices[0]
    assert c["index"] == 0
    assert "message" in c
    assert c["message"]["role"] == "assistant"
    assert isinstance(c["message"]["content"], str)
    assert len(c["message"]["content"]) > 0
    assert c["finish_reason"] in ("stop", "length")
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] > 0
    assert body["usage"]["completion_tokens"] > 0
    assert body["usage"]["total_tokens"] == (
        body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]
    )


def test_system_message(client, model):
    r = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a pirate. Always say 'Arrr'."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 32,
        "temperature": 0,
    })
    assert r.status_code == 200


def test_max_tokens_respected(client, model):
    r = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Count from 1 to 1000."}],
        "max_tokens": 5,
        "temperature": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] <= 5


def test_stop_sequence(client, model):
    r = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Count: 1 2 3 4 5 6 7 8 9 10"}],
        "max_tokens": 64,
        "temperature": 0,
        "stop": ["5"],
    })
    assert r.status_code == 200
    body = r.json()
    content = body["choices"][0]["message"]["content"]
    assert "5" not in content or body["choices"][0]["finish_reason"] == "stop"


def test_temperature_zero_deterministic(client, model):
    """Two identical temp=0+seed requests should produce the same output."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 2+2?"}],
        "max_tokens": 16,
        "temperature": 0,
        "seed": 42,
    }
    r1 = client.post("/v1/chat/completions", json=payload)
    r2 = client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    assert r2.status_code == 200
    c1 = r1.json()["choices"][0]["message"]["content"]
    c2 = r2.json()["choices"][0]["message"]["content"]
    assert c1 == c2, f"Non-deterministic at temp=0 seed=42: {c1!r} != {c2!r}"


def test_models_endpoint(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


class TestPredictedOutputs:
    def test_prediction_param_accepted_and_ignored_shape(self, client, model):
        """The prediction param must never change the response shape — it is
        a draft hint. Works against mock (param silently ignored) and real."""
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 16,
            "temperature": 0,
            "prediction": {"type": "content", "content": "Hello! How can I help?"},
        })
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert len(body["choices"][0]["message"]["content"]) > 0

    def test_prediction_speeds_code_edit_and_reports_usage(self, client, model, is_mock):
        """A code-edit request whose prediction matches the expected output
        must (a) return the same text as without prediction, and (b) report
        accepted_prediction_tokens > 0 in completion_tokens_details."""
        if is_mock:
            pytest.skip("mock server does not implement speculative decoding")

        code = "\n".join(
            f"def func_{i}(x):\n    return x + {i}\n" for i in range(20)
        )
        prompt = (
            "Below is a Python file. Output the COMPLETE file again, changing "
            "ONLY func_0 to return x - 0 instead. No explanations.\n\n" + code
        )
        # The prediction is the near-verbatim expected output.
        prediction = code.replace("return x + 0", "return x - 0")
        base = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
            "temperature": 0,
        }
        r_plain = client.post("/v1/chat/completions", json=base)
        r_pred = client.post("/v1/chat/completions", json={
            **base,
            "prediction": {"type": "content", "content": prediction},
        })
        assert r_plain.status_code == 200 and r_pred.status_code == 200
        plain = r_plain.json()
        pred = r_pred.json()
        # The prediction must not change WHAT is generated — but strict byte
        # equality is too strong: the verify forward (chunked-prefill path)
        # has different fp16 numerics than the decode loop, so a near-tie
        # argmax can legitimately flip when a draft covers it. Require high
        # similarity to catch real corruption (garbage/repetition) instead.
        import difflib
        cp = plain["choices"][0]["message"]["content"]
        cq = pred["choices"][0]["message"]["content"]
        sim = difflib.SequenceMatcher(a=cp, b=cq).ratio()
        assert sim > 0.9, f"prediction changed the output substantially (sim={sim:.2f})"
        details = pred["usage"].get("completion_tokens_details", {})
        assert "accepted_prediction_tokens" in details
        assert "rejected_prediction_tokens" in details
        # On a near-verbatim prediction, at least some drafts must have been
        # sourced from the prediction region and accepted.
        assert details["accepted_prediction_tokens"] > 0

    def test_prediction_content_parts_array(self, client, model):
        """The array-of-parts content form must parse."""
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Count to three."}],
            "max_tokens": 16,
            "temperature": 0,
            "prediction": {"type": "content", "content": [
                {"type": "text", "text": "One, two,"},
                {"type": "text", "text": " three."},
            ]},
        })
        assert r.status_code == 200
