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
