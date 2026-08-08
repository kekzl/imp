"""Tests for SSE streaming in /v1/chat/completions."""

import json
import pytest
import httpx

from conftest import parse_sse, BASE_URL


def test_streaming_chunk_shape(client, model):
    r = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers.get("content-type", "")
    events = parse_sse(r.text)
    assert len(events) >= 1

    # First chunk should have role
    first = events[0]
    assert first["object"] == "chat.completion.chunk"
    assert "id" in first
    assert "created" in first
    delta = first["choices"][0]["delta"]
    assert delta.get("role") == "assistant" or "content" in delta

    # Last chunk should have finish_reason
    last = events[-1]
    assert last["choices"][0]["finish_reason"] in ("stop", "length")


def test_streaming_content_matches_nonstream(client, model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "What is 1+1?"}],
        "max_tokens": 16,
        "temperature": 0,
        "seed": 123,
    }
    # Non-streaming
    r1 = client.post("/v1/chat/completions", json=payload)
    expected = r1.json()["choices"][0]["message"]["content"]

    # Streaming
    r2 = client.post("/v1/chat/completions", json={**payload, "stream": True})
    events = parse_sse(r2.text)
    content = ""
    for ev in events:
        delta = ev["choices"][0].get("delta", {})
        content += delta.get("content", "")
    assert content == expected


def test_stream_nonstream_agree_across_truncation_points(client, model):
    """Transport must not change content, at any truncation point.

    test_streaming_content_matches_nonstream above asserts the same invariant
    but pins max_tokens=16 on "What is 1+1?", which never truncates inside a
    multi-byte character. Against the mock it cannot: mock_server.py has no
    tokenizer and emits whole ASCII words, so the assertion passes without
    exercising the code that fails.

    Sweeping max_tokens walks the truncation point across the generation. When
    it lands mid-character the non-streaming path emits U+FFFD while the
    streaming path holds the incomplete bytes back (#1310), so the two
    transports return different bytes for one request.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "temperature": 0,
        "seed": 123,
    }
    for max_tokens in range(1, 9):
        body = {**payload, "max_tokens": max_tokens}

        r1 = client.post("/v1/chat/completions", json=body)
        assert r1.status_code == 200
        nonstream = r1.json()["choices"][0]["message"]["content"]

        r2 = client.post("/v1/chat/completions", json={**body, "stream": True})
        assert r2.status_code == 200
        stream = ""
        for ev in parse_sse(r2.text):
            stream += (ev["choices"][0].get("delta") or {}).get("content") or ""

        assert stream == nonstream, (
            f"max_tokens={max_tokens}: transports disagree\n"
            f"  non-stream: {nonstream!r} ({nonstream.encode('utf-8').hex()})\n"
            f"  stream:     {stream!r} ({stream.encode('utf-8').hex()})"
        )
        assert "\ufffd" not in nonstream, (
            f"max_tokens={max_tokens}: non-streaming content carries U+FFFD, "
            f"a character no generated token produced: {nonstream!r}"
        )


def test_streaming_include_usage(client, model):
    r = client.post("/v1/chat/completions", json={
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 8,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    })
    assert r.status_code == 200
    events = parse_sse(r.text)
    # Find the usage chunk (last event or event with usage)
    usage_events = [e for e in events if e.get("usage")]
    assert len(usage_events) >= 1
    usage = usage_events[-1]["usage"]
    assert usage["prompt_tokens"] > 0
    assert usage["completion_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streaming_done_marker(base_url, model):
    """Stream must end with 'data: [DONE]'."""
    lines = []
    with httpx.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 32,
            "temperature": 0,
            "stream": True,
        },
        timeout=30.0,
    ) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            lines.append(line)
            if "data: [DONE]" in line:
                break
    assert any("data: [DONE]" in l for l in lines)
