"""
#2420: POST /v1/requests/{id}/end_thinking ends a running request's think block.

What it tests:   unknown or finished id is 404 request_not_found (param "id"), repeatable; GET is
                 not a route (`nomodel`: mock and the shipping binary). Mock with --prefill-ms: a
                 held streaming request named by X-Request-Id answers 200 "ending" twice (idempotent),
                 then streams content and no reasoning; without the call the same request reasons;
                 after the finish the id is 404. Live (real model, GPU): a streaming chat and a
                 streaming /v1/messages request in their think phase, ended by the response id, stop
                 reasoning and stream the answer.
What it does NOT test: the engine's forced closer (CPU: EndThinking.* in
                 tests/test_think_stop_logic.cpp; registry: EndThinkingRegistry.* in
                 tests/test_end_thinking.cpp).
"""

import json
import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

import conftest

MOCK_MODEL = "mock-model-v1"  # mock_server.MOCK_MODEL_ID
HOLD_MS = 1500


def _end(client, rid):
    return client.post(f"/v1/requests/{rid}/end_thinking")


def _assert_not_found(r):
    assert r.status_code == 404, r.text
    err = r.json()["error"]
    assert err["code"] == "request_not_found", err
    assert err["param"] == "id", err


def _sse_events(lines):
    for line in lines:
        if line.startswith("data: ") and line != "data: [DONE]":
            yield json.loads(line[len("data: "):])


@pytest.mark.nomodel
class TestEndThinkingRoute:

    def test_unknown_id_is_404_every_time(self, client):
        for _ in range(2):
            _assert_not_found(_end(client, "chatcmpl-not-running"))

    def test_needs_post(self, client):
        assert client.get("/v1/requests/chatcmpl-x/end_thinking").status_code in (404, 405)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def held_mock():
    """A mock that holds every request HOLD_MS before its first token (--prefill-ms)."""
    if not conftest.USE_MOCK:
        pytest.skip("mock-only: needs a request held before its first token")
    port = _free_port()
    proc = subprocess.Popen([sys.executable, "-m", "mock_server", "--port", str(port),
                             "--prefill-ms", str(HOLD_MS)],
                            cwd=os.path.dirname(__file__), stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    try:
        while True:
            if proc.poll() is not None:
                pytest.fail(f"mock exited {proc.returncode}: {proc.stderr.read().decode()[-500:]}")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("mock did not answer /health within 30 s")
            time.sleep(0.1)
        with httpx.Client(base_url=base, timeout=30.0) as c:
            yield c
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _stream_chat(base_url, rid, out):
    body = {"model": MOCK_MODEL, "max_tokens": 16, "stream": True, "reasoning_effort": "high",
            "messages": [{"role": "user", "content": "Hi"}]}
    r = httpx.post(base_url + "/v1/chat/completions", json=body,
                   headers={"X-Request-Id": rid}, timeout=30.0)
    out["status"] = r.status_code
    out["events"] = list(_sse_events(r.text.splitlines()))


def _deltas(events, key):
    return [c["delta"][key] for e in events for c in e.get("choices", []) if c["delta"].get(key)]


class TestEndThinkingHeldRequest:

    def _run(self, client, rid, end):
        out = {}
        t = threading.Thread(target=_stream_chat, args=(str(client.base_url), rid, out))
        t.start()
        acks = []
        if end:
            deadline = time.monotonic() + HOLD_MS / 1000.0
            while time.monotonic() < deadline:
                r = _end(client, rid)
                if r.status_code == 200:
                    acks = [r, _end(client, rid)]
                    break
                time.sleep(0.05)
        t.join(timeout=30)
        assert out.get("status") == 200, out
        return acks, out["events"]

    def test_end_while_held_drops_reasoning_and_is_idempotent(self, held_mock):
        acks, events = self._run(held_mock, "et-held-1", end=True)
        assert len(acks) == 2, "the held request never answered 200"
        for r in acks:
            assert r.json() == {"id": "et-held-1", "object": "request.end_thinking", "status": "ending"}
        assert _deltas(events, "reasoning_content") == []
        assert _deltas(events, "content")
        _assert_not_found(_end(held_mock, "et-held-1"))

    def test_control_without_end_reasons(self, held_mock):
        _, events = self._run(held_mock, "et-held-2", end=False)
        assert _deltas(events, "reasoning_content")
        assert _deltas(events, "content")


class TestEndThinkingLive:
    """Real model with a think block (Qwen3.x <think> or gpt-oss Harmony): GPU, make test-server."""

    LIMIT_AFTER_ACK = 256  # reasoning deltas allowed after the 200: one in-flight graph burst

    @pytest.fixture(autouse=True)
    def _real_only(self, is_mock, has_model):
        if is_mock or not has_model:
            pytest.skip("needs a real model that reasons")

    def test_chat_stream_ends_think_and_answers(self, client, model):
        body = {"model": model, "max_tokens": 4096, "stream": True,
                "messages": [{"role": "user",
                              "content": "Think carefully, step by step: how many primes are below 200?"}]}
        reasoning_after, content, acked = 0, "", None
        with client.stream("POST", "/v1/chat/completions", json=body) as r:
            assert r.status_code == 200
            for ev in _sse_events(r.iter_lines()):
                for c in ev.get("choices", []):
                    d = c.get("delta", {})
                    if d.get("reasoning_content"):
                        if acked is None:
                            acked = [_end(client, ev["id"]), _end(client, ev["id"])]
                        else:
                            reasoning_after += 1
                    if d.get("content"):
                        content += d["content"]
        assert acked is not None, "the model never streamed reasoning"
        for a in acked:
            assert a.status_code == 200, a.text
            assert a.json()["status"] in ("ending", "already_closed")
        assert reasoning_after <= self.LIMIT_AFTER_ACK, reasoning_after
        assert content.strip(), "no answer after the think block was ended"

    def test_messages_stream_ends_think_by_message_id(self, client, model):
        body = {"model": model, "max_tokens": 4096, "stream": True,
                "thinking": {"type": "enabled", "budget_tokens": 3000},
                "messages": [{"role": "user",
                              "content": "Think carefully, step by step: how many primes are below 200?"}]}
        msg_id, acked, text = None, None, ""
        with client.stream("POST", "/v1/messages", json=body) as r:
            assert r.status_code == 200
            for ev in _sse_events(r.iter_lines()):
                if ev.get("type") == "message_start":
                    msg_id = ev["message"]["id"]
                delta = ev.get("delta", {})
                if delta.get("type") == "thinking_delta" and acked is None:
                    acked = _end(client, msg_id)
                if delta.get("type") == "text_delta":
                    text += delta["text"]
        assert acked is not None and acked.status_code == 200, acked and acked.text
        assert text.strip(), "no answer after the think block was ended"
        _assert_not_found(_end(client, msg_id))
