"""
#2408: --max-queued-tokens.

What it tests:   with prompt A holding the queue (mock --prefill-ms), a request whose prompt tokens
                 push the queued total over the cap gets 429 at once, rate_limit_error (OpenAI
                 envelope) with Retry-After: 1, on /v1/chat/completions, /v1/completions and
                 /v1/responses; one that fits is admitted; after A leaves the queue the refused one
                 is admitted. Cap 0 admits everything. imp_queued_prompt_tokens on /metrics.
                 Real binary (model-less): flag accepted, gauge 0, garbage refused at startup.
What it does NOT test: the /v1/messages envelope (the mock does not route /v1/messages; the C++
                 shape test is QueuedTokenGate.RefusalIs429OverloadedWithRetryAfterInEachDialect).
"""

import os
import socket
import subprocess
import sys
import threading
import time

import httpx
import pytest

import conftest

MODEL = "mock-model-v1"  # mock_server.MOCK_MODEL_ID
CAP = 100
HOLD_MS = 3000
BIG, OVER = 80, 30  # mock prompt tokens: 80 + 30 > CAP refused, 80 + 20 == CAP admitted
# /v1/responses counts json.dumps([["user", text]]): 14 chars around the text.
_OVERHEAD = {"/v1/responses": 14}


def _prompt(path: str, n: int) -> str:
    """A prompt the mock counts as exactly n tokens (1 per 4 chars) on `path`."""
    return "x" * (4 * n - _OVERHEAD.get(path, 0))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_cmd(port: int, *extra: str) -> list[str]:
    if conftest.SERVER_BIN:
        return [conftest.SERVER_BIN, "--host", "127.0.0.1", "--port", str(port), *extra]
    return [sys.executable, "-m", "mock_server", "--port", str(port), *extra]


@pytest.fixture
def server(request):
    """Start a server with request.param as extra args, yield its base URL."""
    port = _free_port()
    proc = subprocess.Popen(_server_cmd(port, *request.param), cwd=os.path.dirname(__file__),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}"
    deadline = time.monotonic() + 30
    try:
        while True:
            if proc.poll() is not None:
                pytest.fail(f"server exited {proc.returncode}: {proc.stderr.read().decode()[-500:]}")
            try:
                if httpx.get(base + "/health", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                pytest.fail("server did not answer /health within 30 s")
            time.sleep(0.1)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _gauge(base: str) -> int:
    for line in httpx.get(base + "/metrics", timeout=5).text.splitlines():
        if line.startswith("imp_queued_prompt_tokens "):
            return int(float(line.split()[1]))
    raise AssertionError("imp_queued_prompt_tokens missing from /metrics")


def _post(base: str, path: str, prompt: str) -> httpx.Response:
    if path == "/v1/chat/completions":
        body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": 2}
    elif path == "/v1/completions":
        body = {"model": MODEL, "prompt": prompt, "max_tokens": 2}
    else:
        body = {"model": MODEL, "input": prompt, "store": False}
    return httpx.post(base + path, json=body, timeout=30)


def _hold(base: str, path: str, prompt: str) -> threading.Thread:
    """Send `prompt` in the background and return once its tokens show on the gauge."""
    t = threading.Thread(target=_post, args=(base, path, prompt), daemon=True)
    t.start()
    deadline = time.monotonic() + 5
    while _gauge(base) == 0:
        assert time.monotonic() < deadline, "the holding request never reached the queue"
        time.sleep(0.02)
    return t


GEN_PATHS = ["/v1/chat/completions", "/v1/completions", "/v1/responses"]
needs_mock = pytest.mark.skipif(bool(conftest.SERVER_BIN), reason="needs generation (mock --prefill-ms)")


@needs_mock
class TestQueuedTokenCap:

    @pytest.mark.parametrize("path", GEN_PATHS)
    @pytest.mark.parametrize("server", [("--max-queued-tokens", str(CAP), "--prefill-ms", str(HOLD_MS))],
                             indirect=True)
    def test_over_cap_refused_at_once_under_cap_admitted(self, server, path):
        # The responses mock counts the JSON-wrapped conversation, so hold with the same body shape.
        holder = _hold(server, path, _prompt(path, BIG))
        held = _gauge(server)
        assert held == BIG

        t0 = time.monotonic()
        r = _post(server, path, _prompt(path, OVER))
        elapsed = time.monotonic() - t0
        assert r.status_code == 429, r.text
        assert elapsed < 1.0, f"refusal took {elapsed:.2f} s, it must not wait behind the queue"
        assert r.headers.get("retry-after") == "1"
        err = r.json()["error"]
        assert err["type"] == "rate_limit_error"
        assert "--max-queued-tokens" in err["message"]
        assert "type" not in r.json(), "OpenAI envelope has no top-level type"
        assert _gauge(server) == held, "a refusal must not count its tokens"

        fits = _prompt(path, CAP - held)
        assert _post(server, path, fits).status_code == 200, "queued + n == cap is admitted"

        holder.join(timeout=30)
        assert _gauge(server) == 0
        assert _post(server, path, _prompt(path, OVER)).status_code == 200, "admitted once the queue drained"

    @pytest.mark.parametrize("server", [("--max-queued-tokens", str(CAP), "--prefill-ms", str(HOLD_MS))],
                             indirect=True)
    def test_empty_queue_admits_a_prompt_over_the_cap(self, server):
        t0 = time.monotonic()
        r = _post(server, "/v1/completions", "w" * (4 * 5 * CAP))
        assert r.status_code == 200, r.text
        assert time.monotonic() - t0 >= HOLD_MS / 1000.0 * 0.9

    @pytest.mark.parametrize("server", [("--prefill-ms", str(HOLD_MS))], indirect=True)
    def test_cap_off_by_default_admits_everything(self, server):
        holder = _hold(server, "/v1/chat/completions", _prompt("", 4000))
        assert _gauge(server) >= 4000, "gauge counts with the cap off"
        r = _post(server, "/v1/chat/completions", _prompt("", 1500))
        assert r.status_code == 200, r.text
        holder.join(timeout=30)


@pytest.mark.nomodel
class TestQueuedTokenFlag:

    @pytest.mark.parametrize("server", [("--max-queued-tokens", "65536")], indirect=True)
    def test_flag_accepted_and_gauge_exported(self, server):
        assert _gauge(server) == 0

    @pytest.mark.parametrize("bad", ["-1", "64k"])
    def test_garbage_refused_at_startup(self, bad):
        p = subprocess.run(_server_cmd(_free_port(), "--max-queued-tokens", bad),
                           capture_output=True, timeout=30, cwd=os.path.dirname(__file__))
        assert p.returncode == 1
        assert b"integer >= 0" in p.stderr
