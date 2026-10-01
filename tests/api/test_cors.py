"""
#2402: --cors-origins.

What it tests:   Access-Control-Allow-* and Vary on a GET and on a preflight OPTIONS for the
                 default (no flag), an exact-origin list and "*", on the real binary
                 (model-less) and on the mock.
What it does NOT test: a browser enforcing the headers.
"""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

import conftest

A = "https://a.example"
B = "https://b.example"
PREFLIGHT = {"Access-Control-Request-Method": "POST",
             "Access-Control-Request-Headers": "content-type, authorization"}


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


def _get(base: str, origin: str | None) -> httpx.Response:
    headers = {"Origin": origin} if origin else {}
    return httpx.get(base + "/v1/models", headers=headers, timeout=5)


def _preflight(base: str, origin: str) -> httpx.Response:
    return httpx.options(base + "/v1/chat/completions", headers={"Origin": origin, **PREFLIGHT},
                         timeout=5)


def _no_cors(r: httpx.Response):
    for h in ("access-control-allow-origin", "access-control-allow-methods",
              "access-control-allow-headers"):
        assert h not in r.headers, (h, dict(r.headers))


@pytest.mark.nomodel
class TestCorsOrigins:

    @pytest.mark.parametrize("server", [()], indirect=True)
    def test_default_sends_no_cors_headers(self, server):
        for r in (_get(server, A), _get(server, None), _preflight(server, A)):
            _no_cors(r)
            assert "vary" not in r.headers
        assert _preflight(server, A).status_code == 204

    @pytest.mark.parametrize("server", [("--cors-origins", A)], indirect=True)
    def test_listed_origin_is_echoed(self, server):
        for r in (_get(server, A), _preflight(server, A)):
            assert r.headers.get("access-control-allow-origin") == A, dict(r.headers)
            assert r.headers.get("vary") == "Origin"
            assert "Authorization" in r.headers.get("access-control-allow-headers", "")
            assert "POST" in r.headers.get("access-control-allow-methods", "")
        assert _preflight(server, A).status_code == 204

    @pytest.mark.parametrize("server", [("--cors-origins", A)], indirect=True)
    def test_unlisted_origin_gets_nothing(self, server):
        for r in (_get(server, B), _preflight(server, B), _get(server, None)):
            _no_cors(r)
            assert r.headers.get("vary") == "Origin"

    @pytest.mark.parametrize("server", [("--cors-origins", "*")], indirect=True)
    def test_wildcard_restores_any_origin(self, server):
        for r in (_get(server, A), _get(server, B), _preflight(server, B), _get(server, None)):
            assert r.headers.get("access-control-allow-origin") == "*", dict(r.headers)
            assert "vary" not in r.headers
