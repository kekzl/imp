"""
Tests for server lifecycle: OOM handling, error resilience, disconnect behavior.

What it tests:   503 on OOM, error response format, client disconnect handling.
What it does NOT test: Actual VRAM management, CUDA errors, kernel failures.
External state:  Running imp-server or mock server (OOM tests need mock --oom).
"""

import json
import os
import signal
import subprocess
import sys
import time

import httpx
import pytest

import conftest
from mock_server import MOCK_MODEL_ID, run_server


class TestOOMHandling:
    """Test server behavior under simulated OOM pressure.
    Only runs against mock server with OOM mode."""

    @pytest.fixture(scope="class")
    def oom_server(self):
        """Start a separate mock server in OOM mode."""
        port = 9098
        server = run_server(port=port, latency_ms=1, oom=True)
        url = f"http://127.0.0.1:{port}"
        # Wait for it to be ready
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{url}/health", timeout=2)
                if r.status_code == 200:
                    break
            except httpx.ConnectError:
                time.sleep(0.1)
        yield url
        server.shutdown()

    def test_oom_returns_503(self, oom_server):
        """OOM should produce 503, not 500 or crash."""
        r = httpx.post(f"{oom_server}/v1/chat/completions", json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
        }, timeout=10)
        assert r.status_code == 503

    def test_oom_has_retry_after(self, oom_server):
        """503 response should include Retry-After header."""
        r = httpx.post(f"{oom_server}/v1/chat/completions", json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
        }, timeout=10)
        assert r.status_code == 503
        assert "retry-after" in {k.lower() for k in r.headers.keys()}

    def test_oom_returns_json_error(self, oom_server):
        """503 body should be a JSON error object."""
        r = httpx.post(f"{oom_server}/v1/chat/completions", json={
            "model": MOCK_MODEL_ID,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 8,
        }, timeout=10)
        body = r.json()
        assert "error" in body
        assert "message" in body["error"]

    def test_health_ok_during_oom(self, oom_server):
        """/health should still return 200 even under OOM."""
        r = httpx.get(f"{oom_server}/health", timeout=5)
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestErrorResilience:
    """Test that error responses don't break subsequent requests."""

    def test_error_then_success(self, client, model):
        """After a 400 error, the next valid request should succeed."""
        # Send bad request
        r1 = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": "not an array",
        })
        assert r1.status_code == 400

        # Send good request
        r2 = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        })
        assert r2.status_code == 200
        assert "choices" in r2.json()

    def test_404_then_success(self, client, model):
        """After a 404 (wrong model), valid request succeeds."""
        r1 = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model.gguf",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert r1.status_code == 404

        r2 = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
        })
        assert r2.status_code == 200

    def test_multiple_errors_no_crash(self, client, model):
        """Rapid-fire invalid requests don't crash the server."""
        for _ in range(20):
            client.post("/v1/chat/completions",
                        content="invalid json{{{",
                        headers={"content-type": "application/json"})

        # Server should still be alive
        r = client.get("/health")
        assert r.status_code == 200


class TestClientDisconnect:
    """Test behavior when client drops connection mid-stream."""

    def test_disconnect_mid_stream_server_survives(self, base_url, model):
        """Close connection mid-stream; server should still handle next request."""
        # Start a streaming request and close it after first chunk
        try:
            with httpx.stream(
                "POST",
                f"{base_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Count from 1 to 100."}],
                    "max_tokens": 32,
                    "stream": True,
                },
                timeout=10.0,
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data: ") and line[6:].strip() != "[DONE]":
                        break  # got first chunk, close connection
        except Exception:
            pass  # connection close may raise

        # Give server a moment to clean up
        time.sleep(0.5)

        # Server should still handle new requests
        with httpx.Client(base_url=base_url, timeout=30.0) as c:
            r = c.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 4,
            })
            assert r.status_code == 200
            assert "choices" in r.json()


class TestGracefulShutdown:
    """Test SIGTERM handling on mock server."""

    @pytest.mark.skipif(not conftest.USE_MOCK, reason="Shutdown test only runs against mock server")
    def test_sigterm_exits_cleanly(self):
        """Mock server should exit within 2s of SIGTERM."""
        proc = subprocess.Popen(
            [sys.executable, "-m", "mock_server", "--port", "9097"],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait for server to start
        deadline = time.monotonic() + 5
        started = False
        while time.monotonic() < deadline:
            try:
                r = httpx.get("http://127.0.0.1:9097/health", timeout=1)
                if r.status_code == 200:
                    started = True
                    break
            except httpx.ConnectError:
                time.sleep(0.1)

        assert started, "Mock server did not start"

        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)

        # Should exit within 2 seconds
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            pytest.fail("Server did not exit within 2s of SIGTERM")

        assert proc.returncode == 0 or proc.returncode == -signal.SIGTERM
