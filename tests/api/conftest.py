import os
import time

import httpx
import pytest

from mock_server import MOCK_MODEL_ID, run_server


BASE_URL = os.environ.get("IMP_TEST_URL", "http://localhost:8080")
MODEL = os.environ.get("IMP_TEST_MODEL", "")

# Auto-start mock server when no real server URL or model is configured
USE_MOCK = os.environ.get("IMP_USE_MOCK", "0") == "1" or (not MODEL and BASE_URL == "http://localhost:8080")
MOCK_PORT = int(os.environ.get("IMP_MOCK_PORT", "9099"))


def wait_for_server(url: str, timeout: float = 120.0):
    """Block until the server's /health endpoint returns 200."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{url}/health", timeout=5)
            if r.status_code == 200:
                return
        except httpx.ConnectError:
            pass
        time.sleep(0.5 if USE_MOCK else 2)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s")


# Session-scoped mock server (started once, shared across all tests)
_mock_server = None


def pytest_configure(config):
    """Start mock server if needed, before any test collection."""
    global _mock_server, BASE_URL, MODEL
    if USE_MOCK:
        _mock_server = run_server(port=MOCK_PORT, latency_ms=5)
        BASE_URL = f"http://127.0.0.1:{MOCK_PORT}"
        MODEL = MOCK_MODEL_ID


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def model():
    if not MODEL:
        pytest.skip("IMP_TEST_MODEL not set and mock mode disabled")
    return MODEL


@pytest.fixture(scope="session")
def client(base_url):
    wait_for_server(base_url, timeout=10 if USE_MOCK else 120)
    with httpx.Client(base_url=base_url, timeout=60.0) as c:
        yield c


@pytest.fixture(scope="session")
def is_mock():
    """True if running against mock server (skip tests that need real model)."""
    return USE_MOCK


@pytest.fixture(scope="session", autouse=True)
def warmup(client, model, is_mock):
    """Send warmup requests to prime cuBLAS autotuning and stabilize output."""
    if is_mock:
        return  # Mock doesn't need warmup
    for _ in range(2):
        client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 16,
            "temperature": 0,
        })


def parse_sse(response_text: str):
    """Parse SSE text into a list of JSON objects (skipping [DONE])."""
    import json
    events = []
    for line in response_text.splitlines():
        if line.startswith("data: "):
            data = line[6:]
            if data.strip() == "[DONE]":
                continue
            events.append(json.loads(data))
    return events
