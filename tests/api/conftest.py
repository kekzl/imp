import os
import subprocess
import time

import httpx
import pytest

from mock_server import MOCK_MODEL_ID, run_server


BASE_URL = os.environ.get("IMP_TEST_URL", "http://localhost:8080")
MODEL = os.environ.get("IMP_TEST_MODEL", "")

# IMP_SERVER_BIN=<path to imp-server>: start the SHIPPING binary model-less and
# test it directly. This is the lane a GPU-less runner can have (#1302) — the
# server answers its whole request-validation surface before it ever looks for a
# model, so every 4xx the mock claims can be checked against the real thing.
# Generation needs a GPU and stays out: those tests are the ones NOT marked
# `nomodel`, and the lane selects `-m nomodel`.
SERVER_BIN = os.environ.get("IMP_SERVER_BIN", "")
SERVER_PORT = int(os.environ.get("IMP_SERVER_PORT", "9098"))

# Auto-start mock server when no real server URL or model is configured
USE_MOCK = not SERVER_BIN and (
    os.environ.get("IMP_USE_MOCK", "0") == "1" or (not MODEL and BASE_URL == "http://localhost:8080")
)
MOCK_PORT = int(os.environ.get("IMP_MOCK_PORT", "9099"))

# True when a model is resident, i.e. tests may ask for generated tokens.
# A model-less real server is deliberately not one.
HAS_MODEL = not SERVER_BIN


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
_real_server = None


def pytest_configure(config):
    """Start the server under test (mock or real binary) before collection."""
    global _mock_server, _real_server, BASE_URL, MODEL
    if SERVER_BIN:
        # No --model: the binary serves the validation surface and answers 503
        # on anything that would need weights. stdout/stderr are inherited so a
        # start-up failure is visible in the job log instead of a bare timeout.
        _real_server = subprocess.Popen(
            [SERVER_BIN, "--host", "127.0.0.1", "--port", str(SERVER_PORT)]
        )
        BASE_URL = f"http://127.0.0.1:{SERVER_PORT}"
        # Any name: nothing resolves on a model-less server, and the tests that
        # send one only care about what happens BEFORE model resolution.
        MODEL = MODEL or "no-model-loaded"
    elif USE_MOCK:
        _mock_server = run_server(port=MOCK_PORT, latency_ms=5)
        BASE_URL = f"http://127.0.0.1:{MOCK_PORT}"
        MODEL = MOCK_MODEL_ID


def pytest_unconfigure(config):
    if _real_server is not None and _real_server.poll() is None:
        _real_server.terminate()
        try:
            _real_server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _real_server.kill()


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
    wait_for_server(base_url, timeout=120 if HAS_MODEL and not USE_MOCK else 10)
    with httpx.Client(base_url=base_url, timeout=60.0) as c:
        yield c


@pytest.fixture(scope="session")
def is_mock():
    """True if running against mock server (skip tests that need real model)."""
    return USE_MOCK


@pytest.fixture(scope="session")
def has_model():
    """False on the model-less real-binary lane: no request can generate."""
    return HAS_MODEL


@pytest.fixture(scope="session", autouse=True)
def warmup(client, model, is_mock, has_model):
    """Send warmup requests to prime cuBLAS autotuning and stabilize output."""
    if is_mock or not has_model:
        return  # Mock needs no warmup; a model-less server has nothing to warm
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
