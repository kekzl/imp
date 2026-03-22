import os
import time

import httpx
import pytest


BASE_URL = os.environ.get("IMP_TEST_URL", "http://localhost:8080")
MODEL = os.environ.get("IMP_TEST_MODEL", "")


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
        time.sleep(2)
    raise TimeoutError(f"Server at {url} not ready after {timeout}s")


@pytest.fixture(scope="session")
def base_url():
    return BASE_URL


@pytest.fixture(scope="session")
def model():
    if not MODEL:
        pytest.skip("IMP_TEST_MODEL not set")
    return MODEL


@pytest.fixture(scope="session")
def client(base_url):
    wait_for_server(base_url)
    with httpx.Client(base_url=base_url, timeout=60.0) as c:
        yield c


@pytest.fixture(scope="session", autouse=True)
def warmup(client, model):
    """Send warmup requests to prime cuBLAS autotuning and stabilize output."""
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
