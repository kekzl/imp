"""
#2206: Responses API `store: true` and `previous_response_id`.

What it tests:   validation and 404 envelopes, GET/DELETE routes, /metrics gauges,
                 the --responses-store-* flags (both lanes, model-less); two-turn
                 continuation equal to a stateless resend, TTL expiry, cap eviction,
                 store=false not retrievable (mock: generation needs a model).
What it does NOT test: token identity on a real model. That needs a GPU:
                 scripts/accept_2206.sh.
"""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

import conftest
from mock_server import run_server


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _server_cmd(port: int, *extra: str) -> list[str]:
    """The binary under test on the real lane, the mock script otherwise."""
    if conftest.SERVER_BIN:
        return [conftest.SERVER_BIN, "--host", "127.0.0.1", "--port", str(port), *extra]
    return [sys.executable, "-m", "mock_server", "--port", str(port), *extra]


def _start(port: int, *extra: str) -> subprocess.Popen:
    proc = subprocess.Popen(_server_cmd(port, *extra), cwd=os.path.dirname(__file__),
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited {proc.returncode}: {proc.stderr.read().decode()[-500:]}")
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=1).status_code == 200:
                return proc
        except httpx.TransportError:
            pass
        time.sleep(0.1)
    proc.kill()
    pytest.fail("server did not answer /health within 30 s")


def _stop(proc: subprocess.Popen):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _err(r: httpx.Response) -> dict:
    body = r.json()
    assert "error" in body, r.text
    return body["error"]


def _metric(text: str, name: str) -> float:
    for line in text.splitlines():
        if line.startswith(name + " "):
            return float(line.split()[1])
    raise AssertionError(f"{name} missing from /metrics")


# ---------------------------------------------------------------------------
# Both lanes: everything decided before a model is needed.
# ---------------------------------------------------------------------------

@pytest.mark.nomodel
class TestStoreValidation:
    def test_unknown_previous_response_id_is_404(self, client, model):
        r = client.post("/v1/responses", json={
            "model": model, "input": "x", "previous_response_id": "resp_does_not_exist"})
        assert r.status_code == 404, r.text
        e = _err(r)
        assert e["code"] == "response_not_found"
        assert e["param"] == "previous_response_id"
        assert e["type"] == "invalid_request_error"
        assert "resp_does_not_exist" in e["message"]

    @pytest.mark.parametrize("body,param", [
        ({"store": "yes"}, "store"),
        ({"store": 1}, "store"),
        ({"previous_response_id": 5}, "previous_response_id"),
        ({"previous_response_id": ""}, "previous_response_id"),
    ])
    def test_wrong_types_are_400(self, client, model, body, param):
        r = client.post("/v1/responses", json={"model": model, "input": "x", **body})
        assert r.status_code == 400, r.text
        assert _err(r).get("param") == param

    def test_store_true_is_not_refused(self, client, model, has_model):
        # Model-less binary: 503 from generation, past every store check; mock: 200.
        r = client.post("/v1/responses", json={"model": model, "input": "x", "store": True,
                                               "max_output_tokens": 4})
        assert r.status_code != 400, r.text
        assert r.status_code == (200 if has_model else 503), r.text

    @pytest.mark.parametrize("method", ["GET", "DELETE"])
    def test_unknown_id_route_is_404(self, client, method):
        r = client.request(method, "/v1/responses/resp_nope")
        assert r.status_code == 404, r.text
        e = _err(r)
        assert e["code"] == "response_not_found"
        assert "resp_nope" in e["message"]

    def test_metrics_gauges_present(self, client):
        text = client.get("/metrics").text
        for name in ("imp_responses_store_entries", "imp_responses_store_bytes",
                     "imp_responses_store_evictions_total", "imp_responses_store_expired_total"):
            assert _metric(text, name) >= 0


@pytest.mark.nomodel
class TestStoreFlags:
    def test_zero_limit_disables_store(self, model):
        port = _free_port()
        proc = _start(port, "--responses-store-max-entries", "0")
        try:
            r = httpx.post(f"http://127.0.0.1:{port}/v1/responses", timeout=10,
                           json={"model": model, "input": "x", "store": True})
            assert r.status_code == 400, r.text
            assert _err(r).get("param") == "store"
            r = httpx.post(f"http://127.0.0.1:{port}/v1/responses", timeout=10,
                           json={"model": model, "input": "x", "store": False, "max_output_tokens": 4})
            assert r.status_code != 400, r.text
        finally:
            _stop(proc)

    @pytest.mark.parametrize("flag", ["--responses-store-ttl", "--responses-store-max-entries",
                                      "--responses-store-max-mib"])
    @pytest.mark.parametrize("value", ["-1", "10s"])
    def test_bad_value_refuses_to_start(self, flag, value):
        proc = subprocess.run(_server_cmd(_free_port(), flag, value),
                              cwd=os.path.dirname(__file__), capture_output=True, timeout=30)
        assert proc.returncode == 1, proc.stderr.decode()[-500:]
        assert b"integer >= 0" in proc.stderr

    def test_get_requires_api_key(self):
        if not conftest.SERVER_BIN:
            pytest.skip("the mock has no auth; real binary lane only")
        port = _free_port()
        proc = _start(port, "--api-key", "sekrit")
        try:
            base = f"http://127.0.0.1:{port}"
            for method in ("GET", "DELETE"):
                assert httpx.request(method, base + "/v1/responses/resp_x", timeout=5).status_code == 401
                ok = httpx.request(method, base + "/v1/responses/resp_x", timeout=5,
                                   headers={"Authorization": "Bearer sekrit"})
                assert ok.status_code == 404, ok.text
        finally:
            _stop(proc)


# ---------------------------------------------------------------------------
# Mock only: needs a generated response to store.
# ---------------------------------------------------------------------------

@pytest.fixture
def own_mock(is_mock):
    """A private mock with custom store limits, so counters start at 0."""
    if not is_mock:
        pytest.skip("stored responses need generation; GPU lane: scripts/accept_2206.sh")
    servers = []

    def make(**limits):
        srv = run_server(port=0, latency_ms=0, **limits)
        servers.append(srv)
        return httpx.Client(base_url=f"http://127.0.0.1:{srv.server_address[1]}", timeout=10)

    yield make
    for srv in servers:
        srv.shutdown()


def _text(resp: dict) -> str:
    return "".join(p["text"] for it in resp["output"] if it["type"] == "message"
                   for p in it["content"] if p["type"] == "output_text")


def _ask(c: httpx.Client, model: str, **fields) -> httpx.Response:
    return c.post("/v1/responses", json={"model": model, **fields})


class TestStoreMock:
    def test_two_turns_equal_stateless_resend(self, own_mock, model):
        c = own_mock()
        t1 = _ask(c, model, input="first question", store=True)
        assert t1.status_code == 200, t1.text
        r1 = t1.json()
        assert r1["store"] is True and r1["previous_response_id"] is None

        t2 = _ask(c, model, input="second question", previous_response_id=r1["id"])
        assert t2.status_code == 200, t2.text
        r2 = t2.json()
        assert r2["previous_response_id"] == r1["id"]

        stateless = [{"role": "user", "content": "first question"}, *r1["output"],
                     {"role": "user", "content": "second question"}]
        s2 = _ask(c, model, input=stateless, store=False).json()
        assert _text(r2) == _text(s2)
        assert r2["usage"]["input_tokens"] == s2["usage"]["input_tokens"]
        assert _text(r2) != _text(r1)

        got = c.get(f"/v1/responses/{r1['id']}")
        assert got.status_code == 200, got.text
        assert got.json()["output"] == r1["output"]

    def test_store_false_is_not_retrievable(self, own_mock, model):
        c = own_mock()
        r = _ask(c, model, input="x", store=False).json()
        assert r["store"] is False
        assert c.get(f"/v1/responses/{r['id']}").status_code == 404
        cont = _ask(c, model, input="y", previous_response_id=r["id"])
        assert cont.status_code == 404
        assert _err(cont)["code"] == "response_not_found"
        # absent `store` is not stored either (imp default, unlike OpenAI's)
        r = _ask(c, model, input="x").json()
        assert c.get(f"/v1/responses/{r['id']}").status_code == 404

    def test_ttl_expiry_is_404(self, own_mock, model):
        c = own_mock(responses_store_ttl=0.3)
        r = _ask(c, model, input="x", store=True).json()
        assert c.get(f"/v1/responses/{r['id']}").status_code == 200
        time.sleep(0.5)
        cont = _ask(c, model, input="y", previous_response_id=r["id"])
        assert cont.status_code == 404, cont.text
        assert _err(cont)["code"] == "response_not_found"
        m = c.get("/metrics").text
        assert _metric(m, "imp_responses_store_expired_total") == 1
        assert _metric(m, "imp_responses_store_entries") == 0

    def test_cap_evicts_least_recently_used(self, own_mock, model):
        c = own_mock(responses_store_max_entries=2)
        ids = [_ask(c, model, input=f"q{i}", store=True).json()["id"] for i in range(2)]
        assert c.get(f"/v1/responses/{ids[0]}").status_code == 200  # ids[1] is now LRU
        ids.append(_ask(c, model, input="q2", store=True).json()["id"])
        assert c.get(f"/v1/responses/{ids[1]}").status_code == 404
        assert c.get(f"/v1/responses/{ids[0]}").status_code == 200
        assert c.get(f"/v1/responses/{ids[2]}").status_code == 200
        m = c.get("/metrics").text
        assert _metric(m, "imp_responses_store_evictions_total") == 1
        assert _metric(m, "imp_responses_store_entries") == 2

    def test_delete_then_404(self, own_mock, model):
        c = own_mock()
        r = _ask(c, model, input="x", store=True).json()
        d = c.delete(f"/v1/responses/{r['id']}")
        assert d.status_code == 200, d.text
        assert d.json() == {"id": r["id"], "object": "response.deleted", "deleted": True}
        assert c.get(f"/v1/responses/{r['id']}").status_code == 404
        assert c.delete(f"/v1/responses/{r['id']}").status_code == 404
