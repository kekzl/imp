"""
#2199: POST /admin/lora/{load,unload} and --idle-unload-seconds.

What it tests:   request validation and error envelopes of both routes, the
                 load -> use -> unload -> refused cycle (mock), the flag parse
                 and its /health report, auth and rate limit on the new routes
                 (real binary).
What it does NOT test: device memory, the suspend itself, resume latency. Those
                 need a GPU: scripts/accept_2199.sh.
"""

import os
import socket
import subprocess
import sys
import time

import httpx
import pytest

import conftest


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


@pytest.mark.nomodel
class TestLoraRouteValidation:
    """Body validation answers before any model or engine is looked at."""

    @pytest.mark.parametrize("body", [{}, {"path": ""}, {"path": 5}, {"path": None}])
    def test_load_requires_path(self, client, body):
        r = client.post("/admin/lora/load", json=body)
        assert r.status_code == 400, r.text
        assert "path" in _err(r)["message"]

    def test_load_rejects_non_object_body(self, client):
        r = client.post("/admin/lora/load", content=b"[1,2]",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400, r.text

    def test_load_rejects_empty_name(self, client):
        r = client.post("/admin/lora/load", json={"path": "/tmp/x", "name": ""})
        assert r.status_code == 400, r.text
        assert "name" in _err(r)["message"]

    @pytest.mark.parametrize("body", [{}, {"id": "1"}, {"id": 1.5}, {"name": 3}])
    def test_unload_requires_id_or_name(self, client, body):
        r = client.post("/admin/lora/unload", json=body)
        assert r.status_code == 400, r.text

    def test_unload_unknown_id_is_404(self, client):
        r = client.post("/admin/lora/unload", json={"id": 987654})
        assert r.status_code == 404, r.text
        err = _err(r)
        assert "987654" in err["message"] and "not loaded" in err["message"]
        assert err.get("code") == "lora_not_found"

    def test_unload_unknown_name_is_404(self, client):
        r = client.post("/admin/lora/unload", json={"name": "never-loaded"})
        assert r.status_code == 404, r.text
        assert "never-loaded" in _err(r)["message"]

    def test_load_without_model_is_409(self, client, has_model):
        if has_model:
            pytest.skip("model-less real lane only")
        r = client.post("/admin/lora/load", json={"path": "/tmp/adapter"})
        assert r.status_code == 409, r.text
        assert "no model loaded" in _err(r)["message"]


class TestLoraCycleMock:
    """Load, select, unload, refused: the contract a client sees (mock)."""

    @pytest.fixture(autouse=True)
    def _mock_only(self, is_mock):
        if not is_mock:
            pytest.skip("real adapters need a GPU: scripts/accept_2199.sh")

    def _chat(self, client, model, lora):
        return client.post("/v1/chat/completions", json={
            "model": model, "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4, "lora": lora})

    def test_load_use_unload_refuse(self, client, model, tmp_path):
        adapter = tmp_path / "style-a"
        adapter.mkdir()
        r = client.post("/admin/lora/load", json={"path": str(adapter)})
        assert r.status_code == 200, r.text
        body = r.json()
        assert isinstance(body["id"], int) and body["id"] >= 1
        assert body["name"] == "style-a"
        lora_id = body["id"]

        assert self._chat(client, model, "style-a").status_code == 200

        dup = client.post("/admin/lora/load", json={"path": str(adapter)})
        assert dup.status_code == 409, dup.text
        assert _err(dup).get("code") == "lora_already_loaded"

        r = client.post("/admin/lora/unload", json={"id": lora_id})
        assert r.status_code == 200, r.text
        assert r.json() == {"id": lora_id, "name": "style-a", "unloaded": True}

        refused = self._chat(client, model, "style-a")
        assert refused.status_code == 400, refused.text
        err = _err(refused)
        assert err.get("code") == "lora_not_loaded"
        assert "style-a" in err["message"] and "not loaded" in err["message"]

        again = client.post("/admin/lora/unload", json={"id": lora_id})
        assert again.status_code == 404, again.text

    def test_ids_are_not_reused_and_name_unload_works(self, client, tmp_path):
        adapter = tmp_path / "b"
        adapter.mkdir()
        first = client.post("/admin/lora/load", json={"path": str(adapter), "name": "b1"}).json()["id"]
        assert client.post("/admin/lora/unload", json={"name": "b1"}).status_code == 200
        second = client.post("/admin/lora/load", json={"path": str(adapter), "name": "b1"}).json()["id"]
        assert second > first
        assert client.post("/admin/lora/unload", json={"id": second}).status_code == 200

    def test_load_missing_path_is_400(self, client, tmp_path):
        r = client.post("/admin/lora/load", json={"path": str(tmp_path / "absent")})
        assert r.status_code == 400, r.text
        assert _err(r).get("code") == "lora_load_failed"


class TestLoraDroppedOnSwapMock:
    """#2217: a model swap drops every adapter; the swap needs a model, so mock only."""

    SWAP_TO = "mock-model-v2"

    @pytest.fixture(autouse=True)
    def _mock_only(self, is_mock):
        if not is_mock:
            pytest.skip("a swap needs a loaded model: scripts/accept_2199.sh")

    def test_swap_drops_adapters(self, model, tmp_path):
        adapter = tmp_path / "style-s"
        adapter.mkdir()
        port = _free_port()
        proc = _start(port, "--swap-model", self.SWAP_TO)
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
                def chat(m):
                    return c.post("/v1/chat/completions", json={
                        "model": m, "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 4, "lora": "style-s"})

                r = c.post("/admin/lora/load", json={"path": str(adapter)})
                assert r.status_code == 200, r.text
                lora_id = r.json()["id"]
                assert chat(model).status_code == 200

                refused = chat(self.SWAP_TO)
                assert refused.status_code == 400, refused.text
                assert _err(refused).get("code") == "lora_not_loaded"
                # Swapping back does not bring it back.
                assert chat(model).status_code == 400

                for body in ({"id": lora_id}, {"name": "style-s"}):
                    gone = c.post("/admin/lora/unload", json=body)
                    assert gone.status_code == 404, gone.text
                    assert _err(gone).get("code") == "lora_not_found"
        finally:
            _stop(proc)


@pytest.mark.nomodel
class TestIdleUnloadFlag:
    """--idle-unload-seconds: default off, parsed strictly, reported on /health."""

    def test_default_is_off(self, client):
        body = client.get("/health").json()
        assert body["idle_unload_seconds"] == 0
        assert body["idle_suspended"] is False

    def test_flag_value_reaches_health(self):
        port = _free_port()
        proc = _start(port, "--idle-unload-seconds", "7")
        try:
            body = httpx.get(f"http://127.0.0.1:{port}/health", timeout=5).json()
            assert body["idle_unload_seconds"] == 7
            assert body["idle_suspended"] is False
        finally:
            _stop(proc)

    @pytest.mark.parametrize("value", ["-3", "30s", "abc"])
    def test_bad_value_refuses_to_start(self, value):
        proc = subprocess.run(_server_cmd(_free_port(), "--idle-unload-seconds", value),
                              cwd=os.path.dirname(__file__), capture_output=True, timeout=30)
        assert proc.returncode == 1, proc.stderr.decode()[-500:]
        assert b"integer >= 0" in proc.stderr


@pytest.mark.nomodel
class TestAdminLoraGuards:
    """The new routes sit behind the same pre-routing auth and rate limit as /admin/suspend."""

    @pytest.fixture(autouse=True)
    def _real_only(self):
        if not conftest.SERVER_BIN:
            pytest.skip("the mock has no auth or rate limit; real binary lane only")

    def test_api_key_required(self):
        port = _free_port()
        proc = _start(port, "--api-key", "sekrit")
        try:
            base = f"http://127.0.0.1:{port}"
            for route in ("/admin/lora/load", "/admin/lora/unload", "/admin/suspend"):
                r = httpx.post(base + route, json={"id": 1, "path": "/x"}, timeout=5)
                assert r.status_code == 401, (route, r.text)
            ok = httpx.post(base + "/admin/lora/unload", json={"id": 1},
                            headers={"Authorization": "Bearer sekrit"}, timeout=5)
            assert ok.status_code == 404, ok.text
        finally:
            _stop(proc)

    def test_rate_limited(self):
        port = _free_port()
        proc = _start(port, "--rate-limit", "2")
        try:
            base = f"http://127.0.0.1:{port}"
            codes = [httpx.post(base + "/admin/lora/unload", json={"id": 1}, timeout=5).status_code
                     for _ in range(4)]
            assert codes[:2] == [404, 404], codes
            assert codes[2:] == [429, 429], codes
        finally:
            _stop(proc)
