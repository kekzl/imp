"""Fill-in-the-middle (#2201): POST /infill (llama.cpp shape) and `suffix` on /v1/completions."""

import time

import httpx
import pytest

from conftest import parse_sse
from mock_server import MOCK_MODEL_ID, run_server

PREFIX = "def add(a, b):\n    "
SUFFIX = "\n\nprint(add(1, 2))\n"


@pytest.mark.nomodel
class TestInfillValidation:
    """Refused before model resolution: same answer on the mock and the model-less binary."""

    @pytest.mark.parametrize("field,value", [
        ("input_prefix", 5),
        ("input_suffix", ["x"]),
        ("prompt", {"a": 1}),
    ])
    def test_non_string_field_is_400(self, client, field, value):
        body = {"input_prefix": PREFIX, "input_suffix": SUFFIX, field: value}
        r = client.post("/infill", json=body)
        assert r.status_code == 400, r.text
        assert field in r.json()["error"]["message"]

    @pytest.mark.parametrize("extra", [
        "not-a-list",
        [{"filename": "a.py"}],
        [{"filename": 3, "text": "x"}],
        ["x"],
    ])
    def test_bad_input_extra_is_400(self, client, extra):
        r = client.post("/infill", json={"input_prefix": PREFIX, "input_extra": extra})
        assert r.status_code == 400, r.text
        assert "input_extra" in r.json()["error"]["message"]

    def test_route_exists(self, client, has_model):
        # Model-less binary: 503 (no weights), never 404.
        r = client.post("/infill", json={"input_prefix": PREFIX, "input_suffix": SUFFIX, "max_tokens": 4})
        assert r.status_code != 404, r.text
        if not has_model:
            assert r.status_code == 503, r.text

    def test_completions_suffix_non_string_is_400(self, client, model):
        r = client.post("/v1/completions", json={"model": model, "prompt": PREFIX, "suffix": 7})
        assert r.status_code == 400, r.text
        assert r.json()["error"].get("param") == "suffix"

    def test_completions_suffix_with_token_prompt_is_400(self, client, model):
        r = client.post("/v1/completions", json={"model": model, "prompt": [1, 2, 3], "suffix": SUFFIX})
        assert r.status_code == 400, r.text
        assert r.json()["error"].get("param") == "suffix"


class TestFimNotSupported:
    """A loaded model without FIM tokens: 400 fim_not_supported on both entry points."""

    @pytest.fixture(scope="class")
    def nofim_server(self):
        server = run_server(port=0, latency_ms=1, fim=False)  # port 0: never the real lane's port
        url = f"http://127.0.0.1:{server.server_address[1]}"
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=2).status_code == 200:
                    break
            except httpx.ConnectError:
                time.sleep(0.1)
        yield url
        server.shutdown()

    def test_infill_400(self, nofim_server):
        r = httpx.post(f"{nofim_server}/infill", json={"input_prefix": PREFIX, "input_suffix": SUFFIX})
        assert r.status_code == 400, r.text
        assert r.json()["error"]["code"] == "fim_not_supported"

    def test_completions_suffix_400(self, nofim_server):
        r = httpx.post(f"{nofim_server}/v1/completions",
                       json={"model": MOCK_MODEL_ID, "prompt": PREFIX, "suffix": SUFFIX})
        assert r.status_code == 400, r.text
        err = r.json()["error"]
        assert err["code"] == "fim_not_supported"
        assert err["param"] == "suffix"

    def test_completions_without_suffix_still_works(self, nofim_server):
        for extra in ({}, {"suffix": ""}):
            r = httpx.post(f"{nofim_server}/v1/completions",
                           json={"model": MOCK_MODEL_ID, "prompt": PREFIX, "max_tokens": 4, **extra})
            assert r.status_code == 200, r.text


def _fim_or_skip(r):
    # A real-model run on a non-FIM model answers the documented 400; that is the contract above.
    if r.status_code == 400 and r.json().get("error", {}).get("code") == "fim_not_supported":
        pytest.skip("loaded model has no FIM tokens")


class TestInfillShape:
    def test_infill_nonstream(self, client):
        r = client.post("/infill", json={"input_prefix": PREFIX, "input_suffix": SUFFIX, "max_tokens": 8,
                                         "temperature": 0})
        _fim_or_skip(r)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion"
        text = body["choices"][0]["text"]
        assert isinstance(text, str)
        assert body["content"] == text  # llama.cpp clients read `content`
        assert body["stop"] is True
        assert body["choices"][0]["finish_reason"] in ("stop", "length")
        assert body["usage"]["completion_tokens"] <= 8

    def test_infill_stream(self, client):
        r = client.post("/infill", json={"input_prefix": PREFIX, "input_suffix": SUFFIX, "max_tokens": 8,
                                         "stream": True, "temperature": 0})
        _fim_or_skip(r)
        assert r.status_code == 200, r.text
        assert r.headers["content-type"].startswith("text/event-stream")
        assert r.text.rstrip().endswith("data: [DONE]")
        events = [e for e in parse_sse(r.text) if e.get("choices")]
        assert events, r.text
        for e in events:
            assert e["object"] == "text_completion"
            assert e["content"] == e["choices"][0]["text"]
        assert events[-1]["choices"][0]["finish_reason"] in ("stop", "length")
        assert events[-1]["stop"] is True
        assert all(e["stop"] is False for e in events[:-1])

    def test_infill_input_extra_and_n_predict(self, client):
        r = client.post("/infill", json={
            "input_prefix": PREFIX, "input_suffix": SUFFIX, "prompt": "",
            "input_extra": [{"filename": "util.py", "text": "X = 1\n"}, {"text": "Y = 2\n"}],
            "n_predict": 4, "temperature": 0,
        })
        _fim_or_skip(r)
        assert r.status_code == 200, r.text
        assert r.json()["usage"]["completion_tokens"] <= 4

    def test_completions_suffix(self, client, model):
        r = client.post("/v1/completions", json={"model": model, "prompt": PREFIX, "suffix": SUFFIX,
                                                 "max_tokens": 8, "temperature": 0})
        _fim_or_skip(r)
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "text_completion"
        assert isinstance(body["choices"][0]["text"], str)
        assert "content" not in body  # the llama.cpp mirror is /infill only
