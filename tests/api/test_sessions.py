"""
#2407: request field `session_id` and POST /v1/sessions/{id}/close.

What it tests:   session_id is accepted on /v1/chat/completions, /v1/completions and /v1/responses
                 (mock, generation); a malformed or wrong-typed session_id is a 400 naming the field
                 on all four dialects incl. /v1/messages (Anthropic envelope); the close endpoint
                 answers 200 {"id","object":"session","closed":true} for any valid id, idempotent,
                 and 400 for an id outside [A-Za-z0-9._:-]{1,128}. The 400s and the close endpoint
                 are `nomodel`: CI runs them against the shipping binary.
What it does NOT test: the pin itself (KV eviction under pressure). CPU: KVCacheManagerTest.Session*
                 in tests/test_kv_cache.cpp; GPU: the 2-turn agent run named in the PR.
"""

import pytest

BAD_IDS = ["", "a/b", "has space", "s" * 129, "ümlaut"]
GOOD_ID = "agent-7.run:3_x"


def _bodies(model):
    return {
        "/v1/chat/completions": {"model": model, "max_tokens": 2,
                                 "messages": [{"role": "user", "content": "Hi"}]},
        "/v1/completions": {"model": model, "max_tokens": 2, "prompt": "Hi"},
        "/v1/responses": {"model": model, "max_output_tokens": 16, "input": "Hi"},
        "/v1/messages": {"model": model, "max_tokens": 2,
                         "messages": [{"role": "user", "content": "Hi"}]},
    }


def _assert_field_400(r, path):
    assert r.status_code == 400, r.text
    body = r.json()
    if path == "/v1/messages":
        assert body.get("type") == "error", body
    assert body["error"]["type"] == "invalid_request_error", body
    assert "session_id" in body["error"]["message"], body


@pytest.mark.nomodel
class TestSessionIdValidation:

    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/responses",
                                      "/v1/messages"])
    @pytest.mark.parametrize("bad", BAD_IDS + [7, ["a"]], ids=repr)
    def test_bad_session_id_is_400(self, client, model, is_mock, path, bad):
        if is_mock and path == "/v1/messages":
            pytest.skip("mock server does not implement /v1/messages (#1329)")
        r = client.post(path, json={**_bodies(model)[path], "session_id": bad})
        _assert_field_400(r, path)


@pytest.mark.nomodel
class TestSessionClose:

    def test_close_valid_id_is_200_and_idempotent(self, client):
        for _ in range(2):
            r = client.post(f"/v1/sessions/{GOOD_ID}/close")
            assert r.status_code == 200, r.text
            assert r.json() == {"id": GOOD_ID, "object": "session", "closed": True}

    @pytest.mark.parametrize("bad", ["has%20space", "s" * 129, "%C3%BC"])
    def test_close_bad_id_is_400(self, client, bad):
        r = client.post(f"/v1/sessions/{bad}/close")
        assert r.status_code == 400, r.text
        assert "session_id" in r.json()["error"]["message"]

    def test_close_needs_post(self, client):
        assert client.get(f"/v1/sessions/{GOOD_ID}/close").status_code in (404, 405)


class TestSessionIdAccepted:
    """Generation with a valid session_id: the field rides along, the reply is unchanged."""

    @pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/completions", "/v1/responses"])
    def test_valid_session_id_generates(self, client, model, path):
        r = client.post(path, json={**_bodies(model)[path], "session_id": GOOD_ID})
        assert r.status_code == 200, r.text
