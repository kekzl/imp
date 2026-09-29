"""/v1/decide and /v1/score contract (#2198): 400s, response shape, probs sum to 1, cached_tokens per mode.
Not tested here: that the probabilities match the model (scripts/accept_2198.sh, GPU).
"""

import pytest

EVIDENCE = "The invoice lists 3 items, total 120 EUR, paid on 2026-03-01 by bank transfer."


def decide_body(model, **over):
    body = {
        "model": model,
        "evidence": EVIDENCE,
        "items": [
            {"id": "paid", "criterion": "Was the invoice paid?", "options": ["yes", "no", "unclear"]},
            {"id": "method", "criterion": "Payment method?", "options": ["cash", "card", "bank transfer"]},
        ],
    }
    body.update(over)
    return body


def score_body(model, **over):
    body = {"model": model, "messages": [{"role": "user", "content": "Pick A or B."}], "candidates": ["A", "B"]}
    body.update(over)
    return body


def assert_400(r, needle):
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert needle in err["message"], err["message"]


# Validation runs before model resolution, so the model-less binary answers all of these.
@pytest.mark.nomodel
class TestDecideValidation:
    def test_route_registered(self, client, model, has_model):
        r = client.post("/v1/decide", json=decide_body(model))
        # Model-less: past validation the request needs weights (503), never an unknown route (404).
        assert r.status_code == (200 if has_model else 503), r.text

    def test_more_than_16_options(self, client, model):
        items = [{"criterion": "c", "options": [f"o{i}" for i in range(17)]}]
        assert_400(client.post("/v1/decide", json=decide_body(model, items=items)), "maximum is 16")

    def test_exactly_16_options_pass_validation(self, client, model, has_model):
        items = [{"criterion": "c", "options": [f"o{i}" for i in range(16)]}]
        r = client.post("/v1/decide", json=decide_body(model, items=items))
        assert r.status_code == (200 if has_model else 503), r.text

    def test_empty_options(self, client, model):
        items = [{"criterion": "c", "options": []}]
        assert_400(client.post("/v1/decide", json=decide_body(model, items=items)), "options must be an array")

    def test_single_option(self, client, model):
        items = [{"criterion": "c", "options": ["only"]}]
        assert_400(client.post("/v1/decide", json=decide_body(model, items=items)), "options must be an array")

    def test_non_string_option(self, client, model):
        items = [{"criterion": "c", "options": ["a", 2]}]
        assert_400(client.post("/v1/decide", json=decide_body(model, items=items)), "only strings")

    def test_mode_shared_not_implemented(self, client, model):
        r = client.post("/v1/decide", json=decide_body(model, mode="shared"))
        assert_400(r, "not implemented yet, see #2198")

    def test_unknown_mode(self, client, model):
        assert_400(client.post("/v1/decide", json=decide_body(model, mode="batch")), '"mode" must be one of')

    def test_missing_evidence(self, client, model):
        body = decide_body(model)
        del body["evidence"]
        assert_400(client.post("/v1/decide", json=body), '"evidence"')

    def test_empty_items(self, client, model):
        assert_400(client.post("/v1/decide", json=decide_body(model, items=[])), '"items"')

    def test_missing_criterion(self, client, model):
        items = [{"options": ["a", "b"]}]
        assert_400(client.post("/v1/decide", json=decide_body(model, items=items)), "criterion")


@pytest.mark.nomodel
class TestScoreValidation:
    def test_route_registered(self, client, model, has_model):
        r = client.post("/v1/score", json=score_body(model))
        assert r.status_code == (200 if has_model else 503), r.text

    def test_prompt_and_messages_both(self, client, model):
        assert_400(client.post("/v1/score", json=score_body(model, prompt="x")), "exactly one of")

    def test_neither_prompt_nor_messages(self, client, model):
        body = score_body(model)
        del body["messages"]
        assert_400(client.post("/v1/score", json=body), "exactly one of")

    def test_one_candidate(self, client, model):
        assert_400(client.post("/v1/score", json=score_body(model, candidates=["A"])), '"candidates"')

    def test_bad_candidate_type(self, client, model):
        assert_400(client.post("/v1/score", json=score_body(model, candidates=["A", 1.5])), "each candidate")

    def test_empty_candidate_string(self, client, model):
        assert_400(client.post("/v1/score", json=score_body(model, candidates=["A", ""])), "must not be empty")

    def test_mode_shared_not_implemented(self, client, model):
        assert_400(client.post("/v1/score", json=score_body(model, mode="shared")), "not implemented yet, see #2198")


# Tokenizer guards need a loaded vocabulary: mock (mock tokenizer rules) and real server only.
class TestScoreTokenGuards:
    def test_multi_token_candidate(self, client, model):
        r = client.post("/v1/score", json=score_body(model, candidates=["A", "Xylophonequarkzz"]))
        assert_400(r, "scoring needs exactly one")

    def test_boundary_merge_after_colon(self, client, model):
        body = score_body(model, prompt="Answer:", candidates=["A", "B"])
        del body["messages"]
        assert_400(client.post("/v1/score", json=body), "merges with the end of the prompt")

    def test_boundary_merge_after_angle_bracket(self, client, model):
        body = score_body(model, prompt="<answer>", candidates=["A", "B"])
        del body["messages"]
        assert_400(client.post("/v1/score", json=body), "merges with the end of the prompt")

    def test_candidate_id_out_of_range(self, client, model):
        r = client.post("/v1/score", json=score_body(model, candidates=[1, 10**9]))
        assert_400(r, "outside the vocabulary")


class TestDecideResponse:
    def test_shape_and_probs(self, client, model):
        r = client.post("/v1/decide", json=decide_body(model))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "decide"
        assert body["mode_used"] == "serial"
        assert [it["id"] for it in body["items"]] == ["paid", "method"]
        for it in body["items"]:
            assert list(it["probs"]) == ["A", "B", "C"]
            assert abs(sum(it["probs"].values()) - 1.0) <= 1e-5
            assert it["argmax"] == "ABC"[it["argmax_index"]]
            assert it["probs"][it["argmax"]] == max(it["probs"].values())
            assert it["prompt_tokens"] > 0
        assert body["usage"]["prompt_tokens"] == sum(it["prompt_tokens"] for it in body["items"])
        assert body["usage"]["cached_tokens"] == sum(it["cached_tokens"] for it in body["items"])

    def test_serial_reuses_evidence_prefix(self, client, model):
        body = client.post("/v1/decide", json=decide_body(model, mode="serial")).json()
        assert body["items"][1]["cached_tokens"] > 0

    def test_direct_reports_no_cache(self, client, model):
        body = client.post("/v1/decide", json=decide_body(model, mode="direct")).json()
        assert body["mode_used"] == "direct"
        assert all(it["cached_tokens"] == 0 for it in body["items"])

    def test_default_ids_are_indices(self, client, model):
        items = [{"criterion": "c", "options": ["x", "y"]}]
        body = client.post("/v1/decide", json=decide_body(model, items=items)).json()
        assert body["items"][0]["id"] == 0


class TestScoreResponse:
    def test_shape_and_probs(self, client, model):
        r = client.post("/v1/score", json=score_body(model, candidates=["A", "B", 42]))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["object"] == "score"
        assert [c["candidate"] for c in body["candidates"]] == ["A", "B", 42]
        assert body["candidates"][2]["token_id"] == 42
        assert abs(sum(c["prob"] for c in body["candidates"]) - 1.0) <= 1e-5
        probs = [c["prob"] for c in body["candidates"]]
        assert probs[body["argmax_index"]] == max(probs)
        assert body["prompt_tokens"] > 0
