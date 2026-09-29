"""Prompt logprobs on /v1/completions (#2207).

`nomodel` tests run against the mock and the model-less imp-server binary (parameter validation
happens before weights are needed). Shape tests need generated output: mock lane and `make test-server`.
"""

import math

import pytest

PROMPT = "The quick brown fox jumps over the lazy dog."


@pytest.mark.nomodel
class TestPromptLogprobsValidation:
    @pytest.mark.parametrize("value", [21, -1, 100, True, 1.5, "3", [1]])
    def test_out_of_range_or_wrong_type_is_400(self, client, model, value):
        r = client.post("/v1/completions", json={"model": model, "prompt": "hi", "prompt_logprobs": value})
        assert r.status_code == 400, r.text
        assert "prompt_logprobs" in r.json()["error"]["message"]

    @pytest.mark.parametrize("value", [0, 1, 20, None])
    def test_in_range_passes_validation(self, client, model, value):
        r = client.post("/v1/completions", json={"model": model, "prompt": "hi", "max_tokens": 1,
                                                 "prompt_logprobs": value})
        assert r.status_code != 400, r.text

    def test_stream_with_prompt_logprobs_is_400(self, client, model):
        r = client.post("/v1/completions", json={"model": model, "prompt": "hi", "stream": True,
                                                 "prompt_logprobs": 1})
        assert r.status_code == 400, r.text
        assert "stream" in r.json()["error"]["message"]

    @pytest.mark.parametrize("logprobs", [True, 2])
    def test_stream_with_echo_and_logprobs_is_400(self, client, model, logprobs):
        r = client.post("/v1/completions", json={"model": model, "prompt": "hi", "stream": True,
                                                 "echo": True, "logprobs": logprobs})
        assert r.status_code == 400, r.text
        assert "stream" in r.json()["error"]["message"]

    def test_stream_with_echo_alone_passes_validation(self, client, model):
        r = client.post("/v1/completions", json={"model": model, "prompt": "hi", "stream": True,
                                                 "echo": True, "max_tokens": 1})
        assert r.status_code != 400, r.text


def _complete(client, model, **fields):
    r = client.post("/v1/completions", json={"model": model, "prompt": PROMPT, "max_tokens": 4,
                                             "temperature": 0, **fields})
    assert r.status_code == 200, r.text
    return r.json()


class TestPromptLogprobsShape:
    def test_vllm_shape_one_entry_per_prompt_token_first_null(self, client, model):
        body = _complete(client, model, prompt_logprobs=2)
        plp = body["choices"][0]["prompt_logprobs"]
        assert len(plp) == body["usage"]["prompt_tokens"]
        assert plp[0] is None
        for entry in plp[1:]:
            assert 1 <= len(entry) <= 3, entry  # prompt token + top 2
            assert all(k.lstrip("-").isdigit() for k in entry)
            ranks = sorted(v["rank"] for v in entry.values())
            assert ranks[0] >= 1
            for v in entry.values():
                assert v["logprob"] <= 1e-4 and math.isfinite(v["logprob"])
                assert isinstance(v["decoded_token"], str)

    def test_prompt_logprobs_zero_keeps_only_the_prompt_token(self, client, model):
        plp = _complete(client, model, prompt_logprobs=0)["choices"][0]["prompt_logprobs"]
        assert plp[0] is None
        assert all(len(entry) == 1 for entry in plp[1:])

    def test_echo_logprobs_prompt_tokens_precede_completion(self, client, model):
        body = _complete(client, model, echo=True, logprobs=2)
        choice = body["choices"][0]
        lp = choice["logprobs"]
        n_prompt = body["usage"]["prompt_tokens"]
        n = len(lp["tokens"])
        assert n >= n_prompt
        assert len(lp["token_logprobs"]) == len(lp["top_logprobs"]) == len(lp["text_offset"]) == n
        assert lp["token_logprobs"][0] is None and lp["top_logprobs"][0] is None
        for v in lp["token_logprobs"][1:]:
            assert v <= 1e-4 and math.isfinite(v)
        for top in lp["top_logprobs"][1:n_prompt]:
            assert len(top) <= 2
        assert choice["text"].startswith(PROMPT)
        assert lp["text_offset"][0] == 0
        assert lp["text_offset"] == sorted(lp["text_offset"])
        assert lp["text_offset"][-1] <= len(choice["text"].encode())

    def test_both_forms_agree(self, client, model):
        body = _complete(client, model, echo=True, logprobs=1, prompt_logprobs=1)
        choice = body["choices"][0]
        plp, echo = choice["prompt_logprobs"], choice["logprobs"]["token_logprobs"]
        assert len(plp) == body["usage"]["prompt_tokens"]
        for pos in range(1, len(plp)):
            assert any(abs(v["logprob"] - echo[pos]) < 1e-6 for v in plp[pos].values()), pos
