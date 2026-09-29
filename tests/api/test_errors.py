"""Tests for error handling and parameter validation.

Everything here that is marked `nomodel` runs in two lanes: against the Python
mock (CI, `IMP_USE_MOCK=1`) and against the shipping `imp-server` binary started
without a model (CI, `IMP_SERVER_BIN=...`). Parameter validation happens before
the server looks for weights, so the second lane needs no GPU — and it is the
only one that says anything about `tools/imp-server/` (#1302).
"""

import os

import pytest


@pytest.mark.nomodel
class TestParameterValidation:
    def test_invalid_json(self, client):
        r = client.post(
            "/v1/chat/completions",
            content="not json{{{",
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert "error" in r.json()

    def test_missing_messages(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
        })
        assert r.status_code == 400

    def test_messages_not_array(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": "not an array",
        })
        assert r.status_code == 400
        assert "array" in r.json()["error"]["message"].lower()

    def test_empty_messages(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [],
        })
        assert r.status_code == 400

    @pytest.mark.parametrize("temp", [-0.1, 2.5, 3.0])
    def test_temperature_out_of_range(self, client, model, temp):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "temperature": temp,
        })
        assert r.status_code == 400
        assert "temperature" in r.json()["error"]["message"].lower()

    @pytest.mark.parametrize("top_p", [-0.1, 1.5])
    def test_top_p_out_of_range(self, client, model, top_p):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "top_p": top_p,
        })
        assert r.status_code == 400
        assert "top_p" in r.json()["error"]["message"].lower()

    def test_max_tokens_zero(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 0,
        })
        assert r.status_code == 400
        assert "max_tokens" in r.json()["error"]["message"].lower()

    def test_max_tokens_negative(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": -5,
        })
        assert r.status_code == 400

    @pytest.mark.parametrize("n", [0, 5, 100])
    def test_n_out_of_range(self, client, model, n):
        """/v1/chat/completions accepts n in [1,4] and rejects the rest.

        This used to assert that n=2 is a 400 — which is what the mock does and
        the shipping server does NOT: `handlers.cpp` validates n against [1,4]
        and `handlers_chat_core.cpp` runs n independent generations. The
        assertion was green for a year because only the mock ever answered it.
        """
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "n": n,
        })
        assert r.status_code == 400
        assert "n" in r.json()["error"]["message"].lower()

    def test_max_tokens_null_is_not_a_crash(self, client, model):
        """`max_tokens: null` is the SDK default for "unset", not an error.

        It must be treated as absent — never as a parse failure and never as a
        dropped connection.
        """
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": None,
        })
        # 200 with weights, 503 model-less — never 400, never a disconnect.
        assert r.status_code in (200, 503), r.text[:200]

    @pytest.mark.parametrize("body", ["[1,2,3]", "42", '"a string"', "null"])
    def test_non_object_json_body(self, client, body):
        """A well-formed JSON body that is not an object is a 400, not a hang-up."""
        r = client.post(
            "/v1/chat/completions",
            content=body,
            headers={"content-type": "application/json"},
        )
        assert r.status_code == 400
        assert "error" in r.json()


class TestParameterAcceptance:
    """The other half of validation: values at the boundary must be ACCEPTED.

    Not `nomodel` — proving acceptance means producing a completion, which needs
    weights. A model-less server answers 503 here, which is not the contract.
    """

    def test_temperature_boundary_valid(self, client, model):
        """temperature=0 and temperature=2 should be accepted."""
        for t in [0, 2.0]:
            r = client.post("/v1/chat/completions", json={
                "model": model,
                "messages": [{"role": "user", "content": "Hi"}],
                "temperature": t,
                "max_tokens": 1,
            })
            assert r.status_code == 200, f"temperature={t} should be valid"

    def test_n_equals_1_valid(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 1,
            "max_tokens": 1,
            "temperature": 0,
        })
        assert r.status_code == 200

    def test_n_within_range_returns_that_many_choices(self, client, model):
        """n=2 is accepted and produces two choices (the contract the mock denied)."""
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 2,
            "max_tokens": 1,
            "temperature": 0,
        })
        assert r.status_code == 200
        assert len(r.json()["choices"]) == 2


@pytest.mark.nomodel
class TestWrongFieldType:
    """A wrong-typed scalar answered 400 "[json.exception.type_error.302] type must be number, but
    is string" without the field, and /v1/messages answered 500. The binary names the field."""

    @pytest.mark.parametrize("path,body,field", [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "temperature": "hot"}, "temperature"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": 5}]}, "messages[0].content"),
        ("/v1/completions", {"prompt": "hi", "echo": "yes"}, "echo"),
        ("/v1/responses", {"input": "hi", "max_output_tokens": "10"}, "max_output_tokens"),
        ("/v1/embeddings", {"input": "hi", "encoding_format": 5}, "encoding_format"),
        ("/v1/messages", {"max_tokens": "10", "messages": [{"role": "user", "content": "hi"}]}, "max_tokens"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "logprobs": "yes"}, "logprobs"),
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                  "stream_options": {"include_usage": "yes"}}, "stream_options.include_usage"),
        ("/v1/chat/completions", {"messages": [1]}, "messages[0]"),
        ("/v1/responses", {"input": "hi", "tools": [1]}, "tools[0]"),
        ("/v1/messages", {"max_tokens": 8, "messages": [1]}, "messages[0]"),
        ("/v1/messages/count_tokens", {"messages": [1]}, "messages[0]"),
        ("/v1/rerank", {"query": "q", "documents": ["a"], "top_n": "2"}, "top_n"),
        ("/v1/rerank", {"query": "q", "documents": ["a"], "return_documents": "yes"}, "return_documents"),
    ])
    def test_names_the_field(self, client, model, is_mock, path, body, field):
        if is_mock:
            pytest.skip("the mock does not reproduce the binary's parser")
        r = client.post(path, json=dict(body, model=model), headers={"anthropic-version": "2023-06-01"})
        assert r.status_code == 400, r.text
        msg = r.json()["error"]["message"]
        assert f'"{field}"' in msg and "json.exception" not in msg, msg


@pytest.mark.nomodel
def test_assistant_tool_call_with_text_parts_is_accepted(client, model, is_mock):
    """Assistant content as text parts next to tool_calls is valid OpenAI; it answered a raw 400."""
    if is_mock:
        pytest.skip("the mock does not reproduce the binary's parser")
    r = client.post("/v1/chat/completions", json={"model": model, "max_tokens": 4, "messages": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "calling"}],
         "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": [{"type": "text", "text": "ok"}]},
    ]})
    assert r.status_code != 400, r.text


@pytest.mark.nomodel
@pytest.mark.parametrize("path,extra", [
    ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
    ("/v1/completions", {"prompt": "hi"}),
])
def test_null_fields_mean_absent(client, model, is_mock, path, extra):
    """OpenAI fields are nullable; "seed": null and friends answered a raw 400 type_error.302."""
    if is_mock:
        pytest.skip("the mock does not reproduce the binary's parser")
    nulls = {k: None for k in ("seed", "temperature", "top_p", "n", "presence_penalty", "frequency_penalty",
                               "logprobs", "top_logprobs", "stream", "stop", "max_tokens", "echo")}
    if path == "/v1/completions":
        nulls.pop("top_logprobs")
    r = client.post(path, json=dict(extra, model=model, **nulls))
    assert r.status_code != 400, r.text


@pytest.mark.nomodel
class TestSamplerRange:
    """repetition_penalty 0 or < 0, min_p and penalties outside their range passed the parser; the
    penalty kernel divides positive logits by repetition_penalty."""

    @pytest.mark.parametrize("path,extra", [
        ("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}),
        ("/v1/completions", {"prompt": "hi"}),
    ])
    @pytest.mark.parametrize("field,value", [
        ("repetition_penalty", 0), ("repetition_penalty", -1), ("min_p", 5), ("presence_penalty", 100),
        ("frequency_penalty", -100),
    ])
    def test_rejects_out_of_range(self, client, model, is_mock, path, extra, field, value):
        if is_mock:
            pytest.skip("the mock does not reproduce the binary's parser")
        r = client.post(path, json=dict(extra, model=model, **{field: value}),
                        headers={"anthropic-version": "2023-06-01"})
        assert r.status_code == 400, r.text
        assert f'"{field}"' in r.json()["error"]["message"], r.text


@pytest.mark.nomodel
class TestModelField:
    def test_missing_model_field(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert r.status_code == 400
        assert "model" in r.json()["error"]["message"].lower()

    def test_missing_model_completions(self, client):
        r = client.post("/v1/completions", json={
            "prompt": "Hello",
        })
        assert r.status_code == 400
        assert "model" in r.json()["error"]["message"].lower()


@pytest.mark.nomodel
class TestUnknownEndpoint:
    """An unmatched route must answer in the dialect of the path it was on.

    httplib's built-in 404 has a zero-length body, so a client doing
    `r.json()["error"]["message"]` on a typo'd path got a JSON parse error and
    no reason. The mock has always sent an envelope here; nothing checked that
    the server did too (#1302).
    """

    @pytest.mark.parametrize("path", ["/v1/nope", "/v1/chat/completion", "/nonsense"])
    def test_unknown_path_returns_json_error(self, client, path):
        r = client.post(path, json={})
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert body["error"]["message"]

    def test_unknown_messages_path_uses_anthropic_envelope(self, client):
        """Anthropic clients read `type` at the top level, not `error.message`."""
        r = client.post("/v1/messages/nope", json={})
        assert r.status_code == 404
        body = r.json()
        assert body.get("type") == "error"
        assert body["error"]["message"]


class TestUnknownModel:
    @pytest.mark.parametrize("name", [
        "/tmp/x.gguf",                 # absolute path
        "../../etc/passwd",            # traversal
        "./models/x.gguf",             # relative path
        "org/x.gguf",                  # a relative file, not an HF repo id
        "a/b/c",                       # multi-segment
    ])
    def test_path_shaped_model_name_is_not_resolved(self, client, name):
        """A request may name a model, never a path (AUDIT_arch_2026 F2-1).

        Until 2026-09-05 any name containing '/' reached fs::exists and a
        readable .gguf anywhere on the box was loaded, evicting the resident
        model. Now every path shape answers like an unknown name: 404 with the
        envelope on a loaded server, 503 on a model-less one, never 200 or 500.
        """
        r = client.post("/v1/chat/completions", json={
            "model": name,
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert r.status_code in (404, 503), r.text
        body = r.json()
        assert "error" in body
        assert body["error"]["message"]

    def test_chat_completions_unknown_model(self, client):
        r = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model-xyz.gguf",
            "messages": [{"role": "user", "content": "Hi"}],
        })
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert "not found" in body["error"]["message"].lower()

    def test_completions_unknown_model(self, client):
        r = client.post("/v1/completions", json={
            "model": "nonexistent-model-xyz.gguf",
            "prompt": "Hello",
        })
        assert r.status_code == 404
        body = r.json()
        assert "error" in body
        assert "not found" in body["error"]["message"].lower()


@pytest.mark.nomodel
class TestCompletionsEndpoint:
    def test_missing_prompt(self, client, model):
        r = client.post("/v1/completions", json={
            "model": model,
        })
        assert r.status_code == 400

    def test_temperature_out_of_range(self, client, model):
        r = client.post("/v1/completions", json={
            "model": model,
            "prompt": "Hello",
            "temperature": 3.0,
        })
        assert r.status_code == 400

    def test_n_greater_than_1(self, client, model):
        r = client.post("/v1/completions", json={
            "model": model,
            "prompt": "Hello",
            "n": 3,
        })
        assert r.status_code == 400


@pytest.mark.nomodel
class TestPerRequestCaps:
    """The per-request work caps (#1616, #1617, --max-images-per-request) had
    no test in any lane (AUDIT_arch_2026 F2-5). Defaults: n 8, messages 10000,
    logit_bias 1024, images 8, rerank documents 512. Every check runs before
    the server looks for a model, so the model-less lane sees the 400.
    """

    def test_n_above_max_n(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 1000,
        })
        assert r.status_code == 400
        assert '"n"' in r.json()["error"]["message"]

    def test_messages_above_cap(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "x"}] * 20000,
        })
        assert r.status_code == 400
        assert "messages" in r.json()["error"]["message"]

    # speculative object form ({"mtp_k": N}) with depth outside range must 400 and NAME the
    # range, not be parsed and silently dropped (#1384 class).
    @pytest.mark.parametrize("value", [
        "yes",                  # wrong type entirely
        1,                      # a number is not a boolean
        {"mtp_k": -1},          # below the range
        {"mtp_k": 99},          # above any chain buffer
        {"mtp_k": "2"},         # a string depth
        {"mtp_k": 1.5},         # a fractional depth
    ])
    def test_speculative_field_rejects_bad_shapes(self, client, model, value):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "speculative": value,
        })
        assert r.status_code == 400, r.text
        assert "speculative" in r.json()["error"]["message"]

    def test_speculative_depth_error_names_the_range(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "speculative": {"mtp_k": 99},
        })
        assert r.status_code == 400
        msg = r.json()["error"]["message"]
        assert "0.." in msg, msg

    @pytest.mark.parametrize("value", [True, False, {"mtp_k": 0}, {"mtp_k": 2}, {}])
    def test_speculative_field_accepts_both_forms(self, client, model, value):
        # Accepted shapes must not be 400. Whether the request then generates
        # is a different lane's business (the model-less server answers 503).
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 4,
            "speculative": value,
        })
        assert r.status_code != 400, r.text

    def test_logit_bias_above_cap(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "logit_bias": {str(i): 1 for i in range(5000)},
        })
        assert r.status_code == 400
        assert "logit_bias" in r.json()["error"]["message"]

    @pytest.mark.parametrize("bias", [{"abc": 1}, {"12abc": 1}, {"-5": 1}, {"7": 101}, {"7": "1"}, [1, 2]])
    def test_malformed_logit_bias_is_refused_not_dropped(self, client, model, bias):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "logit_bias": bias,
        })
        assert r.status_code == 400, r.text
        assert "logit_bias" in r.json()["error"]["message"]

    def test_images_above_cap(self, client, model):
        part = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": [part] * 9}],
        })
        assert r.status_code == 400
        assert "images" in r.json()["error"]["message"]

    def test_rerank_documents_above_cap(self, client, model, is_mock):
        if is_mock:
            pytest.skip("the mock has no /rerank")
        r = client.post("/rerank", json={
            "model": model,
            "query": "q",
            "documents": ["d"] * 1000,
        })
        assert r.status_code == 400
        assert "documents" in r.json()["error"]["message"]


def _client_error_lines(capsys, needle, timeout=3.0):
    """Server stderr lines carrying `needle`, collected until one shows up or `timeout` passes."""
    import time
    seen = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        seen += capsys.readouterr().err
        if needle in seen:
            time.sleep(0.3)  # a second line for the same response would arrive with the first
            seen += capsys.readouterr().err
            break
        time.sleep(0.05)
    return [ln for ln in seen.splitlines() if needle in ln]


@pytest.mark.nomodel
@pytest.mark.skipif(not os.environ.get("IMP_SERVER_BIN"), reason="reads the real binary's stderr")
class TestClientErrorLogLine:
    """#2279: every 4xx logs exactly one server line with status, route and the reason sent."""

    @pytest.mark.parametrize("path,body", [
        ("/v1/chat/completions", {"messages": "not an array"}),
        ("/v1/messages", {"max_tokens": 8, "messages": "not an array"}),
    ])
    def test_4xx_logs_one_line_with_the_reason(self, client, model, capsys, path, body):
        capsys.readouterr()
        r = client.post(path, json={"model": model, **body})
        assert 400 <= r.status_code < 500, r.text
        reason = r.json()["error"]["message"]
        lines = _client_error_lines(capsys, f"HTTP {r.status_code} POST {path}")
        assert len(lines) == 1, lines
        assert reason[:60] in lines[0], lines[0]

    def test_unknown_route_404_is_logged(self, client, capsys):
        capsys.readouterr()
        r = client.get("/v1/no-such-route-2279")
        assert r.status_code == 404
        lines = _client_error_lines(capsys, "HTTP 404 GET /v1/no-such-route-2279")
        assert len(lines) == 1, lines
        assert "Unknown endpoint" in lines[0], lines[0]
