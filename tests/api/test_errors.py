"""Tests for error handling and parameter validation."""

import pytest


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

    def test_n_greater_than_1(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 2,
        })
        assert r.status_code == 400
        assert "n" in r.json()["error"]["message"].lower()

    def test_n_equals_1_valid(self, client, model):
        r = client.post("/v1/chat/completions", json={
            "model": model,
            "messages": [{"role": "user", "content": "Hi"}],
            "n": 1,
            "max_tokens": 1,
            "temperature": 0,
        })
        assert r.status_code == 200


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


class TestUnknownModel:
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
