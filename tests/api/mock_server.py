"""
Mock imp server for GPU-free CI testing.

Implements the OpenAI-compatible API surface of imp-server without any model
loading, CUDA, or GPU requirements. Returns deterministic pseudo-random tokens
at configurable latency.

What it tests:   HTTP contract, SSE streaming format, JSON schema, error codes,
                 concurrency, lifecycle.
What it does NOT test: Model correctness, numerical precision, KV cache, CUDA kernels.
External state:  None (standalone process).

Usage:
    python mock_server.py [--port 9090] [--latency-ms 10] [--fail-rate 0.0]

    --port          Listen port (default: 9090)
    --latency-ms    Per-token delay in ms (default: 5)
    --fail-rate     Fraction of requests that return 500 (for resilience testing)
    --oom           Simulate OOM: all inference requests return 503
"""

import argparse
import collections
import hashlib
import json
import math
import os
import random
import signal
import sys
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

# Predictable vocabulary for mock responses
MOCK_VOCAB = [
    "Hello", " world", "!", " The", " quick", " brown", " fox", " jumps",
    " over", " the", " lazy", " dog", ".", " I", " am", " a", " helpful",
    " assistant", ".", " How", " can", " I", " help", " you", " today", "?",
    "\n", " Yes", " No", " Maybe", " 42", " is", " the", " answer",
]

MOCK_MODEL_ID = "mock-model-v1"
MOCK_MAX_SEQ_LEN = 32768  # mirrors the server's context-length probes
FIM_NOT_SUPPORTED = ("fill-in-the-middle is not supported by this model: its tokenizer has no FIM "
                     "prefix/suffix/middle tokens")

_server_instance = None
_shutdown_event = threading.Event()

# Track active connections for graceful shutdown
_active_requests = threading.Semaphore(1000)
_active_count = 0
_active_lock = threading.Lock()


class MockMetrics:
    def __init__(self):
        self.requests_total = 0
        self.requests_failed = 0
        self.tokens_prompt_total = 0
        self.tokens_completion_total = 0
        self.lock = threading.Lock()
        self.start_time = time.monotonic()

    def inc_request(self):
        with self.lock:
            self.requests_total += 1

    def inc_failed(self):
        with self.lock:
            self.requests_failed += 1

    def add_tokens(self, prompt: int, completion: int):
        with self.lock:
            self.tokens_prompt_total += prompt
            self.tokens_completion_total += completion


metrics = MockMetrics()


class MockConfig:
    """Per-server configuration (avoids class variable pollution across instances)."""
    def __init__(self, latency_ms=5, fail_rate=0.0, oom=False, idle_unload_seconds=0, fim=True,
                 responses_store_ttl=3600.0, responses_store_max_entries=1000,
                 responses_store_max_bytes=256 << 20, swap_models=None):
        self.latency_ms = latency_ms
        # Loaded model has FIM tokens (#2201); False = /infill and `suffix` answer 400 fim_not_supported.
        self.fim = fim
        self.fail_rate = fail_rate
        self.oom_mode = oom
        # --idle-unload-seconds (#2199): reported on /health; the mock never suspends.
        self.idle_unload_seconds = idle_unload_seconds
        # /admin/lora/{load,unload} (#2199): name -> {"id", "path"}; ids never reused.
        self.loras = {}
        self.next_lora_id = 1
        self.lora_lock = threading.Lock()
        # --swap-model NAME (repeatable): extra resolvable models, the mock's --models-dir.
        self.current_model = MOCK_MODEL_ID
        self.swap_models = set(swap_models or ())
        # Responses store (#2206), same limits as --responses-store-*; TTL may be fractional here.
        self.rs_ttl = responses_store_ttl
        self.rs_max_entries = responses_store_max_entries
        self.rs_max_bytes = responses_store_max_bytes
        self.rs_items = collections.OrderedDict()  # id -> (expires, bytes, entry); end = most recent
        self.rs_bytes = 0
        self.rs_evictions = 0
        self.rs_expired = 0
        self.rs_lock = threading.Lock()


class MockHandler(BaseHTTPRequestHandler):
    # Default config, overridden per-server via make_handler_class()
    config = MockConfig()

    def log_message(self, format, *args):
        # Suppress default logging for cleaner test output
        pass

    def end_headers(self):
        # Mirror the real server's trace propagation (post-routing echo):
        # a client-sent X-Request-Id comes back on every response, sanitized
        # to printable ASCII and capped at 128 chars + "..." marker.
        rid = self.headers.get("X-Request-Id")
        if rid:
            out = "".join(c if 0x20 <= ord(c) < 0x7F else "." for c in rid[:128])
            if len(rid) > 128:
                out += "..."
            self.send_header("X-Request-Id", out)
        super().end_headers()

    def _send_json(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str, error_type: str = "invalid_request_error",
                    param: str | None = None, code: str | None = None):
        # Anthropic clients read `type` at the top level; imp-server switches
        # envelope by path (main.cpp pre-routing + error handler), so the mock
        # has to as well or the two disagree on every /v1/messages error.
        if urlparse(self.path).path.startswith("/v1/messages"):
            if error_type == "invalid_request_error" and status == 404:
                error_type = "not_found_error"
            self._send_json(status, {"type": "error",
                                     "error": {"type": error_type, "message": message}})
            return
        err = {"message": message, "type": error_type}
        if param is not None:
            err["param"] = param
        if code is not None:
            err["code"] = code
        self._send_json(status, {"error": err})

    def _check_model(self, model: str) -> bool:
        with self.config.lora_lock:
            if model == self.config.current_model:
                return True
            # handlers.cpp ensure_model_loaded: a swap drops every LoRA adapter (#2217).
            if model == MOCK_MODEL_ID or model in self.config.swap_models:
                self.config.current_model = model
                self.config.loras.clear()
                return True
        self._send_error(404, f"Model '{model}' not found. Loaded: {self.config.current_model}")
        return False

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.end_headers()

    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/health":
            self._send_json(200, {
                "status": "ok",
                "model_loaded": True,
                "queue_depth": 0,
                "suspended": False,
                "idle_suspended": False,
                "idle_unload_seconds": self.config.idle_unload_seconds,
            })
        elif path == "/ready":
            # The mock always has its model: readiness is 200. The 503 branch
            # (no_model / suspended / swapping / draining) is the real
            # binary's, exercised by the model-less Real API lane.
            self._send_json(200, {"ready": True, "model_loaded": True, "suspended": False})
        elif path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": MOCK_MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "imp",
                    "max_model_len": MOCK_MAX_SEQ_LEN,        # vLLM convention
                    "meta": {"n_ctx_train": MOCK_MAX_SEQ_LEN,   # llama.cpp convention
                             # the loaded template's reasoning_effort list (imp; absent without one)
                             "reasoning_effort": {"values": ["xhigh", "medium", "low"], "default": "xhigh"}},
                }],
            })
        elif path == "/props":
            self._send_json(200, {
                "model_path": MOCK_MODEL_ID,
                "total_slots": 64,
                "n_ctx": MOCK_MAX_SEQ_LEN,
                "default_generation_settings": {"n_ctx": MOCK_MAX_SEQ_LEN},
            })
        elif path == "/info":
            self._send_json(200, {
                "model_id": MOCK_MODEL_ID,
                "max_total_tokens": MOCK_MAX_SEQ_LEN,
                "max_input_tokens": MOCK_MAX_SEQ_LEN - 1,
            })
        elif path.startswith("/v1/responses/") and "/" not in path[len("/v1/responses/"):]:
            self._handle_responses_get(path[len("/v1/responses/"):])
        elif path == "/metrics":
            uptime = time.monotonic() - metrics.start_time
            with self.config.rs_lock:
                self._rs_purge_locked()
                rs = (len(self.config.rs_items), self.config.rs_bytes,
                      self.config.rs_evictions, self.config.rs_expired)
            body = (
                f"# HELP imp_uptime_seconds Server uptime\n"
                f"# TYPE imp_uptime_seconds gauge\n"
                f"imp_uptime_seconds {uptime:.1f}\n"
                f"# HELP imp_requests_total Total requests\n"
                f"# TYPE imp_requests_total counter\n"
                f"imp_requests_total {metrics.requests_total}\n"
                f"# HELP imp_requests_failed_total Failed requests\n"
                f"# TYPE imp_requests_failed_total counter\n"
                f"imp_requests_failed_total {metrics.requests_failed}\n"
                f"# HELP imp_tokens_prompt_total Total prompt tokens\n"
                f"# TYPE imp_tokens_prompt_total counter\n"
                f"imp_tokens_prompt_total {metrics.tokens_prompt_total}\n"
                f"# HELP imp_tokens_completion_total Total completion tokens\n"
                f"# TYPE imp_tokens_completion_total counter\n"
                f"imp_tokens_completion_total {metrics.tokens_completion_total}\n"
                f"# HELP imp_model_loaded Model loaded\n"
                f"# TYPE imp_model_loaded gauge\n"
                f'imp_model_loaded{{model="mock"}} 1\n'
                + "".join(
                    f'imp_endpoint_requests_total{{endpoint="{ep}"}} 0\n'
                    f'imp_endpoint_ttft_seconds_bucket{{endpoint="{ep}",le="+Inf"}} 0\n'
                    for ep in ("chat_completions", "completions", "messages", "responses", "embeddings", "rerank")
                )
                + "".join(
                    f"# HELP imp_spec_{m} Speculative decoding, all sources\n"
                    f"# TYPE imp_spec_{m} counter\n"
                    f"imp_spec_{m} 0\n"
                    for m in ("drafted_total", "accepted_total", "verify_steps_total",
                              "miss_steps_total")
                )
                + "".join(
                    f"# HELP imp_spec_{src}_{m} Speculative decoding by draft source\n"
                    f"# TYPE imp_spec_{src}_{m} counter\n"
                    f"imp_spec_{src}_{m} 0\n"
                    for src in ("mtp", "ngram")
                    for m in ("verify_steps_total", "drafted_total", "accepted_total",
                              "emitted_total", "verify_wall_ms_total")
                )
                + f"# HELP imp_queue_depth Queue depth\n"
                f"# HELP imp_queue_depth Queue depth\n"
                f"# TYPE imp_queue_depth gauge\n"
                f"imp_queue_depth 0\n"
                f"# HELP imp_decode_batch_last_rows Sequences in the most recent decode step\n"
                f"# TYPE imp_decode_batch_last_rows gauge\n"
                f"imp_decode_batch_last_rows 0\n"
                f"# HELP imp_streaming_kv_auto_enables_total StreamingLLM auto-enable events\n"
                f"# TYPE imp_streaming_kv_auto_enables_total counter\n"
                f"imp_streaming_kv_auto_enables_total 0\n"
                f"# HELP imp_prefix_cache_evictions_total Cached prefix blocks reclaimed\n"
                f"# TYPE imp_prefix_cache_evictions_total counter\n"
                f"imp_prefix_cache_evictions_total 0\n"
                f"# TYPE imp_responses_store_entries gauge\n"
                f"imp_responses_store_entries {rs[0]}\n"
                f"# TYPE imp_responses_store_bytes gauge\n"
                f"imp_responses_store_bytes {rs[1]}\n"
                f"# TYPE imp_responses_store_evictions_total counter\n"
                f"imp_responses_store_evictions_total {rs[2]}\n"
                f"# TYPE imp_responses_store_expired_total counter\n"
                f"imp_responses_store_expired_total {rs[3]}\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            data = body.encode()
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._send_error(404, f"Unknown endpoint: {path}")

    def do_POST(self):
        path = urlparse(self.path).path

        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length) if content_length > 0 else b""

        if path == "/v1/chat/completions":
            self._handle_chat_completions(raw_body)
        elif path == "/v1/completions":
            self._handle_completions(raw_body)
        elif path == "/v1/decide":
            self._handle_decide(raw_body)
        elif path == "/v1/score":
            self._handle_score(raw_body)
        elif path == "/infill":
            self._handle_infill(raw_body)
        elif path == "/v1/responses":
            self._handle_responses(raw_body)
        elif path == "/tokenize":
            self._handle_tokenize(raw_body)
        elif path == "/detokenize":
            self._handle_detokenize(raw_body)
        elif path == "/admin/lora/load":
            self._handle_lora_load(raw_body)
        elif path == "/admin/lora/unload":
            self._handle_lora_unload(raw_body)
        else:
            self._send_error(404, f"Unknown endpoint: {path}")

    def _parse_json_body(self, raw: bytes) -> dict | None:
        try:
            body = json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return None
        # Non-dict JSON body must return 400 (nlohmann type_error.306), not a dropped connection.
        # Match status only, not wording.
        if not isinstance(body, dict):
            self._send_error(400, "request body must be a JSON object")
            return None
        return body

    def _validate_sampling(self, body: dict) -> bool:
        """Validate sampling parameters. Returns False and sends error if invalid."""
        if "messages" in body and body["messages"] is not None and not isinstance(body["messages"], list):
            self._send_error(400, '"messages" must be an array')
            return False
        if "temperature" in body:
            t = body["temperature"]
            if not isinstance(t, (int, float)) or t < 0 or t > 2:
                self._send_error(400, '"temperature" must be between 0 and 2')
                return False
        if "top_p" in body:
            p = body["top_p"]
            if not isinstance(p, (int, float)) or p < 0 or p > 1:
                self._send_error(400, '"top_p" must be between 0 and 1')
                return False
        if "max_tokens" in body and body["max_tokens"] is not None:
            mt = body["max_tokens"]
            if not isinstance(mt, int) or mt < 1:
                self._send_error(400, '"max_tokens" must be at least 1')
                return False
        if "n" in body:
            n = body["n"]
            # /v1/chat/completions accepts n in [1,4] (independent generations); /v1/completions still
            # rejects n>1 in its own handler. See handlers.cpp validate_sampling_params,
            # handlers_chat_core.cpp.
            if not isinstance(n, int) or n < 1 or n > 4:
                self._send_error(400, '"n" must be between 1 and 4.')
                return False
        if not self._validate_speculative(body):
            return False
        return True

    def _validate_speculative(self, body: dict) -> bool:
        """imp extension "speculative": true|false|{"mtp_k": N}.

        Mirrors parse_spec_field_ in tools/imp-server/handlers_internal.h. The
        mock arms no MTP head, so the bound is the device chain cap
        (kSpecRequestMaxMtpK = 16) - which is also what a model-less imp-server
        answers, so both lanes assert the same message.
        """
        if "speculative" not in body:
            return True
        sp = body["speculative"]
        if isinstance(sp, bool):
            return True
        if not isinstance(sp, dict):
            self._send_error(
                400, '"speculative" must be a boolean or an object of the form {"mtp_k": N}')
            return False
        if "mtp_k" not in sp:
            return True  # an empty object asks for nothing
        k = sp["mtp_k"]
        if not isinstance(k, int) or isinstance(k, bool):
            self._send_error(400, '"speculative.mtp_k" must be an integer in 0..16')
            return False
        if k < 0 or k > 16:
            self._send_error(
                400,
                '"speculative.mtp_k" is %d, outside the accepted range 0..16 '
                "(no MTP head is armed; the bound is the device chain cap)" % k,
            )
            return False
        return True

    def _generate_tokens(self, seed: int, max_tokens: int) -> list[str]:
        """Generate deterministic pseudo-random token strings."""
        rng = random.Random(seed)
        n = min(max_tokens, 32)  # cap for mock
        tokens = []
        for _ in range(n):
            tokens.append(rng.choice(MOCK_VOCAB))
        return tokens

    def _parse_prompt_logprobs(self, body: dict):
        """Mirror of tools/imp-server/prompt_logprobs.cpp parse_prompt_logprobs (#2207); None = 400 sent."""
        pl = body.get("prompt_logprobs")
        if pl is None:
            pl = -1
        elif isinstance(pl, bool) or not isinstance(pl, int) or not 0 <= pl <= 20:
            self._send_error(400, f'"prompt_logprobs" must be an integer in [0, 20], got {pl!r}')
            return None
        lp = body.get("logprobs")
        req_logprobs = (lp is True) or (isinstance(lp, int) and not isinstance(lp, bool) and lp > 0)
        top = body.get("top_logprobs") or 0
        if isinstance(lp, int) and not isinstance(lp, bool):
            top = max(top, lp)
        echo_top = min(max(top, 0), 20) if (body.get("echo") and req_logprobs) else -1
        if body.get("stream") and (pl >= 0 or echo_top >= 0):
            self._send_error(400, 'prompt logprobs ("prompt_logprobs", or "echo" with "logprobs") '
                                  'are not supported with "stream": true')
            return None
        return {"prompt_logprobs": pl, "echo_top": echo_top}

    @staticmethod
    def _mock_prompt_pieces(prompt: str) -> list[str]:
        """1 token per 4 chars, the last piece takes the remainder (matches prompt_tokens)."""
        n = max(1, len(prompt) // 4)
        return [prompt[i * 4:(i + 1) * 4] if i < n - 1 else prompt[i * 4:] for i in range(n)]

    @staticmethod
    def _mock_prompt_logprobs(pieces: list[str], top: int) -> list:
        out = [None]
        for i, piece in enumerate(pieces[1:], start=1):
            entry = {str(100 + i): {"logprob": -1.0, "rank": top + 1, "decoded_token": piece}}
            for k in range(top):
                entry[str(900 + k)] = {"logprob": -0.1 * (k + 1), "rank": k + 1, "decoded_token": f"alt{k}"}
            out.append(entry)
        return out

    @staticmethod
    def _mock_echo_logprobs(pieces: list[str], completion: list[str], top: int, text: str) -> dict:
        tokens, lps, tops, offsets, cursor = [], [], [], [], 0
        for i, piece in enumerate(pieces + completion):
            tokens.append(piece)
            offsets.append(cursor)
            if i == 0:
                lps.append(None)
                tops.append(None)
            else:
                lps.append(-1.0)
                tops.append({f"alt{k}": -0.1 * (k + 1) for k in range(top)})
            if piece and text.startswith(piece, cursor):
                cursor += len(piece)
        return {"tokens": tokens, "token_logprobs": lps, "top_logprobs": tops, "text_offset": offsets}

    def _send_coded_error(self, status: int, message: str, param: str, code: str | None = None):
        err = {"message": message, "type": "invalid_request_error", "param": param}
        if code:
            err["code"] = code
        self._send_json(status, {"error": err})

    # Mirrors tools/imp-server/handlers_admin.cpp handle_lora_load. The mock has no
    # model to check shapes against, so "loadable" means the path exists.
    def _handle_lora_load(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        path = body.get("path")
        if not isinstance(path, str) or not path:
            self._send_coded_error(400, "'path' (non-empty string) is required", "path")
            return
        if "name" in body:
            name = body["name"]
            if not isinstance(name, str) or not name:
                self._send_coded_error(400, "'name' must be a non-empty string", "name")
                return
        else:
            name = os.path.splitext(os.path.basename(path.rstrip("/")))[0]
        with self.config.lora_lock:
            if name in self.config.loras:
                self._send_coded_error(
                    409, f"LoRA adapter '{name}' is already loaded (id {self.config.loras[name]['id']}); "
                    "unload it first or pass another 'name'", "name", "lora_already_loaded")
                return
            if not os.path.exists(path):
                self._send_coded_error(400, f"LoRA adapter load failed for '{path}'", "path",
                                       "lora_load_failed")
                return
            lora_id = self.config.next_lora_id
            self.config.next_lora_id += 1
            self.config.loras[name] = {"id": lora_id, "path": path}
        self._send_json(200, {"id": lora_id, "name": name, "path": path, "loaded": True})

    def _handle_lora_unload(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        by_id = "id" in body
        if by_id and (not isinstance(body["id"], int) or isinstance(body["id"], bool)):
            self._send_coded_error(400, "'id' must be an integer", "id")
            return
        if not by_id and not isinstance(body.get("name"), str):
            self._send_coded_error(400, "'id' (integer) or 'name' (string) is required", "id")
            return
        with self.config.lora_lock:
            if by_id:
                name = next((n for n, e in self.config.loras.items() if e["id"] == body["id"]), None)
                what = f"id {body['id']}"
            else:
                name = body["name"] if body["name"] in self.config.loras else None
                what = f"'{body['name']}'"
            if name is None:
                self._send_coded_error(404, f"LoRA adapter {what} is not loaded",
                                       "id" if by_id else "name", "lora_not_found")
                return
            entry = self.config.loras.pop(name)
        self._send_json(200, {"id": entry["id"], "name": name, "unloaded": True})

    def _handle_chat_completions(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        if not self._validate_sampling(body):
            return
        messages = body.get("messages", [])
        if not messages:
            self._send_error(400, "messages array is required and must not be empty")
            return
        # The server's per-request caps at their defaults (handlers_chat_params.cpp:
        # kMaxMessages, --max-logit-bias, --max-images-per-request). The nomodel
        # lane asserts them on both ends (AUDIT_arch_2026 F2-5).
        if len(messages) > 10000:
            self._send_error(400, "messages array exceeds maximum of 10000 entries")
            return
        logit_bias = body.get("logit_bias")
        if isinstance(logit_bias, dict) and len(logit_bias) > 1024:
            self._send_error(400, "logit_bias has too many entries, above the server limit of 1024 (--max-logit-bias)")
            return
        # tools/imp-server/logit_bias.cpp: a malformed entry is a 400, not a dropped bias.
        if logit_bias is not None and (
            not isinstance(logit_bias, dict)
            or any(not k.isdigit() or isinstance(v, bool) or not isinstance(v, (int, float))
                   or not -100 <= v <= 100 for k, v in logit_bias.items())
        ):
            self._send_error(400, "logit_bias must map token ids to numbers in [-100, 100]")
            return
        n_images = sum(
            1
            for m in messages
            if isinstance(m, dict) and isinstance(m.get("content"), list)
            for p in m["content"]
            if isinstance(p, dict) and p.get("type") == "image_url"
        )
        if n_images > 8:
            self._send_error(400, "request carries more than 8 images, the server limit (--max-images-per-request)")
            return

        model = body.get("model", "")
        if not model:
            self._send_error(400, '"model" is required')
            return
        if not self._check_model(model):
            return
        # After the model check, as in handlers_chat_core.cpp: a swap drops the adapter table first.
        lora = body.get("lora")
        if lora:
            with self.config.lora_lock:
                known = lora in self.config.loras
            if not known:
                self._send_coded_error(
                    400, f"LoRA adapter '{lora}' is not loaded (POST /admin/lora/load, or --lora "
                    "NAME=PATH at startup)", "lora", "lora_not_loaded")
                return

        # Simulate OOM
        if self.config.oom_mode:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "5")
            err = json.dumps({"error": {"message": "Out of memory", "type": "server_error"}}).encode()
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        # Simulate random failures
        if self.config.fail_rate > 0 and random.random() < self.config.fail_rate:
            metrics.inc_failed()
            self._send_error(500, "Simulated failure", "server_error")
            return

        metrics.inc_request()

        # `max_tokens: null` is "unset" in every OpenAI SDK. Passing None
        # through reached min(None, 32) and killed the connection.
        max_tokens = body.get("max_tokens") or 16
        seed = body.get("seed", 42)
        stream = body.get("stream", False)
        include_usage = False
        if "stream_options" in body and isinstance(body["stream_options"], dict):
            include_usage = body["stream_options"].get("include_usage", False)

        # Count prompt tokens (rough: 1 token per 4 chars)
        prompt_text = " ".join(m.get("content", "") or "" for m in messages if isinstance(m.get("content"), str))
        prompt_tokens = max(1, len(prompt_text) // 4)

        tokens = self._generate_tokens(seed, max_tokens)
        completion_tokens = len(tokens)
        # Reasoning request must produce a reasoning channel, reasoning_tokens in usage, and
        # imp_finish_detail when budget exhausts before an answer (handlers_chat_core.cpp).
        # Tiny max_tokens is the real exhaustion case: content empty, reasoning non-empty.
        reasoning_tokens: list[str] = []
        if body.get("reasoning_effort"):
            n_reason = len(tokens) if max_tokens <= 8 else len(tokens) // 2
            reasoning_tokens, tokens = tokens[:n_reason], tokens[n_reason:]
        reasoning = "".join(reasoning_tokens)
        content = "".join(tokens)

        metrics.add_tokens(prompt_tokens, completion_tokens)

        req_id = f"mock-{int(time.time())}-{random.randint(0, 9999)}"
        created = int(time.time())

        if stream:
            self._stream_chat_response(req_id, created, model, tokens,
                                       prompt_tokens, include_usage, reasoning_tokens)
        else:
            # n independent generations, like handlers_chat_core.cpp. The mock
            # repeats the same text; what the suite checks is the choice count
            # and the index numbering.
            n_choices = body.get("n", 1)

            def choice(i):
                msg = {"role": "assistant", "content": content}
                if reasoning:
                    msg["reasoning_content"] = reasoning
                c = {
                    "index": i,
                    "message": msg,
                    "finish_reason": "stop" if completion_tokens < max_tokens else "length",
                }
                if reasoning and not content:
                    c["imp_finish_detail"] = "reasoning_budget_exhausted"
                return c

            usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens * n_choices,
                "total_tokens": prompt_tokens + completion_tokens * n_choices,
                # Mirrors the server: cached_tokens is present on a miss (0), #1980.
                "prompt_tokens_details": {"cached_tokens": 0},
            }
            if reasoning_tokens:
                usage["completion_tokens_details"] = {
                    "reasoning_tokens": len(reasoning_tokens) * n_choices
                }
            self._send_json(200, {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [choice(i) for i in range(n_choices)],
                "usage": usage,
            })

    def _stream_chat_response(self, req_id: str, created: int, model: str,
                              tokens: list[str], prompt_tokens: int,
                              include_usage: bool, reasoning_tokens: list[str] | None = None):
        reasoning_tokens = reasoning_tokens or []
        completion_tokens = len(tokens) + len(reasoning_tokens)

        # Build full SSE body first so we can set Content-Length.
        # This makes httpx's non-streaming .post() work correctly.
        # For real streaming tests, use httpx.stream() which reads progressively.
        parts: list[str] = []

        # First chunk: role
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {"role": "assistant", "content": ""},
                "finish_reason": None,
            }],
        }
        parts.append(f"data: {json.dumps(chunk)}\n\n")

        # Reasoning chunks (delta.reasoning_content, never delta.content)
        for token in reasoning_tokens:
            parts.append("data: " + json.dumps({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"reasoning_content": token},
                    "finish_reason": None,
                }],
            }) + "\n\n")

        # Content chunks
        for i, token in enumerate(tokens):
            is_last = (i == len(tokens) - 1)
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": "stop" if is_last else None,
                }],
            }
            parts.append(f"data: {json.dumps(chunk)}\n\n")

        # The answer never started: the server's final chunk carries the
        # finish_reason and the exhaustion detail with an empty delta
        # (handlers_chat_stream.cpp).
        if reasoning_tokens and not tokens:
            parts.append("data: " + json.dumps({
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "length",
                    "imp_finish_detail": "reasoning_budget_exhausted",
                }],
            }) + "\n\n")

        # Usage chunk (if requested)
        if include_usage:
            usage_chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
            if reasoning_tokens:
                usage_chunk["usage"]["completion_tokens_details"] = {
                    "reasoning_tokens": len(reasoning_tokens)
                }
            parts.append(f"data: {json.dumps(usage_chunk)}\n\n")

        # DONE sentinel
        parts.append("data: [DONE]\n\n")

        body = "".join(parts).encode()

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()

        try:
            self.wfile.write(body)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _handle_completions(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        if not self._validate_sampling(body):
            return

        # Multi-choice generation is chat-only in imp-server; /v1/completions
        # refuses n>1 rather than silently returning one choice.
        if body.get("n", 1) > 1:
            self._send_error(
                400, "n>1 is not supported on /v1/completions; request one completion per call")
            return

        plp = self._parse_prompt_logprobs(body)
        if plp is None:
            return

        model = body.get("model", "")
        if not model:
            self._send_error(400, '"model" is required')
            return
        if not self._check_model(model):
            return

        prompt = body.get("prompt", "")
        if not prompt:
            self._send_error(400, '"prompt" is required')
            return

        # `suffix` = fill-in-the-middle (#2201), same checks as fim_request.cpp.
        suffix = body.get("suffix")
        if suffix is not None and not isinstance(suffix, str):
            self._send_error(400, '"suffix" must be a string', param="suffix")
            return
        if suffix:
            if isinstance(prompt, list) and not (len(prompt) == 1 and isinstance(prompt[0], str)):
                self._send_error(400, '"suffix" needs a text prompt', param="suffix")
                return
            if not self.config.fim:
                self._send_error(400, FIM_NOT_SUPPORTED, param="suffix", code="fim_not_supported")
                return

        if self.config.oom_mode:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "5")
            err = json.dumps({"error": {"message": "Out of memory", "type": "server_error"}}).encode()
            self.send_header("Content-Length", str(len(err)))
            self.end_headers()
            self.wfile.write(err)
            return

        metrics.inc_request()

        # `max_tokens: null` is "unset" in every OpenAI SDK. Passing None
        # through reached min(None, 32) and killed the connection.
        max_tokens = body.get("max_tokens") or 16
        seed = body.get("seed", 42)
        tokens = self._generate_tokens(seed, max_tokens)
        content = "".join(tokens)
        prompt_tokens = max(1, len(prompt) // 4)
        choice = {
            "index": 0,
            "text": (prompt + content) if body.get("echo") else content,
            "finish_reason": "stop" if len(tokens) < max_tokens else "length",
        }
        pieces = self._mock_prompt_pieces(prompt) if isinstance(prompt, str) else ["tok"] * prompt_tokens
        pieces = pieces[:prompt_tokens]
        if plp["prompt_logprobs"] >= 0:
            choice["prompt_logprobs"] = self._mock_prompt_logprobs(pieces, plp["prompt_logprobs"])
        if plp["echo_top"] >= 0:
            choice["logprobs"] = self._mock_echo_logprobs(pieces, tokens, plp["echo_top"], choice["text"])

        self._send_json(200, {
            "id": f"mock-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [choice],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        })

    def _handle_infill(self, raw: bytes):
        """POST /infill (llama.cpp shape, #2201): OpenAI text_completion + top-level content/stop."""
        body = self._parse_json_body(raw)
        if body is None:
            return
        if not self._validate_sampling(body):
            return
        for key in ("input_prefix", "input_suffix", "prompt"):
            if key in body and body[key] is not None and not isinstance(body[key], str):
                self._send_error(400, f'"{key}" must be a string')
                return
        extra = body.get("input_extra")
        if extra is not None:
            ok = isinstance(extra, list) and all(
                isinstance(c, dict) and isinstance(c.get("text"), str)
                and isinstance(c.get("filename", ""), (str, type(None))) for c in extra)
            if not ok:
                self._send_error(400, "each \"input_extra\" entry must be {\"filename\": string, \"text\": string}")
                return
        model = body.get("model") or MOCK_MODEL_ID  # llama.cpp clients send no model
        if not self._check_model(model):
            return
        if not self.config.fim:
            self._send_error(400, FIM_NOT_SUPPORTED, code="fim_not_supported")
            return

        metrics.inc_request()
        n_predict = body.get("n_predict")
        max_tokens = body.get("max_tokens") or (n_predict if isinstance(n_predict, int) and n_predict > 0 else 16)
        tokens = self._generate_tokens(body.get("seed", 42), max_tokens)
        finish = "stop" if len(tokens) < max_tokens else "length"
        req_id = f"mock-{int(time.time())}"
        created = int(time.time())

        def frame(text, finish_reason):
            return {"id": req_id, "object": "text_completion", "created": created, "model": model,
                    "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish_reason}],
                    "content": text, "stop": finish_reason is not None}

        if not body.get("stream", False):
            resp = frame("".join(tokens), finish)
            resp["usage"] = {"prompt_tokens": 8, "completion_tokens": len(tokens),
                             "total_tokens": 8 + len(tokens)}
            self._send_json(200, resp)
            return

        parts = [f"data: {json.dumps(frame(t, None))}\n\n" for t in tokens]
        parts.append(f"data: {json.dumps(frame('', finish))}\n\n")
        parts.append("data: [DONE]\n\n")
        out = "".join(parts).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        try:
            self.wfile.write(out)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- /v1/responses with store (#2206), mirrors handlers_responses.cpp ----
    # Output is a digest of the flattened conversation, so equal replies mean equal transcripts.

    def _rs_error(self, status: int, message: str, param: str | None, code: str | None = None):
        err = {"message": message, "type": "invalid_request_error"}
        if param:
            err["param"] = param
        if code:
            err["code"] = code
        self._send_json(status, {"error": err})

    def _rs_purge_locked(self):
        now = time.monotonic()
        cfg = self.config
        for rid in [k for k, v in cfg.rs_items.items() if v[0] <= now]:
            cfg.rs_bytes -= cfg.rs_items.pop(rid)[1]
            cfg.rs_expired += 1

    def _rs_get(self, rid: str):
        with self.config.rs_lock:
            self._rs_purge_locked()
            slot = self.config.rs_items.get(rid)
            if slot is None:
                return None
            self.config.rs_items.move_to_end(rid)
            return slot[2]

    def _rs_put(self, rid: str, entry: dict):
        cfg = self.config
        size = len(rid) + len(json.dumps(entry))
        with cfg.rs_lock:
            if size > cfg.rs_max_bytes:
                return
            self._rs_purge_locked()
            cfg.rs_items[rid] = (time.monotonic() + cfg.rs_ttl, size, entry)
            cfg.rs_bytes += size
            while len(cfg.rs_items) > cfg.rs_max_entries or cfg.rs_bytes > cfg.rs_max_bytes:
                cfg.rs_bytes -= cfg.rs_items.popitem(last=False)[1][1]
                cfg.rs_evictions += 1

    @staticmethod
    def _rs_flatten(items: list) -> list:
        """Chat messages the transform (responses.cpp) builds; reasoning items are skipped."""
        out = []
        for it in items:
            t = it.get("type", "message" if "role" in it else "")
            if t == "message":
                c = it.get("content", "")
                if isinstance(c, list):
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                out.append([it.get("role", "user"), c])
            elif t == "function_call":
                out.append(["assistant_call", it.get("name", ""), it.get("arguments", "{}")])
            elif t == "function_call_output":
                out.append(["tool", it.get("call_id", ""), str(it.get("output", ""))])
        return out

    def _handle_responses(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        store = body.get("store")
        if store is not None and not isinstance(store, bool):
            self._rs_error(400, '"store" must be a boolean', "store")
            return
        cfg = self.config
        enabled = cfg.rs_ttl > 0 and cfg.rs_max_entries > 0 and cfg.rs_max_bytes > 0
        # Absent means true (OpenAI default); on a disabled store it means stateless.
        store = enabled if store is None else store
        if store and not enabled:
            self._rs_error(400, "store=true is disabled on this server", "store")
            return
        new_input = body.get("input")
        items = [{"role": "user", "content": new_input}] if isinstance(new_input, str) else list(new_input or [])
        prev_id = body.get("previous_response_id")
        if prev_id is not None:
            if not isinstance(prev_id, str) or not prev_id:
                self._rs_error(400, '"previous_response_id" must be a non-empty string', "previous_response_id")
                return
            prev = self._rs_get(prev_id)
            if prev is None:
                self._rs_error(404, f"Previous response with id '{prev_id[:128]}' not found.",
                               "previous_response_id", "response_not_found")
                return
            items = prev["input_items"] + prev["output_items"] + items
        model = body.get("model", MOCK_MODEL_ID)
        if not self._check_model(model):
            return
        if body.get("stream"):
            self._rs_error(400, "the mock does not stream /v1/responses", "stream")
            return

        metrics.inc_request()
        convo = self._rs_flatten(items)
        if body.get("instructions"):
            convo.insert(0, ["system", body["instructions"]])
        digest = hashlib.sha256(json.dumps(convo).encode()).hexdigest()[:16]
        text = f"mock reply {digest} after {len(convo)} messages"
        in_tok = max(1, len(json.dumps(convo)) // 4)
        out_tok = len(text.split())
        metrics.add_tokens(in_tok, out_tok)
        rid = f"resp_mock{random.getrandbits(64):016x}"
        output = [{"type": "message", "id": "msg_mock0", "status": "completed", "role": "assistant",
                   "content": [{"type": "output_text", "text": text, "annotations": []}]}]
        response = {
            "id": rid, "object": "response", "created_at": int(time.time()), "model": model,
            "status": "completed", "error": None, "incomplete_details": None, "output": output,
            "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
            "store": store, "previous_response_id": prev_id,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
                      "input_tokens_details": {"cached_tokens": 0},
                      "output_tokens_details": {"reasoning_tokens": 0}},
        }
        self._send_json(200, response)
        if store:
            self._rs_put(rid, {"input_items": items, "output_items": output, "response": response})

    def _handle_responses_get(self, rid: str):
        entry = self._rs_get(rid)
        if entry is None:
            self._rs_error(404, f"Response with id '{rid[:128]}' not found.", None, "response_not_found")
            return
        self._send_json(200, entry["response"])

    def do_DELETE(self):
        path = urlparse(self.path).path
        if not path.startswith("/v1/responses/"):
            self._send_error(404, f"Unknown endpoint: {path}")
            return
        rid = path[len("/v1/responses/"):]
        with self.config.rs_lock:
            self._rs_purge_locked()
            slot = self.config.rs_items.pop(rid, None)
            if slot is not None:
                self.config.rs_bytes -= slot[1]
        if slot is None:
            self._rs_error(404, f"Response with id '{rid[:128]}' not found.", None, "response_not_found")
            return
        self._send_json(200, {"id": rid, "object": "response.deleted", "deleted": True})

    def _handle_tokenize(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        # `content` (llama.cpp) or `prompt` (vLLM), as the server takes them (#1980).
        text = body.get("content") or body.get("prompt") or ""
        if not isinstance(text, str) or not text:
            self._send_error(400, "\"content\" (or its alias \"prompt\") is required")
            return
        # Mock: 1 token per 4 chars
        n_tokens = max(1, len(text) // 4)
        tokens = list(range(100, 100 + n_tokens))
        self._send_json(200, {"tokens": tokens})

    # /v1/decide + /v1/score (#2198). Validation mirrors handlers_decide.cpp. Mock tokenizer:
    # a candidate string is one token iff it is one character after optional leading spaces;
    # a raw prompt ending in ':' or '>' merges with an alphanumeric candidate (Qwen ":A").
    def _score_mode(self, body: dict) -> str | None:
        mode = body.get("mode", "auto")
        if mode not in ("auto", "serial", "direct", "shared"):
            self._send_error(400, '"mode" must be one of auto, serial, direct, shared')
            return None
        return "serial" if mode == "auto" else mode

    @staticmethod
    def _mock_probs(seed: str, n: int) -> list[float]:
        rng = random.Random(seed)
        w = [rng.random() + 0.01 for _ in range(n)]
        s = sum(w)
        return [x / s for x in w]

    def _handle_decide(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        mode = self._score_mode(body)
        if mode is None:
            return
        evidence = body.get("evidence")
        if not isinstance(evidence, str):
            self._send_error(400, '"evidence" (string) is required')
            return
        system = body.get("system", "Answer only with the letter of the correct option.")
        if not isinstance(system, str):
            self._send_error(400, '"system" must be a string')
            return
        items = body.get("items")
        if not isinstance(items, list) or not items:
            self._send_error(400, '"items" (non-empty array) is required')
            return
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                self._send_error(400, f"items[{i}] must be an object")
                return
            if not isinstance(it.get("criterion"), str):
                self._send_error(400, f"items[{i}].criterion (string) is required")
                return
            opts = it.get("options")
            if not isinstance(opts, list) or len(opts) < 2:
                self._send_error(400, f"items[{i}].options must be an array of 2 to 16 strings")
                return
            if len(opts) > 16:
                self._send_error(400, f"items[{i}].options has {len(opts)} entries, the maximum is 16 (letters A to P)")
                return
            if not all(isinstance(o, str) for o in opts):
                self._send_error(400, f"items[{i}].options must contain only strings")
                return
        model = body.get("model", MOCK_MODEL_ID)
        if not self._check_model(model):
            return
        evidence_tokens = max(1, len(system + evidence) // 4)
        out, total_prompt, total_cached = [], 0, 0
        for i, it in enumerate(items):
            letters = [chr(ord("A") + k) for k in range(len(it["options"]))]
            probs = self._mock_probs(evidence + it["criterion"], len(letters))
            best = max(range(len(probs)), key=probs.__getitem__)
            prompt_tokens = evidence_tokens + max(1, len(it["criterion"] + "".join(it["options"])) // 4) + 8
            cached = evidence_tokens if (mode in ("serial", "shared") and i > 0) else 0
            total_prompt += prompt_tokens
            total_cached += cached
            out.append({"id": it.get("id", i), "probs": dict(zip(letters, probs)), "argmax": letters[best],
                        "argmax_index": best, "prompt_tokens": prompt_tokens, "cached_tokens": cached})
        self._send_json(200, {"object": "decide", "model": model, "mode_used": mode, "items": out,
                              "usage": {"prompt_tokens": total_prompt, "cached_tokens": total_cached,
                                        "total_tokens": total_prompt}})

    def _handle_score(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        mode = self._score_mode(body)
        if mode is None:
            return
        has_prompt, has_messages = "prompt" in body, "messages" in body
        if has_prompt == has_messages:
            self._send_error(400, 'exactly one of "prompt" (string) or "messages" (array) is required')
            return
        if has_prompt and (not isinstance(body["prompt"], str) or not body["prompt"]):
            self._send_error(400, '"prompt" must be a non-empty string')
            return
        if has_messages:
            msgs = body["messages"]
            if not isinstance(msgs, list) or not msgs or not all(
                    isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str)
                    for m in msgs):
                self._send_error(400, 'each message must be {"role": string, "content": string}')
                return
        cands = body.get("candidates")
        if not isinstance(cands, list) or len(cands) < 2:
            self._send_error(400, '"candidates" must be an array of 2 or more token strings or token ids')
            return
        if len(cands) > 256:
            self._send_error(400, f'"candidates" has {len(cands)} entries, the maximum is 256')
            return
        for c in cands:
            if isinstance(c, bool) or not isinstance(c, (str, int)):
                self._send_error(400, "each candidate must be a token string or an integer token id")
                return
            if isinstance(c, str) and not c:
                self._send_error(400, "a candidate string must not be empty")
                return
        model = body.get("model", MOCK_MODEL_ID)
        if not self._check_model(model):
            return
        text = body["prompt"] if has_prompt else "".join(m["content"] for m in body["messages"])
        ids = []
        for c in cands:
            if isinstance(c, str):
                if len(c.lstrip(" ")) != 1:
                    self._send_error(400, f'candidate "{c}" is {len(c)} tokens in this tokenizer, scoring needs exactly one')
                    return
                if has_prompt and text[-1] in ":>" and c[0].isalnum():
                    self._send_error(400, f'candidate "{c}" merges with the end of the prompt: '
                                          'tokenize(prefix + candidate) != tokenize(prefix) + [id]')
                    return
                ids.append(1000 + ord(c[-1]) + (100000 if c[0] == " " else 0))
            else:
                if c < 0 or c >= 151669:
                    self._send_error(400, f"candidate token id {c} is outside the vocabulary [0, 151669)")
                    return
                ids.append(c)
        if len(set(ids)) != len(ids):
            self._send_error(400, "a candidate token id appears twice")
            return
        probs = self._mock_probs(text, len(ids))
        best = max(range(len(probs)), key=probs.__getitem__)
        prompt_tokens = max(1, len(text) // 4)
        self._send_json(200, {
            "object": "score", "model": model, "mode_used": mode,
            "candidates": [{"candidate": c, "token_id": t, "logit": math.log(p), "prob": p}
                           for c, t, p in zip(cands, ids, probs)],
            "argmax_index": best, "prompt_tokens": prompt_tokens, "cached_tokens": 0,
            "usage": {"prompt_tokens": prompt_tokens, "cached_tokens": 0, "total_tokens": prompt_tokens}})

    def _handle_detokenize(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        tokens = body.get("tokens", [])
        # Mock: each token -> "tok"
        text = "tok" * len(tokens)
        self._send_json(200, {"content": text})


class ThreadedHTTPServer(HTTPServer):
    """HTTPServer that handles each request in a new thread."""
    daemon_threads = True
    allow_reuse_address = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self.process_request_thread,
                             args=(request, client_address))
        t.daemon = True
        t.start()

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def make_handler_class(config: MockConfig):
    """Create a handler class with its own config (avoids class variable sharing)."""
    class Handler(MockHandler):
        pass
    Handler.config = config
    return Handler


def run_server(port: int = 9090, latency_ms: int = 5,
               fail_rate: float = 0.0, oom: bool = False, fim: bool = True,
               **store_limits) -> ThreadedHTTPServer:
    """Start the mock server and return the server instance. port=0 picks a free port
    (read it from server.server_address); store_limits are MockConfig responses_store_* kwargs."""
    config = MockConfig(latency_ms=latency_ms, fail_rate=fail_rate, oom=oom, fim=fim, **store_limits)
    handler_class = make_handler_class(config)

    server = ThreadedHTTPServer(("127.0.0.1", port), handler_class)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def main():
    parser = argparse.ArgumentParser(description="Mock imp server for testing")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--latency-ms", type=int, default=5)
    parser.add_argument("--fail-rate", type=float, default=0.0)
    parser.add_argument("--oom", action="store_true")
    parser.add_argument("--idle-unload-seconds", type=str, default="0")
    parser.add_argument("--swap-model", action="append", default=[])
    for flag in ("--responses-store-ttl", "--responses-store-max-entries", "--responses-store-max-mib"):
        parser.add_argument(flag, type=str, default=None)
    args = parser.parse_args()

    # Same contract as tools/imp-server/args.cpp: integer >= 0, else exit 1.
    try:
        idle = int(args.idle_unload_seconds)
    except ValueError:
        idle = -1
    if idle < 0:
        print(f"--idle-unload-seconds expects an integer >= 0, got '{args.idle_unload_seconds}'",
              file=sys.stderr, flush=True)
        sys.exit(1)

    store = {"responses_store_ttl": 3600, "responses_store_max_entries": 1000,
             "responses_store_max_mib": 256}
    for key in store:
        raw = getattr(args, key)
        if raw is None:
            continue
        try:
            store[key] = int(raw)
        except ValueError:
            store[key] = -1
        if store[key] < 0:
            flag = "--" + key.replace("_", "-")
            print(f"{flag} expects an integer >= 0, got '{raw}'", file=sys.stderr, flush=True)
            sys.exit(1)

    config = MockConfig(latency_ms=args.latency_ms, fail_rate=args.fail_rate, oom=args.oom,
                        idle_unload_seconds=idle,
                        responses_store_ttl=store["responses_store_ttl"],
                        responses_store_max_entries=store["responses_store_max_entries"],
                        responses_store_max_bytes=store["responses_store_max_mib"] << 20,
                        swap_models=args.swap_model)
    handler_class = make_handler_class(config)
    server = ThreadedHTTPServer(("127.0.0.1", args.port), handler_class)

    shutdown_event = threading.Event()

    def handle_signal(sig, frame):
        print("\nShutting down mock server...", flush=True)
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    print(f"Mock imp server listening on http://127.0.0.1:{args.port}", flush=True)
    print(f"  Model: {MOCK_MODEL_ID}", flush=True)
    print(f"  Latency: {args.latency_ms}ms/token", flush=True)
    if args.oom:
        print(f"  OOM mode: enabled (all inference returns 503)", flush=True)

    # Run server in a thread so signal handlers can fire on main thread
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()

    shutdown_event.wait()
    server.shutdown()
    sys.exit(0)


if __name__ == "__main__":
    main()
