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
import json
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
    def __init__(self, latency_ms=5, fail_rate=0.0, oom=False):
        self.latency_ms = latency_ms
        self.fail_rate = fail_rate
        self.oom_mode = oom


class MockHandler(BaseHTTPRequestHandler):
    # Default config, overridden per-server via make_handler_class()
    config = MockConfig()

    def log_message(self, format, *args):
        # Suppress default logging for cleaner test output
        pass

    def _send_json(self, status: int, body: dict):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error(self, status: int, message: str, error_type: str = "invalid_request_error"):
        self._send_json(status, {"error": {"message": message, "type": error_type}})

    def _check_model(self, model: str) -> bool:
        if model != MOCK_MODEL_ID:
            self._send_error(404, f"Model '{model}' not found. Loaded: {MOCK_MODEL_ID}")
            return False
        return True

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
            })
        elif path == "/v1/models":
            self._send_json(200, {
                "object": "list",
                "data": [{
                    "id": MOCK_MODEL_ID,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "imp",
                    "max_model_len": MOCK_MAX_SEQ_LEN,        # vLLM convention
                    "meta": {"n_ctx_train": MOCK_MAX_SEQ_LEN},  # llama.cpp convention
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
        elif path == "/metrics":
            uptime = time.monotonic() - metrics.start_time
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
                f"imp_model_loaded 1\n"
                f"# HELP imp_queue_depth Queue depth\n"
                f"# TYPE imp_queue_depth gauge\n"
                f"imp_queue_depth 0\n"
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
        elif path == "/tokenize":
            self._handle_tokenize(raw_body)
        elif path == "/detokenize":
            self._handle_detokenize(raw_body)
        else:
            self._send_error(404, f"Unknown endpoint: {path}")

    def _parse_json_body(self, raw: bytes) -> dict | None:
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError) as e:
            self._send_error(400, f"Invalid JSON: {e}")
            return None

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
            if n != 1:
                self._send_error(400, '"n" must be 1. n > 1 is not supported.')
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

        model = body.get("model", "")
        if not model:
            self._send_error(400, '"model" is required')
            return
        if not self._check_model(model):
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

        max_tokens = body.get("max_tokens", 16)
        seed = body.get("seed", 42)
        stream = body.get("stream", False)
        include_usage = False
        if "stream_options" in body and isinstance(body["stream_options"], dict):
            include_usage = body["stream_options"].get("include_usage", False)

        # Count prompt tokens (rough: 1 token per 4 chars)
        prompt_text = " ".join(m.get("content", "") or "" for m in messages if isinstance(m.get("content"), str))
        prompt_tokens = max(1, len(prompt_text) // 4)

        tokens = self._generate_tokens(seed, max_tokens)
        content = "".join(tokens)
        completion_tokens = len(tokens)

        metrics.add_tokens(prompt_tokens, completion_tokens)

        req_id = f"mock-{int(time.time())}-{random.randint(0, 9999)}"
        created = int(time.time())

        if stream:
            self._stream_chat_response(req_id, created, model, tokens,
                                       prompt_tokens, include_usage)
        else:
            self._send_json(200, {
                "id": req_id,
                "object": "chat.completion",
                "created": created,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": "stop" if completion_tokens < max_tokens else "length",
                }],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            })

    def _stream_chat_response(self, req_id: str, created: int, model: str,
                              tokens: list[str], prompt_tokens: int,
                              include_usage: bool):
        completion_tokens = len(tokens)

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
                },
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

        max_tokens = body.get("max_tokens", 16)
        seed = body.get("seed", 42)
        tokens = self._generate_tokens(seed, max_tokens)
        content = "".join(tokens)
        prompt_tokens = max(1, len(prompt) // 4)

        self._send_json(200, {
            "id": f"mock-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "text": content,
                "finish_reason": "stop" if len(tokens) < max_tokens else "length",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": len(tokens),
                "total_tokens": prompt_tokens + len(tokens),
            },
        })

    def _handle_tokenize(self, raw: bytes):
        body = self._parse_json_body(raw)
        if body is None:
            return
        text = body.get("content", body.get("text", ""))
        # Mock: 1 token per 4 chars
        n_tokens = max(1, len(text) // 4)
        tokens = list(range(100, 100 + n_tokens))
        self._send_json(200, {"tokens": tokens})

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
               fail_rate: float = 0.0, oom: bool = False) -> ThreadedHTTPServer:
    """Start the mock server and return the server instance."""
    config = MockConfig(latency_ms=latency_ms, fail_rate=fail_rate, oom=oom)
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
    args = parser.parse_args()

    config = MockConfig(latency_ms=args.latency_ms, fail_rate=args.fail_rate, oom=args.oom)
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
