#!/usr/bin/env python3
"""Streaming stand-in for imp-server for developing the web UI without a GPU.

Frames follow tools/imp-server/handlers_chat_stream.cpp: role chunk, one delta
per token (reasoning_content, then content), finish_reason chunk, usage chunk
with prompt_tokens_details / completion_tokens_details, [DONE]. Unlike
tests/api/mock_server.py it streams for real (one chunked write per token), so
TTFT, inter-token latency and mid-stream failures are observable.

Serves GET / from --html so the page runs same-origin, as in production.
Prompt keywords steer a run: nothink, length, thinkonly, fail400, fail500,
failmid (socket dropped after 12 tokens), slow (2.5 s TTFT), long, big.
Naming a model with loaded=false swaps it in after --swap-delay seconds;
--boot-delay N answers /v1/models with an empty list for N seconds.

    docker run --rm -p 9099:9099 -v "$PWD":/src -w /src python:3.12-slim \
        python3 tools/imp-server/webui/dev/mock_server.py --port 9099
    # then open http://localhost:9099/ or run drive.js (see its header)
"""
import argparse, json, os, random, socket, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ARGS = None
STATE = {"loaded": None, "boot_until": 0.0, "lock": threading.Lock()}

REASONING = ("Let me think about this. The user wants a short answer with an example. "
             "I should mention the key point first, then show code, then summarise. "
             "Also worth checking the edge case where the list is empty.").split(" ")

ANSWER = """Here is the short version, with **the key point first**.

## Why it works

The engine streams one delta per token, so the client can measure inter-token
latency directly. Use `stream_options.include_usage` to get server-side counts.

```python
def median(xs):
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2
```

- prompt tokens come from the tokenizer, not the client
- *cached* tokens are the prefix-cache hit
- reasoning tokens are counted separately

That covers it. Tell me if you want the long version, which also handles `a < b && c > d` safely.
"""


def tokenize(text):
    # ~4 chars per token, but keep whitespace attached like a BPE would.
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= 4 and (ch == " " or ch == "\n"):
            out.append(cur)
            cur = ""
    if cur:
        out.append(cur)
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *a):
        sys.stderr.write("%s %s\n" % (self.command, self.path))

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _reqid(self):
        rid = self.headers.get("X-Request-Id") or ("req_" + uuid.uuid4().hex[:12])
        self.send_header("X-Request-Id", rid)
        return rid

    def _json(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self._cors()
        self._reqid()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _error(self, status, message, etype="invalid_request_error", code=None):
        self._json(status, {"error": {"message": message, "type": etype, "code": code, "param": None}})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _booting(self):
        return time.monotonic() < STATE["boot_until"]

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            with open(ARGS.html, "rb") as f:
                raw = f.read()
            self.send_response(200)
            self._cors()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/health":
            self._json(200, {"status": "ok", "model_loaded": not self._booting() and STATE["loaded"] is not None,
                             "queue_depth": 0, "suspended": False})
        elif path == "/v1/models":
            data = []
            if not self._booting():
                for m in ARGS.models:
                    entry = {"id": m, "object": "model", "created": int(time.time()), "owned_by": "imp",
                             "loaded": m == STATE["loaded"]}
                    if m == STATE["loaded"]:
                        entry["max_model_len"] = ARGS.ctx
                        entry["meta"] = {"n_ctx_train": ARGS.ctx}
                    data.append(entry)
            self._json(200, {"object": "list", "data": data})
        else:
            self._error(404, "unmatched route: GET %s" % path, code="not_found")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if path != "/v1/chat/completions":
            self._error(404, "unmatched route: POST %s" % path, code="not_found")
            return
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self._error(400, "invalid JSON body")
            return
        self._chat(body)

    def _chat(self, body):
        if self._booting() or STATE["loaded"] is None and not body.get("model"):
            self._error(503, "no model loaded", etype="server_error", code="model_not_loaded")
            return
        model = body.get("model") or STATE["loaded"]
        if model not in ARGS.models and model != "imp":
            self._error(404, "model '%s' not found" % model, code="model_not_found")
            return
        if model != "imp" and model != STATE["loaded"]:
            time.sleep(ARGS.swap_delay)  # drain + swap
            with STATE["lock"]:
                STATE["loaded"] = model
        model = STATE["loaded"]

        msgs = body.get("messages") or []
        last = (msgs[-1].get("content") if msgs else "") or ""
        if isinstance(last, list):
            last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
        low = last.lower()
        max_tokens = int(body.get("max_tokens") or 512)
        if max_tokens < 1 or max_tokens > 65536:
            self._error(400, "max_tokens must be between 1 and 65536", code="invalid_value")
            return
        if "fail400" in low:
            self._error(400, "temperature must be between 0 and 2 (got %s)" % body.get("temperature"),
                        code="invalid_value")
            return
        if "fail500" in low:
            self._error(500, "internal error: CUDA error 700 (illegal address)", etype="server_error")
            return

        think = bool(body.get("enable_thinking", False)) and "nothink" not in low
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        reasoning = REASONING if think else []
        if "thinkonly" in low:
            reasoning = (REASONING * 4)
            content = []
        elif "long" in low:
            content = tokenize(ANSWER * 6)
        elif "big" in low:
            content = tokenize(ANSWER * 40)
        else:
            content = tokenize(ANSWER)
        if "length" in low:
            content = content[: max(1, max_tokens // 2)]
        # Budget: reasoning + content share max_tokens, like the server.
        budget = max_tokens
        finish = "stop"
        if len(reasoning) >= budget:
            reasoning = reasoning[:budget]
            content = []
            finish = "length"
        elif len(reasoning) + len(content) > budget or "length" in low or "thinkonly" in low:
            content = content[: max(0, budget - len(reasoning))]
            finish = "length"

        prompt_tokens = sum(len(str(m.get("content", ""))) // 4 + 4 for m in msgs) + 12
        cached = 0
        if len(msgs) > 2:
            cached = (prompt_tokens - len(last) // 4 - 4) // 16 * 16

        rid = "chatcmpl-" + uuid.uuid4().hex[:16]
        created = int(time.time())
        self.send_response(200)
        self._cors()
        self._reqid()
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def frame(delta, finish_reason=None, usage=None):
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [] if usage is not None else
                   [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}
            if usage is not None:
                obj["usage"] = usage
            data = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(("%x\r\n" % len(data)).encode() + data + b"\r\n")
            self.wfile.flush()

        itl = ARGS.itl / 1000.0
        try:
            time.sleep(2.5 if "slow" in low else ARGS.ttft / 1000.0)
            frame({"role": "assistant"})
            n_emitted = 0
            for tok in reasoning:
                frame({"reasoning_content": tok + " "})
                n_emitted += 1
                time.sleep(itl * random.uniform(0.7, 1.4))
            for i, tok in enumerate(content):
                if "failmid" in low and i == 12:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True
                    return
                frame({"content": tok})
                n_emitted += 1
                # one visible stall per run, like a page-in
                time.sleep(0.25 if i == len(content) // 3 else itl * random.uniform(0.7, 1.4))
            frame({}, finish_reason=finish)
            if include_usage:
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": n_emitted,
                         "total_tokens": prompt_tokens + n_emitted}
                if cached:
                    usage["prompt_tokens_details"] = {"cached_tokens": cached}
                if reasoning:
                    usage["completion_tokens_details"] = {"reasoning_tokens": len(reasoning)}
                frame(None, usage=usage)
            data = b"data: [DONE]\n\n"
            self.wfile.write(("%x\r\n" % len(data)).encode() + data + b"\r\n0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write("client went away\n")


def main():
    global ARGS
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9099)
    p.add_argument("--html", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "index.html"))
    p.add_argument("--models", nargs="+", default=["Qwen3.8-27B-NVFP4-vllm", "Llama-3.2-3B-Q8_0.gguf", "gemma-3-12b-it-NVFP4"])
    p.add_argument("--ctx", type=int, default=32768)
    p.add_argument("--itl", type=float, default=11.0, help="ms between tokens")
    p.add_argument("--ttft", type=float, default=180.0, help="ms before first token")
    p.add_argument("--swap-delay", type=float, default=2.0)
    p.add_argument("--boot-delay", type=float, default=0.0, help="seconds of model-less start")
    ARGS = p.parse_args()
    STATE["loaded"] = ARGS.models[0]
    STATE["boot_until"] = time.monotonic() + ARGS.boot_delay
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), H)
    srv.daemon_threads = True
    sys.stderr.write("mock imp on :%d serving %s\n" % (ARGS.port, ARGS.html))
    srv.serve_forever()


if __name__ == "__main__":
    main()
