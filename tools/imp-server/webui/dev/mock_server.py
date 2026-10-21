#!/usr/bin/env python3
"""Streaming stand-in for imp-server for developing the web UI without a GPU.

Frames follow tools/imp-server/handlers_chat_stream.cpp: role chunk, one delta
per token (reasoning_content, then content), finish_reason chunk, usage chunk
with prompt_tokens_details / completion_tokens_details, [DONE]. Unlike
tests/api/mock_server.py it streams for real (one chunked write per token), so
TTFT, inter-token latency and mid-stream failures are observable.

Serves GET / from --html so the page runs same-origin, as in production.
Prompt keywords steer a run: nothink, length, thinkonly, fail400, fail500,
failmid (socket dropped after 12 tokens), slow (2.5 s TTFT), long, big,
fast (3x token rate, no stall), deepthink (long reasoning, for end_thinking).
Naming a model with loaded=false swaps it in after --swap-delay seconds;
--boot-delay N answers /v1/models with an empty list for N seconds;
--answer-file / --reasoning-file replace the built-in texts (README capture).
Also mirrors GET /metrics (names from handlers_misc.cpp / metrics_memory.cpp, plus
imp_gpu_power_watts / imp_gpu_energy_joules_total unless --no-power), /props, /info,
/health, POST /tokenize, POST /v1/requests/{id}/end_thinking, streamed logprobs + logit_lens,
continue_final_message,
response_format (answers JSON) and image_url parts (400 vision_unavailable on a
model whose name contains "Llama").

    docker run --rm -p 9099:9099 -v "$PWD":/src -w /src python:3.12-slim \
        python3 tools/imp-server/webui/dev/mock_server.py --port 9099
    # then open http://localhost:9099/ or run drive.py (see its header)
"""
import argparse, json, math, os, random, socket, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ARGS = None
MOCK_LAYERS = 24  # logit lens rows
STATE = {"loaded": None, "boot_until": 0.0, "lock": threading.Lock(), "started": time.time()}
# /metrics counters; energy integrates power (idle vs streaming) between reads.
METRICS = {"requests": 0, "failed": 0, "prompt": 0, "completion": 0, "cached": 0, "active": 0,
           "energy": 0.0, "energy_t": time.monotonic(), "last_ttft_ms": 0}
ACTIVE = {}  # request id -> threading.Event set by end_thinking


def power_now():
    return 72.0 + (380.0 if METRICS["active"] else 0.0) + random.uniform(-4, 4)


def tick_energy():
    with STATE["lock"]:
        now = time.monotonic()
        METRICS["energy"] += power_now() * (now - METRICS["energy_t"])
        METRICS["energy_t"] = now

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

    def _reqid(self, own=None):
        # Like the server: the client id if sent, else the completion id on generation routes.
        rid = self.headers.get("X-Request-Id") or own or ("req_" + uuid.uuid4().hex[:12])
        self.send_header("X-Request-Id", rid)
        return rid

    def _text(self, status, text, ctype):
        raw = text.encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _metrics(self):
        tick_energy()
        loaded = STATE["loaded"] or ""
        active = METRICS["active"]
        total_blocks = ARGS.ctx // 16 * 4
        used = min(total_blocks, 180 + active * 900 + METRICS["requests"] * 12)
        lines = [
            ("imp_uptime_seconds", int(time.time() - STATE["started"])),
            ("imp_requests_total", METRICS["requests"]),
            ("imp_requests_failed_total", METRICS["failed"]),
            ("imp_tokens_prompt_total", METRICS["prompt"]),
            ("imp_tokens_completion_total", METRICS["completion"]),
            ("imp_tokens_cached_total", METRICS["cached"]),
            ("imp_last_ttft_ms", METRICS["last_ttft_ms"]),
            ("imp_decode_batch_last_rows", active),
            ('imp_model_loaded{model="%s"}' % loaded, 1 if loaded else 0),
            ("imp_queue_depth", active),
            ("imp_queue_waiting", 0),
            ("imp_queue_running", active),
            ("imp_queued_prompt_tokens", 0),
            ("imp_kv_blocks_total", total_blocks),
            ("imp_kv_blocks_used", used),
            ("imp_kv_blocks_cached", max(0, used - active * 900)),
            ("imp_vram_budget_bytes", 0),
            ("imp_vram_own_bytes", 27_412_000_000 + active * 180_000_000),
            ("imp_vram_own_peak_bytes", 28_190_000_000),
        ]
        if not ARGS.no_power:
            lines += [("imp_gpu_power_watts", round(power_now(), 1)),
                      ("imp_gpu_energy_joules_total", round(METRICS["energy"], 1))]
        out = []
        for name, v in lines:
            base = name.split("{")[0]
            out.append("# TYPE %s %s" % (base, "counter" if base.endswith("_total") else "gauge"))
            out.append("%s %s" % (name, v))
        return "\n".join(out) + "\n"

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
                             "queue_depth": METRICS["active"], "suspended": False, "idle_suspended": False})
        elif path == "/metrics":
            self._text(200, self._metrics(), "text/plain; version=0.0.4")
        elif path == "/props":  # handlers.cpp handle_props
            self._json(200, {"model_path": STATE["loaded"] or "", "total_slots": 8, "n_ctx": ARGS.ctx,
                             "default_generation_settings": {"n_ctx": ARGS.ctx}})
        elif path == "/info":  # handlers.cpp handle_info
            self._json(200, {"model_id": STATE["loaded"] or "", "max_total_tokens": ARGS.ctx,
                             "max_input_tokens": ARGS.ctx - 1})
        elif path == "/v1/models":
            data = []
            if not self._booting():
                for m in ARGS.models:
                    entry = {"id": m, "object": "model", "created": int(time.time()), "owned_by": "imp",
                             "loaded": m == STATE["loaded"]}
                    if m == STATE["loaded"]:
                        entry["max_model_len"] = ARGS.ctx
                        entry["meta"] = {"n_ctx_train": ARGS.ctx,
                                         "reasoning_effort": {"values": ["xhigh", "medium", "low"], "default": "xhigh"}}
                    data.append(entry)
            self._json(200, {"object": "list", "data": data})
        else:
            self._error(404, "unmatched route: GET %s" % path, code="not_found")

    def do_POST(self):
        path = urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        if path.startswith("/v1/requests/") and path.endswith("/end_thinking"):  # handlers_admin.cpp
            rid = path[len("/v1/requests/"):-len("/end_thinking")]
            ev = ACTIVE.get(rid)
            if ev is None:
                self._error(404, "No running request with id '%s'." % rid[:128], code="request_not_found")
                return
            status = "already_closed" if ev.is_set() else "ending"
            ev.set()
            self._json(200, {"id": rid, "object": "request.end_thinking", "status": status})
            return
        if path not in ("/v1/chat/completions", "/tokenize"):
            self._error(404, "unmatched route: POST %s" % path, code="not_found")
            return
        try:
            body = json.loads(raw or b"{}")
        except Exception:
            self._error(400, "invalid JSON body")
            return
        if path == "/tokenize":  # handlers_misc.cpp handle_tokenize
            content = body.get("content") or body.get("prompt") or ""
            if not content:
                self._error(400, "\"content\" (or its alias \"prompt\") is required")
                return
            self._json(200, {"tokens": [hash(t) % 151936 for t in tokenize(content)]})
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
        # handlers_chat_params.cpp: continue_final_message needs a final assistant turn with text;
        # the reply holds only the new text, keywords come from the last user turn.
        prefill = body.get("continue_final_message") is True
        if prefill and (not msgs or msgs[-1].get("role") != "assistant" or not msgs[-1].get("content")):
            self._error(400, "continue_final_message needs a final assistant message with text and no tool_calls",
                        code="invalid_value")
            return
        users = [m for m in msgs if m.get("role") == "user"]
        last = ((users[-1] if prefill and users else msgs[-1]).get("content") if msgs else "") or ""
        if isinstance(last, list):
            last = " ".join(p.get("text", "") for p in last if isinstance(p, dict))
        low = last.lower()
        max_tokens = int(body.get("max_tokens") or ARGS.ctx)  # no field: the context end, as imp-server
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
        tail = msgs[-1].get("content") if msgs else None
        has_image = isinstance(tail, list) and any(isinstance(p, dict) and p.get("type") == "image_url" for p in tail)
        if has_image and "llama" in model.lower():
            self._error(400, "The loaded model has no vision tower; image_url parts cannot be read.",
                        code="vision_unavailable")
            return

        rf = (body.get("response_format") or {}).get("type")
        think = bool(body.get("enable_thinking", False)) and "nothink" not in low and not rf and not prefill
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        want_lp = bool(body.get("logprobs"))
        n_top = int(body.get("top_logprobs") or 0)
        ll = body.get("logit_lens")
        n_lens = 5 if ll is True else int(ll) if isinstance(ll, int) and not isinstance(ll, bool) else 0
        if n_lens:
            want_lp = True
        reasoning = REASONING if think else []
        if "deepthink" in low and think:
            reasoning = REASONING * 10
        if "thinkonly" in low:
            reasoning = (REASONING * 4)
            content = []
        elif prefill:
            content = tokenize(" and from this token on the branch takes its own way. " + ANSWER[:160])
        elif rf in ("json_object", "json_schema"):
            content = tokenize(json.dumps({"answer": "streaming sends one delta per token", "tokens": 42,
                                           "images_seen": int(has_image)}, indent=1))
        elif has_image:
            content = tokenize("I see the picture you attached: a small pink imp holding a fireball. " + ANSWER[:200])
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
        rid = self._reqid(rid)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def frame(delta, finish_reason=None, usage=None, logprobs=None):
            choice = {"index": 0, "delta": delta, "finish_reason": finish_reason}
            if logprobs is not None:
                choice["logprobs"] = logprobs
            obj = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": model,
                   "choices": [] if usage is not None else [choice]}
            if usage is not None:
                obj["usage"] = usage
            data = ("data: " + json.dumps(obj) + "\n\n").encode()
            self.wfile.write(("%x\r\n" % len(data)).encode() + data + b"\r\n")
            self.wfile.flush()

        def lp_for(tok):
            # handlers_chat_stream.cpp: {"content": [{token, logprob, bytes, top_logprobs: [...]}]}
            p = random.choice([0.97, 0.9, 0.75, 0.5, 0.2, 0.06]) if tok.strip() else 0.99
            lp = math.log(p)
            alts = [{"token": tok, "logprob": lp, "bytes": list(tok.encode())}]
            rest = 1.0 - p
            for k, alt in enumerate([" the", " a", "ing", ".", " and"][: max(0, n_top - 1)]):
                q = max(rest * (0.6 / (k + 1)), 1e-6)
                alts.append({"token": alt, "logprob": math.log(q), "bytes": list(alt.encode())})
            alts.sort(key=lambda a: -a["logprob"])
            entry = {"token": tok, "logprob": lp, "bytes": list(tok.encode()), "top_logprobs": alts[:n_top]}
            if n_lens > 0:
                entry["logit_lens"] = lens_for(tok)
            return {"content": [entry]}

        def lens_for(tok):
            # utils.cpp logit_lens_json: one {rank, logprob, entropy, top} per layer, the sampled
            # token climbing to rank 1 at a random settle layer.
            settle = random.randint(MOCK_LAYERS // 3, MOCK_LAYERS - 2)
            out = []
            for layer in range(MOCK_LAYERS):
                rank = 1 if layer >= settle else max(2, (settle - layer) * 37)
                lp = math.log(0.15 + 0.8 * (layer - settle + 1) / (MOCK_LAYERS - settle + 1)) if rank == 1 \
                    else math.log(0.5 / rank)
                top = [{"token": t, "logprob": math.log(0.3 / (k + 1))}
                       for k, t in enumerate([" the", " of", "ing", ".", " and"][:n_lens])]
                if rank == 1:
                    top = [{"token": tok, "logprob": lp}] + top[: n_lens - 1]
                out.append({"rank": rank, "logprob": lp, "entropy": 4.5 * (1 - layer / MOCK_LAYERS) + 0.3, "top": top})
            return out

        itl = ARGS.itl / 1000.0 / (3.0 if "fast" in low else 1.0)
        ended = threading.Event()
        ACTIVE[rid] = ended
        with STATE["lock"]:
            METRICS["requests"] += 1
        tick_energy()
        with STATE["lock"]:
            METRICS["active"] += 1
        t0 = time.monotonic()
        try:
            time.sleep(2.5 if "slow" in low else ARGS.ttft / 1000.0)
            METRICS["last_ttft_ms"] = int((time.monotonic() - t0) * 1000)
            frame({"role": "assistant"})
            n_emitted = 0
            for tok in reasoning:
                if ended.is_set():  # end_thinking: the think block closes, the answer follows
                    break
                frame({"reasoning_content": tok + " "})
                n_emitted += 1
                time.sleep(itl * random.uniform(0.7, 1.4))
            ended.set()
            for i, tok in enumerate(content):
                if "failmid" in low and i == 12:
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.close_connection = True
                    return
                frame({"content": tok}, logprobs=lp_for(tok) if want_lp else None)
                n_emitted += 1
                # one visible stall per run, like a page-in
                time.sleep(0.25 if i == len(content) // 3 and "fast" not in low else itl * random.uniform(0.7, 1.4))
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
            with STATE["lock"]:
                METRICS["prompt"] += prompt_tokens
                METRICS["completion"] += n_emitted
                METRICS["cached"] += cached
        except (BrokenPipeError, ConnectionResetError):
            sys.stderr.write("client went away\n")
        finally:
            ACTIVE.pop(rid, None)
            tick_energy()
            with STATE["lock"]:
                METRICS["active"] -= 1


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
    p.add_argument("--no-power", action="store_true", help="omit imp_gpu_power_watts / imp_gpu_energy_joules_total")
    p.add_argument("--boot-delay", type=float, default=0.0, help="seconds of model-less start")
    p.add_argument("--answer-file", help="markdown the model answers with, instead of the built-in text")
    p.add_argument("--reasoning-file", help="text the model thinks with, instead of the built-in text")
    ARGS = p.parse_args()
    global ANSWER, REASONING
    if ARGS.answer_file:
        ANSWER = open(ARGS.answer_file, encoding="utf-8").read()
    if ARGS.reasoning_file:
        REASONING = open(ARGS.reasoning_file, encoding="utf-8").read().split(" ")
    STATE["loaded"] = ARGS.models[0]
    STATE["boot_until"] = time.monotonic() + ARGS.boot_delay
    srv = ThreadingHTTPServer(("0.0.0.0", ARGS.port), H)
    srv.daemon_threads = True
    sys.stderr.write("mock imp on :%d serving %s\n" % (ARGS.port, ARGS.html))
    srv.serve_forever()


if __name__ == "__main__":
    main()
