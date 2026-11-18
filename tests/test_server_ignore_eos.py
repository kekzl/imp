#!/usr/bin/env python3
"""ignore_eos on /v1/completions and /v1/chat/completions: the request runs to
exactly max_tokens (EOS and stop tokens ignored), without the flag the same
prompt with an early-stop instruction ends before max_tokens. Needs a running
imp-server (IMP_HOST/IMP_PORT/IMP_MODEL, defaults localhost:8080 and
Qwen3-8B-NVFP4-cortecs). Exit 1 on any failure."""
import json
import os
import sys
import urllib.request

HOST = os.environ.get("IMP_HOST", "localhost")
PORT = int(os.environ.get("IMP_PORT", "8080"))
M = os.environ.get("IMP_MODEL", "Qwen3-8B-NVFP4-cortecs")
BASE = f"http://{HOST}:{PORT}"
N = 48
fails = []


def post(path, body):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def check(label, cond, detail):
    print(f"{'ok  ' if cond else 'FAIL'} {label}: {detail}")
    if not cond:
        fails.append(label)


# A prompt the model answers in a handful of tokens.
prompt = "Reply with the single word OK and nothing else."
# completions
j = post("/v1/completions", {"model": M, "prompt": prompt, "max_tokens": N, "temperature": 0,
                             "ignore_eos": True})
c = j["usage"]["completion_tokens"]
check("completions ignore_eos runs to max_tokens", c == N, f"completion_tokens={c} want {N}")
check("completions ignore_eos finish_reason=length", j["choices"][0].get("finish_reason") == "length",
      f"finish_reason={j['choices'][0].get('finish_reason')}")
# Control on the chat endpoint (template + EOS): a raw-prompt completion
# has no chat template and a base-style continuation can run past N.
j = post("/v1/chat/completions", {"model": M, "messages": [{"role": "user", "content": prompt}],
                                  "max_tokens": N, "temperature": 0,
                                  "chat_template_kwargs": {"enable_thinking": False}})
c0 = j["usage"]["completion_tokens"]
check("chat without the flag stops early", c0 < N, f"completion_tokens={c0}")
# chat
msgs = [{"role": "user", "content": prompt}]
j = post("/v1/chat/completions", {"model": M, "messages": msgs, "max_tokens": N, "temperature": 0,
                                  "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}})
c = j["usage"]["completion_tokens"]
check("chat ignore_eos runs to max_tokens", c == N, f"completion_tokens={c} want {N}")
# streaming completions: the token count survives the SSE path
body = json.dumps({"model": M, "prompt": prompt, "max_tokens": N, "temperature": 0, "ignore_eos": True,
                   "stream": True, "stream_options": {"include_usage": True}}).encode()
req = urllib.request.Request(BASE + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
usage, chunks = {}, 0
with urllib.request.urlopen(req, timeout=300) as r:
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        d = json.loads(line[5:].strip())
        if d.get("usage"):
            usage = d["usage"]
        if d.get("choices") and d["choices"][0].get("text"):
            chunks += 1
check("streaming completions ignore_eos usage", usage.get("completion_tokens") == N,
      f"usage.completion_tokens={usage.get('completion_tokens')} chunks={chunks}")


def stream_chat(extra):
    """Streams a chat request; returns (reasoning deltas, content deltas)."""
    b = {"model": M, "messages": [{"role": "user", "content": "Name three rivers."}], "max_tokens": N,
         "temperature": 0, "stream": True, **extra}
    rq = urllib.request.Request(BASE + "/v1/chat/completions", data=json.dumps(b).encode(),
                                headers={"Content-Type": "application/json"})
    n_r = n_c = 0
    with urllib.request.urlopen(rq, timeout=300) as r:
        for raw in r:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            d = json.loads(line[5:].strip())
            delta = (d.get("choices") or [{}])[0].get("delta") or {}
            n_r += bool(delta.get("reasoning_content") or delta.get("reasoning"))
            n_c += bool(delta.get("content"))
    return n_r, n_c


def last_ttft_ms():
    with urllib.request.urlopen(BASE + "/metrics", timeout=30) as r:
        for line in r.read().decode().splitlines():
            if line.startswith("imp_last_ttft_ms "):
                return float(line.split()[1])
    return -1.0


# Streaming chat: the <think> opener is a stop id the ignore_eos branch swallowed, so the whole chain of
# thought streamed as content (Flint: 0 reasoning deltas vs 33 without the flag) and ttft read 0.0.
ctl_r, _ = stream_chat({})
ie_r, ie_c = stream_chat({"ignore_eos": True})
check("streaming chat ignore_eos keeps reasoning split", ctl_r == 0 or ie_r > 0,
      f"reasoning deltas {ie_r} (without the flag {ctl_r}), content deltas {ie_c}")
t = last_ttft_ms()
check("streaming chat ignore_eos sets ttft", t > 0.0, f"imp_last_ttft_ms={t}")

print(f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
