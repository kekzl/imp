#!/usr/bin/env python3
# MTP spec -> batched decode -> spec switching on one imp-server: a long greedy chat (LONG tokens, streamed)
# decodes alone, SHORT_N short chats join mid-way and finish, the long one decodes alone again.
# Reads imp_spec_accepted_total around each phase and the long answer's text. Exit 0 when the long request
# drafted while alone (phase 1) and its text passes a repetition check. Phase 3 is reported: a binding that
# decoded in a batch unbinds (imp-server log "head cache behind"), so it reads 0 today.
# Usage: mtp_switch_probe.py PORT; env MODEL_NAME, LONG (900), SHORT (120), SHORT_N (3), JOIN_AT (150 tokens).
import json
import os
import re
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1])
MODEL = os.environ.get("MODEL_NAME", "Qwen3.8-27B-NVFP4-vllm")
LONG = int(os.environ.get("LONG", "900"))
SHORT = int(os.environ.get("SHORT", "120"))
SHORT_N = int(os.environ.get("SHORT_N", "3"))
JOIN_AT = int(os.environ.get("JOIN_AT", "150"))
URL = f"http://127.0.0.1:{PORT}"


def accepted():
    with urllib.request.urlopen(f"{URL}/metrics", timeout=30) as r:
        text = r.read().decode()
    return sum(float(m.group(1)) for m in re.finditer(r"^imp_spec_accepted_total(?:\{[^}]*\})?\s+(\S+)$", text, re.M))


def chat(prompt, n, stream, on_token=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": n,
            "temperature": 0, "stream": stream, "ignore_eos": True}
    req = urllib.request.Request(f"{URL}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    out = []
    with urllib.request.urlopen(req, timeout=900) as resp:
        if not stream:
            msg = json.load(resp)["choices"][0]["message"]
            return (msg.get("reasoning_content") or "") + (msg.get("content") or "")
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    out.append(piece)
                    if on_token:
                        on_token(len(out))
    return "".join(out)


marks = {}
joined = threading.Event()


def on_long_token(n):
    if n == JOIN_AT and not joined.is_set():
        marks["a_join"] = accepted()
        joined.set()


result = {}
a0 = accepted()
lt = threading.Thread(target=lambda: result.update(long=chat(
    "Write a long, detailed history of the printing press, from Gutenberg to offset lithography.", LONG, True,
    on_long_token)))
lt.start()
joined.wait(timeout=600)
shorts = [threading.Thread(target=lambda i=i: result.update({f"s{i}": chat(
    f"In two sentences, what is {['a mutex', 'a semaphore', 'a barrier', 'a spinlock'][i % 4]}?", SHORT, False)}))
    for i in range(SHORT_N)]
for t in shorts:
    t.start()
for t in shorts:
    t.join()
a_shorts_done = accepted()
lt.join()
a_end = accepted()
text = result.get("long", "")
words = text.split()
max_run = max((len(m.group(0).split()) for m in re.finditer(r"(\b\S+\b)(\s+\1\b)+", text)), default=1)
phase1 = marks.get("a_join", a0) - a0
phase3 = a_end - a_shorts_done
print(f"accepted phase1(alone) {phase1:.0f} phase2(batched) {a_shorts_done - marks.get('a_join', a0):.0f} "
      f"phase3(alone again) {phase3:.0f}; long words {len(words)} max_word_run {max_run}")
for i in range(SHORT_N):
    print(f"short{i}: {result.get(f's{i}', '')[:120]!r}")
print("long tail:", repr(text[-200:]))
ok = phase1 > 0 and max_run <= 4 and len(words) > 100
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
