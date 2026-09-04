#!/usr/bin/env python3
"""/metrics latency histograms are fed by EVERY generation path.

Before this battery the four histograms (request duration, TTFT, queue wait,
inter-token) were observed by the chat streaming driver and, minus ITL, the
chat non-stream loop; /v1/completions (both modes) fed none of them, which is
the endpoint every serving harness in tools/analysis/ uses. Also checks the
preemption counters and the last-batch gauge exist and that the gauge reads 0
once the server idles. Needs a running imp-server (IMP_HOST/IMP_PORT/IMP_MODEL,
defaults localhost:8080 and Qwen3-8B-NVFP4-cortecs). Exit 1 on any failure."""
import json
import os
import re
import sys
import time
import urllib.request

HOST = os.environ.get("IMP_HOST", "localhost")
PORT = int(os.environ.get("IMP_PORT", "8080"))
M = os.environ.get("IMP_MODEL", "Qwen3-8B-NVFP4-cortecs")
BASE = f"http://{HOST}:{PORT}"
N = 24
fails = []


def check(label, cond, detail):
    print(f"{'ok  ' if cond else 'FAIL'} {label}: {detail}")
    if not cond:
        fails.append(label)


def metrics():
    with urllib.request.urlopen(BASE + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        m = re.match(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)\s+(-?[0-9.eE+]+)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out, text


def post(path, body, stream):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        if not stream:
            return json.loads(r.read())
        n = 0
        for raw in r:
            if raw.startswith(b"data:") and b"[DONE]" not in raw:
                n += 1
        return n


HIST = ("imp_request_duration_seconds_count", "imp_ttft_seconds_count",
        "imp_queue_time_seconds_count", "imp_inter_token_seconds_count")

CASES = [
    ("chat non-stream", "/v1/chat/completions",
     {"model": M, "messages": [{"role": "user", "content": "Count from one to thirty in words."}],
      "max_tokens": N, "temperature": 0, "ignore_eos": True}, False),
    ("chat stream", "/v1/chat/completions",
     {"model": M, "messages": [{"role": "user", "content": "Count from one to thirty in words."}],
      "max_tokens": N, "temperature": 0, "stream": True, "ignore_eos": True}, True),
    ("completions non-stream", "/v1/completions",
     {"model": M, "prompt": "One two three four five six seven", "max_tokens": N,
      "temperature": 0, "ignore_eos": True}, False),
    ("completions stream", "/v1/completions",
     {"model": M, "prompt": "One two three four five six seven", "max_tokens": N,
      "temperature": 0, "stream": True, "ignore_eos": True}, True),
]

for label, path, body, stream in CASES:
    m0, _ = metrics()
    post(path, body, stream)
    m1, _ = metrics()
    d = {k: m1.get(k, 0) - m0.get(k, 0) for k in HIST}
    check(f"{label}: duration/ttft/queue observed once",
          d[HIST[0]] == 1 and d[HIST[1]] == 1 and d[HIST[2]] == 1, d)
    # N tokens delivered => N-1 gaps; a stop token swallowed by the loop may
    # cost one, so demand most of them rather than the exact count.
    check(f"{label}: ITL observed per token", d[HIST[3]] >= N - 4, f"{d[HIST[3]]:.0f} gaps for {N} tokens")

_, text = metrics()
for name in ("imp_streaming_kv_auto_enables_total", "imp_prefix_cache_evictions_total",
             "imp_decode_batch_last_rows", "imp_kv_pressure_rejections_total"):
    check(f"series {name} present", re.search(rf"^{name} ", text, re.M) is not None, name)

time.sleep(1.0)
m, _ = metrics()
check("imp_decode_batch_last_rows is 0 when idle", m.get("imp_decode_batch_last_rows", -1) == 0,
      m.get("imp_decode_batch_last_rows"))

print(f"{len(fails)} failure(s)")
sys.exit(1 if fails else 0)
