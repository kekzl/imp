#!/usr/bin/env python3
"""Decoder inter-token latency while k other sessions ingest (#1643).

One streaming decoder, then k concurrent long ingests. Reports the decoder's
inter-token latency (what the ingests interfere with) and each ingest's wall
time (what bounding them costs). This is the harness behind the
`runtime.prefill_batch_decode_cap` numbers in src/runtime/config.h.

It exists because the sibling knob's pinned numbers were taken against ONE
active decoder and one ingest, and nothing in the tree could re-run them - so
the k-dependence they miss stayed invisible.

Two traps it already walked into, both of which silently produce a
"no difference" result:

  - The decoder must still be running when the ingests land. At 120 tokens it
    finished in 0.4 s, before the ingests started, and every arm looked equal.
  - Every ingest prompt must be unique per arm AND per run, or the prefix cache
    serves it: the same prompt came back at 260 ms against 1200 ms cold, i.e.
    it measured a cache hit and no prefill at all. Hence NONCE.

Usage:  bench_prefill_latency.py [url] [model] [k] [nonce]
The engine reads the cap at startup, so the two arms are two server runs.
"""
import json, statistics, sys, threading, time, urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8099"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "Qwen3-8B-Q8_0.gguf"
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3
# Every ingest prompt must be unique across arms AND within a run: the second
# measurement came back at 260 ms against 1200 ms because the prefix cache had
# the previous run's prompt, i.e. it measured a cache hit and no prefill at all.
NONCE = sys.argv[4] if len(sys.argv) > 4 else "0"
INGEST_WORDS = 5200          # ~7k tokens
DECODE_TOKENS = 600

def post(path, body, stream=False):
    req = urllib.request.Request(URL + path, data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=600)

def decoder(out):
    body = {"model": MODEL, "max_tokens": DECODE_TOKENS, "temperature": 0, "stream": True,
            "messages": [{"role": "user",
                          "content": "Count slowly from one to sixty. Nonce %s." % NONCE}]}
    deltas, last, n = [], None, 0
    r = post("/v1/chat/completions", body, stream=True)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            d = json.loads(payload)
        except Exception:
            continue
        delta = d.get("choices", [{}])[0].get("delta", {})
        if not (delta.get("content") or delta.get("reasoning_content")):
            continue
        now = time.perf_counter()
        if last is not None:
            deltas.append((now - last) * 1000.0)
        last = now
        n += 1
    out["deltas"] = deltas
    out["tokens"] = n

def ingest(idx, out):
    prompt = ("w%s%d " % (NONCE, idx)) + ("token " * INGEST_WORDS) + \
             "\nSummarise the text above in one word."
    body = {"model": MODEL, "max_tokens": 4, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.perf_counter()
    try:
        post("/v1/chat/completions", body).read()
        out[idx] = (time.perf_counter() - t0) * 1000.0
    except Exception as e:
        out[idx] = float("nan")
        sys.stderr.write("ingest %d failed: %s\n" % (idx, e))

dec = {}
t = threading.Thread(target=decoder, args=(dec,))
t.start()
time.sleep(0.3)                      # ingests must land while the decoder still runs
ing = {}
threads = [threading.Thread(target=ingest, args=(i, ing)) for i in range(K)]
for th in threads: th.start()
for th in threads: th.join()
t.join()

d = sorted(dec.get("deltas") or [0.0])
def pct(p):
    if not d: return 0.0
    return d[min(len(d) - 1, int(round(p / 100.0 * (len(d) - 1))))]
print(json.dumps({
    "decode_tokens": dec.get("tokens", 0),
    # p95 hides this: only a handful of gaps are affected, and those are the
    # stutter a user sees. The count over 50 ms is the honest signal.
    "inter_token_ms": {"median": round(statistics.median(d), 1) if d else 0,
                       "p95": round(pct(95), 1), "p99": round(pct(99), 1),
                       "max": round(max(d), 1) if d else 0},
    "gaps_over_50ms": sum(1 for x in d if x > 50.0),
    "gaps_over_100ms": sum(1 for x in d if x > 100.0),
    "ingest_ms": [round(ing[i], 1) for i in sorted(ing)],
}, indent=None))
