#!/usr/bin/env python3
# Sequential streaming chat requests, one at a time, real prompts: per request TTFT (send -> first
# delta) and decode tok/s ((completion_tokens - 1) / (last delta - first delta)); MEDIAN line at the end.
# Usage: seq_stream_client.py PORT [TAG]; env MODEL_NAME, GEN (400), IGNORE_EOS (1), ROUNDS (1), THINK (0/1).
import json
import os
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"
MODEL = os.environ.get("MODEL_NAME", "")
if not MODEL:  # the loaded model: imp-server answers 400 without a "model" field
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=30) as r:
        MODEL = next(m["id"] for m in json.load(r)["data"] if m.get("loaded", True))
GEN = int(os.environ.get("GEN", "400"))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "1") == "1"
ROUNDS = int(os.environ.get("ROUNDS", "1"))
THINK = os.environ.get("THINK", "")  # "0" / "1" sets enable_thinking; unset = the model default
PROMPTS = [
    ("prose", "Write a detailed essay about the history of the Hanseatic League, its trade routes, its decline, "
     "and its legacy in northern European cities."),
    ("code", "Implement a lock-free single-producer single-consumer ring buffer in C++20 with std::atomic, "
     "explain the memory orderings, and add a small benchmark main()."),
    ("chat", "Hey, I'm planning a 5 day trip to Japan in November with my partner. We like food, hiking and "
     "quiet places. Can you suggest an itinerary and some tips?"),
    ("math", "Prove that there are infinitely many primes of the form 4k+3, then explain where the same "
     "argument fails for primes of the form 4k+1."),
    ("debug", "My Python service leaks memory slowly under load. It uses asyncio, aiohttp and a global dict as "
     "a cache. Walk me through how to find and fix the leak, with concrete commands."),
    ("summary", "Explain to a new engineer how a paged KV cache works in an LLM inference server, why it "
     "exists, and what prefix caching adds on top."),
]


def run(name, prompt):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": GEN, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    if MODEL:
        body["model"] = MODEL
    if THINK:
        body["enable_thinking"] = THINK == "1"
    if IGNORE_EOS:
        body["ignore_eos"] = True
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    deltas = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    deltas += 1
    n = usage["completion_tokens"] if usage else deltas
    ttft_ms = (t_first - t0) * 1e3
    tps = (n - 1) / (t_last - t_first) if t_last > t_first else 0.0
    print(f"{TAG} {name:8s} tokens {n:4d}  ttft {ttft_ms:8.1f} ms  decode {tps:7.2f} tok/s", flush=True)
    return ttft_ms, tps


def main():
    ttfts, rates = [], []
    for _ in range(ROUNDS):
        for name, prompt in PROMPTS:
            t, r = run(name, prompt)
            ttfts.append(t)
            rates.append(r)
    print(f"{TAG} MEDIAN decode {statistics.median(rates):.2f} tok/s (min {min(rates):.2f} max {max(rates):.2f}) "
          f"ttft {statistics.median(ttfts):.1f} ms (min {min(ttfts):.1f} max {max(ttfts):.1f}) n={len(rates)}",
          flush=True)


if __name__ == "__main__":
    main()
