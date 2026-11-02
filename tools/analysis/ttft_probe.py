#!/usr/bin/env python3
"""Sequential TTFT probe: phases of /v1/completions requests with exact token-id prompts.

usage: ttft_probe.py PORT MODEL OUT_JSON  (driven by tools/analysis/coldstart_arm.sh)
Each prompt is a list of token ids, unique from its first id (no prefix-cache hit).
"""
import json
import random
import sys
import time
import urllib.request

PORT, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/completions"
rng = random.Random(1234)
seq = [0]


def prompt(n):
    seq[0] += 1
    head = 1000 + seq[0] * 7
    return [head] + [rng.randint(2000, 20000) for _ in range(n - 1)]


def one(n, max_tokens, phase):
    body = json.dumps({"model": MODEL, "prompt": prompt(n), "max_tokens": max_tokens, "temperature": 0,
                       "stream": True, "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line and ttft is None:
                ttft = (time.perf_counter() - t0) * 1000.0
    total = (time.perf_counter() - t0) * 1000.0
    return {"phase": phase, "n": n, "max_tokens": max_tokens, "client_ttft_ms": ttft, "total_ms": total}


phases = [
    ("dec_cold", [24, 20, 20, 23, 21], 128),
    ("same24", [24] * 8, 1),
    ("vary17_31", [rng.randint(17, 31) for _ in range(8)], 1),
    ("vary40_64", [rng.randint(40, 64) for _ in range(8)], 1),
    ("vary65_200", [rng.randint(65, 200) for _ in range(8)], 1),
    ("dec_again", [24, 20, 20, 23, 21], 128),
]
recs = []
for name, lens, mt in phases:
    for n in lens:
        recs.append(one(n, mt, name))
        print(json.dumps(recs[-1]), flush=True)
with open(OUT, "w") as f:
    json.dump(recs, f, indent=1)
