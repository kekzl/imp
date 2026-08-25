#!/usr/bin/env python3
# conc_client.py - N concurrent unique short prompts against a running
# imp-server, 300-token greedy gens, aggregate tok/s = sum completion
# tokens / wall. Usage: conc_client.py PORT CONC WAVES [TAG].
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1])
CONC = int(sys.argv[2])
WAVES = int(sys.argv[3])
TAG = sys.argv[4] if len(sys.argv) > 4 else "x"
GEN = 300

results = []


def one(i, wave, out):
    prompt = (f"[{TAG}-w{wave}-r{i}] Explain topic {i * 7 + wave}: how a {i}-way set-associative "
              f"CPU cache interacts with a TLB during a page-crossing load.")
    body = json.dumps({
        "model": "Qwen3.8-27B-NVFP4",
        "prompt": prompt,
        "max_tokens": GEN,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.loads(r.read())
        out[i] = (j.get("usage", {}).get("completion_tokens", 0), time.time() - t0)
    except Exception as e:
        out[i] = (0, time.time() - t0)
        print(f"ERR r{i}: {e}", file=sys.stderr)


for wave in range(WAVES):
    out = {}
    threads = [threading.Thread(target=one, args=(i, wave, out)) for i in range(CONC)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    toks = sum(v[0] for v in out.values())
    agg = toks / wall
    results.append(agg)
    print(f"wave{wave}: {toks} tok in {wall:.2f}s = {agg:.1f} tok/s aggregate", flush=True)

results.sort()
print(f"MEDIAN {results[len(results) // 2]:.1f}")
