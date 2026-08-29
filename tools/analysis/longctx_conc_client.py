#!/usr/bin/env python3
# longctx_conc_client.py - N concurrent LONG prompts (~15.5k tokens from the
# NIAH filler, unique head per stream) against a running imp-server.
# Usage: longctx_conc_client.py PORT CONC GEN TAG
# Prints: WAVE tag conc gen sum_completion wall_s
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1])
CONC = int(sys.argv[2])
GEN = int(sys.argv[3])
TAG = sys.argv[4] if len(sys.argv) > 4 else "x"
TARGET_CHARS = int(__import__("os").environ.get("TARGET_CHARS", "62000"))  # ~chars/4 tokens

raw = (Path(__file__).resolve().parents[1] / "eval/niah/data/filler.txt").read_text(
    encoding="utf-8", errors="replace")
filler = (raw * ((TARGET_CHARS // len(raw)) + 2))

with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=30) as r:
    MODEL_ID = json.loads(r.read())["data"][0]["id"]

out = {}

def one(i):
    off = (i * 7919) % len(raw)
    body_txt = (f"[stream-{i}-{TAG}] You will summarize the following text.\n\n" +
                (filler[off:] + filler[:off])[:TARGET_CHARS] +
                "\n\nSummarize the main activities described above in one paragraph.")
    body = json.dumps({
        "model": MODEL_ID,
        "prompt": body_txt,
        "max_tokens": GEN,
        "temperature": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            j = json.loads(r.read())
        out[i] = j.get("usage", {}).get("completion_tokens", 0)
    except Exception as e:
        out[i] = 0
        print(f"ERR r{i}: {e}", file=sys.stderr)

threads = [threading.Thread(target=one, args=(i,)) for i in range(CONC)]
t0 = time.time()
for t in threads:
    t.start()
for t in threads:
    t.join()
wall = time.time() - t0
print(f"WAVE {TAG} conc={CONC} gen={GEN} sum_completion={sum(out.values())} wall_s={wall:.2f}")
