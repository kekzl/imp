#!/usr/bin/env python3
# conc_client.py - N concurrent unique short prompts against a running
# imp-server, 300-token greedy gens, aggregate tok/s = sum completion
# tokens / wall. Usage: conc_client.py PORT CONC WAVES [TAG] [PLEN].
# PLEN > 0 prepends ~PLEN tokens of filler prose after the unique header
# (the long-prompt shape); GEN=<n> in the environment overrides the 300.
import json
import os
import sys
import threading
import time
import urllib.request

PORT = int(sys.argv[1])
CONC = int(sys.argv[2])
WAVES = int(sys.argv[3])
TAG = sys.argv[4] if len(sys.argv) > 4 else "x"
PLEN = int(sys.argv[5]) if len(sys.argv) > 5 else 0
GEN = int(os.environ.get("GEN", "300"))
MODEL = os.environ.get("MODEL_NAME", "Qwen3.8-27B-NVFP4-vllm")
FILLER_SENTENCES = [
    "A translation lookaside buffer caches recent virtual-to-physical page mappings so most loads skip the page walk.",
    "When a load crosses a page boundary the two halves may map to different frames and need two translations.",
    "Set-associative caches index a set by address bits and compare the tag against every way in that set.",
    "Write-back caches mark a line dirty and defer the memory write until the line is evicted.",
    "A hardware prefetcher watches the stream of misses and issues loads for the lines it expects next.",
    "Store buffers let the core retire a store before the cache accepts it, which hides write latency.",
    "Out-of-order cores track dependencies with a reorder buffer and commit results in program order.",
    "Branch predictors keep global and local histories and combine them to guess the next instruction address.",
    "A memory barrier orders earlier loads and stores against later ones as seen by other cores.",
    "Cache coherence protocols such as MESI keep one writer or many readers per line across cores.",
    "Huge pages cut TLB pressure by mapping two megabytes or a gigabyte with one entry.",
    "Non-uniform memory access means a core reaches its local memory faster than a remote node's.",
    "Speculative loads that miss can still warm the cache even when the branch was mispredicted.",
    "A victim cache holds lines evicted from a direct-mapped cache to soften conflict misses.",
    "Inclusive last-level caches keep a copy of every line the private caches hold, which simplifies snooping.",
    "Line fill buffers track outstanding misses so several loads can wait on the same line.",
    "Page table walkers cache intermediate levels so a walk after a TLB miss rarely touches memory four times.",
    "Alignment matters because a misaligned access that straddles two lines costs two cache lookups.",
    "Software prefetch instructions hint the address early, but a wrong hint only wastes bandwidth.",
    "Virtually indexed physically tagged caches overlap the TLB lookup with the set index computation.",
]


def filler(n_tokens):
    # ~20 tokens per sentence; varied technical prose, so the first answer token
    # is not a coin flip between EOS and text (a repeated pangram made about half
    # of the 1000-token prompts answer with an immediate EOS, 2026-09-02).
    n = max(1, n_tokens // 20)
    return " ".join(FILLER_SENTENCES[k % len(FILLER_SENTENCES)] for k in range(n)) + " "

results = []


def one(i, wave, out):
    prompt = (f"[{TAG}-w{wave}-r{i}] Explain topic {i * 7 + wave}: how a {i}-way set-associative "
              f"CPU cache interacts with a TLB during a page-crossing load.")
    if PLEN > 0:
        prompt = f"[{TAG}-w{wave}-r{i}] " + filler(PLEN) + prompt
    body = json.dumps({
        "model": MODEL,
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
        u = j.get("usage", {})
        out[i] = (u.get("completion_tokens", 0), time.time() - t0, u.get("prompt_tokens", 0))
    except Exception as e:
        out[i] = (0, time.time() - t0, 0)
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
    ptoks = sum(v[2] for v in out.values()) / max(1, len(out))
    print(f"wave{wave}: {toks} tok in {wall:.2f}s = {agg:.1f} tok/s aggregate"
          f" (prompt {ptoks:.0f} tok avg)", flush=True)

results.sort()
print(f"MEDIAN {results[len(results) // 2]:.1f}")
