#!/usr/bin/env python3
# burst_stream_client.py - N concurrent streaming completions against a running
# OpenAI-compatible server (imp-server or vLLM), unique prompts per stream and
# wave, greedy gens; per wave: aggregate tok/s (sum of emitted tokens / wall),
# TTFT p50/p90/max, inter-token latency (ITL) p50/p95/max over every token of
# every stream, and the count of ITL gaps over 100 ms. The prompt shape is the
# one of conc_client.py (PLEN filler tokens after a unique header).
# Usage: burst_stream_client.py PORT CONC WAVES [TAG] [PLEN]; env GEN=<n> (300),
# IGNORE_EOS=1 (every stream runs to GEN tokens, imp and vLLM both accept it).
import json
import os
import statistics
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
IGNORE_EOS = os.environ.get("IGNORE_EOS", "0") == "1"  # imp/vLLM: run every stream to GEN tokens
# Protection scenario of runtime.prefill_chunk_decode_cap: LATE_PLEN > 0 adds one
# stream with a LATE_PLEN-token prompt that arrives LATE_DELAY seconds into the
# wave while the CONC short streams decode; the wave line then also reports the
# short streams' ITL max / p95 during the ingest and the late stream's TTFT.
LATE_PLEN = int(os.environ.get("LATE_PLEN", "0"))
LATE_DELAY = float(os.environ.get("LATE_DELAY", "2.0"))
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


def one(i, wave, out, plen=None, delay=0.0):
    if delay > 0:
        time.sleep(delay)
    prompt = (f"[{TAG}-w{wave}-r{i}] Explain topic {i * 7 + wave}: how a {i}-way set-associative "
              f"CPU cache interacts with a TLB during a page-crossing load.")
    n_fill = PLEN if plen is None else plen
    if n_fill > 0:
        prompt = f"[{TAG}-w{wave}-r{i}] " + filler(n_fill) + prompt
    body = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": GEN,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **({"ignore_eos": True} if IGNORE_EOS else {}),
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    stamps = []
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    j = json.loads(payload)
                except ValueError:
                    continue
                if j.get("usage"):
                    usage = j["usage"]
                ch = j.get("choices") or []
                if ch and ch[0].get("text"):
                    stamps.append(time.perf_counter())
    except Exception as e:  # noqa: BLE001
        print(f"ERR r{i}: {e}", file=sys.stderr)
    out[i] = (t0, stamps, usage)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round((len(xs) - 1) * p)))
    return xs[k]


for wave in range(WAVES):
    out = {}
    threads = [threading.Thread(target=one, args=(i, wave, out)) for i in range(CONC)]
    if LATE_PLEN > 0:
        threads.append(threading.Thread(target=one, args=(CONC, wave, out, LATE_PLEN, LATE_DELAY)))
    t_wave = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_wave
    ttfts, itls, toks, ptoks = [], [], 0, []
    for idx in range(CONC):
        t0, stamps, usage = out[idx]
        if stamps:
            ttfts.append(stamps[0] - t0)
            itls.extend(b - a for a, b in zip(stamps, stamps[1:]))
        toks += usage.get("completion_tokens", len(stamps)) if usage else len(stamps)
        if usage:
            ptoks.append(usage.get("prompt_tokens", 0))
    slow = sum(1 for g in itls if g > 0.1)
    late = ""
    if LATE_PLEN > 0 and CONC in out:
        t0l, st_l, u_l = out[CONC]
        # ITL of the SHORT streams while the late prompt was being ingested
        # (from its arrival to its first token).
        win_lo, win_hi = t0l, (st_l[0] if st_l else t0l)
        in_win = []
        for i in range(CONC):
            t0i, sti, _ = out[i]
            in_win.extend(b - a for a, b in zip(sti, sti[1:]) if win_lo <= b <= win_hi)
        late = (f" | LATE {u_l.get('prompt_tokens', 0)} tok: TTFT {((st_l[0] - t0l) * 1e3) if st_l else 0:.0f} ms,"
                f" short-stream ITL during ingest p95 {pct(in_win, 0.95) * 1e3:.1f} max"
                f" {max(in_win) * 1e3 if in_win else 0:.0f} ms over {len(in_win)} tokens")
    print(f"wave{wave}: {toks} tok in {wall:.2f}s = {toks / wall:.1f} tok/s aggregate"
          f" | prompt {statistics.mean(ptoks) if ptoks else 0:.0f} tok avg"
          f" | TTFT p50 {pct(ttfts, 0.5) * 1e3:.0f} p90 {pct(ttfts, 0.9) * 1e3:.0f}"
          f" max {max(ttfts) * 1e3 if ttfts else 0:.0f} ms"
          f" | ITL p50 {pct(itls, 0.5) * 1e3:.1f} p95 {pct(itls, 0.95) * 1e3:.1f}"
          f" max {max(itls) * 1e3 if itls else 0:.0f} ms"
          f" | gaps>100ms {slow}/{len(itls)}{late}", flush=True)
