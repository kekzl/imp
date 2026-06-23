#!/usr/bin/env python3
"""Agent-shaped benchmark harness for imp-server.

The standard pp/tg protocol measures raw prefill/decode. Agents feel something
different: time-to-first-token and inter-token latency *under concurrency*, with
a large static prefix (system + tool defs) reused across turns via prompt
caching, plus a short dynamic suffix.

This harness measures what agents feel:
  - TTFT p50/p90/p99 at 1 / 4 / 16 / 64 concurrent streams
  - ITL (inter-token latency) p50/p90/p99 under the same concurrency
  - Warm-vs-cold cache TTFT for a shared static prefix (cache_control pinned)

Stdlib only (urllib + threading) so it runs on the clean host against the
dockerised server. Streaming SSE is parsed to time the *first* token precisely.

Example:
  python3 tools/agent_bench.py --url http://localhost:8080 \
      --model Qwen3-4B-Instruct-2507-Q8_0.gguf --concurrency 1,4,16,64
"""
import argparse
import json
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# A chunky static prefix standing in for an agent's system prompt + tool
# definitions. Repeated to ~2-3K tokens so prompt caching has something to bite.
TOOL_BLOCK = """You are a coding agent with access to the following tools.
- read_file(path: string): returns the file contents.
- write_file(path: string, content: string): overwrites the file.
- run_shell(cmd: string): runs a shell command, returns stdout/stderr.
- search(query: string, k: int): returns the top-k matching code spans.
- list_dir(path: string): lists directory entries with sizes and mtimes.
Always think step by step, prefer minimal diffs, and never fabricate file paths.
Follow the project conventions exactly and keep changes surgical and reviewable.
"""


def static_prefix(repeat: int) -> str:
    return ("SYSTEM INSTRUCTIONS (cached static prefix)\n" + TOOL_BLOCK * repeat +
            "\nEnd of system instructions. Answer user turns concisely.\n")


def percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] * (1 - frac) + xs[hi] * frac


def stream_request(url, model, system, user, max_tokens, cache_prompt):
    """Fire one streaming chat request. Returns (ttft_s, itl_mean_s, e2e_s, n_tok)."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": True,
        "cache_prompt": cache_prompt,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = None
    n_tok = 0
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                n_tok += 1
    t_end = time.perf_counter()
    if t_first is None:
        return None
    ttft = t_first - t0
    e2e = t_end - t0
    itl = (t_end - t_first) / max(n_tok - 1, 1)
    return ttft, itl, e2e, n_tok


def run_level(url, model, system, user, max_tokens, concurrency, n_requests):
    results = []
    lock = threading.Lock()

    def worker(i):
        try:
            r = stream_request(url, model, system, f"{user} (variant {i})", max_tokens, True)
            if r:
                with lock:
                    results.append(r)
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"  request {i} failed: {e}\n")

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(worker, range(n_requests)))
    wall = time.perf_counter() - t0
    return results, wall


def fmt_ms(xs):
    return (f"p50={percentile(xs, 0.5)*1000:7.1f}  p90={percentile(xs, 0.9)*1000:7.1f}  "
            f"p99={percentile(xs, 0.99)*1000:7.1f}  max={max(xs)*1000:7.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--requests-per-level", type=int, default=0,
                    help="0 = 4x concurrency (so every level does real fan-out)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prefix-repeat", type=int, default=24,
                    help="repeats of the tool block in the static prefix (~tokens)")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    system = static_prefix(args.prefix_repeat)
    approx_prefix_tok = len(system) // 4
    print(f"# imp agent-bench  model={args.model}")
    print(f"# static prefix ~{approx_prefix_tok} tokens (cache_prompt pinned), "
          f"max_tokens={args.max_tokens}\n")

    # Warm the model + clocks + prime the prefix cache (discarded).
    sys.stderr.write("warming up (clocks + prefix cache)...\n")
    for _ in range(2):
        stream_request(args.url, args.model, system, "Warmup turn.", args.max_tokens, True)

    # --- Cold vs warm cache TTFT (single stream, unique cold prefix) ---
    # Nonce at the START so the entire block-hash chain is fresh (block hashes
    # are parent-chained; a trailing marker would leave the prefix blocks warm
    # from the warmup above and understate the cache win).
    cold_system = f"FRESH-COLD-{time.time_ns()}\n" + system
    cold = stream_request(args.url, args.model, cold_system, "First turn.", args.max_tokens, True)
    warm = stream_request(args.url, args.model, cold_system, "Second turn.", args.max_tokens, True)
    print("## Prefix cache (single stream)")
    if cold and warm:
        print(f"  cold TTFT = {cold[0]*1000:7.1f} ms   warm TTFT = {warm[0]*1000:7.1f} ms   "
              f"speedup = {cold[0]/warm[0]:4.2f}x\n")

    # --- TTFT / ITL under concurrency ---
    print("## TTFT / ITL under concurrency (ms)")
    for c in levels:
        n = args.requests_per_level or max(c * 4, 8)
        results, wall = run_level(args.url, args.model, system, "Summarize the tool list.",
                                  args.max_tokens, c, n)
        if not results:
            print(f"  c={c:3d}: no successful requests")
            continue
        ttfts = [r[0] for r in results]
        itls = [r[1] for r in results]
        thru = len(results) / wall
        print(f"  c={c:3d} (n={len(results):3d})  TTFT {fmt_ms(ttfts)}")
        print(f"            {'':17s}ITL  {fmt_ms(itls)}   throughput={thru:5.1f} req/s")
    print()


if __name__ == "__main__":
    main()
