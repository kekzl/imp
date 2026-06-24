#!/usr/bin/env python3
"""Agent-shaped benchmark harness for imp-server.

The standard pp/tg protocol measures raw prefill/decode. Agents feel something
different: time-to-first-token and inter-token latency *under concurrency*, with
a large static prefix (system + tool defs) reused across turns via prompt
caching, plus a short dynamic suffix.

CONCURRENCY VIA OS PROCESSES (curl), NOT Python threads. An earlier threaded
version under-reported badly: 16+ concurrent SSE streams parsed in Python threads
serialize on the GIL, so the measured TTFT was the *client's* parsing throughput,
not the server's. Driving each request with a separate `curl` process (true
parallelism, no shared interpreter) is the only honest way to load the server
from a single box. TTFT is measured as curl's time_total for a max_tokens=1
request (= prefill + one decode = time to the first content token).

Example:
  python3 tools/agent_bench.py --url http://localhost:8080 \
      --model Qwen3-4B-Instruct-2507-Q8_0.gguf --concurrency 1,4,16,64
"""
import argparse
import json
import subprocess
import sys
import time

TOOL_BLOCK = ("Tool spec: read_file write_file run_shell search list_dir; "
              "follow conventions, minimal diffs, never fabricate paths. ")


def static_prefix(repeat: int) -> str:
    return "SYSTEM INSTRUCTIONS (cached static prefix)\n" + TOOL_BLOCK * repeat + "\n"


def percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def body(model, system, user, max_tokens, cache_prompt, stream=False):
    return json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": stream,
        "cache_prompt": cache_prompt,
    })


def curl_popen(url, payload, write_fmt="%{time_total}"):
    """Launch a curl process (non-blocking). Returns the Popen handle."""
    return subprocess.Popen(
        ["curl", "-s", "-o", "/dev/null", "-w", write_fmt + "\n",
         url + "/v1/chat/completions", "-H", "Content-Type: application/json",
         "-d", payload],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def curl_one(url, payload, write_fmt="%{time_total}"):
    p = curl_popen(url, payload, write_fmt)
    out, _ = p.communicate(timeout=180)
    try:
        return float(out.strip())
    except (ValueError, AttributeError):
        return None


def run_concurrent(url, model, system, c, max_tokens, prefix_label):
    """Fire c requests as concurrent curl processes; return list of time_total (s)."""
    procs = [curl_popen(url, body(model, system, f"go {prefix_label}-{i}", max_tokens, True))
             for i in range(c)]
    out = []
    for p in procs:
        try:
            s, _ = p.communicate(timeout=180)
            out.append(float(s.strip()))
        except Exception:  # noqa: BLE001
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--concurrency", default="1,4,16,64")
    ap.add_argument("--itl-tokens", type=int, default=64,
                    help="token count for the ITL measurement run")
    ap.add_argument("--prefix-repeat", type=int, default=180,
                    help="repeats of the tool block in the static prefix (~tokens)")
    args = ap.parse_args()

    levels = [int(x) for x in args.concurrency.split(",")]
    system = static_prefix(args.prefix_repeat)
    print(f"# imp agent-bench  model={args.model}  (curl-process concurrency)")
    print(f"# static prefix ~{len(system)//4} tokens (cache_prompt pinned)\n")

    # Warm model + clocks + prime the shared prefix cache (discarded).
    sys.stderr.write("warming up...\n")
    for _ in range(2):
        curl_one(args.url, body(args.model, system, "warmup", 8, True))

    # --- Cold vs warm cache TTFT (max_tokens=1 = prefill + 1 token) ---
    cold_system = f"FRESH-COLD-{time.time_ns()}\n" + system
    cold = curl_one(args.url, body(args.model, cold_system, "first", 1, True))
    warm = curl_one(args.url, body(args.model, cold_system, "second", 1, True))
    print("## Prefix cache (single request, TTFT = max_tokens=1 time_total)")
    if cold and warm:
        print(f"  cold = {cold*1000:7.1f} ms   warm = {warm*1000:7.1f} ms   "
              f"speedup = {cold/warm:4.2f}x\n")

    # --- TTFT under concurrency (max_tokens=1) ---
    print("## TTFT under concurrency (ms) — first content token, warm shared prefix")
    ttft1 = {}
    for c in levels:
        ts = run_concurrent(args.url, args.model, system, c, 1, f"t{c}")
        if not ts:
            print(f"  c={c:3d}: no successful requests")
            continue
        ttft1[c] = percentile(ts, 0.5)
        print(f"  c={c:3d} (n={len(ts):3d})  p50={percentile(ts,0.5)*1000:7.1f}  "
              f"p90={percentile(ts,0.9)*1000:7.1f}  p99={percentile(ts,0.99)*1000:7.1f}  "
              f"max={max(ts)*1000:7.1f}")

    # --- ITL under concurrency: (time_total(K) - TTFT) / (K-1) ---
    K = args.itl_tokens
    print(f"\n## Mean ITL under concurrency (ms) — from max_tokens={K} runs")
    for c in levels:
        ts = run_concurrent(args.url, args.model, system, c, K, f"i{c}")
        if not ts or c not in ttft1:
            continue
        itls = [(t - ttft1[c]) / (K - 1) for t in ts if t > ttft1[c]]
        if itls:
            print(f"  c={c:3d}  ITL p50={percentile(itls,0.5)*1000:6.1f}  "
                  f"p90={percentile(itls,0.9)*1000:6.1f}  "
                  f"aggregate={c/percentile(ts,0.5):6.1f} tok/s")
    print()


if __name__ == "__main__":
    main()
