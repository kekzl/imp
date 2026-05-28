#!/usr/bin/env python3
"""Concurrent load harness for the imp HTTP server.

Measures aggregate decode throughput, TTFT and end-to-end latency percentiles
across concurrency levels — exposes whether continuous batching scales and where
the server serializes. Stdlib only (urllib + threads) so it runs anywhere.

Usage:
  python3 load_test.py --url http://localhost:18080 --api-key KEY \
      --levels 1,4,8,16,32 --requests-per-level 64 --max-tokens 128
"""
import argparse, json, statistics, threading, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

PROMPTS = [
    "Explain how a CPU cache works in three sentences.",
    "Write a short paragraph about the ocean.",
    "List five prime numbers and explain what makes them prime.",
    "Describe the water cycle briefly.",
    "What is the capital of Japan and one fact about it?",
    "Summarize how photosynthesis works.",
    "Give a short definition of entropy in physics.",
    "Explain recursion with a tiny example.",
]


def one_request(url, api_key, prompt, max_tokens, stream):
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": stream,
    }
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url + "/v1/chat/completions", data=data, headers=headers)
    t0 = time.perf_counter()
    ttft = None
    completion_tokens = 0
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            if stream:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    try:
                        j = json.loads(payload)
                        delta = j["choices"][0].get("delta", {})
                        if delta.get("content"):
                            completion_tokens += 1
                    except Exception:
                        pass
                t_end = time.perf_counter()
            else:
                j = json.loads(resp.read().decode())
                t_end = time.perf_counter()
                completion_tokens = j.get("usage", {}).get("completion_tokens", 0)
                ttft = t_end - t0
        return {"ok": True, "latency": t_end - t0, "ttft": ttft, "tokens": completion_tokens}
    except Exception as e:
        return {"ok": False, "err": str(e), "latency": time.perf_counter() - t0}


def run_level(url, api_key, concurrency, n_requests, max_tokens, stream):
    results = []
    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [
            ex.submit(one_request, url, api_key, PROMPTS[i % len(PROMPTS)], max_tokens, stream)
            for i in range(n_requests)
        ]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t_start
    ok = [r for r in results if r["ok"]]
    errs = [r for r in results if not r["ok"]]
    tot_tokens = sum(r["tokens"] for r in ok)
    lats = sorted(r["latency"] for r in ok) if ok else [0.0]
    ttfts = sorted(r["ttft"] for r in ok if r["ttft"] is not None)

    def pct(xs, p):
        if not xs:
            return 0.0
        k = max(0, min(len(xs) - 1, int(round((p / 100.0) * (len(xs) - 1)))))
        return xs[k]

    return {
        "concurrency": concurrency,
        "n": n_requests,
        "ok": len(ok),
        "err": len(errs),
        "wall_s": wall,
        "agg_tok_s": tot_tokens / wall if wall > 0 else 0,
        "req_s": len(ok) / wall if wall > 0 else 0,
        "lat_p50": pct(lats, 50),
        "lat_p99": pct(lats, 99),
        "ttft_p50": pct(ttfts, 50),
        "ttft_p99": pct(ttfts, 99),
        "sample_err": errs[0]["err"] if errs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:18080")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--levels", default="1,4,8,16,32")
    ap.add_argument("--requests-per-level", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--warmup", type=int, default=4)
    args = ap.parse_args()

    if args.warmup:
        for i in range(args.warmup):
            one_request(args.url, args.api_key, PROMPTS[i % len(PROMPTS)], 16, args.stream)

    levels = [int(x) for x in args.levels.split(",")]
    print(f"{'conc':>5} {'ok/err':>8} {'wall_s':>8} {'agg_tok/s':>10} {'req/s':>7} "
          f"{'lat_p50':>8} {'lat_p99':>8} {'ttft_p50':>9} {'ttft_p99':>9}")
    rows = []
    for c in levels:
        r = run_level(args.url, args.api_key, c, args.requests_per_level, args.max_tokens, args.stream)
        rows.append(r)
        print(f"{r['concurrency']:>5} {str(r['ok'])+'/'+str(r['err']):>8} {r['wall_s']:>8.2f} "
              f"{r['agg_tok_s']:>10.1f} {r['req_s']:>7.2f} {r['lat_p50']:>8.3f} {r['lat_p99']:>8.3f} "
              f"{r['ttft_p50']:>9.3f} {r['ttft_p99']:>9.3f}")
        if r["sample_err"]:
            print(f"      sample error: {r['sample_err']}")
        time.sleep(3)
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
