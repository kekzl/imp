#!/usr/bin/env python3
"""Needle-in-a-haystack retrieval gate past 16K (#1022).

Drives a running imp-server: builds a filler haystack to a target token length,
plants a unique needle at a given depth, asks for it, and asserts the answer
carries the needle. A CORRECTNESS gate (retrieval success), independent of
timing — safe to run without a pinned-perf host. Complements the TTFT gates
(which need a verified-healthy host to pin numbers).

Extends the ≤16K coverage to 32K/64K. Needs a server whose max_seq_len admits
the longest prompt (e.g. `--set runtime.max_seq_len=40000` for 32K). On a
sliding-window model this exercises the SWA global layers carrying long-range
recall (`--set kv_cache.swa_sizing=true`).

Usage:
  python3 tools/analysis/niah_check.py --url http://localhost:8080 --model NAME \
      --lengths 16000,32000 --depths 0.1,0.5,0.9
"""
import argparse
import json
import sys
import time
import urllib.request

FILLER = ("The quarterly logistics report notes routine warehouse throughput and "
          "nominal transit times across the regional distribution network. ")


def approx_tokens(s):
    # ~0.75 words/token for English; good enough to size the haystack.
    return int(len(s.split()) / 0.75)


def build_prompt(needle, target_tokens, depth):
    reps = max(1, int(target_tokens / approx_tokens(FILLER)))
    body = [FILLER] * reps
    at = min(len(body), max(0, int(len(body) * depth)))
    body.insert(at, needle + " ")
    haystack = "".join(body)
    q = (" Question: What is the secret vault access code mentioned earlier? "
         "Answer with ONLY the code, nothing else.")
    return haystack + q


def post(url, body):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="16000,32000")
    ap.add_argument("--depths", default="0.1,0.5,0.9")
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",") if x]
    depths = [float(x) for x in args.depths.split(",") if x]

    fails = 0
    total = 0
    t0 = time.monotonic()
    for length in lengths:
        for depth in depths:
            total += 1
            code = f"ZEBRA-{length}-{int(depth*100):02d}"
            needle = f"The secret vault access code is {code}."
            prompt = build_prompt(needle, length, depth)
            # Budget covers a thinking model's reasoning + the short answer;
            # /no_think suppresses it on models that honor the hint (Qwen3).
            body = {"model": args.model, "temperature": 0.0, "max_tokens": 512,
                    "messages": [{"role": "user", "content": prompt + " /no_think"}]}
            try:
                rsp = post(args.url, body)
                ptok = rsp.get("usage", {}).get("prompt_tokens", "?")
                ans = (rsp["choices"][0]["message"].get("content") or "")
                ok = code in ans
            except Exception as e:  # noqa: BLE001 — report any transport/parse failure as a fail
                ptok, ans, ok = "?", f"ERROR {e}", False
            if not ok:
                fails += 1
            mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
            print(f"  [{mark}] len~{length} depth={depth:.1f} ptok={ptok}: "
                  f"want {code} got {ans[:40]!r}")
    dt = time.monotonic() - t0
    print("=" * 60)
    print(f"niah_check: {total} probes, {fails} FAIL ({dt:.0f}s)")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
