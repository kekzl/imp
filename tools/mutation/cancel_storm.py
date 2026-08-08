#!/usr/bin/env python3
"""Cancel-storm probe: does aborting requests mid-flight change the survivors?

The invariant: greedy generation is a pure function of its prompt. A request
that runs alongside 45 others which are cancelled mid-decode must produce the
same bytes as the same request run by itself. If it does not, cancellation is
leaking state across sequences — the shape of #1044/#1045, where a stale prefix
hash silently corrupted another request's KV.

Nothing in the suite covers this: `tests/api/test_concurrency.py` runs against
the mock in CI (#1302), and the C++ scheduler tests exercise `HandlesCancel` at
the bookkeeping level, never end to end under load.

Phases:
  A  each survivor prompt alone, greedy      -> reference bytes
  B  survivors + N victims launched together, victims aborted mid-stream
  C  compare survivors against their references, byte for byte

Also probes the prefix-cache half-eviction case: submit, cancel, immediately
resubmit the same prompt, and check the resubmission matches its reference.

Usage: cancel_storm.py --base http://127.0.0.1:8099 --model <id> [--victims 45]
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time

import httpx

SURVIVORS = [
    "List the first five prime numbers.",
    "What is the capital of France?",
    "Name three primary colours.",
    "Write the word banana five times.",
    "Count from one to seven.",
]


def stream_once(base, model, prompt, max_tokens, abort_after=None, timeout=120):
    """Stream a completion. If abort_after is set, drop the connection after
    that many seconds and return None."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 1234,
        "stream": True,
    }
    out = []
    t0 = time.monotonic()
    try:
        with httpx.Client(base_url=base, timeout=timeout) as c:
            with c.stream("POST", "/v1/chat/completions", json=body) as r:
                for line in r.iter_lines():
                    if abort_after is not None and time.monotonic() - t0 > abort_after:
                        return None  # closes the connection on the way out
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    delta = (json.loads(payload)["choices"][0].get("delta") or {})
                    out.append(delta.get("content") or "")
    except Exception:  # noqa: BLE001 — an aborted stream is the point
        return None
    return "".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--victims", type=int, default=45)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--no-cancel", action="store_true",
                    help="let the victims run to completion — isolates concurrency from cancellation")
    args = ap.parse_args()

    print("phase A: reference runs, one at a time")
    ref = {}
    for p in SURVIVORS:
        ref[p] = stream_once(args.base, args.model, p, args.max_tokens)
        assert ref[p], f"reference run produced nothing for {p!r}"
        print(f"  {p[:34]:<34} {len(ref[p]):4d} chars")

    bad = 0
    for rnd in range(1, args.rounds + 1):
        print(f"\nphase B round {rnd}: {len(SURVIVORS)} survivors + {args.victims} victims"
              f"{' (no cancel)' if args.no_cancel else ''}")
        with cf.ThreadPoolExecutor(max_workers=args.victims + len(SURVIVORS)) as ex:
            futs = {}
            for i in range(args.victims):
                futs[ex.submit(stream_once, args.base, args.model,
                               f"Tell me a long story about topic number {i}.",
                               512, None if args.no_cancel else 0.35)] = ("victim", i)
            for p in SURVIVORS:
                futs[ex.submit(stream_once, args.base, args.model, p,
                               args.max_tokens)] = ("survivor", p)

            got = {}
            for f in cf.as_completed(futs):
                kind, key = futs[f]
                if kind == "survivor":
                    got[key] = f.result()

        for p in SURVIVORS:
            g = got.get(p)
            if g != ref[p]:
                bad += 1
                print(f"  DIVERGED {p!r}\n    alone: {ref[p]!r}\n    storm: {g!r}")
            else:
                print(f"  ok       {p[:34]:<34} {len(g or ''):4d} chars")

    print("\nphase C: cancel then immediately resubmit the same prompt")
    for p in SURVIVORS[:3]:
        stream_once(args.base, args.model, p, 512, abort_after=0.25)
        again = stream_once(args.base, args.model, p, args.max_tokens)
        if again != ref[p]:
            bad += 1
            print(f"  DIVERGED after cancel+resubmit {p!r}\n    alone: {ref[p]!r}\n    after: {again!r}")
        else:
            print(f"  ok       {p[:34]:<34} (cancel -> resubmit matches)")

    print(f"\n{bad} divergence(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
