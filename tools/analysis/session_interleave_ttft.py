#!/usr/bin/env python3
"""Interleaved multi-session TTFT probe for hybrid prefix caching (#2409, #2485).

N sessions share one system prompt (--shared-words), each adds its own context in turn 1
(--prefix-words), turns run round-robin with the full history resent. Prints TTFT per turn;
compare `cached=` in the server log. stdlib only, exit 0 always.

  python3 tools/analysis/session_interleave_ttft.py --url http://127.0.0.1:8090 \
      --shared-words 2000 --prefix-words 1500 --sessions 2,8 --turns 3 [--print-replies]
"""
import argparse
import hashlib
import json
import random
import statistics
import time
import urllib.request

WORDS = (
    "river stone lantern copper meadow signal harbor engine violet thunder orchard canvas "
    "glacier ribbon falcon marble compass ember willow quartz beacon saddle tundra cobalt"
).split()


def context(seed, n_words):
    rng = random.Random(seed)
    return f"Session {seed} notes: " + " ".join(rng.choice(WORDS) for _ in range(n_words))


def ttft(url, model, messages, max_tokens):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0, "stream": True}
    req = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    first, text = None, []
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])["choices"][0]["delta"]
            piece = d.get("content") or d.get("reasoning_content") or ""
            if piece and first is None:
                first = time.perf_counter() - t0
            text.append(d.get("content") or "")
    return (first if first is not None else time.perf_counter() - t0) * 1e3, "".join(text)


def run(args, n):
    shared = context(7, args.shared_words) if args.shared_words > 0 else ""
    hist = [[{"role": "system", "content": "You are a terse assistant. " + shared}] for _ in range(n)]
    per_turn = {t: [] for t in range(1, args.turns + 1)}
    for t in range(1, args.turns + 1):
        for s in range(n):
            own = context(1000 * n + s, args.prefix_words) + "\n" if t == 1 else ""
            hist[s].append({"role": "user", "content": f"{own}Turn {t}: name three words from the notes. /no_think"})
            ms, reply = ttft(args.url, args.model, hist[s], args.max_tokens)
            hist[s].append({"role": "assistant", "content": reply})
            per_turn[t].append(ms)
            if args.print_replies:
                print(f"reply sessions={n} s={s} t={t} md5={hashlib.md5(reply.encode()).hexdigest()[:12]} {reply!r}")
    for t, v in per_turn.items():
        print(f"sessions={n} turn={t} ttft_ms median={statistics.median(v):.1f} min={min(v):.1f} max={max(v):.1f}")
    return statistics.median(per_turn[2]) if args.turns >= 2 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8090")
    ap.add_argument("--model", default="default")
    ap.add_argument("--sessions", default="2,8")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--prefix-words", type=int, default=3000)
    ap.add_argument("--shared-words", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--print-replies", action="store_true", help="per reply: md5 + text (on/off parity)")
    args = ap.parse_args()
    if args.model == "default":
        with urllib.request.urlopen(args.url + "/v1/models", timeout=30) as r:
            args.model = json.loads(r.read())["data"][0]["id"]
    t2 = {}
    for n in (int(x) for x in args.sessions.split(",")):
        t2[n] = run(args, n)
    ns = sorted(t2)
    if len(ns) >= 2 and t2[ns[0]]:
        print(f"turn2 median ratio sessions={ns[-1]} / sessions={ns[0]}: {t2[ns[-1]] / t2[ns[0]]:.3f}")


if __name__ == "__main__":
    main()
