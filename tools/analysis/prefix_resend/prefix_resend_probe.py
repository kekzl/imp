"""Prefix-cache resend probe (#2152): turn 2 of a fixed conversation, sent N times against one
server. Send 0 restores the turn-1 prefix, sends 1.. restore the whole turn-2 prompt, so each send
prefills a different tail. Greedy output must be identical across sends.

Turn-1 answers are fixed files (water_cycle_*.txt), so the prompt does not depend on the model's own
turn-1 output. Stdlib only. Exit 0 = every conversation identical and every later send restored.

usage: prefix_resend_probe.py --url http://localhost:8080 [--sends 3] [a1.txt ...]
"""
import argparse
import json
import pathlib
import sys
import urllib.request

SYS = ("You are a precise assistant. Answer in full sentences and explain your reasoning "
       "step by step so the reader can follow along without prior knowledge of the topic. "
       "Prefer concrete physical mechanisms over vague generalities, and name the quantities involved.")
Q1 = "Describe the water cycle in about 150 words."
Q2 = "Now explain which step is most affected by climate change and why."


def post(url, path, body=None):
    req = urllib.request.Request(url + path, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))


def chat(url, model, msgs, n):
    r = post(url, "/v1/chat/completions", {"model": model, "messages": msgs, "max_tokens": n, "temperature": 0,
                                           "chat_template_kwargs": {"enable_thinking": False}})
    u = r.get("usage") or {}
    return (r["choices"][0]["message"]["content"], u.get("prompt_tokens"),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--sends", type=int, default=3)
    ap.add_argument("--no-warm", action="store_true", help="skip the turn-1 request (no turn-1 prefix blocks)")
    ap.add_argument("a1", nargs="*", default=sorted(str(p) for p in pathlib.Path(__file__).parent.glob("water_cycle_*.txt")))
    args = ap.parse_args()
    model = post(args.url, "/v1/models")["data"][0]["id"]
    bad = 0
    for a1_file in args.a1:
        base = [{"role": "system", "content": SYS}, {"role": "user", "content": Q1}]
        if not args.no_warm:
            chat(args.url, model, base, 250)  # caches the turn-1 prefix, as a real conversation does
        msgs = base + [{"role": "assistant", "content": pathlib.Path(a1_file).read_text()},
                       {"role": "user", "content": Q2}]
        outs, info = [], []
        for _ in range(args.sends):
            text, prompt, cached = chat(args.url, model, msgs, 120)
            outs.append(text)
            info.append((prompt, cached))
        same = all(o == outs[0] for o in outs) and bool(outs[0].strip())
        restored = all(c > 0 for _, c in info[1:])
        ok = same and restored
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} a1={pathlib.Path(a1_file).name} "
              + " | ".join(f"p={p} c={c}" for p, c in info) + f" identical={same} restored={restored}", flush=True)
        if not same:
            for k, o in enumerate(outs):
                print(f"  send{k}: {o[:80]!r}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
