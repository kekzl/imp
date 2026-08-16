#!/usr/bin/env python3
"""Deep multi-turn probe: one growing session, checked reply by reply.

`degen_suite.py`'s multi-turn category runs three turns with one recall, which
catches turn-2 garble. It cannot catch what only appears once a session has
grown: a recurrent state that decays, a recall that stops reaching turn 1, or a
thinking model that spends its whole token budget before answering.

This drives ONE session, resent in full each turn, mixes topics deliberately so
a stuck state shows as an off-topic reply, and asks for facts planted at the
start. `--filler N` inserts N neutral turns before the final recalls, so the
same facts have to survive an arbitrarily long context.

Every reply is checked for the failure shapes in the `check-degeneration` skill:
empty content, early abort, verbatim repetition, n-gram loops, and a script
change mid-stream. On failure it also reports `finish_reason` and how much went
to the reasoning channel, because those separate a real defect from a budget
that thinking consumed (see TROUBLESHOOTING.md).

Found with it, 2026-08-16 on Qwen3.8-27B: at `--max-tokens 260` a 74-turn
session produced empty replies and one single-token non-Latin answer; at 600 the
same session is 74/74 clean. The same conversation degenerates on vLLM too, so
it is the model, not the engine.

stdlib only, same as degen_suite. Exit 0 = clean, 1 = failures.

  python3 tools/analysis/multiturn_deep.py --url http://localhost:8080 \
      --model <id> --filler 60 --max-tokens 600
"""
import argparse
import json
import re
import sys
import time
import urllib.request

FACTS = {
    "project": ("Kestrel", "my project is called Kestrel"),
    "gpu": ("RTX 5090", "the target GPU is an RTX 5090"),
    "port": ("8347", "the service listens on port 8347"),
}

# (prompt, kind, expected_substring)
TURNS = [
    ("Remember these three facts for later: my project is called Kestrel, "
     "the target GPU is an RTX 5090, and the service listens on port 8347. "
     "Just confirm you have them.", "setup", None),
    ("Explain why the sky is blue, in three sentences.", "topic", "scatter"),
    ("What is my project called? Answer with the name only.", "recall", "Kestrel"),
    ("Write a Python function that returns the nth Fibonacci number.", "code", "def "),
    ("Now switch topic completely: name three staple crops and one region "
     "each where they dominate.", "topic", None),
    ("Explain the difference between a mutex and a semaphore, briefly.", "topic", None),
    ("Which GPU did I say I was targeting? Answer with the model only.", "recall", "5090"),
    ("Write two sentences of a product description for a waterproof backpack.", "topic", None),
    ("What is 17 * 23? Give the number only.", "math", "391"),
    ("Summarise, in one sentence, what Rayleigh scattering explains.", "topic", "scatter"),
    ("List the first five prime numbers.", "math", "11"),
    ("Which port did I mention at the very beginning? Number only.", "recall", "8347"),
    ("Name the three facts I asked you to remember, as a short list.", "recall", "Kestrel"),
    ("In one sentence: what have we talked about in this conversation?", "wrapup", None),
]


def post(url, model, messages, max_tokens, timeout):
    body = json.dumps({
        "model": model, "messages": messages, "temperature": 0,
        "max_tokens": max_tokens, "stream": False,
    }).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return payload, time.time() - t0


def degenerate(text, min_words=3):
    """Failure shapes from the check-degeneration skill. Returns a reason or None.

    min_words is per turn: a prompt that says "answer with the name only" is
    satisfied by one word, and the skill's >=10-token rule explicitly exempts
    single-word factual prompts. Applying it anyway reports a correct answer as
    an early abort, which is how this probe first read a clean run as 4 failures.
    """
    if not text or not text.strip():
        return "empty content"
    words = text.split()
    if len(words) < min_words:
        return f"early abort ({len(words)} words, expected >={min_words})"
    for i in range(len(words) - 4):
        if len(set(words[i:i + 5])) == 1:
            return f"token repeated 5x: {words[i]!r}"
    grams = {}
    for i in range(len(words) - 2):
        g = " ".join(words[i:i + 3]).lower()
        grams[g] = grams.get(g, 0) + 1
        if grams[g] > 3:
            return f"3-gram repeated {grams[g]}x: {g!r}"
    # A script change mid-stream is the "structurally valid garbage" class.
    if re.search(r"[一-鿿Ѐ-ӿ؀-ۿ]", text):
        return "non-latin script in an English reply"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-tokens", type=int, default=220)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--filler", type=int, default=0,
                    help="extra topic turns inserted before the final recalls, so "
                         "the planted facts are recalled across a much longer session")
    a = ap.parse_args()

    turns = list(TURNS)
    if a.filler:
        # Insert neutral topic turns before the last three (the deep recalls),
        # so the facts from turn 1 have to survive a long, growing context.
        fill = [(f"Explain concept number {i} in two sentences: "
                 + ["caching", "backpressure", "idempotency", "sharding", "quorum",
                    "leader election", "write-ahead logging", "vector clocks",
                    "bloom filters", "consistent hashing", "circuit breakers",
                    "exponential backoff", "copy-on-write", "memory barriers",
                    "tail latency", "head-of-line blocking"][i % 16],
                 "topic", None) for i in range(a.filler)]
        turns = turns[:-3] + fill + turns[-3:]

    messages = []
    failures = []
    print(f"{'turn':>4}  {'kind':<7} {'prompt_tok':>10} {'gen':>5} {'s':>6}  verdict")
    for i, (prompt, kind, expect) in enumerate(turns, 1):
        messages.append({"role": "user", "content": prompt})
        payload, dt = post(a.url, a.model, messages, a.max_tokens, a.timeout)
        choice = payload["choices"][0]
        msg = choice["message"]
        text = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or "").strip()
        finish = choice.get("finish_reason")
        usage = payload.get("usage", {})
        messages.append({"role": "assistant", "content": text or "(empty)"})

        problems = []
        d = degenerate(text, 1 if kind in ("recall", "math") else 10)
        if d:
            problems.append(d)
        if expect and expect.lower() not in text.lower():
            problems.append(f"missing {expect!r}")
        verdict = "ok" if not problems else "FAIL: " + "; ".join(problems)
        if problems:
            # An empty content field is ambiguous: a real defect, or the whole
            # token budget spent in the reasoning channel. Record what separates
            # the two instead of guessing later.
            detail = (f"finish={finish} reasoning_chars={len(reasoning)} "
                      f"completion_tok={usage.get('completion_tokens')}")
            failures.append((i, kind, verdict + " | " + detail,
                             (text or reasoning)[-160:]))
        print(f"{i:>4}  {kind:<7} {usage.get('prompt_tokens', 0):>10} "
              f"{usage.get('completion_tokens', 0):>5} {dt:>6.1f}  {verdict}")

    print()
    if failures:
        print(f"MULTITURN: {len(failures)} of {len(turns)} turns FAILED")
        for i, kind, verdict, snippet in failures:
            print(f"  turn {i} ({kind}): {verdict}\n    {snippet!r}")
        return 1
    print(f"MULTITURN: all {len(turns)} turns clean, "
          f"context grew to {messages and 'see table'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
