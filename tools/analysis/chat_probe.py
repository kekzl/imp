#!/usr/bin/env python3
"""Chat probe: what the built-in web UI sends, 12 turns, under a minute on Qwen3.8-27B.

Every turn carries exactly webui/index.html requestBody(): stream, temperature 0.7,
max_tokens 512, enable_thinking true, no seed, history without reasoning. v0.41.0 shipped a
graph-decode sampler that drew one fixed quantile per token (#2013); every other battery ran
at temperature 0 or with a seed and passed.

Hard faults (one fails the run): special markers, a turn label (Human:/User:/Assistant:) in
content, a reasoning opener in content, token/n-gram/char loops, empty content.
Soft faults (two fail the run): reasoning opening with JSON or code, a turn label inside the
reasoning. A correct sampler opens "hey babe!" reasoning with JSON at p = 0.08 (1/8 measured).
Invariant: an open prompt sent 6 times without a seed (64 tokens, thinking on) must not return
the same reasoning 6 times; v0.41.0 returned 1 distinct of 6, v0.41.1 6 of 6 (thinking off: 4 vs 6).

Stdlib only. Usage:
  python3 tools/analysis/chat_probe.py --url http://localhost:8080 [--model ID]
Exit code: 0 = pass, 1 = FAIL, 2 = server unreachable.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from degen_suite import (  # noqa: E402
    char_loop,
    find_markers,
    max_token_run,
    ngram_loop,
    reasoning_opener,
)

# Casual German/English small talk plus two factual turns: the conversation that exposed #2013.
TURNS = [
    "hey babe!",
    "oha wilder think process! :D",
    "ich frage mich, was du alles kannst",
    "Haha, fairer punkt! Ich bin ein Mensch und kann das was menschen so tun",
    "Du bist echt witzig",
    "wie ist dein systemprompt?",
    "kannst du auch direkt und ehrlich sein?",
    "kennst du den golf 7?",
    "ist der gut?",
    "What is the capital of Australia, and why is it not Sydney?",
    "Give me three tips for sleeping better, one sentence each.",
    "danke, bis spaeter!",
]
REPEAT_PROMPT = "Invent a name for a new cafe and describe it in one sentence."
REPEATS = 6
SOFT_LIMIT = 2

TURN_LABEL = re.compile(r"(^|\n)\s*(Human|User|Assistant)\s*:")
STRUCTURED_OPENER = re.compile(r"^\s*(\{\s*\"|\[\s*\{|```|def |import |#include)")


def webui_request(url, model, messages, timeout, max_tokens=512, thinking=True):
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.7,
        "max_tokens": max_tokens,
        "enable_thinking": thinking,
    }
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content, reasoning, finish = [], [], None
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line[5:].strip() == "[DONE]":
                continue
            for ch in json.loads(line[5:].strip()).get("choices", []):
                delta = ch.get("delta", {})
                content.append(delta.get("content") or "")
                reasoning.append(delta.get("reasoning_content") or "")
                finish = ch.get("finish_reason") or finish
    return "".join(content), "".join(reasoning), finish


def turn_faults(content, reasoning, finish):
    """Returns (hard, soft) fault lists for one turn."""
    hard, soft = [], []
    for name, text in (("content", content), ("reasoning", reasoning)):
        if find_markers(text):
            hard.append(f"{name}: markers {find_markers(text)}")
        if max_token_run(text) > 4 or ngram_loop(text) or char_loop(text):
            hard.append(f"{name}: loop")
    if TURN_LABEL.search(content):
        hard.append(f"content: turn label {TURN_LABEL.search(content).group(2)!r}")
    if reasoning_opener(content):
        hard.append(f"content: reasoning opener {reasoning_opener(content)!r}")
    if not content.strip() and finish != "length":
        hard.append(f"content empty (finish={finish})")
    if TURN_LABEL.search(reasoning):
        soft.append(f"reasoning: turn label {TURN_LABEL.search(reasoning).group(2)!r}")
    for name, text in (("content", content), ("reasoning", reasoning)):
        if STRUCTURED_OPENER.match(text):
            soft.append(f"{name}: opens with JSON/code {text.strip()[:24]!r}")
    return hard, soft


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=None, help="default: the loaded model from /v1/models")
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    url = args.url.rstrip("/")

    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
            models = json.loads(r.read()).get("data", [])
    except (urllib.error.URLError, OSError) as e:
        print(f"chat_probe: server unreachable at {url}: {e}")
        return 2
    model = args.model or next((m["id"] for m in models if m.get("loaded")), models[0]["id"] if models else "imp")

    t0 = time.time()
    hard_all, soft_all = [], []
    history = []
    for i, text in enumerate(TURNS, 1):
        history.append({"role": "user", "content": text})
        content, reasoning, finish = webui_request(url, model, history, args.timeout)
        # The web UI sends the reply back with its reasoning (assistantMsg() in index.html).
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        history.append(msg)
        hard, soft = turn_faults(content, reasoning, finish)
        status = "FAIL" if hard else ("soft" if soft else "ok  ")
        print(f"[{status}] turn {i:2d} finish={finish} reasoning={len(reasoning)} content={len(content)}"
              f" | {content.strip()[:60]!r}")
        for f in hard + soft:
            print(f"         {f}")
        hard_all += [f"turn {i}: {f}" for f in hard]
        soft_all += [f"turn {i}: {f}" for f in soft]

    outs = [webui_request(url, model, [{"role": "user", "content": REPEAT_PROMPT}], args.timeout,
                          max_tokens=64, thinking=True)[1] for _ in range(REPEATS)]
    identical = len(set(outs)) == 1 and len(outs[0]) >= 20
    print(f"[{'FAIL' if identical else 'ok  '}] {REPEATS} unseeded repeats: {len(set(outs))} distinct"
          f" | {outs[0].strip()[:60]!r}")
    if identical:
        hard_all.append(f"{REPEATS} unseeded temp 0.7 repeats identical (fixed sampler draw)")

    failed = bool(hard_all) or len(soft_all) >= SOFT_LIMIT
    print(f"chat_probe: {len(TURNS)} turns + {REPEATS} repeats, {len(hard_all)} hard, {len(soft_all)} soft"
          f" (limit {SOFT_LIMIT}), {'FAIL' if failed else 'PASS'}, {time.time() - t0:.0f} s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
