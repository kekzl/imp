#!/usr/bin/env python3
"""Extended degeneration suite for imp-server — on-demand, server-level.

Probes a RUNNING imp-server through /v1/chat/completions (stream and
non-stream) for the failure classes that keep recurring in production and
that the C-API GTest battery (tests/test_degeneration.cpp) cannot see,
because they live in the server layer: chat template, reasoning/think
separation, channel stripping, stop handling, streaming protocol.

Categories (select with --only / --skip):
  repetition       repetition loops, stuck tokens, high-temp stress
  think-leak       reasoning in `content` instead of `reasoning_content`,
                   think tags in user-visible output, truncated-think spill
  special-tokens   raw template/control markers in content
  adherence        prompt-blindness + gross hallucination (exact-answer tasks)
  long-context     needle echo at ~2-3k tokens (catches RoPE/echo bugs)
  multi-turn       state carry across turns, turn-2 garble
  stream           stream/non-stream consistency, SSE termination

Stdlib-only (urllib) — runs on the clean host, no venv, no container.

Usage:
  python3 tools/analysis/degen_suite.py                       # localhost:8080
  python3 tools/analysis/degen_suite.py --url http://host:8081
  python3 tools/analysis/degen_suite.py --only think-leak,repetition
  python3 tools/analysis/degen_suite.py --quick               # short probes
  python3 tools/analysis/degen_suite.py --json report.json
  python3 tools/analysis/degen_suite.py --skip-deterministic  # e.g. Qwen3.6
                                        (non-deterministic at temp=0 — skips
                                        the stream==non-stream equality check)

Exit code: 0 = all pass, 1 = at least one FAIL, 2 = server unreachable.
"""

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Markers that must NEVER appear in user-visible content. Covers ChatML,
# Gemma channels/turns, Llama instruct, generic pads. (PR #442 regression
# class: turn markers sampled at high temperature.)
SPECIAL_MARKERS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<pad>", "<unk>",
    "<|channel>", "<channel|>", "<start_of_turn>", "<end_of_turn>",
    "[INST]", "[/INST]", "<<SYS>>", "<|user|>", "<|assistant|>",
    "<think>", "</think>",
]

# Meta-reasoning openers: if a *reasoning* model puts these at the start of
# `content`, its think output leaked into the user-visible channel (e.g. the
# truncated-think spill: max_tokens exhausted before </think>, non-stream
# path dumps the whole buffer as content).
REASONING_OPENERS = [
    "the user wants", "the user is asking", "the user asks",
    "let me think", "let's think", "i should respond", "i need to figure",
    "thinking process", "my goal is to", "first, i need to",
    "der nutzer möchte", "der user möchte", "ich soll",
]


class Server:
    def __init__(self, url, model, timeout):
        self.url = url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path, body):
        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        return urllib.request.urlopen(req, timeout=self.timeout)

    def chat(self, messages, max_tokens=256, temperature=0.0, seed=42, **kw):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }
        body.update(kw)
        with self._post("/v1/chat/completions", body) as r:
            data = json.loads(r.read())
        msg = data["choices"][0]["message"]
        return {
            "content": msg.get("content") or "",
            "reasoning": msg.get("reasoning_content") or "",
            "finish": data["choices"][0].get("finish_reason"),
            "raw": data,
        }

    def chat_stream(self, messages, max_tokens=256, temperature=0.0, seed=42, **kw):
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
            "stream": True,
        }
        body.update(kw)
        content, reasoning, finish = [], [], None
        got_done = False
        with self._post("/v1/chat/completions", body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    got_done = True
                    break
                chunk = json.loads(payload)
                for ch in chunk.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
        return {
            "content": "".join(content),
            "reasoning": "".join(reasoning),
            "finish": finish,
            "done": got_done,
        }


# ---------------------------------------------------------------------------
# Detectors

def max_token_run(text):
    """Longest run of one whitespace-separated token repeated consecutively."""
    toks = text.split()
    best = run = 1
    for a, b in zip(toks, toks[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    return best if toks else 0


def ngram_loop(text, n=4, threshold=4):
    """True if any n-gram repeats >= threshold times consecutively."""
    toks = text.split()
    if len(toks) < n * threshold:
        return False
    for i in range(len(toks) - n * threshold + 1):
        gram = toks[i:i + n]
        reps = 1
        j = i + n
        while toks[j:j + n] == gram:
            reps += 1
            j += n
        if reps >= threshold:
            return True
    return False


def char_loop(text, min_len=6, threshold=6):
    """Catch sub-token loops like 'ababababab' that token-level checks miss."""
    return re.search(r"(.{%d,}?)\1{%d,}" % (min_len, threshold - 1), text) is not None


def unique_ratio(text):
    toks = text.split()
    return len(set(toks)) / len(toks) if toks else 1.0


def find_markers(text):
    return [m for m in SPECIAL_MARKERS if m in text]


def reasoning_opener(text):
    head = text.strip().lower()[:80]
    return next((p for p in REASONING_OPENERS if head.startswith(p)), None)


# ---------------------------------------------------------------------------
# Check registry

class Suite:
    def __init__(self, srv, quick, skip_deterministic):
        self.srv = srv
        self.quick = quick
        self.skip_det = skip_deterministic
        self.results = []  # (category, name, status, detail)
        self.is_reasoning = False

    def record(self, cat, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        self.results.append((cat, name, status, detail))
        mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))

    def skip(self, cat, name, why):
        self.results.append((cat, name, "SKIP", why))
        print(f"  [\033[33mSKIP\033[0m] {name} — {why}")

    def probe_reasoning(self):
        """One short probe: does this model emit reasoning_content?"""
        r = self.srv.chat_stream(
            [{"role": "user", "content": "What is 2+2? Answer with the number only."}],
            max_tokens=200,
        )
        self.is_reasoning = bool(r["reasoning"])
        print(f"Model emits reasoning_content: {self.is_reasoning}")
        return r

    # -- repetition ---------------------------------------------------------
    def cat_repetition(self):
        n = 200 if self.quick else 512
        r = self.srv.chat(
            [{"role": "user", "content":
              "Write a short story about a lighthouse keeper. Keep going until you run out of space."}],
            max_tokens=n, temperature=0.7,
        )
        text = r["reasoning"] + " " + r["content"]
        self.record("repetition", "no stuck token (run<=6)",
                    max_token_run(text) <= 6, f"max run={max_token_run(text)}")
        self.record("repetition", "no 4-gram loop", not ngram_loop(text),
                    text[-160:].replace("\n", " "))
        self.record("repetition", "no char-level loop", not char_loop(text),
                    text[-160:].replace("\n", " "))
        self.record("repetition", "vocabulary diversity > 0.25",
                    unique_ratio(text) > 0.25, f"ratio={unique_ratio(text):.2f}")

        # High-temperature stress: must still terminate loop-free. (Sampler
        # NaN / banned-mask bugs surface here first.)
        r = self.srv.chat(
            [{"role": "user", "content": "Describe an imaginary animal."}],
            max_tokens=n, temperature=1.2, seed=7,
        )
        text = r["reasoning"] + " " + r["content"]
        self.record("repetition", "temp=1.2 loop-free",
                    not ngram_loop(text) and max_token_run(text) <= 8,
                    text[-160:].replace("\n", " "))
        self.record("repetition", "temp=1.2 finish_reason set",
                    r["finish"] in ("stop", "length"), f"finish={r['finish']}")

    # -- think-leak ---------------------------------------------------------
    def cat_think_leak(self):
        if not self.is_reasoning:
            # Still meaningful: content must not contain literal think tags.
            r = self.srv.chat([{"role": "user", "content": "Hi, who are you?"}],
                              max_tokens=120)
            self.record("think-leak", "no think tags (non-reasoning model)",
                        "<think>" not in r["content"] and "</think>" not in r["content"])
            return

        # 1. Normal request with enough budget to exit the think phase.
        n = 400 if self.quick else 800
        q = [{"role": "user", "content": "What is the capital of France? One word."}]
        r = self.srv.chat(q, max_tokens=n)
        self.record("think-leak", "non-stream: no think tags in content",
                    not any(t in r["content"] for t in ("<think>", "</think>")),
                    r["content"][:120])
        op = reasoning_opener(r["content"])
        self.record("think-leak", "non-stream: content is answer, not reasoning",
                    op is None, f"content starts with reasoning opener {op!r}: "
                    f"{r['content'][:120]}")
        self.record("think-leak", "non-stream: reasoning separated",
                    bool(r["reasoning"]) or op is None,
                    "no reasoning_content field but content looks like reasoning")

        # 2. Truncated think: budget too small to reach </think>. The whole
        #    buffer is reasoning — it must NOT be emitted as content.
        #    (Production bug class: non-stream path dumped it into content
        #    while the streaming path correctly labelled reasoning_content.)
        r = self.srv.chat(q, max_tokens=24)
        op = reasoning_opener(r["content"])
        self.record("think-leak", "truncated think does not spill into content",
                    op is None, f"content={r['content'][:120]!r}")

        # 3. Streaming: think must arrive as delta.reasoning_content only.
        r = self.srv.chat_stream(q, max_tokens=n)
        self.record("think-leak", "stream: no think tags in content deltas",
                    not any(t in r["content"] for t in ("<think>", "</think>")),
                    r["content"][:120])
        op = reasoning_opener(r["content"])
        self.record("think-leak", "stream: content deltas are answer, not reasoning",
                    op is None, f"{r['content'][:120]!r}")

        # 4. Thinking disabled: no reasoning anywhere in the visible answer.
        r = self.srv.chat(q, max_tokens=200, enable_thinking=False)
        op = reasoning_opener(r["content"])
        self.record("think-leak", "enable_thinking=false: clean content",
                    op is None and "<think>" not in r["content"],
                    f"{r['content'][:120]!r}")

    # -- special-tokens -----------------------------------------------------
    def cat_special_tokens(self):
        prompts = [
            "Repeat after me: hello world",
            "Write the word END followed by nothing else.",
            "List three colors.",
        ]
        for i, p in enumerate(prompts):
            r = self.srv.chat([{"role": "user", "content": p}],
                              max_tokens=300, temperature=0.9, seed=100 + i)
            leaked = find_markers(r["content"])
            self.record("special-tokens", f"no raw markers (probe {i+1})",
                        not leaked, f"leaked={leaked} content={r['content'][:100]!r}")

    # -- adherence (prompt-blindness / gross hallucination) ------------------
    def cat_adherence(self):
        n = 600 if self.is_reasoning else 80
        tasks = [
            ("echo literal", "Reply with exactly this string and nothing else: BANANA42", "BANANA42"),
            ("arithmetic", "What is 17 + 25? Reply with the number only.", "42"),
            ("extraction",
             "Context: Der Sicherheitscode für das Lager lautet 7391. "
             "Frage: Wie lautet der Sicherheitscode? Antworte nur mit dem Code.", "7391"),
            ("instruction",
             "Answer with exactly one word, the capital of France.", "Paris"),
        ]
        for name, prompt, expect in tasks:
            r = self.srv.chat([{"role": "user", "content": prompt}], max_tokens=n)
            visible = r["content"]
            self.record("adherence", f"{name}: answer contains {expect!r}",
                        expect.lower() in visible.lower(),
                        f"content={visible[:140]!r} finish={r['finish']}")

    # -- long-context needle echo --------------------------------------------
    def cat_long_context(self):
        filler_unit = (
            "The maintenance log notes routine checks of pumps, valves and "
            "filters performed during the shift without irregularities. "
        )
        reps = 30 if self.quick else 120  # ~0.6k / ~2.5k tokens of filler
        needle = "ZEBRA-9134"
        prompt = (
            f"Remember this code: {needle}.\n\n"
            + filler_unit * reps
            + "\n\nWhat was the code I asked you to remember? Reply with the code only."
        )
        n = 600 if self.is_reasoning else 60
        try:
            r = self.srv.chat([{"role": "user", "content": prompt}], max_tokens=n)
        except urllib.error.HTTPError as e:
            self.skip("long-context", "needle echo", f"server rejected: {e}")
            return
        self.record("long-context", f"needle echo across ~{reps*14} filler tokens",
                    needle in r["content"],
                    f"content={r['content'][:140]!r}")

    # -- multi-turn state ----------------------------------------------------
    def cat_multi_turn(self):
        n = 600 if self.is_reasoning else 100
        msgs = [{"role": "user", "content": "My lucky number is 271. Just acknowledge briefly."}]
        r1 = self.srv.chat(msgs, max_tokens=n)
        msgs.append({"role": "assistant", "content": r1["content"] or "Noted."})
        msgs.append({"role": "user", "content": "Name one primary color. One word."})
        r2 = self.srv.chat(msgs, max_tokens=n)
        text2 = r2["content"]
        self.record("multi-turn", "turn 2 not garbled",
                    not ngram_loop(text2) and max_token_run(text2) <= 6
                    and not find_markers(text2),
                    text2[:140])
        msgs.append({"role": "assistant", "content": text2 or "Red."})
        msgs.append({"role": "user", "content": "What was my lucky number? Reply with the number only."})
        r3 = self.srv.chat(msgs, max_tokens=n)
        self.record("multi-turn", "turn 3 recalls turn-1 fact (271)",
                    "271" in r3["content"], f"content={r3['content'][:140]!r}")

    # -- stream protocol -----------------------------------------------------
    def cat_stream(self):
        q = [{"role": "user", "content": "List the numbers from 1 to 10, comma separated."}]
        n = 700 if self.is_reasoning else 120
        s = self.srv.chat_stream(q, max_tokens=n)
        self.record("stream", "SSE terminates with [DONE]", s["done"])
        self.record("stream", "finish_reason chunk present",
                    s["finish"] in ("stop", "length"), f"finish={s['finish']}")
        self.record("stream", "stream produced visible content",
                    len(s["content"].strip()) > 0,
                    f"reasoning={len(s['reasoning'])}ch content empty")

        if self.skip_det:
            self.skip("stream", "stream == non-stream at temp=0",
                      "model flagged non-deterministic at temp=0")
            return
        ns = self.srv.chat(q, max_tokens=n)
        self.record("stream", "stream == non-stream at temp=0",
                    s["content"].strip() == ns["content"].strip(),
                    f"stream={s['content'][:80]!r} nonstream={ns['content'][:80]!r}")


CATEGORIES = {
    "repetition": Suite.cat_repetition,
    "think-leak": Suite.cat_think_leak,
    "special-tokens": Suite.cat_special_tokens,
    "adherence": Suite.cat_adherence,
    "long-context": Suite.cat_long_context,
    "multi-turn": Suite.cat_multi_turn,
    "stream": Suite.cat_stream,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default=None,
                    help="model id (default: first entry of /v1/models)")
    ap.add_argument("--only", default=None,
                    help="comma-separated categories to run")
    ap.add_argument("--skip", default=None,
                    help="comma-separated categories to skip")
    ap.add_argument("--quick", action="store_true", help="shorter probes")
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--json", dest="json_out", default=None,
                    help="write machine-readable report to this path")
    ap.add_argument("--skip-deterministic", action="store_true",
                    help="skip temp=0 stream/non-stream equality "
                         "(e.g. Qwen3.6 is non-deterministic at temp=0)")
    args = ap.parse_args()

    # Resolve model from the server (strict semantics: it serves exactly one).
    try:
        with urllib.request.urlopen(args.url.rstrip("/") + "/v1/models",
                                    timeout=10) as r:
            models = json.loads(r.read())["data"]
    except (urllib.error.URLError, OSError) as e:
        print(f"Server unreachable at {args.url}: {e}", file=sys.stderr)
        return 2
    if args.model:
        model = args.model
    elif models:
        model = models[0]["id"]
    else:
        print("Server has no model loaded", file=sys.stderr)
        return 2

    selected = list(CATEGORIES)
    if args.only:
        selected = [c for c in args.only.split(",") if c in CATEGORIES]
    if args.skip:
        selected = [c for c in selected if c not in args.skip.split(",")]

    print(f"degen_suite: url={args.url} model={model} "
          f"categories={','.join(selected)} quick={args.quick}")
    srv = Server(args.url, model, args.timeout)
    suite = Suite(srv, args.quick, args.skip_deterministic)

    t0 = time.time()
    try:
        suite.probe_reasoning()
    except urllib.error.HTTPError as e:
        body = e.read()[:300].decode("utf-8", "replace")
        print(f"Reasoning probe failed: HTTP {e.code}: {body}", file=sys.stderr)
        print("Server is up but cannot serve chat requests — check model state "
              "and docker logs.", file=sys.stderr)
        return 2
    for cat in selected:
        print(f"\n== {cat} ==")
        try:
            CATEGORIES[cat](suite)
        except urllib.error.HTTPError as e:
            body = e.read()[:300].decode("utf-8", "replace")
            suite.record(cat, "category aborted", False, f"HTTP {e.code}: {body}")
        except (urllib.error.URLError, OSError) as e:
            suite.record(cat, "category aborted", False,
                         f"server connection lost: {e} — possible crash, "
                         f"check docker logs / RestartCount")

    fails = [r for r in suite.results if r[2] == "FAIL"]
    print(f"\n{'='*60}")
    print(f"degen_suite: {len(suite.results)} checks, "
          f"{len(fails)} FAIL, {sum(1 for r in suite.results if r[2]=='SKIP')} skipped "
          f"({time.time()-t0:.0f}s)")
    for cat, name, status, detail in fails:
        print(f"  FAIL [{cat}] {name}: {detail[:200]}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "url": args.url, "model": model,
                "quick": args.quick, "duration_s": round(time.time() - t0, 1),
                "results": [
                    {"category": c, "name": n, "status": s, "detail": d}
                    for c, n, s, d in suite.results
                ],
            }, f, indent=1)
        print(f"report written to {args.json_out}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
