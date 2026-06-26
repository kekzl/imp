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

There are two modes:
  * default — the hand-coded category probes below (deep, server-protocol focused).
  * --corpus — the large data-driven adversarial battery in degen_corpus.jsonl
    (~250 prompts across the same failure classes; grow it by editing JSONL,
    not Python). Each record declares its own prompt/messages, params, and the
    checks to apply (no_loop, no_markers, no_think_tags, no_reasoning_opener,
    contains, needle, max_words, json_parse, finish_set, vocab_div, stream_eq).

Usage:
  python3 tools/analysis/degen_suite.py                       # localhost:8080
  python3 tools/analysis/degen_suite.py --url http://host:8081
  python3 tools/analysis/degen_suite.py --only think-leak,repetition
  python3 tools/analysis/degen_suite.py --corpus              # full ~250-prompt battery
  python3 tools/analysis/degen_suite.py --corpus --only adherence,long-context
  python3 tools/analysis/degen_suite.py --quick               # short probes
  python3 tools/analysis/degen_suite.py --json report.json
  python3 tools/analysis/degen_suite.py --skip-deterministic  # e.g. Qwen3.6
                                        (non-deterministic at temp=0 — skips
                                        the stream==non-stream equality check)

Exit code: 0 = all pass, 1 = at least one FAIL, 2 = server unreachable.
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

DEFAULT_CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "degen_corpus.jsonl")

# ---------------------------------------------------------------------------
# Markers that must NEVER appear in user-visible content. Covers ChatML,
# Gemma channels/turns, Llama instruct, generic pads. (PR #442 regression
# class: turn markers sampled at high temperature.)
SPECIAL_MARKERS = [
    "<|im_start|>", "<|im_end|>", "<|endoftext|>", "<pad>", "<unk>",
    "<|channel>", "<channel|>", "<start_of_turn>", "<end_of_turn>",
    "[INST]", "[/INST]", "<<SYS>>", "<|user|>", "<|assistant|>",
    "<think>", "</think>",
    # NUL byte: byte-level-BPE models can emit the 0x00 byte token —
    # observed live on Qwen3.6-NVFP4 (alternating token/NUL right after
    # server start). Never valid in text output.
    "\x00",
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

    def messages(self, messages, max_tokens=256, temperature=0.0, **kw):
        """Anthropic /v1/messages, DEFAULT path (no reasoning-format flags)."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        body.update(kw)
        with self._post("/v1/messages", body) as r:
            data = json.loads(r.read())
        thinking, text = [], []
        for block in data.get("content", []):
            if block.get("type") == "thinking":
                thinking.append(block.get("thinking", ""))
            elif block.get("type") == "text":
                text.append(block.get("text", ""))
        return {
            "thinking": "".join(thinking),
            "text": "".join(text),
            "stop_reason": data.get("stop_reason"),
            "usage": data.get("usage", {}),
            "raw": data,
        }

    def messages_stream(self, messages, max_tokens=256, temperature=0.0, **kw):
        """Anthropic /v1/messages with stream=true; collects block deltas."""
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        body.update(kw)
        thinking, text = [], []
        stop_reason = None
        events = set()
        with self._post("/v1/messages", body) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                ev = json.loads(payload)
                events.add(ev.get("type", ""))
                if ev.get("type") == "content_block_delta":
                    d = ev.get("delta", {})
                    if d.get("type") == "thinking_delta":
                        thinking.append(d.get("thinking", ""))
                    elif d.get("type") == "text_delta":
                        text.append(d.get("text", ""))
                elif ev.get("type") == "message_delta":
                    stop_reason = ev.get("delta", {}).get("stop_reason", stop_reason)
        return {
            "thinking": "".join(thinking),
            "text": "".join(text),
            "stop_reason": stop_reason,
            "events": events,
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


def parses_as_json(text):
    """True if `text` (after stripping markdown fences / surrounding prose)
    contains a parseable JSON value. Used by the corpus 'json_parse' check."""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        s = s[4:] if s[:4].lower() == "json" else s
        s = s.strip()
    try:
        json.loads(s)
        return True
    except (json.JSONDecodeError, ValueError):
        pass
    # Fall back to the widest bracketed span (handles leading/trailing prose).
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = s.find(open_c), s.rfind(close_c)
        if 0 <= i < j:
            try:
                json.loads(s[i:j + 1])
                return True
            except (json.JSONDecodeError, ValueError):
                continue
    return False


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
        """Probe: does this model emit reasoning_content?

        Two attempts (stream + non-stream) — reasoning models are not
        deterministic about WHEN they think (Qwen3.6 sometimes answers a
        trivial question with zero reasoning tokens), so a single probe
        misclassifies and downstream categories then use too-small budgets.
        """
        q = [{"role": "user", "content":
              "Is 17 a prime number? Explain briefly, then answer yes or no."}]
        r = self.srv.chat_stream(q, max_tokens=400)
        if not r["reasoning"]:
            r2 = self.srv.chat(q, max_tokens=400, seed=7)
            r["reasoning"] = r["reasoning"] or r2["reasoning"]
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
        n = 1000 if self.is_reasoning else 80
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
        n = 1000 if self.is_reasoning else 60
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
        n = 1000 if self.is_reasoning else 100
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
        n = 1000 if self.is_reasoning else 120
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


    # -- anthropic-thinking ---------------------------------------------------
    # /v1/messages DEFAULT path (audit 2026-05-31 T8: thinking blocks were only
    # ever confirmed WITH --reasoning-format; the default path was unconfirmed).
    def cat_anthropic_thinking(self):
        q = [{"role": "user", "content": "What is the capital of France? One word."}]
        n = 400 if self.quick else 800

        if not self.is_reasoning:
            r = self.srv.messages(q, max_tokens=120)
            self.record("anthropic-thinking", "non-reasoning: text block, no think tags",
                        bool(r["text"].strip()) and "<think>" not in r["text"],
                        f"text={r['text'][:80]!r}")
            return

        # 1. Non-stream default: reasoning must arrive as a `thinking` content
        #    block, the answer as a `text` block — with NO request flags.
        r = self.srv.messages(q, max_tokens=n)
        self.record("anthropic-thinking", "default non-stream: thinking block present",
                    bool(r["thinking"].strip()), f"blocks={[b.get('type') for b in r['raw'].get('content', [])]}")
        self.record("anthropic-thinking", "default non-stream: text block is the answer",
                    bool(r["text"].strip()) and reasoning_opener(r["text"]) is None,
                    f"text={r['text'][:100]!r}")
        self.record("anthropic-thinking", "default non-stream: no literal think tags",
                    "<think>" not in r["text"] and "</think>" not in r["text"],
                    r["text"][:100])
        self.record("anthropic-thinking", "default non-stream: usage tokens present",
                    r["usage"].get("input_tokens", 0) > 0 and r["usage"].get("output_tokens", 0) > 0,
                    str(r["usage"]))

        # 2. Streaming default: thinking_delta blocks, then text_delta.
        r = self.srv.messages_stream(q, max_tokens=n)
        self.record("anthropic-thinking", "default stream: thinking_delta events",
                    bool(r["thinking"].strip()), f"events={sorted(r['events'])}")
        self.record("anthropic-thinking", "default stream: text deltas are answer",
                    bool(r["text"].strip()) and reasoning_opener(r["text"]) is None
                    and "<think>" not in r["text"],
                    f"text={r['text'][:100]!r}")


    # -- data-driven corpus -------------------------------------------------
    # Runs the large adversarial prompt battery in tools/analysis/degen_corpus.jsonl.
    # Each record declares its own prompt/messages, params, and the checks to
    # apply — so the battery grows by editing JSONL, not Python.
    def _eval_checks(self, rec, content, reasoning, finish, stream_content):
        """Apply a record's declared checks to one response. Yields (ok, why)."""
        checks = rec.get("checks", [])
        expect = rec.get("expect", "")
        text = content  # checks run on the USER-VISIBLE channel only
        for c in checks:
            if c == "not_empty":
                yield bool(text.strip()), f"empty (content={text[:60]!r})"
            elif c == "no_loop":
                run, ng, ch = max_token_run(text), ngram_loop(text), char_loop(text)
                yield (run <= 6 and not ng and not ch), \
                    f"loop run={run} ngram={ng} char={ch}: {text[-120:]!r}"
            elif c == "no_markers":
                m = find_markers(text)
                yield (not m), f"leaked markers {m}: {text[:80]!r}"
            elif c == "finish_set":
                yield (finish in ("stop", "length")), f"finish={finish}"
            elif c == "vocab_div":
                ur = unique_ratio(text)
                yield (ur > 0.25), f"vocab ratio={ur:.2f}"
            elif c == "no_think_tags":
                bad = [t for t in ("<think>", "</think>") if t in text]
                yield (not bad), f"think tags {bad} in content: {text[:80]!r}"
            elif c == "no_reasoning_opener":
                op = reasoning_opener(text)
                yield (op is None), f"reasoning opener {op!r}: {text[:80]!r}"
            elif c == "contains":
                yield (expect.lower() in text.lower()), \
                    f"missing {expect!r} (content={text[:100]!r})"
            elif c == "needle":
                yield (expect.lower() in text.lower()), \
                    f"needle {expect!r} not recalled (content={text[:100]!r})"
            elif c == "max_words":
                lim = rec.get("limit", 3)
                wc = len(text.split())
                yield (wc <= lim), f"{wc} words > limit {lim}: {text[:80]!r}"
            elif c == "json_parse":
                yield parses_as_json(text), f"not valid JSON: {text[:100]!r}"
            elif c == "stream_eq":
                if self.skip_det:
                    continue
                yield (stream_content.strip() == text.strip()), \
                    f"stream!=nonstream: {stream_content[:60]!r} vs {text[:60]!r}"

    def run_corpus(self, records, only_cats=None):
        for rec in records:
            cat = rec.get("cat", "?")
            if only_cats and cat not in only_cats:
                continue
            messages = rec.get("messages") or [
                {"role": "user", "content": rec.get("prompt", "")}]
            mt = int(rec.get("max_tokens", 256))
            # think-leak truncation probes NEED their tiny budget; everything
            # else gets enough room for a reasoning model to finish thinking.
            if cat != "think-leak" and self.is_reasoning:
                mt = max(mt, 800)
            if self.quick and cat != "think-leak":
                mt = min(mt, 256)
            kw = {}
            if "enable_thinking" in rec:
                kw["enable_thinking"] = rec["enable_thinking"]
            temp = float(rec.get("temperature", 0.0))
            seed = int(rec.get("seed", 42))
            need_stream = "stream_eq" in rec.get("checks", [])
            try:
                r = self.srv.chat(messages, max_tokens=mt, temperature=temp,
                                  seed=seed, **kw)
            except urllib.error.HTTPError as e:
                self.record(cat, rec.get("id", "?"), False,
                            f"HTTP {e.code}: {e.read()[:120].decode('utf-8','replace')}")
                continue
            except (urllib.error.URLError, OSError) as e:
                self.record(cat, rec.get("id", "?"), False,
                            f"connection lost: {e} (possible crash)")
                continue
            stream_content = ""
            if need_stream:
                try:
                    stream_content = self.srv.chat_stream(
                        messages, max_tokens=mt, temperature=temp, seed=seed,
                        **kw)["content"]
                except (urllib.error.URLError, OSError):
                    pass
            content = r["reasoning"] + " " + r["content"] if cat == "repetition" \
                else r["content"]
            fails = [why for ok, why in self._eval_checks(
                rec, content, r["reasoning"], r["finish"], stream_content) if not ok]
            self.record(cat, rec.get("id", "?"), not fails,
                        " | ".join(fails)[:300])


CATEGORIES = {
    "repetition": Suite.cat_repetition,
    "think-leak": Suite.cat_think_leak,
    "special-tokens": Suite.cat_special_tokens,
    "adherence": Suite.cat_adherence,
    "long-context": Suite.cat_long_context,
    "multi-turn": Suite.cat_multi_turn,
    "stream": Suite.cat_stream,
    "anthropic-thinking": Suite.cat_anthropic_thinking,
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
    ap.add_argument("--corpus", nargs="?", const=DEFAULT_CORPUS, default=None,
                    metavar="PATH",
                    help="run the large JSONL prompt battery instead of the "
                         "hand-coded categories (default file: degen_corpus.jsonl). "
                         "Filter its categories with --only.")
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
    if args.corpus:
        try:
            with open(args.corpus, encoding="utf-8") as f:
                records = [json.loads(ln) for ln in f if ln.strip()]
        except (OSError, json.JSONDecodeError) as e:
            print(f"Cannot load corpus {args.corpus}: {e}", file=sys.stderr)
            return 2
        only = set(args.only.split(",")) if args.only else None
        shown = [r for r in records if not only or r.get("cat") in only]
        print(f"\n== corpus: {len(shown)} prompts from {os.path.basename(args.corpus)} ==")
        suite.run_corpus(records, only)
    else:
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
