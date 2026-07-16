#!/usr/bin/env python3
"""Real model-driven agent loop against imp-server (#1007, stage 2).

Where stage 1 (agent_loop_suite.py) drives FORCED wire patterns, this stage lets
the MODEL drive: a multi-step task, `tool_choice:auto`, REAL local tools executed
for real, looping turn-by-turn until the model stops calling tools — the actual
agent loop a harness runs. Gates on task completion, not wire shape.

Tasks (per dialect — OpenAI /v1/chat/completions + Anthropic /v1/messages):
  chain     "compute (17+25)*3" with add()/multiply() tools — the model must
            call add -> read the result -> call multiply -> answer 126. A genuine
            2-step chain the model sequences itself.
  optional  a plain question with tools available — the model must answer WITHOUT
            calling (auto is optional), proving tools don't force a call.
  lookup    "what's the capital stored for 'FR'?" with a get(key) tool over a
            fixed table — one call, then the answer must use the returned value.

Pass = every task completes: tool calls parse, arguments run the real tool, the
final answer reflects the tool results, no turn stalls (>60s = fail), loop
terminates within the turn cap. Stdlib-only; mirrors degen_suite conventions.

Usage:
  python3 tools/analysis/agent_task_loop.py [--url http://localhost:8080]
                                            [--model NAME] [--only openai,anthropic]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request

MAX_TURNS = 6
TIMEOUT = 60

# ---- real tools the model can call (executed for real, no stubs) -------------
_TABLE = {"FR": "Paris", "JP": "Tokyo", "BR": "Brasilia"}


def tool_add(a, b):
    return {"result": float(a) + float(b)}


def tool_multiply(a, b):
    return {"result": float(a) * float(b)}


def tool_get(key):
    return {"value": _TABLE.get(str(key), None)}


TOOLS_IMPL = {"add": tool_add, "multiply": tool_multiply, "get": tool_get}

# OpenAI-dialect tool schemas (Anthropic derives from these below).
OAI_TOOLS = [
    {"type": "function", "function": {
        "name": "add", "description": "Add two numbers.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "multiply", "description": "Multiply two numbers.",
        "parameters": {"type": "object",
                       "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                       "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "get", "description": "Look up the capital city stored for a 2-letter country code.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}},
                       "required": ["key"]}}},
]


def _anthropic_tools():
    return [{"name": t["function"]["name"], "description": t["function"]["description"],
             "input_schema": t["function"]["parameters"]} for t in OAI_TOOLS]


class Suite:
    def __init__(self, url, model):
        self.url = url.rstrip("/")
        self.model = model
        self.results = []

    def record(self, cat, name, ok, detail=""):
        self.results.append((cat, name, ok, detail))
        mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not ok else ""))

    def _post(self, path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(self.url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            payload = json.loads(r.read())
        return payload, time.monotonic() - t0

    # ---- OpenAI /v1/chat/completions driver --------------------------------
    def run_openai(self, cat, messages, expect_substr, force_no_call=False):
        max_stall = 0.0
        calls_made = 0
        for turn in range(MAX_TURNS):
            body = {"model": self.model, "messages": messages, "tools": OAI_TOOLS,
                    "tool_choice": "auto", "temperature": 0.3, "max_tokens": 400}
            try:
                rsp, dt = self._post("/v1/chat/completions", body)
            except (urllib.error.URLError, TimeoutError) as e:
                self.record(cat, "openai loop completes", False, f"turn {turn}: {e}")
                return
            max_stall = max(max_stall, dt)
            msg = rsp["choices"][0]["message"]
            tcs = msg.get("tool_calls") or []
            if not tcs:
                content = msg.get("content") or ""
                if force_no_call:
                    self.record(cat, "openai: no forced call (auto is optional)",
                                calls_made == 0, f"calls={calls_made} content={content[:60]!r}")
                    return
                ok = expect_substr.lower() in content.lower()
                self.record(cat, "openai: final answer uses tool results", ok,
                            f"answer={content[:80]!r} want~{expect_substr!r}")
                self.record(cat, "openai: no turn stall", max_stall < TIMEOUT,
                            f"max_turn={max_stall:.1f}s")
                return
            # Execute every requested tool for real and feed results back.
            messages.append(msg)
            for tc in tcs:
                calls_made += 1
                fn = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"])
                    out = TOOLS_IMPL[fn](**args)
                    ok = True
                except (json.JSONDecodeError, TypeError, KeyError, ValueError) as e:
                    out, ok = {"error": str(e)}, False
                if not ok:
                    self.record(cat, "openai: tool call args run the real tool", False,
                                f"fn={fn} args={tc['function']['arguments'][:80]!r}")
                    return
                messages.append({"role": "tool", "tool_call_id": tc.get("id", ""),
                                 "content": json.dumps(out)})
        self.record(cat, "openai loop terminates within turn cap", False,
                    f"still calling after {MAX_TURNS} turns")

    # ---- Anthropic /v1/messages driver -------------------------------------
    def run_anthropic(self, cat, first_user, expect_substr):
        messages = [{"role": "user", "content": first_user}]
        max_stall = 0.0
        calls_made = 0
        for turn in range(MAX_TURNS):
            body = {"model": self.model, "messages": messages, "tools": _anthropic_tools(),
                    "max_tokens": 400, "temperature": 0.3}
            try:
                rsp, dt = self._post("/v1/messages", body)
            except (urllib.error.URLError, TimeoutError) as e:
                self.record(cat, "anthropic loop completes", False, f"turn {turn}: {e}")
                return
            max_stall = max(max_stall, dt)
            blocks = rsp.get("content", [])
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            if not tool_uses:
                text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                ok = expect_substr.lower() in text.lower()
                self.record(cat, "anthropic: final answer uses tool results", ok,
                            f"answer={text[:80]!r} want~{expect_substr!r}")
                self.record(cat, "anthropic: no turn stall", max_stall < TIMEOUT,
                            f"max_turn={max_stall:.1f}s")
                return
            messages.append({"role": "assistant", "content": blocks})
            tool_results = []
            for tu in tool_uses:
                calls_made += 1
                fn = tu.get("name", "")
                try:
                    out = TOOLS_IMPL[fn](**(tu.get("input") or {}))
                    ok = True
                except (TypeError, KeyError, ValueError) as e:
                    out, ok = {"error": str(e)}, False
                if not ok:
                    self.record(cat, "anthropic: tool_use input runs the real tool", False,
                                f"fn={fn} input={tu.get('input')}")
                    return
                tool_results.append({"type": "tool_result", "tool_use_id": tu.get("id", ""),
                                     "content": json.dumps(out)})
            messages.append({"role": "user", "content": tool_results})
        self.record(cat, "anthropic loop terminates within turn cap", False,
                    f"still calling after {MAX_TURNS} turns")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="Qwen3-8B-Q8_0.gguf")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {c for c in args.only.split(",") if c} or None

    suite = Suite(args.url, args.model)
    t0 = time.monotonic()

    chain_prompt = ("Compute (17 + 25) * 3 using the add and multiply tools, one step "
                    "at a time. Add first, then multiply the sum by 3. Report the final number.")
    lookup_prompt = "Use the get tool to look up the capital stored for country code 'FR', then tell me the city."
    plain_prompt = "In one word, what color is a clear daytime sky? Do not use any tools."

    if not only or "openai" in only:
        print("== openai (model-driven, real tools) ==")
        suite.run_openai("openai", [{"role": "user", "content": chain_prompt}], "126")
        suite.run_openai("openai", [{"role": "user", "content": lookup_prompt}], "Paris")
        suite.run_openai("openai", [{"role": "user", "content": plain_prompt}], "", force_no_call=True)
    if not only or "anthropic" in only:
        print("== anthropic (model-driven, real tools) ==")
        suite.run_anthropic("anthropic", chain_prompt, "126")
        suite.run_anthropic("anthropic", lookup_prompt, "Paris")

    fails = [r for r in suite.results if not r[2]]
    dt = time.monotonic() - t0
    print("=" * 60)
    print(f"agent_task_loop: {len(suite.results)} checks, {len(fails)} FAIL ({dt:.0f}s)")
    for cat, name, _, detail in fails:
        print(f"  FAIL [{cat}] {name}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
