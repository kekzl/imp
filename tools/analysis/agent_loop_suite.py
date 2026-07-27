#!/usr/bin/env python3
"""Agent-harness E2E battery (#1007, stage 1).

Drives imp-server with the exact wire patterns real agent harnesses generate —
multi-turn tool loops in all three dialects — as a repeatable gate:

  anthropic-loop   /v1/messages: forced tool_use -> tool_result -> final answer,
                   cache_control prefix reuse across turns, streaming event order
  openai-loop      /v1/chat/completions: tool_choice=required -> role:tool ->
                   final answer; streaming tool_calls delta assembly
  responses-loop   /v1/responses: OpenAI Agents SDK dialect — response.created,
                   function_call_arguments.delta assembly, response.completed
  reasoning-channel  stream vs non-stream must route reasoning to the same
                   channel — with tools the streaming path used to hand the
                   chain of thought to the user as the answer

Pass = every check green: tool calls parse, arguments match the tool's schema,
the final answer uses the tool result, no stream stalls (>30s gap = fail).
Stdlib-only, mirrors tools/analysis/degen_suite.py conventions.

Usage:
  python3 tools/analysis/agent_loop_suite.py [--url http://localhost:8080]
                                             [--only anthropic-loop,reasoning-channel]
Exit codes: 0 = clean, 1 = failures, 2 = server unreachable.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

TOOLS_OAI = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

TOOLS_ANTH = [{
    "name": "get_weather",
    "description": "Get the current weather for a city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

TOOL_RESULT_TEXT = "19 degrees celsius, light rain"
PROMPT = "What's the weather in Paris right now? Use the tool."
SYSTEM = ("You are a terse weather assistant. Always answer with the exact "
          "temperature the tool reports. " + "Padding for cacheable prefix. " * 40)


class Suite:
    def __init__(self, url):
        self.url = url
        self.model = self._model_id()
        self.results = []

    def _model_id(self):
        with urllib.request.urlopen(self.url + "/v1/models", timeout=30) as r:
            return json.load(r)["data"][0]["id"]

    def record(self, cat, name, ok, detail=""):
        self.results.append((cat, name, ok, detail))
        mark = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  [{mark}] {name}" + ("" if ok else f": {detail}"))

    def post(self, path, body, timeout=120):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def post_sse(self, path, body, timeout=120):
        """Collect SSE events; fail on inter-event gaps > 30s (stream stall)."""
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        events = []
        last = time.monotonic()
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                now = time.monotonic()
                if now - last > 30:
                    raise RuntimeError("stream stalled > 30s between events")
                last = now
                line = raw.decode().strip()
                if line.startswith("data: ") and line != "data: [DONE]":
                    events.append(json.loads(line[6:]))
        return events

    # ---- anthropic-loop -----------------------------------------------------

    def run_anthropic_loop(self):
        print("== anthropic-loop ==")
        cat = "anthropic-loop"
        sysblk = [{"type": "text", "text": SYSTEM,
                   "cache_control": {"type": "ephemeral"}}]
        base = {"model": self.model, "max_tokens": 400, "system": sysblk,
                "tools": TOOLS_ANTH,
                "tool_choice": {"type": "tool", "name": "get_weather"}}
        msgs = [{"role": "user", "content": PROMPT}]

        # Turn 1: forced tool call.
        r = self.post("/v1/messages", {**base, "messages": msgs})
        tool_use = next((b for b in r.get("content", []) if b.get("type") == "tool_use"), None)
        self.record(cat, "turn 1: stop_reason is tool_use",
                    r.get("stop_reason") == "tool_use", f"got {r.get('stop_reason')}")
        self.record(cat, "turn 1: tool_use block present", tool_use is not None,
                    f"content={json.dumps(r.get('content'))[:120]}")
        if not tool_use:
            return
        self.record(cat, "turn 1: forced tool name honored",
                    tool_use.get("name") == "get_weather", f"got {tool_use.get('name')}")
        args_ok = isinstance(tool_use.get("input"), dict) and "city" in tool_use["input"]
        self.record(cat, "turn 1: input matches the tool schema", args_ok,
                    f"input={tool_use.get('input')}")

        # Turn 2: tool_result -> final answer must carry the tool's value.
        msgs = msgs + [
            {"role": "assistant", "content": [tool_use]},
            {"role": "user", "content": [{"type": "tool_result",
                                          "tool_use_id": tool_use.get("id", ""),
                                          "content": TOOL_RESULT_TEXT}]},
        ]
        r2 = self.post("/v1/messages",
                       {**base, "tool_choice": {"type": "auto"}, "messages": msgs})
        text = " ".join(b.get("text", "") for b in r2.get("content", [])
                        if b.get("type") == "text")
        self.record(cat, "turn 2: final answer uses the tool result", "19" in text,
                    f"text={text[:120]!r}")
        cache_read = r2.get("usage", {}).get("cache_read_input_tokens", 0)
        self.record(cat, "turn 2: cache_control prefix reused", cache_read > 0,
                    f"cache_read_input_tokens={cache_read}")

        # Streaming: event order + input_json_delta assembly.
        events = self.post_sse("/v1/messages", {**base, "messages":
                               [{"role": "user", "content": PROMPT}], "stream": True})
        types = [e.get("type") for e in events]
        self.record(cat, "stream: message_start first",
                    bool(types) and types[0] == "message_start", f"first={types[:1]}")
        self.record(cat, "stream: message_stop terminates",
                    "message_stop" in types, f"types={types[-3:]}")
        frags = "".join(e["delta"].get("partial_json", "") for e in events
                        if e.get("type") == "content_block_delta" and
                        e.get("delta", {}).get("type") == "input_json_delta")
        ok_json = True
        if frags:
            try:
                json.loads(frags)
            except ValueError:
                ok_json = False
        self.record(cat, "stream: input_json_delta assembles to valid JSON",
                    ok_json and bool(frags), f"frags={frags[:80]!r}")

    # ---- openai-loop --------------------------------------------------------

    def run_openai_loop(self):
        print("== openai-loop ==")
        cat = "openai-loop"
        base = {"model": self.model, "max_tokens": 400, "temperature": 0.0,
                "tools": TOOLS_OAI}
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT}]

        # Turn 1: required tool call (FSM-enforced since #1017).
        r = self.post("/v1/chat/completions",
                      {**base, "tool_choice": "required", "messages": msgs})
        choice = r["choices"][0]
        tcs = choice["message"].get("tool_calls") or []
        self.record(cat, "turn 1: finish_reason is tool_calls",
                    choice.get("finish_reason") == "tool_calls",
                    f"got {choice.get('finish_reason')}")
        self.record(cat, "turn 1: exactly one tool call parses", len(tcs) == 1,
                    f"n={len(tcs)}")
        if not tcs:
            return
        try:
            args = json.loads(tcs[0]["function"]["arguments"])
            args_ok = "city" in args
        except ValueError:
            args_ok = False
        self.record(cat, "turn 1: arguments match the tool schema", args_ok,
                    f"args={tcs[0]['function']['arguments'][:80]}")

        # Turn 2: tool result -> final answer.
        msgs = msgs + [
            {"role": "assistant", "tool_calls": tcs, "content": None},
            {"role": "tool", "tool_call_id": tcs[0]["id"], "content": TOOL_RESULT_TEXT},
        ]
        r2 = self.post("/v1/chat/completions", {**base, "messages": msgs})
        text = r2["choices"][0]["message"].get("content") or ""
        self.record(cat, "turn 2: final answer uses the tool result", "19" in text,
                    f"text={text[:120]!r}")

        # Streaming tool-call delta assembly.
        req = urllib.request.Request(
            self.url + "/v1/chat/completions",
            json.dumps({**base, "tool_choice": "required", "stream": True,
                        "messages": [{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": PROMPT}]}).encode(),
            {"Content-Type": "application/json"})
        name, frags = "", ""
        last = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw in resp:
                now = time.monotonic()
                if now - last > 30:
                    self.record(cat, "stream: no stalls", False, "gap > 30s")
                    return
                last = now
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                d = json.loads(line[6:])
                for tc in (d["choices"][0].get("delta", {}).get("tool_calls") or []):
                    fn = tc.get("function", {})
                    name += fn.get("name", "") or ""
                    frags += fn.get("arguments", "") or ""
        ok_json = False
        try:
            ok_json = "city" in json.loads(frags)
        except ValueError:
            pass
        self.record(cat, "stream: tool_calls deltas assemble to a valid call",
                    name == "get_weather" and ok_json,
                    f"name={name!r} args={frags[:80]!r}")

    # ---- responses-loop -----------------------------------------------------

    def run_responses_loop(self):
        print("== responses-loop ==")
        cat = "responses-loop"
        body = {"model": self.model, "max_output_tokens": 400, "stream": True,
                "tools": [{"type": "function", "name": "get_weather",
                           "description": "Get the current weather for a city",
                           "parameters": TOOLS_OAI[0]["function"]["parameters"]}],
                "tool_choice": "required",
                "input": [{"role": "user", "content": PROMPT}]}
        try:
            events = self.post_sse("/v1/responses", body)
        except urllib.error.HTTPError as e:
            self.record(cat, "endpoint available", False, f"HTTP {e.code}")
            return
        types = [e.get("type") for e in events]
        self.record(cat, "response.created emitted", "response.created" in types,
                    f"types={types[:3]}")
        self.record(cat, "response.completed terminates",
                    "response.completed" in types, f"types={types[-3:]}")
        frags = "".join(e.get("delta", "") for e in events
                        if e.get("type") == "response.function_call_arguments.delta")
        ok_json = False
        try:
            ok_json = "city" in json.loads(frags)
        except ValueError:
            pass
        self.record(cat, "function_call_arguments.delta assembles", ok_json,
                    f"frags={frags[:80]!r}")


    # ---- reasoning-channel --------------------------------------------------

    def run_reasoning_channel(self):
        """Streaming and non-streaming must agree on what is reasoning.

        Found by pointing the real Claude Code binary at imp-server: with tools
        present the same request returned the chain of thought in
        `reasoning_content` without `stream:true`, and as the user-visible
        answer with it. Agent harnesses stream, so the broken half is the half
        every real client sees. Skipped on a model that does not reason.
        """
        print("== reasoning-channel ==")
        cat = "reasoning-channel"
        tools_oai = [{"type": "function", "function": {
            "name": "edit_file", "description": "Edit a file.",
            "parameters": {"type": "object",
                           "properties": {"path": {"type": "string"}},
                           "required": ["path"]}}}]
        prompt = "Edit /tmp/a.py to add a helper function."

        def oai(stream, tools):
            body = {"model": self.model, "max_tokens": 400, "stream": stream,
                    "messages": [{"role": "user", "content": prompt}]}
            if tools:
                body["tools"] = tools
            if not stream:
                m = self.post("/v1/chat/completions", body)["choices"][0]["message"]
                return len(m.get("reasoning_content") or ""), len(m.get("content") or "")
            rc = ct = 0
            for ev in self.post_sse("/v1/chat/completions", body):
                d = ev["choices"][0].get("delta", {})
                rc += len(d.get("reasoning_content") or "")
                ct += len(d.get("content") or "")
            return rc, ct

        # Baseline: no tools. Establishes that this model reasons at all.
        base_ns, _ = oai(False, None)
        if base_ns == 0:
            self.record(cat, "model reasons (skipped: it does not)", True,
                        "no reasoning_content without tools")
            return
        base_st, _ = oai(True, None)
        self.record(cat, "no tools: stream and non-stream agree",
                    base_st > 0, f"non-stream={base_ns} stream={base_st}")

        # The regression: tools present.
        ns_r, ns_c = oai(False, tools_oai)
        st_r, st_c = oai(True, tools_oai)
        if ns_r == 0:
            self.record(cat, "with tools: model reasons (skipped: it does not)", True, "")
            return
        self.record(cat, "with tools: reasoning stays out of content when streaming",
                    st_r > 0,
                    f"non-stream reasoning={ns_r}/content={ns_c}, "
                    f"stream reasoning={st_r}/content={st_c} — the chain of thought "
                    f"was streamed as the answer")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--only", default="",
                    help="comma list: anthropic-loop,openai-loop,responses-loop,reasoning-channel")
    args = ap.parse_args()

    try:
        suite = Suite(args.url)
    except Exception as e:
        print(f"server unreachable at {args.url}: {e}")
        return 2

    only = {c for c in args.only.split(",") if c} or None
    t0 = time.monotonic()
    if not only or "anthropic-loop" in only:
        suite.run_anthropic_loop()
    if not only or "openai-loop" in only:
        suite.run_openai_loop()
    if not only or "responses-loop" in only:
        suite.run_responses_loop()
    if not only or "reasoning-channel" in only:
        suite.run_reasoning_channel()

    fails = [r for r in suite.results if not r[2]]
    dt = time.monotonic() - t0
    print("=" * 60)
    print(f"agent_loop_suite: {len(suite.results)} checks, {len(fails)} FAIL ({dt:.0f}s)")
    for cat, name, _, detail in fails:
        print(f"  FAIL [{cat}] {name}: {detail}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
