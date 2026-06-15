#!/usr/bin/env python3
"""MANUAL gate — part of the local server-test stage (`make test-server`, Stage 3).
Needs a running imp-server with a chat model (default Qwen3-8B-NVFP4-cortecs).

Asserts the Anthropic `/v1/messages` streaming EVENT SEQUENCE that handlers.cpp
emits (AnthropicSSE, handlers.cpp ~3682: `event: <name>\\ndata: <json>\\n\\n`).
The coverage driver only checks the stream does not 5xx; this asserts the
protocol-level ordering the audit named:

  message_start
    (content_block_start  content_block_delta+  content_block_stop)+
  message_delta            <- carries stop_reason
  message_stop             <- terminal

Invariants checked:
  * first event is message_start; last is message_stop;
  * every `event:` line type matches its `data.type`;
  * content blocks are balanced — every delta sits inside an open block, every
    open block is closed, no block opens while another is open;
  * >=1 content_block_delta of type text_delta carrying non-empty text;
  * message_delta appears once, after all content blocks, with a stop_reason;
  * nothing follows message_stop.

Run (stdlib only, no deps):
    python3 tests/test_server_messages_stream.py
Env:
    IMP_BASE   (default http://localhost:8080)
    IMP_MODEL  (default: auto-detected from /v1/models)

Exit code: 0 = sequence valid (PASS), 1 = a regression (FAIL).
"""
import json, os, sys, urllib.request

BASE = os.environ.get("IMP_BASE", "http://localhost:8080").rstrip("/")
_fail = []


def model_id():
    if os.environ.get("IMP_MODEL"):
        return os.environ["IMP_MODEL"]
    data = json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))
    return data["data"][0]["id"]


M = model_id()


def fail(msg):
    _fail.append(msg)
    print(f"  FAIL {msg}")


def check(cond, msg):
    if not cond:
        fail(msg)
    return cond


def stream_events():
    """Yield (event_name, data_obj) from the SSE stream in arrival order.

    Stops as soon as `message_stop` arrives — imp-server keeps the chunked
    connection alive after the terminal event, so reading past it blocks until
    the socket timeout. A timeout is treated as end-of-stream (the gate then
    fails on the missing message_stop rather than crashing on a traceback).
    """
    # A think-model (Qwen3 default-on) emits a `thinking` block first; the
    # Anthropic route does not expose a thinking toggle (anthropic_to_openai_body
    # maps neither `thinking` nor `enable_thinking`), so we don't fight it — we
    # assert the protocol shape AROUND any thinking block and give a budget large
    # enough that the whole think+answer completes with a clean message_stop.
    # `enable_thinking:false` is a harmless no-op here, kept for when the route
    # learns to honour it. The prompt is trivial so reasoning stays short.
    body = {"model": M, "max_tokens": 512, "stream": True, "enable_thinking": False,
            "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}]}
    req = urllib.request.Request(BASE + "/v1/messages", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    ev = None
    try:
        with urllib.request.urlopen(req, timeout=150) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\n").rstrip("\r")
                if line.startswith("event:"):
                    ev = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    payload = line[len("data:"):].strip()
                    try:
                        obj = json.loads(payload) if payload else {}
                    except json.JSONDecodeError:
                        obj = {"__unparseable__": payload}
                    yield ev, obj
                    name, ev = ev, None
                    if name == "message_stop" or obj.get("type") == "message_stop":
                        return
    except TimeoutError:
        print("  (stream read timed out — treating as end-of-stream)")


def main():
    print(f"messages-stream gate: base={BASE} model={M}")
    events = list(stream_events())
    if not check(events, "no SSE events received"):
        return _verdict()

    names = [e for e, _ in events]
    print(f"  events: {names}")

    # event: line type must match data.type
    for e, obj in events:
        t = obj.get("type")
        check(e == t, f"event line {e!r} != data.type {t!r}")

    check(names[0] == "message_start", f"first event is {names[0]!r}, want message_start")
    check(names[-1] == "message_stop", f"last event is {names[-1]!r}, want message_stop")
    check(names.count("message_start") == 1, "message_start not unique")
    check(names.count("message_delta") == 1, "message_delta not exactly once")
    check(names.count("message_stop") == 1, "message_stop not exactly once")

    # message_start payload sanity
    msg0 = events[0][1].get("message", {})
    check(msg0.get("role") == "assistant", "message_start.message.role != assistant")

    # Walk content blocks: balance + delta containment. A thinking block (if the
    # model reasons) and a text block are both valid; we require >=1 text block
    # carrying the answer, and tolerate a preceding thinking block.
    open_block = False
    saw_text_delta = False
    saw_text_block = False
    seen_message_delta = False
    stop_reason = None
    for i, (e, obj) in enumerate(events):
        if e == "content_block_start":
            check(not open_block, f"content_block_start at {i} while a block is open")
            open_block = True
            if obj.get("content_block", {}).get("type") == "text":
                saw_text_block = True
        elif e == "content_block_delta":
            check(open_block, f"content_block_delta at {i} with no open block")
            d = obj.get("delta", {})
            check(d.get("type") in ("text_delta", "thinking_delta", "input_json_delta"),
                  f"content_block_delta at {i} has unknown delta type {d.get('type')!r}")
            if d.get("type") == "text_delta" and d.get("text"):
                saw_text_delta = True
        elif e == "content_block_stop":
            check(open_block, f"content_block_stop at {i} with no open block")
            open_block = False
        elif e == "message_delta":
            seen_message_delta = True
            check(not open_block, "message_delta arrived with a content block still open")
            stop_reason = obj.get("delta", {}).get("stop_reason")
        elif e == "message_stop":
            check(seen_message_delta, "message_stop before message_delta")

    check(not open_block, "a content block was never closed")
    check(saw_text_block, "no text content block in the stream")
    check(saw_text_delta, "no text_delta with non-empty text")
    check(stop_reason not in (None, ""), f"message_delta carried no stop_reason ({stop_reason!r})")

    return _verdict()


def _verdict():
    if _fail:
        print(f"FAIL: {len(_fail)} streaming-sequence assertion(s) regressed")
        return 1
    print("PASS: Anthropic /v1/messages event sequence valid + balanced")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
