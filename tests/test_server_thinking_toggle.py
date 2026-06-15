#!/usr/bin/env python3
"""MANUAL gate — part of the local server-test stage (`make test-server`).
Needs a running imp-server with a THINK-capable model (default Qwen3-8B-NVFP4-cortecs).

Asserts that thinking can actually be turned OFF (and on), end-to-end, on both
dialects. The bug this guards: Qwen3's template renders a closed `<think></think>`
block when thinking is disabled, but handlers.cpp re-enabled thinking on seeing
"<think>" in the prompt tail — so the model reasoned anyway. The fix only
re-enables on an *unclosed* prefix.

Cases (think-model required; auto-skips with exit 0 if the model never reasons):
  * Anthropic /v1/messages  thinking:{type:disabled}  -> NO thinking block
  * Anthropic /v1/messages  thinking:{type:enabled}   -> a thinking block
  * OpenAI  /v1/chat/completions enable_thinking:false+think_budget:0 -> no reasoning_content
  * OpenAI  /v1/chat/completions (default)            -> reasoning_content present
Several prompts are tried because the OLD soft-/no_think suppression was
prompt-dependent — a single prompt could pass by luck.

Run:  python3 tests/test_server_thinking_toggle.py
Env:  IMP_BASE (default http://localhost:8080), IMP_MODEL (auto from /v1/models)
Exit: 0 = pass (or model is non-thinking), 1 = a case regressed.
"""
import json, os, sys, urllib.request

BASE = os.environ.get("IMP_BASE", "http://localhost:8080").rstrip("/")
PROMPTS = [
    "What is 2+2? Reply with just the number.",
    "Reply with exactly: hello world",
    "Name the capital of France in one word.",
]
_fail = []


def _post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def model_id():
    if os.environ.get("IMP_MODEL"):
        return os.environ["IMP_MODEL"]
    return json.load(urllib.request.urlopen(BASE + "/v1/models", timeout=10))["data"][0]["id"]


M = model_id()


def fail(msg):
    _fail.append(msg)
    print(f"  FAIL {msg}")


def anth_thinking_chars(prompt, thinking):
    r = _post("/v1/messages", {"model": M, "max_tokens": 512, "temperature": 0,
                               "thinking": thinking,
                               "messages": [{"role": "user", "content": prompt}]})
    blk = next((b.get("thinking", "") for b in r.get("content", []) if b.get("type") == "thinking"), None)
    return len(blk.strip()) if blk is not None else 0


def oai_reasoning_chars(prompt, **extra):
    body = {"model": M, "max_tokens": 512, "temperature": 0,
            "messages": [{"role": "user", "content": prompt}]}
    body.update(extra)
    m = _post("/v1/chat/completions", body)["choices"][0]["message"]
    rc = m.get("reasoning_content")
    return len(rc.strip()) if rc else 0


def main():
    print(f"thinking-toggle gate: base={BASE} model={M}")

    # Is this even a think-model? If default never reasons, skip (exit 0).
    default_reasons = max(oai_reasoning_chars(p) for p in PROMPTS)
    if default_reasons == 0:
        print("  SKIP: model does not reason by default (not a think-model)")
        return 0
    print(f"  baseline: default reasoning up to {default_reasons} chars")

    for p in PROMPTS:
        # Anthropic: disabled must suppress, enabled must reason.
        dis = anth_thinking_chars(p, {"type": "disabled"})
        if dis > 0:
            fail(f"/v1/messages disabled still reasoned {dis} chars — prompt={p!r}")
        en = anth_thinking_chars(p, {"type": "enabled", "budget_tokens": 256})
        if en == 0:
            fail(f"/v1/messages enabled produced no thinking — prompt={p!r}")

        # OpenAI: both signals off must suppress.
        oai_dis = oai_reasoning_chars(p, enable_thinking=False, think_budget=0)
        if oai_dis > 0:
            fail(f"/v1/chat/completions disabled still reasoned {oai_dis} chars — prompt={p!r}")
        print(f"  ok prompt={p!r}: anth dis={dis} en={en}, oai dis={oai_dis}")

    if _fail:
        print(f"FAIL: {len(_fail)} thinking-toggle case(s) regressed")
        return 1
    print("PASS: thinking disables (no reasoning) and enables (reasoning) on both dialects")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
