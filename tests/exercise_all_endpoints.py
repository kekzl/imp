#!/usr/bin/env python3
"""MANUAL tool — not wired into ctest/CI (TEST_AUDIT.md §7). Needs a running
imp-server on :8080 with a tool-capable model (default Qwen3-8B-NVFP4-cortecs).

Drives every imp-server endpoint and mode once — OpenAI chat/completions/
embeddings, Anthropic /v1/messages, tokenize/detokenize, models/metrics/health,
streaming, tools/function-calling (incl. multi-turn with tool results),
json_schema, logprobs, thinking. Used as the coverage driver by
scripts/coverage_server.sh and as a broad smoke test. Prints status per case;
exits non-zero only on a 5xx (handlers must never 5xx on these valid calls).
"""
import json, os, sys, urllib.request, urllib.error

B = os.environ.get("IMP_BASE", "http://localhost:8080").rstrip("/")
M = os.environ.get("IMP_MODEL", "Qwen3-8B-NVFP4-cortecs")
_bad = []

def call(method, path, obj=None, timeout=120):
    data = json.dumps(obj).encode() if obj is not None else None
    req = urllib.request.Request(B + path, data=data, method=method,
                                 headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read() or b""
    except Exception as e:
        return f"ERR:{e}", b""

def chat(**kw):
    base = {"model": M, "messages": [{"role": "user", "content": "Say PONG"}], "max_tokens": 16}
    base.update(kw)
    return call("POST", "/v1/chat/completions", base)

OAI_TOOLS = [
    {"type": "function", "function": {"name": "get_weather", "description": "weather",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
    {"type": "function", "function": {"name": "calc", "description": "evaluate",
     "parameters": {"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]}}},
]
ANT_TOOLS = [{"name": "get_weather", "description": "weather",
              "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]

cases = [
    ("GET  /health",         lambda: call("GET", "/health")),
    ("GET  /v1/models",      lambda: call("GET", "/v1/models")),
    ("GET  /metrics",        lambda: call("GET", "/metrics")),
    ("OPTIONS preflight",    lambda: call("OPTIONS", "/v1/chat/completions")),
    ("tokenize",             lambda: call("POST", "/tokenize", {"model": M, "content": "hello world foo"})),
    ("detokenize",           lambda: call("POST", "/detokenize", {"model": M, "tokens": [9707, 1879]})),
    ("chat nonstream",       lambda: chat()),
    ("chat temp0",           lambda: chat(temperature=0)),
    ("chat stream",          lambda: chat(stream=True)),
    ("chat system+multiturn", lambda: call("POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "system", "content": "Be terse."}, {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}, {"role": "user", "content": "Say PONG"}], "max_tokens": 16})),
    ("chat logprobs",        lambda: chat(logprobs=True, top_logprobs=3)),
    ("chat penalties",       lambda: chat(frequency_penalty=0.5, presence_penalty=0.3, repetition_penalty=1.1)),
    ("chat min_p/top_k",     lambda: chat(min_p=0.05, top_k=20, top_p=0.9)),
    ("chat seed",            lambda: chat(seed=42, temperature=0.7)),
    ("chat stop seq",        lambda: chat(stop=["PONG"])),
    ("chat n=2",             lambda: chat(n=2)),
    ("chat think off",       lambda: chat(enable_thinking=False)),
    ("chat tools",           lambda: call("POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "user", "content": "weather in Paris?"}], "tools": OAI_TOOLS, "max_tokens": 80})),
    ("chat tool_choice required", lambda: call("POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "user", "content": "weather?"}], "tools": OAI_TOOLS, "tool_choice": "required", "max_tokens": 80})),
    ("chat tool_choice named", lambda: call("POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "user", "content": "2+2?"}], "tools": OAI_TOOLS,
        "tool_choice": {"type": "function", "function": {"name": "calc"}}, "max_tokens": 80})),
    ("chat tool result multiturn", lambda: call("POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
         "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "18C sunny"}], "tools": OAI_TOOLS, "max_tokens": 60})),
    ("chat json_schema",     lambda: call("POST", "/v1/chat/completions", {"model": M, "messages": [
        {"role": "user", "content": "give a city"}], "max_tokens": 60, "response_format": {"type": "json_schema",
        "json_schema": {"name": "c", "schema": {"type": "object", "properties": {"city": {"type": "string"}},
        "required": ["city"]}}}})),
    ("chat json_object",     lambda: chat(response_format={"type": "json_object"},
        messages=[{"role": "user", "content": "JSON with key x"}])),
    ("completions nonstream", lambda: call("POST", "/v1/completions", {"model": M, "prompt": "Say PONG", "max_tokens": 16})),
    ("completions stream",   lambda: call("POST", "/v1/completions", {"model": M, "prompt": "Hi", "max_tokens": 12, "stream": True}, timeout=30)),
    ("completions echo",     lambda: call("POST", "/v1/completions", {"model": M, "prompt": "Hi", "max_tokens": 8, "echo": True})),
    ("embeddings single",    lambda: call("POST", "/v1/embeddings", {"model": M, "input": "hello"})),
    ("embeddings array",     lambda: call("POST", "/v1/embeddings", {"model": M, "input": ["a", "b", "c"]})),
    ("anthropic nonstream",  lambda: call("POST", "/v1/messages", {"model": M, "messages": [
        {"role": "user", "content": "Say PONG"}], "max_tokens": 16})),
    ("anthropic stream",     lambda: call("POST", "/v1/messages", {"model": M, "messages": [
        {"role": "user", "content": "Say PONG"}], "max_tokens": 16, "stream": True}, timeout=30)),
    ("anthropic system+think", lambda: call("POST", "/v1/messages", {"model": M, "system": "Be terse.",
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 32,
        "thinking": {"type": "enabled", "budget_tokens": 64}})),
    ("anthropic tools",      lambda: call("POST", "/v1/messages", {"model": M, "messages": [
        {"role": "user", "content": "weather in Paris?"}], "max_tokens": 80, "tools": ANT_TOOLS})),
    ("anthropic tool result", lambda: call("POST", "/v1/messages", {"model": M, "max_tokens": 60, "tools": ANT_TOOLS,
        "messages": [{"role": "user", "content": "weather in Paris?"},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "18C"}]}]})),
]

for name, fn in cases:
    st, body = fn()
    flag = ""
    if isinstance(st, int) and st >= 500:
        _bad.append((name, st)); flag = "  <-- 5xx!"
    print(f"  {name:30s} -> {st} ({len(body)}B){flag}")

if _bad:
    print(f"FAIL: {len(_bad)} endpoint(s) returned 5xx: {_bad}")
    sys.exit(1)
print("done — no 5xx on any endpoint/mode")
