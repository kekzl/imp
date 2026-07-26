#!/usr/bin/env python3
"""Cross-engine agentic-quality comparison (imp vs any OpenAI-compatible server).

imp can prove it is faster; it could not prove it is more *reliable* for agent
work (docs/roadmap.md gap 7). Speed is published per hero model, but nothing
measured whether the JSON contract or a tool call survives — against another
engine, same model, same requests.

This runs the checks an agent harness actually depends on:

  json_schema     schema-valid object at a REALISTIC token budget
  json_object     parseable JSON at the same budget
  tool_forced     tool_choice=required actually emits a tool call
  tool_args       those arguments parse and carry the required fields
  tool_optional   tool_choice=auto does NOT force a call on a chat turn

Budget is the point, not an afterthought. A think-capable model can spend the
whole budget reasoning and return an empty `content`; whether that happens by
default is a real difference between engines, so every category is measured at
a budget an agent would plausibly set, and `--budget-sweep` shows where each
engine starts succeeding.

Fairness rules baked in:
  * same model file, same sampling params, same prompts on both engines;
  * a failure is recorded with its cause (empty content vs invalid JSON vs no
    call), because "empty because it was still thinking" and "invalid JSON" are
    different verdicts;
  * `--thinking-off` re-runs with thinking disabled via chat_template_kwargs,
    which is how you tell a DEFAULT difference from a CAPABILITY difference.

Usage:
  python3 tools/analysis/agentic_compare.py \
      --engine imp=http://localhost:8080 --engine llama.cpp=http://localhost:8081 \
      --reps 5 --budget 200
  # is a failure just the default? add --thinking-off for a second pass
Exit code is 0 unless an engine was unreachable; this reports, it does not gate.
"""

import argparse
import json
import statistics
import sys
import urllib.error
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["c", "f"]},
            },
            "required": ["city"],
        },
    },
}]

PERSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
    },
}


def post(url, body, timeout=180):
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def model_id(url):
    """Ask the server what it serves. imp validates the `model` field (it can
    swap models), so a placeholder name is correctly rejected with 404 —
    llama.cpp ignores the field. Same request must be valid on both."""
    try:
        with urllib.request.urlopen(url + "/v1/models", timeout=10) as r:
            data = json.loads(r.read().decode())
        entries = data.get("data") or data.get("models") or []
        for e in entries:
            if isinstance(e, dict) and (e.get("id") or e.get("model")):
                return e.get("id") or e.get("model")
    except Exception:
        pass
    return "x"


def base_body(budget, thinking_off, model):
    body = {"model": model, "max_tokens": budget, "temperature": 0}
    if thinking_off:
        # Both engines honour this; it is the documented way to turn a
        # think-capable template off per request.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _msg(resp):
    return (resp.get("choices") or [{}])[0].get("message") or {}


def _finish(resp):
    return (resp.get("choices") or [{}])[0].get("finish_reason") or "?"


def _tokens(resp):
    return (resp.get("usage") or {}).get("completion_tokens", 0)


def check_json(resp, schema_required):
    """Returns (ok, cause). Distinguishes 'never answered' from 'bad JSON'."""
    content = (_msg(resp).get("content") or "").strip()
    if not content:
        why = "empty (spent budget thinking)" if _finish(resp) == "length" else "empty"
        return False, why
    try:
        obj = json.loads(content)
    except Exception:
        return False, "not JSON"
    if schema_required:
        if not isinstance(obj, dict):
            return False, "not an object"
        for k, t in (("name", str), ("age", int)):
            if k not in obj:
                return False, f"missing '{k}'"
            if not isinstance(obj[k], t):
                return False, f"'{k}' wrong type"
    return True, ""


def check_tool(resp, want_call):
    calls = _msg(resp).get("tool_calls") or []
    if not want_call:
        return (not calls), ("forced a call when optional" if calls else "")
    if not calls:
        why = "no tool_call"
        if _finish(resp) == "length":
            why += " (spent budget thinking)"
        return False, why
    return True, ""


def check_tool_args(resp):
    calls = _msg(resp).get("tool_calls") or []
    if not calls:
        return False, "no tool_call"
    raw = (calls[0].get("function") or {}).get("arguments") or ""
    try:
        args = json.loads(raw)
    except Exception:
        return False, "arguments not JSON"
    if "city" not in args:
        return False, "missing required 'city'"
    return True, ""


def check_multiturn(url, body_base, reps, turns=3):
    """Does the JSON contract survive a conversation, not just one shot?

    An agent asks repeatedly with history growing underneath. Template drift,
    KV reuse or a thinking block re-entering on a later turn all show up here
    and nowhere in a single-shot test. Every turn must be schema-valid; the
    turn that first breaks is reported."""
    ok_n, causes = 0, []
    for _ in range(reps):
        msgs = [{"role": "user", "content": "Give me a person object."}]
        broke = ""
        for turn in range(turns):
            body = {**body_base, "messages": msgs, "response_format": PERSON_SCHEMA}
            try:
                resp = post(url, body)
            except Exception as e:
                broke = f"turn {turn + 1}: request failed: {e}"
                break
            ok, why = check_json(resp, True)
            if not ok:
                broke = f"turn {turn + 1}: {why}"
                break
            content = _msg(resp).get("content") or ""
            msgs = msgs + [{"role": "assistant", "content": content},
                           {"role": "user", "content": "Another one, different person."}]
        if broke:
            causes.append(broke)
        else:
            ok_n += 1
    return ok_n, sorted(set(causes))


CASES = {
    "json_schema": lambda b: (
        {**b, "messages": [{"role": "user", "content": "Give me a person object."}],
         "response_format": PERSON_SCHEMA},
        lambda r: check_json(r, True)),
    "json_object": lambda b: (
        {**b, "messages": [{"role": "user", "content":
                            "Reply with a JSON object holding a name and an age."}],
         "response_format": {"type": "json_object"}},
        lambda r: check_json(r, False)),
    "tool_forced": lambda b: (
        {**b, "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
         "tools": TOOLS, "tool_choice": "required"},
        lambda r: check_tool(r, True)),
    "tool_args": lambda b: (
        {**b, "messages": [{"role": "user", "content": "What is the weather in Berlin?"}],
         "tools": TOOLS, "tool_choice": "required"},
        check_tool_args),
    "tool_optional": lambda b: (
        {**b, "messages": [{"role": "user", "content": "Say hello in one word."}],
         "tools": TOOLS, "tool_choice": "auto"},
        lambda r: check_tool(r, False)),
}


def run_engine(name, url, reps, budget, thinking_off, turns=3):
    model = model_id(url)
    print(f"\n=== {name} ({url}) model={model} budget={budget} thinking_off={thinking_off} ===")
    results = {}
    ok_n, causes = check_multiturn(url, base_body(budget, thinking_off, model), reps, turns)
    results["json_multiturn"] = {"ok": ok_n, "n": reps, "tokens": 0, "causes": causes}
    mark = "PASS" if ok_n == reps else ("FAIL" if ok_n == 0 else "FLAKY")
    detail = f"  [{', '.join(causes)}]" if causes else ""
    print(f"  {mark:5} {'json_multiturn':14} {ok_n}/{reps}  {turns} turns each{detail}")
    for case, build in CASES.items():
        ok_n, causes, toks = 0, [], []
        for _ in range(reps):
            body, check = build(base_body(budget, thinking_off, model))
            try:
                resp = post(url, body)
            except Exception as e:
                causes.append(f"request failed: {e}")
                continue
            ok, why = check(resp)
            toks.append(_tokens(resp))
            if ok:
                ok_n += 1
            elif why:
                causes.append(why)
        med = int(statistics.median(toks)) if toks else 0
        results[case] = {"ok": ok_n, "n": reps, "tokens": med,
                         "causes": sorted(set(causes))}
        mark = "PASS" if ok_n == reps else ("FAIL" if ok_n == 0 else "FLAKY")
        detail = f"  [{', '.join(results[case]['causes'])}]" if results[case]["causes"] else ""
        print(f"  {mark:5} {case:14} {ok_n}/{reps}  median {med:4d} tok{detail}")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", action="append", required=True,
                    metavar="NAME=URL", help="repeatable, e.g. imp=http://localhost:8080")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--budget", type=int, default=200,
                    help="max_tokens per request (default 200, a realistic agent budget)")
    ap.add_argument("--turns", type=int, default=3,
                    help="turns in the multi-turn contract check (longer = template drift and KV reuse get more chances to break it)")
    ap.add_argument("--thinking-off", action="store_true",
                    help="disable thinking per request — separates a DEFAULT difference "
                         "from a CAPABILITY difference")
    ap.add_argument("--budget-sweep", type=str, default="",
                    help="comma-separated budgets, e.g. 200,400,800 — shows where each "
                         "engine starts succeeding")
    ap.add_argument("--json", type=str, default="", help="write raw results here")
    args = ap.parse_args()

    engines = []
    for spec in args.engine:
        if "=" not in spec:
            sys.exit(f"--engine wants NAME=URL, got {spec!r}")
        name, url = spec.split("=", 1)
        engines.append((name, url.rstrip("/")))

    for name, url in engines:
        try:
            urllib.request.urlopen(url + "/v1/models", timeout=10).read()
        except Exception as e:
            sys.exit(f"{name} unreachable at {url}: {e}")

    budgets = [int(b) for b in args.budget_sweep.split(",")] if args.budget_sweep \
        else [args.budget]
    out = {}
    for budget in budgets:
        for name, url in engines:
            out[f"{name}@{budget}"] = run_engine(name, url, args.reps, budget,
                                                 args.thinking_off, args.turns)

    print("\n" + "=" * 62)
    cases = list(CASES) + ["json_multiturn"]
    width = max(len(k) for k in out) + 2
    print("summary (passed/total)".ljust(width) + "  ".join(f"{c:>14}" for c in cases))
    for key, res in out.items():
        row = "  ".join(f"{res[c]['ok']}/{res[c]['n']:>12}" for c in cases)
        print(key.ljust(width) + row)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nraw results -> {args.json}")


if __name__ == "__main__":
    main()
