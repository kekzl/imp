#!/usr/bin/env python3
"""Multi-turn agentic replay benchmark for OpenAI-compatible servers.

Simulates the workload coding agents actually generate: ONE session whose
conversation grows turn over turn (system + tool defs, then user/assistant
pairs appended), re-sent in full every turn. What matters to the agent is
the PER-TURN time-to-first-token at growing context depth — i.e. how much
of the previous turns' prefill the server actually reuses (prefix cache) —
plus per-turn decode speed.

This is deliberately different from tools/agent_bench.py (static shared
prefix, concurrent one-shot requests) and tools/analysis/load_test.py
(rotating short prompts): neither grows a prefix across sequential turns.

Protocol per turn i:
  1. messages = [system] + turns[0..i) + [user_i]   (user_i ~ --turn-tokens
     of synthetic code-review text, unique per turn so turns never collide)
  2. POST /v1/chat/completions, stream=true, temperature=0,
     stream_options.include_usage=true, cache_prompt=true (omit via
     --no-cache-prompt for servers that reject unknown fields)
  3. TTFT = first SSE delta carrying content (role-only chunk skipped),
     measured from just before the socket write.
  4. The assistant reply is appended verbatim to the conversation.

Output: per-turn table (prompt tokens, cached tokens, TTFT ms, decode tok/s)
plus aggregate p50/p95 TTFT, and optionally --json for machine consumption.

Example:
  python3 tools/multiturn_bench.py --url http://localhost:8080 \
      --model Qwen3-Coder-30B-A3B-Instruct-FP4 --turns 24 --json out.json
"""
import argparse
import json
import sys
import time
import urllib.request

TOOL_DEFS = (
    "You are a coding agent. Tools: read_file(path), write_file(path, content), "
    "run_shell(cmd), search(pattern), list_dir(path). Follow repository "
    "conventions, produce minimal diffs, never fabricate paths or APIs. "
)

CODE_SNIPPET = """\
def process_batch_{i}(items, config):
    \"\"\"Process a batch of items with retry and backoff.\"\"\"
    results = []
    for attempt in range(config.max_retries):
        try:
            for item in items:
                validated = validate_schema(item, config.schema_{i})
                transformed = apply_transform(validated, config.rules)
                results.append(persist(transformed, config.target_{i}))
            return results
        except TransientError as exc:
            backoff = config.base_delay * (2 ** attempt)
            log.warning("batch %d attempt %d failed: %s", {i}, attempt, exc)
            time.sleep(backoff)
    raise BatchFailed({i}, len(items))
"""


def build_system(repeat: int) -> str:
    return "SYSTEM INSTRUCTIONS\n" + TOOL_DEFS * repeat


def build_user_turn(i: int, snippet_reps: int) -> str:
    code = "".join(CODE_SNIPPET.replace("{i}", str(i * 100 + j)) for j in range(snippet_reps))
    return (
        f"Review chunk {i} of the module below. Point out the single most "
        f"important issue and show the corrected function.\n\n```python\n{code}```"
    )


def percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def run_turn(url, body, timeout):
    """Stream one chat completion; return (ttft_s, text, usage, e2e_s, n_events)."""
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    t_last = t0
    text = []
    usage = None
    n_events = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content")
                if piece:
                    n_events += 1
                    t_last = time.time()
                    if ttft is None:
                        ttft = t_last - t0
                    text.append(delta.get("content") or "")
    return ttft, "".join(text), usage, t_last - t0, n_events


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=24)
    ap.add_argument("--target-ctx", type=int, default=0,
                    help="stop once prompt tokens exceed this (0 = run all turns)")
    ap.add_argument("--system-repeat", type=int, default=60,
                    help="tool-def block repetitions in the system prompt (~25 tok each)")
    ap.add_argument("--snippet-reps", type=int, default=3,
                    help="code snippets per user turn (~170 tok each)")
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--no-cache-prompt", action="store_true",
                    help="omit the cache_prompt field (strict OpenAI-schema servers)")
    ap.add_argument("--json", dest="json_out", default="",
                    help="write per-turn records + aggregates to this file")
    ap.add_argument("--label", default="", help="free-form label recorded in the JSON")
    args = ap.parse_args()

    messages = [{"role": "system", "content": build_system(args.system_repeat)}]
    rows = []

    for i in range(args.turns):
        messages.append({"role": "user", "content": build_user_turn(i, args.snippet_reps)})
        body = {
            "model": args.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if not args.no_cache_prompt:
            body["cache_prompt"] = True
        try:
            ttft, reply, usage, e2e, n_events = run_turn(args.url, body, args.timeout)
        except Exception as exc:
            print(f"turn {i}: request failed: {exc}", file=sys.stderr)
            return 1
        usage = usage or {}
        ptok = usage.get("prompt_tokens", 0)
        ctok = usage.get("completion_tokens", 0)
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        decode_tps = (ctok - 1) / (e2e - ttft) if ttft and e2e > ttft and ctok > 1 else 0.0
        rows.append({
            "turn": i, "prompt_tokens": ptok, "cached_tokens": cached,
            "completion_tokens": ctok, "ttft_ms": (ttft or 0) * 1000,
            "decode_tps": decode_tps, "e2e_s": e2e, "stream_events": n_events,
        })
        print(f"turn {i:3d}  prompt {ptok:7d}  cached {cached:7d}  "
              f"ttft {rows[-1]['ttft_ms']:8.1f} ms  decode {decode_tps:7.1f} tok/s  "
              f"out {ctok:5d}", flush=True)
        messages.append({"role": "assistant", "content": reply})
        if args.target_ctx and ptok >= args.target_ctx:
            break

    ttfts = [r["ttft_ms"] for r in rows]
    tps = [r["decode_tps"] for r in rows if r["decode_tps"] > 0]
    # Cold turn 0 measures raw prefill; steady-state cache behavior starts at 1.
    steady = ttfts[1:] if len(ttfts) > 1 else ttfts
    agg = {
        "turns": len(rows),
        "final_prompt_tokens": rows[-1]["prompt_tokens"] if rows else 0,
        "ttft_ms_p50_steady": percentile(steady, 0.50),
        "ttft_ms_p95_steady": percentile(steady, 0.95),
        "ttft_ms_max": max(ttfts) if ttfts else 0,
        "decode_tps_p50": percentile(tps, 0.50),
        "cached_ratio_last": (rows[-1]["cached_tokens"] / rows[-1]["prompt_tokens"])
        if rows and rows[-1]["prompt_tokens"] else 0.0,
    }
    print("\n# aggregate")
    for k, v in agg.items():
        print(f"  {k:24s} {v:10.1f}" if isinstance(v, float) else f"  {k:24s} {v:10d}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({"label": args.label, "model": args.model, "url": args.url,
                       "params": vars(args), "rows": rows, "aggregate": agg}, f, indent=1)
        print(f"json written: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
