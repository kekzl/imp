#!/usr/bin/env python3
"""Multi-turn agent replay benchmark for imp-server.

agent_bench.py measures TTFT/ITL under *concurrency* with a static prefix.
This measures the other agentic axis: a SINGLE session whose transcript GROWS
turn by turn — the shape a coding agent actually produces. Each turn appends a
user message, an assistant tool call, and a tool result, so the prompt the
server must (re)process is monotonically longer. What the user feels is per-turn
TTFT at increasing depth; what makes it bearable is the prefix cache reusing the
unchanged head of the transcript across turns.

The bench replays the same scripted transcript twice:
  - cache ON:  the growing transcript keeps a stable head across turns, so the
    server-global prefix cache re-reads it from KV and only prefills the new
    suffix -> TTFT stays ~flat with depth.
  - cache OFF: the head is uniquified PER TURN, so every turn presents a novel
    token prefix the cache cannot match -> full re-prefill, TTFT grows ~linearly
    with depth. (NOTE: `cache_prompt=false` alone does NOT disable caching —
    it only controls Anthropic cache_control pinning; the prefix cache is
    server-global via `server.prefix_cache`, default ON. Defeating it by prefix
    novelty is the only in-process way to get a true no-cache baseline.)

TTFT is curl's time_total for a max_tokens=1 request (= full prefill + one
decode = time to first content token). Concurrency is one process per request
(no Python-thread GIL confound; same rationale as agent_bench.py). Turns run
strictly in sequence within a session — the growing transcript is the point.

Example:
  python3 tools/agent_replay_bench.py --url http://localhost:8080 \
      --model Qwen3-Coder-30B-A3B-Instruct-FP4 --turns 20 --turn-tokens 300
"""
import argparse
import json
import subprocess
import sys
import time

TOOL_SPEC = ("Available tools: read_file(path), write_file(path, content), "
             "run_shell(cmd), search(query), list_dir(path). Follow repo "
             "conventions, make minimal diffs, never fabricate paths. ")

# A chunky synthetic "tool result" (~file contents) so each turn adds real
# prefill work, not just a few tokens. ~1 token per ~4 chars.
FILE_BLOB = ("    def process(self, batch, ctx):\n"
             "        # handle the incoming batch under the given context\n"
             "        out = []\n"
             "        for row in batch:\n"
             "            out.append(self._transform(row, ctx))\n"
             "        return out\n") * 6


def system_prompt(repeat: int) -> str:
    return "You are a coding agent operating in a repository.\n" + TOOL_SPEC * repeat


def percentile(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(pos)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (pos - lo)


def build_messages(system, turn, cache_marker):
    """Transcript after `turn` completed turns, plus the next user ask.

    cache_marker uniquifies the system head per replay so a cache-OFF arm (or a
    fresh cache-ON session) is not served stale KV from a previous run.
    """
    msgs = [{"role": "system", "content": cache_marker + system}]
    for i in range(turn):
        msgs.append({"role": "user", "content": f"Step {i}: inspect and edit module {i}."})
        msgs.append({"role": "assistant", "content": f"I'll read module {i} first.\n"
                     f"<tool_call>read_file(\"src/module_{i}.py\")</tool_call>"})
        msgs.append({"role": "user", "content": f"Tool result for read_file src/module_{i}.py:\n{FILE_BLOB}"})
    msgs.append({"role": "user", "content": f"Now make the change for step {turn}."})
    return msgs


def body(model, messages, max_tokens, cache_prompt):
    return json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "stream": False,
        "cache_prompt": cache_prompt,
    })


def curl_ttft(url, payload):
    """One max_tokens=1 request: returns (ttft_s, prompt_tokens, cached_tokens).

    Body and timing come from the SAME request (body to stdout, time_total
    appended after a sentinel) so timing is never confounded by a separate
    priming call. TTFT for max_tokens=1 = full prefill + one decode = time to
    first content token.
    """
    p = subprocess.run(
        ["curl", "-s", "-w", "\n__TIME__%{time_total}",
         url + "/v1/chat/completions", "-H", "Content-Type: application/json",
         "-d", payload],
        capture_output=True, text=True, timeout=300)
    try:
        bodytext, _, tt = p.stdout.rpartition("\n__TIME__")
        ttft = float(tt.strip())
        u = json.loads(bodytext).get("usage", {})
        cached = u.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        return ttft, u.get("prompt_tokens", 0), cached
    except (ValueError, KeyError, AttributeError):
        return None, None, None


def run_arm(url, model, system, turns, turn_tokens, defeat_cache, label):
    session = f"[REPLAY {label} {time.time_ns()}]\n"
    print(f"\n## {label}")
    print(f"  {'turn':>4} {'prompt_tok':>10} {'cached':>8} {'TTFT_ms':>9}  bar")
    ttfts = []
    for turn in range(turns):
        # cache-ON: stable session head reused across turns (cache hits on the
        #   unchanged transcript prefix; ON arm is warmed once below first).
        # cache-OFF: novel head per turn so the prefix cache cannot match ->
        #   full re-prefill of the whole growing transcript every turn.
        marker = session + (f"turn-nonce-{time.time_ns()}\n" if defeat_cache else "")
        msgs = build_messages(system, turn, marker)
        if not defeat_cache:
            # Prime the reusable head once so the timed call reflects a warm
            # cache hit, not the cold first touch of this depth.
            curl_ttft(url, body(model, msgs, 1, True))
        t, ptok, cached = curl_ttft(url, body(model, msgs, 1, True))
        if t is None:
            print(f"  {turn:>4}  request failed")
            continue
        ttfts.append((turn, ptok or 0, cached or 0, t))
        bar = "#" * int(t * 1000 / 25)
        print(f"  {turn:>4} {ptok or 0:>10} {cached or 0:>8} {t*1000:>9.1f}  {bar}")
    return ttfts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=16)
    ap.add_argument("--turn-tokens", type=int, default=300,
                    help="max_tokens for the ITL/generation feel (TTFT uses max_tokens=1)")
    ap.add_argument("--prefix-repeat", type=int, default=8,
                    help="repeats of the tool spec in the system prompt")
    ap.add_argument("--cache", choices=["on", "off", "both"], default="both")
    args = ap.parse_args()

    system = system_prompt(args.prefix_repeat)
    print(f"# imp agent-replay-bench  model={args.model}")
    print(f"# growing transcript, {args.turns} turns, system prefix ~{len(system)//4} tokens")

    sys.stderr.write("warming model + clocks...\n")
    warm = build_messages(system, 0, "[WARM]\n")
    for _ in range(2):
        curl_ttft(args.url, body(args.model, warm, 8, True))

    arms = {}
    if args.cache in ("on", "both"):
        arms["cache-ON"] = run_arm(args.url, args.model, system, args.turns,
                                   args.turn_tokens, False, "cache-ON")
    if args.cache in ("off", "both"):
        arms["cache-OFF"] = run_arm(args.url, args.model, system, args.turns,
                                    args.turn_tokens, True, "cache-OFF")

    print("\n## Summary — per-turn TTFT vs transcript depth")
    for label, rows in arms.items():
        if not rows:
            continue
        ttfts = [r[3] for r in rows]
        first, last = ttfts[0], ttfts[-1]
        print(f"  {label:9s}  turn0={first*1000:7.1f} ms  "
              f"turn{len(ttfts)-1}={last*1000:7.1f} ms  "
              f"growth={last/first:5.2f}x  p50={percentile(ttfts,0.5)*1000:7.1f} ms")
    if "cache-ON" in arms and "cache-OFF" in arms and arms["cache-ON"] and arms["cache-OFF"]:
        on_last = arms["cache-ON"][-1][3]
        off_last = arms["cache-OFF"][-1][3]
        print(f"\n  deepest-turn cache speedup: {off_last/on_last:.2f}x "
              f"({off_last*1000:.0f} ms -> {on_last*1000:.0f} ms)")


if __name__ == "__main__":
    main()
