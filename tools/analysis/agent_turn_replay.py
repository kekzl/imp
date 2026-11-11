#!/usr/bin/env python3
"""Replay recorded agent sessions against imp-server, one model turn per recorded assistant turn.

Each chat walks one session file (Anthropic block format: `{"messages": [{role, content: [text |
thinking | tool_use | tool_result]}]}`): request t sends the recorded messages before assistant turn t
to /v1/messages (stream) and reads the model's own turn; the next request continues from the RECORDED
turn, so every request extends the previous prompt the way an agent loop does (prefix cache hits) and
no tool has to run. Tools are rebuilt from the recorded tool_use blocks (name, argument names, JSON
types). The recording holds no system prompt: --system FILE sends one, else none is sent.

Reports per chat and in total: requests, prompt and output tokens, reasoning share, decode seconds (stream
end minus first delta) and output tokens per decode second, plus output tokens per wall second of the run,
and an md5 over each chat's stream deltas (equal texts print equal md5s: --seed plus runtime.deterministic=true).
Stdlib only.

Usage:
  agent_turn_replay.py --sessions DIR_OR_FILE [...] [--url http://localhost:8080] [--chats 4] [--skip 0]
                       [--turns 20] [--first-turn 2] [--max-tokens 4096] [--think-budget 2048]
                       [--serial] [--system FILE] [--temperature T] [--seed N]
"""
import argparse
import hashlib
import json
import pathlib
import threading
import time
import urllib.request

JSON_TYPE = {str: "string", bool: "boolean", int: "integer", float: "number", list: "array", dict: "object"}


def session_files(paths):
    out = []
    for p in map(pathlib.Path, paths):
        out += sorted(p.rglob("*.json")) if p.is_dir() else [p]
    return out


def load(path):
    try:
        doc = json.load(open(path))
    except (OSError, ValueError):
        return None
    msgs = doc.get("messages") if isinstance(doc, dict) else None
    if not isinstance(msgs, list) or not msgs or not isinstance(msgs[0], dict) or "role" not in msgs[0]:
        return None
    return doc


def tools_of(docs):
    """name -> argument name -> JSON type, over every recorded tool_use block."""
    tools = {}
    for doc in docs:
        for m in doc["messages"]:
            if not isinstance(m.get("content"), list):
                continue
            for b in m["content"]:
                if b.get("type") != "tool_use":
                    continue
                props = tools.setdefault(b["name"], {})
                for k, v in (b.get("input") or {}).items():
                    props.setdefault(k, JSON_TYPE.get(type(v), "string"))
    return [{"name": n, "description": n, "input_schema": {"type": "object", "properties": {
        k: {"type": t} for k, t in sorted(p.items())}}} for n, p in sorted(tools.items())]


def stream_turn(url, body, timeout):
    """One /v1/messages stream: (prompt tok, cached, output tok, ttft s, total s, stop, reasoning tok, md5)."""
    req = urllib.request.Request(url + "/v1/messages", data=json.dumps(body).encode(),
                                 headers={"content-type": "application/json", "anthropic-version": "2023-06-01"})
    t0 = time.monotonic()
    first = None
    n_in = cached = n_out = n_think = 0
    stop = "?"
    digest = hashlib.md5()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            if not raw.startswith(b"data:"):
                continue
            try:
                ev = json.loads(raw[5:])
            except ValueError:
                continue
            kind = ev.get("type")
            if kind == "message_start":
                u = ev.get("message", {}).get("usage", {})
                cached = u.get("cache_read_input_tokens", 0) or 0
                n_in = (u.get("input_tokens", 0) or 0) + cached + (u.get("cache_creation_input_tokens", 0) or 0)
            elif kind == "content_block_start":  # block type + tool name; the tool_use id is per run
                b = ev.get("content_block", {})
                digest.update(f"\0{b.get('type')}\0{b.get('name', '')}\0".encode())
            elif kind == "content_block_delta":  # the strings, not the events: delta cuts follow the verify chunks
                digest.update("".join(v for k, v in sorted(ev.get("delta", {}).items())
                                      if k not in ("type", "signature") and isinstance(v, str)).encode())
                if first is None:
                    first = time.monotonic()
            elif kind == "message_delta":
                u = ev.get("usage", {})
                n_out = u.get("output_tokens", n_out) or n_out
                n_think = u.get("output_tokens_details", {}).get("reasoning_tokens", n_think) or n_think
                if "input_tokens" in u:  # the final split: message_start is sent before the prefix lookup
                    cached = u.get("cache_read_input_tokens", 0) or 0
                    n_in = (u.get("input_tokens", 0) or 0) + cached + (u.get("cache_creation_input_tokens", 0) or 0)
                stop = ev.get("delta", {}).get("stop_reason") or stop
    t1 = time.monotonic()
    return n_in, cached, n_out, (first or t1) - t0, t1 - t0, stop, n_think, digest.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--chats", type=int, default=4)
    ap.add_argument("--turns", type=int, default=20)
    ap.add_argument("--first-turn", type=int, default=2, help="assistant turns skipped at the session start")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--think-budget", type=int, default=2048, help="0 = no thinking block requested")
    ap.add_argument("--model", default="")
    ap.add_argument("--skip", type=int, default=0, help="eligible sessions skipped")
    ap.add_argument("--serial", action="store_true", help="one chat after the other: every request decodes alone")
    ap.add_argument("--system", default="", help="file with a system prompt")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--temperature", type=float, default=None, help="sent only when given (0 = greedy)")
    ap.add_argument("--seed", type=int, default=None, help="sent only when given; request t of a chat sends seed + t")
    a = ap.parse_args()

    docs = [(p, d) for p, d in ((p, load(p)) for p in session_files(a.sessions)) if d]
    tools = tools_of([d for _, d in docs])
    system = open(a.system).read() if a.system else ""
    chats = []
    for path, doc in docs:
        turns = [i for i, m in enumerate(doc["messages"]) if m["role"] == "assistant" and i > 0]
        turns = turns[a.first_turn:a.first_turn + a.turns]
        if len(turns) == a.turns:
            chats.append((doc, turns, path.stem))
    chats = chats[a.skip:a.skip + a.chats]
    if len(chats) < a.chats:
        raise SystemExit(f"only {len(chats)} of {len(docs)} sessions hold {a.first_turn + a.turns} turns "
                         f"past --skip {a.skip}")
    print(f"chats {len(chats)} turns {a.turns} tools {len(tools)} max_tokens {a.max_tokens} "
          f"think_budget {a.think_budget} serial {int(a.serial)} system {len(system)} chars")

    rows = [[] for _ in chats]
    errors = []

    def run(c):
        doc, turns, _ = chats[c]
        for t, i in enumerate(turns):
            body = {"model": a.model or doc.get("model", "imp"), "max_tokens": a.max_tokens, "stream": True,
                    "messages": doc["messages"][:i], "tools": tools}
            if system:
                body["system"] = system
            if a.temperature is not None:
                body["temperature"] = a.temperature
            if a.seed is not None:
                body["seed"] = a.seed + t
            if a.think_budget > 0:
                body["thinking"] = {"type": "enabled", "budget_tokens": a.think_budget}
            try:
                rows[c].append(stream_turn(a.url, body, a.timeout))
            except Exception as e:  # noqa: BLE001 - a failed turn fails the run below
                errors.append(f"chat {c} turn at message {i}: {e}")
                return

    t0 = time.monotonic()
    threads = [threading.Thread(target=run, args=(c,)) for c in range(len(chats))]
    for t in threads:
        t.start()
        if a.serial:
            t.join()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    print("chat | session | requests | prompt tok (last) | cached % | output tok | reasoning % | decode s | "
          "tok/s decode | stops tool_use/end_turn/max_tokens | text md5")
    tot_out = tot_dec = tot_req = tot_think = 0
    for c, rs in enumerate(rows):
        if not rs:
            continue
        n_out = sum(r[2] for r in rs)
        n_think = sum(r[6] for r in rs)
        dec = sum(r[4] - r[3] for r in rs)
        n_in = sum(r[0] for r in rs)
        stops = [sum(1 for r in rs if r[5] == s) for s in ("tool_use", "end_turn", "max_tokens")]
        print(f"{c:4d} | {chats[c][2]} | {len(rs):3d} | {rs[-1][0]:7d} | "
              f"{100.0 * sum(r[1] for r in rs) / max(1, n_in):5.1f} | {n_out:7d} | "
              f"{100.0 * n_think / max(1, n_out):5.1f} | {dec:8.2f} | {n_out / max(dec, 1e-9):7.2f} | "
              f"{stops[0]}/{stops[1]}/{stops[2]} | {hashlib.md5(''.join(r[7] for r in rs).encode()).hexdigest()[:12]}")
        tot_out += n_out
        tot_think += n_think
        tot_dec += dec
        tot_req += len(rs)
    print(f"total | requests {tot_req} | output tok {tot_out} | reasoning {100.0 * tot_think / max(1, tot_out):.1f} % | "
          f"wall {wall:.2f} s | "
          f"{tot_out / max(wall, 1e-9):.2f} tok/s wall | {tot_out / max(tot_dec, 1e-9):.2f} tok/s decode per chat")
    for c, rs in enumerate(rows):
        print(f"turn md5 | {c} | " + " ".join(r[7][:6] for r in rs))
    for e in errors:
        print("ERROR", e)
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
