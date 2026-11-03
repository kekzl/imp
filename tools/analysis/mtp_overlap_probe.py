#!/usr/bin/env python3
# CHATS users (2) on one imp-server whose turns overlap: chat 0 starts, chat c starts once chat c - 1's first answer
# has streamed START deltas, each runs TURNS streamed turns with PAUSES seconds of think time between them
# (assistant turn sent back as `content` only). Per turn: completion id, prompt/cached tokens, decode tok/s,
# MTP drafted/accepted from that request's usage (completion_tokens_details.imp_spec_*), start/end seconds.
# Usage: mtp_overlap_probe.py PORT [TAG]; env MODEL_NAME, TURNS (6), GEN (1500), START (300), PAUSES ("2,6"),
#        CHATS (2), OUT_JSON. mtp_overlap_probe.py --parse SERVER_LOG ROWS_JSON: per turn the MTP decline/unbind reasons of
#        its engine request (joined by "request <id> -> engine req N"), then totals.
import json
import os
import re
import sys
import threading
import time
import urllib.request
from collections import Counter

from mtp_multiturn_probe import SYSTEMS, TURNS_TEXT, max_word_run

REASON_RES = [(re.compile(r"mtp-spec: drafting off for req (\d+) \((.*)\)\s*$"), "unbind"),
              (re.compile(r"mtp-spec: no head cache covers req (\d+)'s prefix-cache resume at \d+ \((.*)\)\s*$"),
               "decline")]


def parse(log_path, rows_path):
    with open(rows_path) as f:
        rows = json.load(f)
    eng, reasons = {}, {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = re.search(r"request (\S+) -> engine req (\d+)", line)
            if m:
                eng[m.group(1)] = int(m.group(2))
                continue
            for rx, kind in REASON_RES:
                m = rx.search(line)
                if m:
                    reasons.setdefault(int(m.group(1)), []).append(f"{kind}: {m.group(2)}")
    total = Counter()
    for r in sorted(rows, key=lambda r: r["t0"]):
        rs = Counter(reasons.get(eng.get(r["id"], -1), []))
        total.update(rs)
        if r["drafted"] > 0:
            total["turns drafted"] += 1
        else:
            total["turns not drafted"] += 1
        print(f"c{r['chat']} turn{r['turn'] + 1} t {r['t0']:.1f}-{r['t1']:.1f} drafted {r['drafted']} "
              f"tps {r['tps']:.2f} reasons {dict(rs) if rs else '-'}")
    print("REASONS_TOTAL", dict(sorted(total.items())))


def table(paths_by_arm):
    # --table ARM=p1.json,p2.json ...: per chat turn the mean tok/s over pairs (min-max), drafted / accepted.
    arms = {}
    for spec in paths_by_arm:
        label, paths = spec.split("=", 1)
        runs = []
        for p in paths.split(","):
            with open(p) as f:
                runs.append(json.load(f))
        arms[label] = runs
    keys = sorted({(r["chat"], r["turn"]) for runs in arms.values() for rows in runs for r in rows})
    print("chat turn | " + " | ".join(f"{a} tps (min-max) drafted accepted" for a in arms))
    for c, t in keys:
        cells = []
        for runs in arms.values():
            got = [r for rows in runs for r in rows if r["chat"] == c and r["turn"] == t]
            tps = [r["tps"] for r in got]
            cells.append(f"{sum(tps) / max(1, len(tps)):.2f} ({min(tps, default=0):.2f}-{max(tps, default=0):.2f}) "
                         f"drafted {sum(r['drafted'] for r in got) // max(1, len(got))} "
                         f"accepted {sum(r['accepted'] for r in got) // max(1, len(got))}")
        print(f"c{c} turn{t + 1} | " + " | ".join(cells))
    for a, runs in arms.items():
        rows = [r for rs in runs for r in rs]
        means = [sum(r["tps"] for r in rs) / max(1, len(rs)) for rs in runs]
        print(f"{a} mean_tps {sum(r['tps'] for r in rows) / max(1, len(rows)):.2f} per-pair "
              f"{' '.join(f'{m:.2f}' for m in means)} drafted_turns "
              f"{sum(1 for r in rows if r['drafted'] > 0)}/{len(rows)} drafted {sum(r['drafted'] for r in rows)} "
              f"accepted {sum(r['accepted'] for r in rows)}")


def turn(url, model, messages, gen, seed, t_base, on_delta=None):
    body = {"model": model, "messages": messages, "max_tokens": gen, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    content, reasoning, usage, finish, cid, n_delta = [], [], None, None, None, 0
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            cid = cid or ev.get("id")
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                finish = ch.get("finish_reason") or finish
                d = ch.get("delta", {})
                if d.get("content") or d.get("reasoning_content"):
                    now = time.perf_counter()
                    t_first = t_first or now
                    t_last = now
                    content.append(d.get("content") or "")
                    reasoning.append(d.get("reasoning_content") or "")
                    n_delta += 1
                    if on_delta:
                        on_delta(n_delta)
    usage = usage or {}
    det = usage.get("completion_tokens_details") or {}
    n = usage.get("completion_tokens", 0)
    tps = (n - 1) / (t_last - t_first) if t_first and t_last > t_first else 0.0
    return {"id": cid, "prompt": usage.get("prompt_tokens", 0),
            "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0), "completion": n,
            "tps": tps, "finish": finish, "drafted": det.get("imp_spec_drafted", 0),
            "accepted": det.get("imp_spec_accepted", 0), "declined": det.get("imp_spec_declined"),
            "t0": t0 - t_base, "t1": time.perf_counter() - t_base, "content": "".join(content),
            "reasoning": "".join(reasoning)}


def main():
    if sys.argv[1] == "--parse":
        parse(sys.argv[2], sys.argv[3])
        return 0
    if sys.argv[1] == "--table":
        table(sys.argv[2:])
        return 0
    port = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "x"
    url = f"http://127.0.0.1:{port}"
    model = os.environ.get("MODEL_NAME", "Qwen3.8-27B-NVFP4-vllm")
    turns = int(os.environ.get("TURNS", "6"))
    gen = int(os.environ.get("GEN", "1500"))
    start = int(os.environ.get("START", "300"))
    pauses = [float(p) for p in os.environ.get("PAUSES", "2,6").split(",")]
    chats = int(os.environ.get("CHATS", "2"))
    go = [threading.Event() for _ in range(chats + 1)]  # go[c]: chat c starts (chat c - 1 streamed START deltas)
    rows, lock, ok = [], threading.Lock(), [True]
    t_base = time.perf_counter()

    def chat(c):
        system = SYSTEMS[c % len(SYSTEMS)] + (f" You are session {c}." if c >= len(SYSTEMS) else "")
        messages = [{"role": "system", "content": system}]
        if c > 0:
            go[c].wait()
        for i, text in enumerate(TURNS_TEXT[:turns]):
            messages.append({"role": "user", "content": text})
            cb = (lambda n: go[c + 1].set() if n >= start else None) if i == 0 else None
            r = turn(url, model, messages, gen, 7 + i, t_base, cb)
            r.update(chat=c, turn=i)
            run = max_word_run(r["content"] + " " + r["reasoning"])
            clean = run <= 4 and len(r["content"].split()) >= 10
            with lock:
                ok[0] &= clean
                rows.append(r)
                print(f"{tag} c{c} turn{i + 1} id {r['id']} t {r['t0']:.1f}-{r['t1']:.1f} prompt {r['prompt']} "
                      f"cached {r['cached']} gen {r['completion']} tps {r['tps']:.2f} drafted {r['drafted']} "
                      f"accepted {r['accepted']} declined {r['declined']} finish {r['finish']} max_run {run} "
                      f"{'ok' if clean else 'DEGEN'}", flush=True)
            messages.append({"role": "assistant", "content": r["content"]})
            go[c + 1].set()  # a chat shorter than START deltas still releases the next one
            time.sleep(pauses[c % len(pauses)])

    threads = [threading.Thread(target=chat, args=(c,)) for c in range(chats)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    per_chat = []
    for c in range(chats):
        rc = [r for r in rows if r["chat"] == c]
        per_chat.append(f"c{c}_mean_tps {sum(r['tps'] for r in rc) / max(1, len(rc)):.2f} "
                        f"c{c}_drafted_turns {sum(1 for r in rc if r['drafted'] > 0)}/{len(rc)}")
    print(f"{tag} SUMMARY {' '.join(per_chat)} "
          f"all_mean_tps {sum(r['tps'] for r in rows) / max(1, len(rows)):.2f} {'PASS' if ok[0] else 'FAIL'}")
    if os.environ.get("OUT_JSON"):
        with open(os.environ["OUT_JSON"], "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if ok[0] else 1


if __name__ == "__main__":
    sys.exit(main())
