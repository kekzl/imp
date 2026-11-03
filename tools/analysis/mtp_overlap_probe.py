#!/usr/bin/env python3
# Two users on one imp-server whose turns overlap: chat 0 starts, chat 1 starts once chat 0's first answer
# has streamed START deltas, each runs TURNS streamed turns with PAUSES seconds of think time between them
# (assistant turn sent back as `content` only). Per turn: completion id, prompt/cached tokens, decode tok/s,
# MTP drafted/accepted from that request's usage (completion_tokens_details.imp_spec_*), start/end seconds.
# Usage: mtp_overlap_probe.py PORT [TAG]; env MODEL_NAME, TURNS (6), GEN (1500), START (300), PAUSES ("2,6"),
#        OUT_JSON. mtp_overlap_probe.py --parse SERVER_LOG ROWS_JSON: per turn the MTP decline/unbind reasons of
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
    port = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "x"
    url = f"http://127.0.0.1:{port}"
    model = os.environ.get("MODEL_NAME", "Qwen3.8-27B-NVFP4-vllm")
    turns = int(os.environ.get("TURNS", "6"))
    gen = int(os.environ.get("GEN", "1500"))
    start = int(os.environ.get("START", "300"))
    pauses = [float(p) for p in os.environ.get("PAUSES", "2,6").split(",")]
    go = threading.Event()
    rows, lock, ok = [], threading.Lock(), [True]
    t_base = time.perf_counter()

    def chat(c):
        messages = [{"role": "system", "content": SYSTEMS[c]}]
        if c > 0:
            go.wait()
        for i, text in enumerate(TURNS_TEXT[:turns]):
            messages.append({"role": "user", "content": text})
            cb = (lambda n: go.set() if n >= start else None) if c == 0 and i == 0 else None
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
            go.set()  # chat 0 shorter than START deltas still releases chat 1
            time.sleep(pauses[c % len(pauses)])

    threads = [threading.Thread(target=chat, args=(c,)) for c in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    per_chat = []
    for c in range(2):
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
