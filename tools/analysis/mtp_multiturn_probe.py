#!/usr/bin/env python3
# Real multi-turn chat on one imp-server, one client: TURNS follow-ups, each streamed, the assistant turn sent
# back as `content` only (what OpenAI clients do; the Qwen3.8 template drops the old thinking). Per turn:
# prompt/cached tokens, TTFT, decode tok/s ((completion_tokens - 1) / (last delta - first delta)), MTP
# drafted/accepted from /metrics, max word run. JSON transcript to OUT_JSON when set.
# Usage: mtp_multiturn_probe.py PORT [TAG]; env MODEL_NAME, TURNS (6), GEN (1500), TEMP (0), SEED (7), OUT_JSON,
#        SIDE (0; 1 = a UI task request between turns, below), CONVS (1; 2 = two chats alternate turns).
#        mtp_multiturn_probe.py --parse SERVER_LOG: per request prompt tokens + MTP unbind reasons, counts.
import json
import os
import re
import sys
import time
import urllib.request
from collections import Counter

TURNS_TEXT = [
    "I'm planning a 5-day trip to Kyoto in early April with my partner. We like food, temples and quiet walks, "
    "not nightlife. Can you suggest a day-by-day itinerary?",
    "Day 3 looks packed. Move the Nara day trip to day 4 and explain what we lose and gain by doing that.",
    "Roughly what should we budget per day for food, transport and entry fees? Give yen and euros.",
    "Write a packing list for that trip as a markdown table with columns item, why, and priority.",
    "My partner is vegetarian. Which of the food suggestions so far need replacing, and with what?",
    "Summarize the final plan in exactly six bullet points, one per day plus one for logistics.",
    "Translate those six bullets into German, keeping place names in their usual spelling.",
    "Last question: which two things on this plan are most likely to go wrong, and what is the backup?",
]
# SIDE=1: after every turn a chat UI's task request (title / follow-up suggestions) over the transcript as text.
SIDE_TASK = ("### Task:\nSuggest 3 short follow-up questions the user might ask next, as a JSON list of strings. "
             "No explanation.\n### Chat History:\n")
SYSTEMS = ["You are a friendly travel planner. Be concrete and practical.",
           "You are a terse assistant for a busy engineer. Prefer lists over prose."]


def parse_log(path):
    reqs = []
    cur = None
    with open(path, errors="replace") as f:
        for line in f:
            if "chat/completions: prompt_msgs=" in line:
                cur = {"msgs": int(re.search(r"prompt_msgs=(\d+)", line).group(1)), "unbind": [], "prompt": None}
                reqs.append(cur)
            elif cur is not None and "mtp-spec: drafting off" in line:
                cur["unbind"].append(re.search(r"drafting off for req \d+ \((.*)\)\s*$", line).group(1))
            elif cur is not None and re.search(r"\] \d+ prompt \+ \d+ completion tokens", line):
                cur["prompt"] = int(re.search(r"\] (\d+) prompt \+", line).group(1))
    total = Counter()
    for i, r in enumerate(reqs):
        c = Counter(r["unbind"])
        total.update(c)
        print(f"req{i} msgs {r['msgs']} prompt {r['prompt']} unbind {dict(c) if c else '-'}")
    print("UNBIND_TOTAL", dict(total) if total else "{}")


def metric(url, name):
    with urllib.request.urlopen(f"{url}/metrics", timeout=30) as r:
        text = r.read().decode()
    return sum(float(m.group(1)) for m in re.finditer(rf"^{name}(?:\{{[^}}]*\}})?\s+(\S+)$", text, re.M))


def max_word_run(text):
    return max((len(m.group(0).split()) for m in re.finditer(r"(\b\S+\b)(\s+\1\b)+", text)), default=1)


def turn(url, model, messages, gen, temp, seed):
    body = {"model": model, "messages": messages, "max_tokens": gen, "temperature": temp, "stream": True,
            "stream_options": {"include_usage": True}}
    if temp > 0:
        body.update({"top_p": 0.95, "seed": seed})
    req = urllib.request.Request(f"{url}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    t_first = t_last = None
    content, reasoning, usage, finish = [], [], None, None
    with urllib.request.urlopen(req, timeout=900) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
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
    n = usage["completion_tokens"] if usage else 0
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    tps = (n - 1) / (t_last - t_first) if t_first and t_last > t_first else 0.0
    return {"prompt": (usage or {}).get("prompt_tokens", 0), "cached": cached, "completion": n,
            "ttft_ms": (t_first - t0) * 1e3 if t_first else 0.0, "tps": tps, "finish": finish,
            "content": "".join(content), "reasoning": "".join(reasoning)}


def main():
    if sys.argv[1] == "--parse":
        parse_log(sys.argv[2])
        return 0
    port = int(sys.argv[1])
    tag = sys.argv[2] if len(sys.argv) > 2 else "x"
    url = f"http://127.0.0.1:{port}"
    model = os.environ.get("MODEL_NAME", "Qwen3.8-27B-NVFP4-vllm")
    turns = int(os.environ.get("TURNS", "6"))
    gen = int(os.environ.get("GEN", "1500"))
    temp = float(os.environ.get("TEMP", "0"))
    seed = int(os.environ.get("SEED", "7"))
    side = os.environ.get("SIDE", "0") == "1"
    convs = int(os.environ.get("CONVS", "1"))
    # CONVS=2: two users alternate turns on the same server, each with its own system prompt.
    chats = [[{"role": "system", "content": SYSTEMS[c]}] if convs > 1 else [] for c in range(convs)]
    rows, ok = [], True
    for i, text in enumerate(TURNS_TEXT[:turns]):
        for c, messages in enumerate(chats):
            messages.append({"role": "user", "content": text})
            d0, a0 = metric(url, "imp_spec_drafted_total"), metric(url, "imp_spec_accepted_total")
            r = turn(url, model, messages, gen, temp, seed + i + 100 * c)
            r["drafted"] = metric(url, "imp_spec_drafted_total") - d0
            r["accepted"] = metric(url, "imp_spec_accepted_total") - a0
            r["turn"] = i
            run = max_word_run(r["content"] + " " + r["reasoning"])
            clean = run <= 4 and len(r["content"].split()) >= 10
            ok &= clean
            print(f"{tag} c{c} turn{i + 1} prompt {r['prompt']} cached {r['cached']} gen {r['completion']} "
                  f"ttft_ms {r['ttft_ms']:.0f} tps {r['tps']:.2f} drafted {r['drafted']:.0f} "
                  f"accepted {r['accepted']:.0f} finish {r['finish']} max_run {run} {'ok' if clean else 'DEGEN'}",
                  flush=True)
            rows.append(r)
            messages.append({"role": "assistant", "content": r["content"]})
            if side:
                history = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
                s = turn(url, model, [{"role": "user", "content": SIDE_TASK + history}], 120, 0, seed)
                print(f"{tag} side{i + 1} prompt {s['prompt']} cached {s['cached']} gen {s['completion']}",
                      flush=True)
    first = [r["tps"] for r in rows if r["turn"] == 0]
    follow = [r for r in rows if r["turn"] > 0]
    print(f"{tag} SUMMARY turn1_tps {sum(first) / len(first):.2f} "
          f"followup_mean_tps {sum(r['tps'] for r in follow) / max(1, len(follow)):.2f} "
          f"followup_drafted_turns {sum(1 for r in follow if r['drafted'] > 0)}/{len(follow)} "
          f"{'PASS' if ok else 'FAIL'}")
    if os.environ.get("OUT_JSON"):
        with open(os.environ["OUT_JSON"], "w") as f:
            json.dump(rows, f, indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
