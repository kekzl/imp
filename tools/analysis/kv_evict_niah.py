#!/usr/bin/env python3
"""KV eviction accuracy probe (roadmap row 48, docs/plans/2026-11-01-kv-eviction.md).

Drives a running imp-server with two long-context tasks on real text (Einstein, "Relativity",
tools/eval/niah/data/filler.txt, wrapped to length):
  niah  - a vault code planted at a depth, asked for at the end; hit = code in the answer.
  cont  - continuation recall (long-context QA): quote the sentence that follows a given
          sentence from the middle of the text; score = word-level F1 against the true sentence.
Same prompts, same order on every arm, temperature 0. Prints one JSON line per probe and a
summary table per length. Stdlib only: run it in python:3.13-slim, never on the host.

Usage:
  python3 kv_evict_niah.py --url http://localhost:8080 --model NAME --arm base \
      --lengths 8000,16000,32000 --out results.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

DEPTHS = [0.1, 0.3, 0.5, 0.7, 0.9]
SEEDS = [1, 2, 3, 4]
CONT_DEPTHS = [0.2, 0.35, 0.5, 0.65, 0.8]
CONT_SEEDS = [1, 2, 3, 4]
CHARS_PER_TOKEN = 4.5  # sizing only; the server's prompt_tokens is reported per probe


def load_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    raw = re.sub(r"\s+", " ", raw).strip()
    return raw


def haystack(text, target_tokens, offset):
    n = int(target_tokens * CHARS_PER_TOKEN)
    out = []
    pos = offset % len(text)
    while sum(len(s) for s in out) < n:
        out.append(text[pos:pos + (n - sum(len(s) for s in out))])
        pos = 0
    hay = "".join(out)
    cut = hay.rfind(". ")
    return hay[:cut + 1] if cut > 0 else hay


def post(url, body, timeout=600):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ask(url, model, prompt, max_tokens):
    body = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt + " /no_think"}]}
    rsp = post(url, body)
    ans = rsp["choices"][0]["message"].get("content") or ""
    ans = re.sub(r"<think>.*?</think>", "", ans, flags=re.S).strip()
    return ans, rsp.get("usage", {}).get("prompt_tokens", -1)


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def f1(pred, gold):
    p, g = words(pred), words(gold)
    if not p or not g:
        return 0.0
    gc = {}
    for w in g:
        gc[w] = gc.get(w, 0) + 1
    common = 0
    for w in p:
        if gc.get(w, 0) > 0:
            common += 1
            gc[w] -= 1
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def sentences(hay):
    # (start, end) spans of sentences with 12..60 words.
    spans = []
    start = 0
    for m in re.finditer(r"[.!?] ", hay):
        end = m.end()
        s = hay[start:end].strip()
        if 12 <= len(s.split()) <= 60:
            spans.append((start, end))
        start = end
    return spans


def niah_probe(url, model, text, length, depth, seed):
    rnd = random.Random(length * 1000 + int(depth * 100) * 10 + seed)
    name = rnd.choice(["Orion", "Vega", "Lyra", "Draco", "Cygnus", "Perseus", "Aquila", "Hydra"])
    code = f"{rnd.randint(100000, 999999)}-{rnd.choice(['amber', 'cobalt', 'violet', 'scarlet', 'jade'])}"
    hay = haystack(text, length, offset=seed * 7919)
    at = hay.find(". ", int(len(hay) * depth))
    at = at + 2 if at > 0 else int(len(hay) * depth)
    needle = f"The secret access code for vault {name} is {code}. "
    prompt = (hay[:at] + needle + hay[at:] +
              f"\n\nQuestion: What is the secret access code for vault {name}? "
              "Answer with only the code.")
    ans, ptok = ask(url, model, prompt, 64)
    return {"task": "niah", "len": length, "depth": depth, "seed": seed, "ptok": ptok,
            "score": 1.0 if code in ans else 0.0, "want": code, "got": ans[:80]}


def cont_probe(url, model, text, length, depth, seed):
    hay = haystack(text, length, offset=seed * 104729)
    spans = sentences(hay)
    # First sentence at or after the depth whose text and successor occur once in the haystack.
    target = int(len(hay) * depth)
    for i, (s0, s1) in enumerate(spans[:-1]):
        if s0 < target:
            continue
        cur = hay[s0:s1].strip()
        nxt_s0, nxt_s1 = spans[i + 1]
        if nxt_s0 != s1:
            continue
        nxt = hay[nxt_s0:nxt_s1].strip()
        if hay.count(cur) == 1 and hay.count(nxt) == 1:
            break
    else:
        return None
    lead = " ".join(cur.split()[:10])
    prompt = (hay + f"\n\nQuestion: In the text above, find the sentence that begins with \"{lead}\". "
              "Quote the sentence that comes immediately after it, word for word, and nothing else.")
    ans, ptok = ask(url, model, prompt, 160)
    return {"task": "cont", "len": length, "depth": depth, "seed": seed, "ptok": ptok,
            "score": f1(ans, nxt), "want": nxt[:80], "got": ans[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--lengths", default="8000,16000,32000")
    ap.add_argument("--text", default="/data/filler.txt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    text = load_text(args.text)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    rows = []
    t0 = time.monotonic()
    with open(args.out, "a", encoding="utf-8") as out:
        for length in lengths:
            probes = [(niah_probe, d, s) for d in DEPTHS for s in SEEDS]
            probes += [(cont_probe, d, s) for d in CONT_DEPTHS for s in CONT_SEEDS]
            for fn, d, s in probes:
                try:
                    r = fn(args.url, args.model, text, length, d, s)
                except Exception as e:  # noqa: BLE001 - a transport failure is a scored miss
                    r = {"task": "niah" if fn is niah_probe else "cont", "len": length, "depth": d,
                         "seed": s, "ptok": -1, "score": 0.0, "want": "", "got": f"ERROR {e}"[:80]}
                if r is None:
                    continue
                r["arm"] = args.arm
                rows.append(r)
                out.write(json.dumps(r) + "\n")
                out.flush()
                print(json.dumps(r), flush=True)
    print(f"=== arm {args.arm} ({time.monotonic() - t0:.0f} s)")
    print("| len | niah hits | niah n | cont F1 mean | cont n | ptok max |")
    print("|---|---|---|---|---|---|")
    for length in lengths:
        n = [r for r in rows if r["len"] == length and r["task"] == "niah"]
        c = [r for r in rows if r["len"] == length and r["task"] == "cont"]
        pt = max([r["ptok"] for r in rows if r["len"] == length] or [-1])
        cf = sum(r["score"] for r in c) / len(c) if c else float("nan")
        print(f"| {length} | {int(sum(r['score'] for r in n))} | {len(n)} | {cf:.4f} | {len(c)} | {pt} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
