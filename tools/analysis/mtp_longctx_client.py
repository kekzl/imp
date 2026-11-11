#!/usr/bin/env python3
# Long-context agent prompts against imp-server: repo docs + source (REPO_DIR) sized by /tokenize to each
# CTX target, a coding question at the end, greedy streaming chat. Per request: prompt tokens, TTFT, decode
# tok/s ((completion - 1) / (last delta - first delta)), MTP drafted/accepted (usage), md5 of the text.
# CONC=2 sends two prompts per length at once (different head line). Texts land in OUT_DIR/<tag>.<ctx>.<i>.txt.
# Usage: mtp_longctx_client.py PORT TAG; env CTX ("8000 20000 40000 80000"), GEN (384), CONC (1), REPS (1),
#        REPO_DIR (/repo), OUT_DIR (.), IGNORE_EOS (1), MTP_K (unset = server default; 0 = speculation off),
#        THINK_BUDGET (fraction of max_tokens; unset = server default, 1 = never forced, 0 = thinking off),
#        STAGGER_MS (0; > 0 starts stream i that long after stream i - 1: a fixed arrival order).
import hashlib
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "x"
CTX = [int(c) for c in os.environ.get("CTX", "8000 20000 40000 80000").split()]
GEN = int(os.environ.get("GEN", "384"))
CONC = int(os.environ.get("CONC", "1"))
REPS = int(os.environ.get("REPS", "1"))
REPO = Path(os.environ.get("REPO_DIR", "/repo"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "."))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "1") == "1"
MTP_K = os.environ.get("MTP_K", "")
THINK_BUDGET = os.environ.get("THINK_BUDGET", "")
STAGGER_S = int(os.environ.get("STAGGER_MS", "0")) / 1000.0
BASE = f"http://127.0.0.1:{PORT}"


def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


with urllib.request.urlopen(BASE + "/v1/models", timeout=30) as r:
    MODEL = next(m["id"] for m in json.load(r)["data"] if m.get("loaded", True))

# Fixed file order: architecture docs first, then the runtime sources an agent would read.
files = sorted((REPO / "docs/internals").glob("*.md")) + sorted((REPO / "src/runtime").glob("*.cpp")) + \
    sorted((REPO / "src/compute").glob("*.cu"))
corpus = "".join(f"\n\n===== {p.relative_to(REPO)} =====\n{p.read_text(encoding='utf-8', errors='replace')}"
                 for p in files)
QUESTION = ("\n\nYou are a coding agent working in this repository. Using only the files above, explain step by "
            "step how a speculative draft is produced and verified, name the functions involved, and propose "
            "one concrete improvement with the code change it needs.")


def n_tokens(text):
    with post("/tokenize", {"content": text}, timeout=300) as r:
        return len(json.load(r)["tokens"])


def build(target, head):
    lo, hi = 1000, len(corpus)
    while hi - lo > 2000:  # chars for target tokens of user content
        mid = (lo + hi) // 2
        if n_tokens(head + corpus[:mid] + QUESTION) < target:
            lo = mid
        else:
            hi = mid
    return head + corpus[:lo] + QUESTION


def one(ctx, i, rep, prompt, res):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": GEN,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    if IGNORE_EOS:
        body["ignore_eos"] = True
    if MTP_K != "":
        body["speculative"] = {"mtp_k": int(MTP_K)}
    if THINK_BUDGET != "":
        body["think_budget"] = float(THINK_BUDGET)
    t0 = time.time()
    first = last = None
    text, usage = [], {}
    with post("/v1/chat/completions", body) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            ev = json.loads(line[5:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("reasoning_content") or "") + (d.get("content") or "")
                if piece:
                    now = time.time()
                    first = first or now
                    last = now
                    text.append(piece)
    out = "".join(text)
    comp = usage.get("completion_tokens", 0)
    det = usage.get("completion_tokens_details") or {}
    dec = (comp - 1) / (last - first) if first and last and last > first else 0.0
    (OUT_DIR / f"{TAG}.{ctx}.{rep}.{i}.txt").write_text(out, encoding="utf-8")
    res[i] = (f"REQ {TAG} ctx {ctx} rep {rep} s{i} prompt {usage.get('prompt_tokens', 0)} comp {comp} "
              f"ttft_s {first - t0 if first else -1:.2f} decode_tps {dec:.2f} "
              f"drafted {det.get('imp_spec_drafted', 0)} accepted {det.get('imp_spec_accepted', 0)} "
              f"md5 {hashlib.md5(out.encode()).hexdigest()[:12]}"), comp, (first or t0), (last or t0)


for rep in range(REPS):
    for ctx in CTX:
        prompts = [build(ctx, f"[session {ctx}-{rep}-{i}] ") for i in range(CONC)]
        res = [None] * CONC
        th = [threading.Thread(target=one, args=(ctx, i, rep, prompts[i], res)) for i in range(CONC)]
        for n, t in enumerate(th):
            if n and STAGGER_S:
                time.sleep(STAGGER_S)
            t.start()
        for t in th:
            t.join()
        for line, *_ in res:
            print(line, flush=True)
        if CONC > 1:  # aggregate decode: tokens of all streams over the span of their deltas
            span = max(r[3] for r in res) - min(r[2] for r in res)
            print(f"AGG {TAG} ctx {ctx} rep {rep} conc {CONC} tok_s {sum(r[1] for r in res) / span:.2f}",
                  flush=True)
