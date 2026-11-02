#!/usr/bin/env python3
# Table for vllm_gap_sweep.sh output: per model and engine the median over trials (min-max) of
# tg128 (1000/TPOT p50), ppN (prompt tokens / mean max_tokens=1 latency), TTFT ms, conc 4/16 aggregate,
# peak VRAM (nvidia-smi max - pre-start baseline), SM clock p50 while util >= 50 %.
# Usage: vllm_gap_table.py OUTDIR
import glob
import json
import os
import re
import statistics
import sys

OUT = sys.argv[1]


def kpi(path):
    try:
        c = json.load(open(path))["results"][0]["client"]
    except (OSError, ValueError, KeyError, IndexError):
        return None
    return c


def median_conc(path):
    try:
        m = re.search(r"^MEDIAN ([0-9.]+)", open(path).read(), re.M)
    except OSError:
        return None
    return float(m.group(1)) if m and float(m.group(1)) > 0 else None


def smi(d):
    try:
        base = int(open(os.path.join(d, "base_mib")).read().strip())
        rows = [r.split(",") for r in open(os.path.join(d, "smi.csv")) if r.count(",") == 5]
        mem = [int(r[1]) for r in rows]
        sm = [int(r[3]) for r in rows if int(r[2]) >= 50]
        return max(mem) - base, (statistics.median(sm) if sm else None)
    except (OSError, ValueError):
        return None, None


def arm(d):
    r = {}
    c = kpi(os.path.join(d, "dec.json"))
    if c and c["tpot_ms"]["n"]:
        r["tg128"] = 1000.0 / c["tpot_ms"]["p50"]
        r["ttft_short"] = c["ttft_ms"]["p50"]
    for pp in (512, 4096, 8192, 15000):
        c = kpi(os.path.join(d, f"pp{pp}.json"))
        if c and c["ok"]:
            # max_tokens=1, closed loop at c=1: wall / ok = mean request latency = TTFT. Not the SSE
            # first-token stamp: Gemma-4 on vLLM can emit an empty-text token, which leaves no stamp.
            ntok = c["input_tokens"] / c["ok"]
            lat_ms = 1000.0 * c["wall_s"] / c["ok"]
            r[f"ttft{pp}"] = lat_ms
            r[f"pp{pp}"] = ntok / (lat_ms / 1000.0)
            r[f"ntok{pp}"] = ntok
    for cc in ("4", "16", "16L"):
        v = median_conc(os.path.join(d, f"conc{cc}.txt"))
        if v:
            r[f"c{cc}"] = v
    r["vram"], r["sm_mhz"] = smi(d)
    return r


KEYS = ["tg128", "ttft_short", "pp512", "ttft512", "pp4096", "ttft4096", "pp8192", "ttft8192",
        "pp15000", "ttft15000", "c4", "c16", "c16L", "vram", "sm_mhz"]
LOWER_BETTER = {"ttft_short", "ttft512", "ttft4096", "ttft8192", "ttft15000", "vram"}


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, None, 0
    return statistics.median(vals), min(vals), max(vals), len(vals)


models = sorted(d for d in os.listdir(OUT) if glob.glob(os.path.join(OUT, d, "*-t[0-9]*")))
rows = []
for m in models:
    res = {}
    for eng in ("imp", "vllm"):
        arms = [arm(d) for d in sorted(glob.glob(os.path.join(OUT, m, f"{eng}-t*")))]
        res[eng] = {k: agg([a.get(k) for a in arms]) for k in KEYS}
        ntoks = [a.get("ntok512") for a in arms if a.get("ntok512")] + [a.get("ntok4096") for a in arms if a.get("ntok4096")]
        res[eng]["_ntok"] = ntoks
    print(f"## {m}")
    print("| metric | imp median (min-max, n) | vLLM median (min-max, n) | imp vs vLLM |")
    print("|---|---:|---:|---:|")
    for k in KEYS:
        i, v = res["imp"][k], res["vllm"][k]
        if i[0] is None and v[0] is None:
            continue

        def f(x):
            return "-" if x[0] is None else f"{x[0]:.1f} ({x[1]:.1f}-{x[2]:.1f}, {x[3]})"
        if i[0] is not None and v[0] is not None and v[0] > 0:
            ratio = i[0] / v[0] - 1
            if k in LOWER_BETTER:
                ratio = v[0] / i[0] - 1  # >0 = imp better
            rel = f"{100 * ratio:+.1f} %"
            if k not in ("vram", "sm_mhz"):
                rows.append((ratio, m, k, i[0], v[0]))
        else:
            rel = "-"
        print(f"| {k} | {f(i)} | {f(v)} | {rel} |")
    print()

print("## ranked (imp vs vLLM, negative = imp loses; TTFT/VRAM inverted so >0 = imp better)")
for ratio, m, k, i, v in sorted(rows):
    print(f"{100 * ratio:+7.1f} %  {m:40s} {k:10s} imp {i:10.1f} vllm {v:10.1f}")
