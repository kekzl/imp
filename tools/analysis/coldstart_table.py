#!/usr/bin/env python3
"""Summarize coldstart_arm.sh dirs: server-side TTFT per ttft_probe phase, or pp tok/s per cell
(vllm_gap_table.py arithmetic) with the pp8192 requests' queue ms and the KV growth lines.

usage: coldstart_table.py DIR [DIR...]
"""
import json
import re
import statistics
import sys

DONE = re.compile(r"\[imp-(\d+)\] (\d+) prompt \+ (\d+) completion tokens, ([0-9.]+) ms "
                  r"\(ttft=([0-9.]+) ms, queue=([0-9.]+) ms\)")


def read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def ttft_arm(d, log, recs):
    rows = sorted((int(m[1]), float(m[5])) for m in DONE.finditer(log))[-len(recs):]
    out = {}
    for r, (_, ttft) in zip(recs, rows):
        out.setdefault(r["phase"], []).append(ttft)
    for ph, v in out.items():
        print(f"  {ph:12s} server ttft {' '.join(f'{x:.1f}' for x in v)}  median {statistics.median(v):.1f}")


def pp_arm(d, log):
    cells = []
    for pp in (512, 4096, 8192, 15000):
        try:
            c = json.load(open(f"{d}/pp{pp}.json"))["results"][0]["client"]
        except (OSError, ValueError, KeyError, IndexError):
            continue
        if c["ok"]:
            lat = 1000.0 * c["wall_s"] / c["ok"]
            cells.append(f"pp{pp} {c['input_tokens'] / c['ok'] / (lat / 1000.0):.0f}")
    q = [float(m[6]) for m in DONE.finditer(log) if 7000 < int(m[2]) < 9000 and m[3] == "1"]
    sync = len(re.findall(r"KV cache: grew \d", log))
    ahead = len(re.findall(r"KV cache: grew ahead", log))
    print(f"  {' | '.join(cells)}")
    print(f"  pp8192 queue ms {' '.join(f'{x:.1f}' for x in q)}; growths sync {sync}, ahead {ahead}")


for d in sys.argv[1:]:
    log = read(f"{d}/server.log") or ""
    mib = [read(f"{d}/{f}") for f in ("base_mib", "ready_mib")]
    vram = int(mib[1]) - int(mib[0]) if all(mib) else -1
    pw = re.search(r"prefill graph prewarm: .*\(([0-9.]+) s\)", log)
    print(f"{d}: ready {read(f'{d}/ready_s')} s, VRAM ready-base {vram} MiB"
          f"{', prefill prewarm ' + pw[1] + ' s' if pw else ''}")
    recs = read(f"{d}/ttft.json")
    if recs:
        ttft_arm(d, log, json.loads(recs))
    else:
        pp_arm(d, log)
