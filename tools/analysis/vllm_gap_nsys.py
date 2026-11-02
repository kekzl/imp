#!/usr/bin/env python3
# Decode-step kernel breakdown of a server nsys sqlite (--cuda-graph-trace=node): a step is the
# interval between consecutive MARK kernels (imp: argmax reduce, vLLM: _gumbel_sample_kernel); steps
# whose kernel count equals the mode are the steady full-batch steps. Prints step period, kernel busy
# share and per-kernel us/step. Usage: vllm_gap_nsys.py SQLITE [TOP=25]
# Env: MARK (regex, default argmax_reduce); SIG=<regex>: kernels matching it are keyed by launch
# signature (grid, block, smem), e.g. SIG=device_kernel splits vLLM's CUTLASS GEMMs by shape.
import collections
import os
import re
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
top = int(sys.argv[2]) if len(sys.argv) > 2 else 25
mark = re.compile(os.environ.get("MARK", "argmax_reduce"))
sig = re.compile(os.environ["SIG"]) if os.environ.get("SIG") else None
rows = [
    (s, e, f"{n} grid({gx},{gy},{gz}) block {bx} smem {sm}" if sig and sig.search(n) else n)
    for s, e, n, gx, gy, gz, bx, sm in db.execute(
        "SELECT k.start, k.end, s.value, k.gridX, k.gridY, k.gridZ, k.blockX, "
        "k.staticSharedMemory + k.dynamicSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start")]
marks = [i for i, r in enumerate(rows) if mark.search(r[2])]
steps = []
for a, b in zip(marks, marks[1:]):
    ks = rows[a + 1:b + 1]
    if not ks:
        continue
    period = rows[b][1] - rows[a][1]
    busy, cur_s, cur_e = 0, None, None
    for s, e, _ in ks:  # union of kernel intervals (PDL overlap)
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    steps.append((len(ks), period, busy, ks))
cnt = collections.Counter(s[0] for s in steps)
mode = cnt.most_common(1)[0][0]
sel = [s for s in steps if s[0] == mode]
# drop the slowest 10 % (prefill-interleaved or host stalls)
sel.sort(key=lambda s: s[1])
sel = sel[: max(1, int(len(sel) * 0.9))]
n = len(sel)
per = collections.defaultdict(float)
calls = collections.Counter()
for _, _, _, ks in sel:
    for s, e, name in ks:
        per[name] += (e - s) / 1e3
        calls[name] += 1
period = statistics.median(s[1] for s in sel) / 1e3
busy = statistics.median(s[2] for s in sel) / 1e3
ksum = sum(per.values()) / n
print(f"steps total {len(steps)}, kernel-count mode {mode} ({cnt[mode]} steps), used {n}")
print(f"step period median {period:.1f} us, kernel union {busy:.1f} us ({100 * busy / period:.1f} %), "
      f"kernel sum {ksum:.1f} us, idle {period - busy:.1f} us")
print("| kernel | calls/step | us/step | share of kernel sum |")
print("|---|---:|---:|---:|")
for name, t in sorted(per.items(), key=lambda kv: -kv[1])[:top]:
    print(f"| `{name[:110]}` | {calls[name] / n:.1f} | {t / n:.1f} | {100 * t / n / ksum:.1f} % |")
