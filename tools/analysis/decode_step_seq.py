#!/usr/bin/env python3
# One steady decode step of a server nsys sqlite (--cuda-graph-trace=node), in launch order: a step is
# the interval between consecutive MARK kernels, steady = kernel-count mode, the median-period step is
# printed. LAYER (regex) marks the first kernel of each layer; rows FROM..TO layers are listed, then a
# per-name table over the whole step for kernels matching CLASS. Usage: decode_step_seq.py SQLITE
# Env: MARK (argmax_reduce), LAYER (paged_attention|fa2_hd512), FROM (2), TO (3),
#      CLASS (rmsnorm|quantize_nvfp4|sanitize|scale_|elementwise|device_copy|fp16_to|add_|copy)
import collections
import os
import re
import sqlite3
import statistics
import sys

db = sqlite3.connect(sys.argv[1])
mark = re.compile(os.environ.get("MARK", "argmax_reduce"))
layer_re = re.compile(os.environ.get("LAYER", "rmsnorm_fp32_in_kernel"))
cls = re.compile(os.environ.get("CLASS", "rmsnorm|quantize_nvfp4|sanitize|scale_|elementwise|device_copy|fp16_to|_add_|copy"))
lo, hi = int(os.environ.get("FROM", "2")), int(os.environ.get("TO", "3"))
rows = list(db.execute(
    "SELECT k.start, k.end, s.value, k.gridX, k.blockX FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"))
marks = [i for i, r in enumerate(rows) if mark.search(r[2])]
steps = []
for a, b in zip(marks, marks[1:]):
    ks = rows[a + 1:b + 1]
    if ks:
        steps.append((len(ks), rows[b][1] - rows[a][1], ks))
mode = collections.Counter(s[0] for s in steps).most_common(1)[0][0]
sel = sorted((s for s in steps if s[0] == mode), key=lambda s: s[1])
sel = sel[: max(1, int(len(sel) * 0.9))]
period_med = statistics.median(s[1] for s in sel)
step = min(sel, key=lambda s: abs(s[1] - period_med))
ks = step[2]
print(f"steady steps {len(sel)}, kernels/step {mode}, median period {period_med / 1e3:.1f} us")
layer = -1
prev_end = ks[0][0]
for s, e, name, gx, bx in ks:
    if layer_re.search(name):
        layer += 1
    if lo <= layer <= hi:
        print(f"L{layer:02d} {(e - s) / 1e3:7.2f} us gap {(s - prev_end) / 1e3:6.2f} grid {gx:5d} blk {bx:4d} {name[:90]}")
    prev_end = e
# per-name totals over all steady steps
per = collections.defaultdict(float)
calls = collections.Counter()
for _, _, kk in sel:
    for s, e, name, _, _ in kk:
        if cls.search(name):
            per[name] += (e - s) / 1e3
            calls[name] += 1
n = len(sel)
tot = sum(per.values()) / n
print(f"class total {tot:.1f} us/step, {sum(calls.values()) / n:.1f} launches/step")
print("| kernel | calls/step | us/step |")
print("|---|---:|---:|")
for name, t in sorted(per.items(), key=lambda kv: -kv[1]):
    print(f"| `{name[:90]}` | {calls[name] / n:.1f} | {t / n:.1f} |")
