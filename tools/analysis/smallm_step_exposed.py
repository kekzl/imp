"""Exposed time per kernel in a decode step (#2414): time a kernel runs past its predecessor end.
Usage: smallm_step_exposed.py <serving.sqlite from smallm_step_nsys.sh> <batch rows>; median over
the middle half of steps whose rmsnorm grid equals the batch. Exposed sums to the step period."""
import sqlite3, statistics, sys
from collections import defaultdict

db = sys.argv[1]
rows = sqlite3.connect(db).execute(
    'SELECT k.start, k.end, k.gridX, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
    'JOIN StringIds s ON s.id = k.shortName ORDER BY k.start').fetchall()
marks = [i for i, r in enumerate(rows) if 'sampl' in r[3].lower() or 'argmax' in r[3].lower()]
steps = []
for a, b in zip(marks, marks[1:]):
    seg = rows[a + 1:b + 1]
    n = sum(1 for r in seg if 'smallm' in r[3])
    steps.append((seg, n, rows[b][0] - rows[a][1]))


def bgrid(seg):
    return max((r[2] for r in seg if r[3].startswith("rmsnorm_fp16_rowblock_nvfp4")), default=0)


steps = [(seg, n, per) for seg, n, per in steps if bgrid(seg) == int(sys.argv[2])]
top = max(s[1] for s in steps)
sel = [s for s in steps if s[1] == top]
sel = sel[len(sel) // 4: 3 * len(sel) // 4]  # middle half
dur = defaultdict(list)
exp = defaultdict(list)
periods = []
for seg, n, per in sel:
    periods.append(per)
    d = defaultdict(float)
    e = defaultdict(float)
    prev_end = seg[0][0]
    for s, en, g, name in seg:
        key = f'{name[:48]} g{g}'
        d[key] += en - s
        e[key] += max(0, en - max(s, prev_end))
        prev_end = max(prev_end, en)
    for k in d:
        dur[k].append(d[k])
        exp[k].append(e[k])
print(f'steps used {len(sel)} (smallm launches {top}), period median {statistics.median(periods)/1e3:.1f} us')
tot_d = tot_e = 0
out = []
for k in dur:
    md, me = statistics.median(dur[k]) / 1e3, statistics.median(exp[k]) / 1e3
    tot_d += md; tot_e += me
    out.append((me, md, k))
for me, md, k in sorted(out, reverse=True)[:25]:
    print(f'{me:9.1f} exposed {md:9.1f} dur  {k}')
print(f'sum exposed {tot_e:.1f} us, sum dur {tot_d:.1f} us')
