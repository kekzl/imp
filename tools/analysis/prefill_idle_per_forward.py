#!/usr/bin/env python3
# Per prefill forward (roadmap row 59), nsys sqlite exported with --cuda-graph-trace=node: from one
# embedding launch to the next, wall vs kernel-union busy and the three widest gaps.
# Usage: prefill_idle_per_forward.py <export.sqlite>
import sqlite3, sys

db = sqlite3.connect(sys.argv[1])
rows = db.execute(
    "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
    "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall()
marks = [i for i, (_, _, n) in enumerate(rows) if n.startswith("embedding_lookup")]
for a, b in zip(marks, marks[1:] + [len(rows)]):
    seg = rows[a:b]
    if len(seg) < 200:
        continue
    t0, t1 = seg[0][0], max(e for _, e, _ in seg)
    union, cs, ce = 0, None, None
    for s, e, _ in seg:
        if cs is None or s > ce:
            if cs is not None:
                union += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    union += ce - cs
    wall = (t1 - t0) / 1e6
    gaps = sorted(((seg[i + 1][0] - max(x[1] for x in seg[: i + 1][-64:])), seg[i][2], seg[i + 1][2])
                  for i in range(len(seg) - 1))[-3:]
    print(f"fwd kernels {len(seg):5d} wall {wall:7.2f} ms busy {union / 1e6:7.2f} ms idle {100 * (1 - union / (t1 - t0)):5.1f} % "
          f"top gaps(us): " + ", ".join(f"{g / 1e3:.0f} {p[:28]}->{n[:28]}" for g, p, n in reversed(gaps)))
