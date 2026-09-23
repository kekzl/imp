"""Host-side attribution of GPU idle time inside one NVTX phase of an nsys sqlite export.
Usage: python host_gaps.py <sqlite> [phase=bench:pp] [top=15]
Prints: GPU busy/idle in the phase, idle-gap histogram, CUDA runtime API time by name, memcpy
count/bytes/time by kind and pageable/pinned source.
"""

import collections
import sqlite3
import sys


def main():
    db = sqlite3.connect(sys.argv[1])
    phase = sys.argv[2] if len(sys.argv) > 2 else "bench:pp"
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 15
    strings = dict(db.execute("SELECT id, value FROM StringIds"))
    cols = {r[1] for r in db.execute("PRAGMA table_info(NVTX_EVENTS)")}
    text = "COALESCE(text,'')" if "text" in cols else "''"
    ranges = [
        (s, e)
        for s, e, t, tid in db.execute(f"SELECT start, end, {text}, textId FROM NVTX_EVENTS WHERE eventType = 59")
        if (t or strings.get(tid, "")) == phase and e is not None
    ]
    if not ranges:
        print(f"no NVTX range {phase}")
        return
    wall = sum(e - s for s, e in ranges)

    def inside(t):
        return any(s <= t <= e for s, e in ranges)

    busy = []
    for table in ("CUPTI_ACTIVITY_KIND_KERNEL", "CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        try:
            busy += [(s, e) for s, e in db.execute(f"SELECT start, end FROM {table}") if inside(s)]
        except sqlite3.OperationalError:
            pass
    busy.sort()
    union, gaps, cur_s, cur_e = 0, [], None, None
    for s, e in busy:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                union += cur_e - cur_s
                gaps.append(s - cur_e)
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        union += cur_e - cur_s
    print(f"phase {phase}: {len(ranges)} ranges, wall {wall / 1e6:.2f} ms, GPU busy {union / 1e6:.2f} ms "
          f"({100 * union / wall:.1f} %), idle {100 - 100 * union / wall:.1f} %")
    buckets = [("<10us", 1e4), ("10-100us", 1e5), ("0.1-1ms", 1e6), (">1ms", float("inf"))]
    hist = collections.Counter()
    for g in gaps:
        for name, lim in buckets:
            if g < lim:
                hist[name] += g
                break
    print("idle gaps between device work (ms): " + ", ".join(f"{n} {hist[n] / 1e6:.2f}" for n, _ in buckets))

    api = collections.defaultdict(lambda: [0, 0])
    for s, e, nid in db.execute("SELECT start, end, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"):
        if inside(s):
            a = api[strings.get(nid, str(nid))]
            a[0] += 1
            a[1] += e - s
    print(f"CUDA runtime API inside the phase (top {top} by time):")
    for name, (n, t) in sorted(api.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {t / 1e6:9.2f} ms  {n:7d} calls  {name}")

    kinds = {1: "HtoD", 2: "DtoH", 8: "DtoD", 10: "PtoP"}
    mem = collections.defaultdict(lambda: [0, 0, 0])
    try:
        for s, e, k, b, src in db.execute("SELECT start, end, copyKind, bytes, srcKind FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            if inside(s):
                m = mem[(kinds.get(k, str(k)), "pageable" if src == 1 else "device/pinned")]
                m[0] += 1
                m[1] += b
                m[2] += e - s
    except sqlite3.OperationalError:
        pass
    print("memcpy inside the phase:")
    for (k, src), (n, b, t) in sorted(mem.items(), key=lambda kv: -kv[1][2]):
        print(f"  {k:5s} {src:14s} {n:7d} copies  {b / 1e6:10.2f} MB  {t / 1e6:8.2f} ms device")


if __name__ == "__main__":
    main()
