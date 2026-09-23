"""Per-phase kernel inventory from an nsys sqlite export of `imp-cli --bench`.

Phase = the `bench:*` NVTX range (tools/imp-cli/mode_bench.cpp) that contains the CPU call which
launched the kernel (joined on correlationId; graph-replayed kernels carry the cudaGraphLaunch id).
Usage: python inventory_extract.py <sqlite> <out.json> [cell label]
"""

import collections
import json
import sqlite3
import sys


def _cols(con, table):
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def _strings(con):
    return dict(con.execute("SELECT id, value FROM StringIds"))


def _ranges(con, strings):
    cols = _cols(con, "NVTX_EVENTS")
    text = "COALESCE(text, '')" if "text" in cols else "''"
    tid = "textId" if "textId" in cols else "NULL"
    out = []
    for start, end, txt, text_id in con.execute(
        f"SELECT start, end, {text}, {tid} FROM NVTX_EVENTS WHERE end IS NOT NULL AND eventType = 59"
    ):
        name = txt or strings.get(text_id, "")
        if name.startswith("bench:"):
            out.append((start, end, name))
    out.sort()
    return out


def _phase_of(ranges, t):
    # Ranges never nest (mode_bench.cpp); linear scan over <= ~10 ranges is cheap enough per call.
    for start, end, name in ranges:
        if start <= t <= end:
            return name
    return None


def _union_ns(intervals):
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def extract(path, label=""):
    con = sqlite3.connect(path)
    strings = _strings(con)
    ranges = _ranges(con, strings)
    launch_t = {}
    for table in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
        try:
            for corr, start in con.execute(f"SELECT correlationId, start FROM {table}"):
                launch_t.setdefault(corr, start)
        except sqlite3.OperationalError:
            pass
    kcols = _cols(con, "CUPTI_ACTIVITY_KIND_KERNEL")
    graph = "graphId" if "graphId" in kcols else "0"
    rows = con.execute(
        "SELECT start, end, correlationId, demangledName, shortName, gridX, gridY, gridZ, blockX, "
        f"blockY, blockZ, registersPerThread, staticSharedMemory, dynamicSharedMemory, {graph} "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    con.close()

    phases = collections.defaultdict(lambda: {"kernels": {}, "intervals": []})
    unphased = 0
    for s, e, corr, dem, short, gx, gy, gz, bx, by, bz, regs, ssm, dsm, gid in rows:
        t = launch_t.get(corr)
        ph = _phase_of(ranges, t) if t is not None else None
        if ph is None:
            unphased += 1
            continue
        p = phases[ph]
        p["intervals"].append((s, e))
        name = strings.get(dem, str(dem))
        k = p["kernels"].setdefault(
            name,
            {
                "short": strings.get(short, str(short)),
                "count": 0,
                "total_ns": 0,
                "max_ns": 0,
                "graph_launches": 0,
                "shapes": collections.Counter(),
                "regs": regs,
                "smem": (ssm or 0) + (dsm or 0),
            },
        )
        k["count"] += 1
        k["total_ns"] += e - s
        k["max_ns"] = max(k["max_ns"], e - s)
        k["graph_launches"] += 1 if gid else 0
        k["shapes"][f"{gx}x{gy}x{gz}/{bx}x{by}x{bz}"] += 1

    out = {"cell": label, "unphased_kernels": unphased, "phases": {}}
    for ph, p in phases.items():
        wall = sum(e - s for s, e, n in ranges if n == ph)
        busy = _union_ns(p["intervals"])
        ksum = sum(k["total_ns"] for k in p["kernels"].values())
        kernels = []
        for name, k in sorted(p["kernels"].items(), key=lambda kv: -kv[1]["total_ns"]):
            kernels.append(
                {
                    "name": name,
                    "short": k["short"],
                    "count": k["count"],
                    "total_ns": k["total_ns"],
                    "mean_ns": k["total_ns"] // k["count"],
                    "max_ns": k["max_ns"],
                    "share": k["total_ns"] / ksum if ksum else 0.0,
                    "graph_frac": k["graph_launches"] / k["count"],
                    "top_shapes": k["shapes"].most_common(3),
                    "regs": k["regs"],
                    "smem": k["smem"],
                }
            )
        out["phases"][ph] = {
            "ranges": sum(1 for r in ranges if r[2] == ph),
            "wall_ns": wall,
            "kernel_union_ns": busy,
            "kernel_sum_ns": ksum,
            "gpu_idle_frac": 1.0 - busy / wall if wall else None,
            "kernels": kernels,
        }
    return out


if __name__ == "__main__":
    res = extract(sys.argv[1], sys.argv[3] if len(sys.argv) > 3 else "")
    with open(sys.argv[2], "w") as f:
        json.dump(res, f, indent=1)
    for ph, p in res["phases"].items():
        print(
            f"{res['cell']} {ph}: wall {p['wall_ns'] / 1e6:.1f} ms, kernels {p['kernel_sum_ns'] / 1e6:.1f} ms,"
            f" idle {100 * (p['gpu_idle_frac'] or 0):.1f} %, {len(p['kernels'])} kernels"
        )
    print(f"unphased kernels: {res['unphased_kernels']}")
