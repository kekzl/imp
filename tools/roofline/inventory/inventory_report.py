"""Joins out/*.json cells into the kernel priority list (markdown on stdout, JSON to argv[2]).

Phase per workload: pp* -> bench:pp, tg* -> bench:tg. A kernel's rank key is its max time share
over all cells; a kernel >= MIN_SHARE in any cell is listed.
Usage: python inventory_report.py <out_dir> <kernels.json> [min_share=0.01]
"""

import glob
import json
import os
import re
import sys


def _phase(cell):
    return "bench:tg" if cell.split("__")[1].startswith("tg") else "bench:pp"


def _label(name):
    # Template instances stay distinct; drop the return type and parameter list for display.
    name = re.sub(r"^void ", "", name)
    depth, cut = 0, len(name)
    for i, ch in enumerate(name):
        depth += ch == "<"
        depth -= ch == ">"
        if ch == "(" and depth == 0:
            cut = i
            break
    return name[:cut]


def main():
    out_dir, out_json = sys.argv[1], sys.argv[2]
    min_share = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01
    cells, kernels = {}, {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*__*.json"))):
        d = json.load(open(path))
        cell = d["cell"] or os.path.basename(path)[:-5]
        ph = d["phases"].get(_phase(cell))
        if not ph:
            continue
        reps = max(ph["ranges"], 1)
        cells[cell] = {
            "wall_ms": ph["wall_ns"] / 1e6 / reps,
            "kernel_ms": ph["kernel_sum_ns"] / 1e6 / reps,
            "idle_frac": ph["gpu_idle_frac"],
        }
        for k in ph["kernels"]:
            e = kernels.setdefault(k["name"], {"label": _label(k["name"]), "cells": {}})
            e["cells"][cell] = {
                "share": k["share"],
                "ms_per_rep": k["total_ns"] / 1e6 / reps,
                "calls_per_rep": k["count"] / reps,
                "mean_us": k["mean_ns"] / 1e3,
                "shape": k["top_shapes"][0][0] if k["top_shapes"] else "",
                "regs": k["regs"],
                "smem": k["smem"],
                "graph_frac": k["graph_frac"],
            }
    rows = []
    for name, e in kernels.items():
        best = max(e["cells"].items(), key=lambda kv: kv[1]["share"])
        if best[1]["share"] < min_share:
            continue
        rows.append({"name": name, "label": e["label"], "max_cell": best[0], "max": best[1],
                     "n_cells_ge1": sum(1 for c in e["cells"].values() if c["share"] >= 0.01),
                     "cells": e["cells"]})
    rows.sort(key=lambda r: -r["max"]["share"])
    json.dump({"cells": cells, "kernels": rows}, open(out_json, "w"), indent=1)

    print("| # | Kernel | max share | cell | us/call | calls/rep | cells >= 1 % | grid/block | regs | smem |")
    print("|---:|---|---:|---|---:|---:|---:|---|---:|---:|")
    for i, r in enumerate(rows, 1):
        m = r["max"]
        label = r["label"] if len(r["label"]) <= 90 else r["label"][:87] + "..."
        label = label.replace("|", "\\|")
        print(f"| {i} | `{label}` | {100 * m['share']:.1f} % | {r['max_cell']} | {m['mean_us']:.1f} |"
              f" {m['calls_per_rep']:.0f} | {r['n_cells_ge1']} | {m['shape']} | {m['regs']} | {m['smem']} |")
    print()
    print("| Cell | phase wall ms/rep | kernel sum ms/rep | GPU idle |")
    print("|---|---:|---:|---:|")
    for c, v in sorted(cells.items()):
        idle = "" if v["idle_frac"] is None else f"{100 * v['idle_frac']:.1f} %"
        print(f"| {c} | {v['wall_ms']:.1f} | {v['kernel_ms']:.1f} | {idle} |")


if __name__ == "__main__":
    main()
