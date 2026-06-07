"""Build the Modul-1 roofline table rows from a run record. Shared by
report/plot/regress/issues. stdlib only."""
import statistics


def _median(vals):
    return statistics.median(vals) if vals else 0.0


def class_aggregate(ncu_kernels, class_name, gpu_cfg):
    """Aggregate all kernels of one class within one restart cell."""
    members = {n: r for n, r in ncu_kernels.items() if r["class"] == class_name}
    if not members:
        return None
    t = sum(r["time_ns"] for r in members.values())
    fl = sum(r["flops"] for r in members.values())
    by = sum(r["dram_bytes"] for r in members.values())
    if t <= 0:
        return None
    # dominant dtype = dtype of the FLOP-heaviest member
    dom = max(members.values(), key=lambda r: r["flops"])
    dtype = dom["dtype"]
    clk_hz = sum(r["sm_clock_mhz"] * 1e6 * r["time_ns"] for r in members.values()) / t

    ai = fl / by if by > 0 else 0.0
    ach_flops = fl / (t * 1e-9)
    ach_bw = by / (t * 1e-9)
    peak_bw = gpu_cfg["dram_peak_gbs"] * 1e9
    fpc = gpu_cfg["flop_per_cycle"][dtype]
    peak_flops_boost = fpc * gpu_cfg["nominal_boost_ghz"] * 1e9
    peak_flops_clk = fpc * clk_hz
    ridge = peak_flops_boost / peak_bw
    bound = "memory" if ai < ridge else "compute"
    pct = 100.0 * (ach_bw / peak_bw if bound == "memory"
                   else (ach_flops / peak_flops_clk if peak_flops_clk else 0.0))
    occ = sum(r["occupancy_pct"] * r["time_ns"] for r in members.values()) / t
    sm_pct = sum(r["sm_pct"] * r["time_ns"] for r in members.values()) / t
    return {
        "sm_pct": sm_pct,
        "class": class_name, "dtype": dtype, "time_ns": t,
        "n_kernels": len(members),
        "n_launches": sum(r.get("n_launches", 0) for r in members.values()),
        "ai": ai, "achieved_gflops": ach_flops / 1e9, "achieved_gbs": ach_bw / 1e9,
        "ridge": ridge, "bound_by": bound, "pct_roofline": pct,
        "occupancy_pct": occ, "sm_clock_mhz": clk_hz / 1e6,
        "peak_gflops_at_clock": peak_flops_clk / 1e9,
        "peak_gbs": gpu_cfg["dram_peak_gbs"],
    }


def build_rows(run, gpu_cfg):
    """-> list of rows: one per (cell, class) with min/med/max over restarts."""
    rows = []
    for cell_key, restarts in sorted(run["cells"].items()):
        model, shape = cell_key.split("|")
        valid = [c for c in restarts if "ncu_kernels" in c]
        if not valid:
            continue
        classes = sorted({r["class"] for c in valid for r in c["ncu_kernels"].values()})
        for cls in classes:
            per_restart = [a for c in valid
                           if (a := class_aggregate(c["ncu_kernels"], cls, gpu_cfg))]
            if not per_restart:
                continue
            med = sorted(per_restart, key=lambda r: r["pct_roofline"])[len(per_restart) // 2]
            row = dict(med)
            row.update({
                "model": model, "shape": shape, "n_restarts": len(per_restart),
                "pct_roofline_min": min(r["pct_roofline"] for r in per_restart),
                "pct_roofline_med": _median([r["pct_roofline"] for r in per_restart]),
                "pct_roofline_max": max(r["pct_roofline"] for r in per_restart),
                "achieved_min": min(_ach(r) for r in per_restart),
                "achieved_med": _median([_ach(r) for r in per_restart]),
                "achieved_max": max(_ach(r) for r in per_restart),
            })
            rows.append(row)
    return rows


def _ach(r):
    return r["achieved_gbs"] if r["bound_by"] == "memory" else r["achieved_gflops"]


def time_share_within_cell(rows):
    """Annotate rows with % of captured-window time within their cell."""
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["model"], r["shape"]), []).append(r)
    for cell_rows in by_cell.values():
        total = sum(r["time_ns"] for r in cell_rows) or 1
        for r in cell_rows:
            r["time_share_pct"] = 100.0 * r["time_ns"] / total
    return rows
