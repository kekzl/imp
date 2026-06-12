"""Build the Module-1 roofline table rows from a run record. Joins the per-cell
ncu metric groups (base/sass_*/tc_* for prefill, full for decode) at kernel-
class level: each group is an independent single-pass capture of the SAME
steady-state window (same --launch-skip/--launch-count), so per-group rates
(FLOP/s, B/s) are comparable; AI is the ratio of those rates. Shared by
report/plot/regress/issues. stdlib only."""
import statistics

from rl_classify import TC_OPS_METRICS, SASS_FLOPS, flops_by_dtype


def _median(vals):
    return statistics.median(vals) if vals else 0.0


def _class_members(group_kernels, class_name):
    return [r for r in group_kernels.values() if r["class"] == class_name]


def _class_time_s(members):
    return sum(r.get("gpu__time_duration.sum", 0.0) for r in members)


def class_flop_rates(ncu_groups, class_name, ncu_cfg=None):
    """{dtype: FLOP/s} summed over all flop-bearing metric groups of one cell.
    Groups without their own time counter (sass) borrow the class time from the
    base group — identical capture window by construction. FP4 classes get
    their FLOPs from the dtype-blind pipe_tensor instruction counter (ops_path
    counts nothing for sm_120 mxf4nvf4 — see config fp4_pipe_comment)."""
    ncu_cfg = ncu_cfg or {}
    fp4_classes = set(ncu_cfg.get("fp4_pipe_classes", []))
    fp4_per_inst = ncu_cfg.get("fp4_flops_per_tensor_inst", 16384)
    base = ncu_groups.get("base") or ncu_groups.get("full") or {}
    base_t = _class_time_s(_class_members(base.get("kernels", {}), class_name))
    rates = {}
    for gname, g in ncu_groups.items():
        if gname == "base":
            continue
        members = _class_members(g["kernels"], class_name)
        t = _class_time_s(members) or base_t
        if t <= 0:
            continue
        agg = {}
        for r in members:
            for k, v in r.items():
                if isinstance(v, (int, float)) and k.endswith(".sum"):
                    agg[k] = agg.get(k, 0.0) + v
        for dtype, fl in flops_by_dtype(agg).items():
            rates[dtype] = rates.get(dtype, 0.0) + fl / t
        pipe_insts = agg.get("smsp__inst_executed_pipe_tensor.sum", 0.0)
        if pipe_insts and class_name in fp4_classes:
            rates["tc_fp4"] = rates.get("tc_fp4", 0.0) + pipe_insts * fp4_per_inst / t
    return rates


def class_aggregate(ncu_groups, class_name, gpu_cfg, ncu_cfg=None, fallback_flop_rates=None):
    """One (cell, restart, class) roofline record, or None.
    fallback_flop_rates: restart-0 FLOP rates for restarts that only measured
    the base group (FLOP counts are workload-deterministic)."""
    base = ncu_groups.get("base") or ncu_groups.get("full")
    if not base:
        return None
    members = _class_members(base["kernels"], class_name)
    if not members:
        return None
    t = _class_time_s(members)
    by = sum(r.get("dram__bytes.sum", 0.0) for r in members)
    if t <= 0:
        return None

    def wavg(key):
        return sum(r.get(key, 0.0) * r.get("gpu__time_duration.sum", 0.0)
                   for r in members) / t

    clk_hz = wavg("gpc__cycles_elapsed.avg.per_second")
    occ = wavg("sm__warps_active.avg.pct_of_peak_sustained_active")
    l1_hit = wavg("l1tex__t_sector_hit_rate.pct")

    rates = class_flop_rates(ncu_groups, class_name, ncu_cfg)
    flops_estimated = False
    if not rates and fallback_flop_rates:
        rates = fallback_flop_rates
        flops_estimated = True
    ach_flops = sum(rates.values())
    dtype = max(rates, key=rates.get) if rates else "cuda_fp32"

    ach_bw = by / t
    ai = ach_flops / ach_bw if ach_bw > 0 else 0.0
    peak_bw = gpu_cfg["dram_peak_gbs"] * 1e9
    # Effective peak for mixed-dtype kernels: FLOP-weighted harmonic mean of
    # the per-dtype peaks (time to issue the mix at peak = sum over dtypes of
    # flops_d / peak_d). Picking only the dominant dtype made the peak flip
    # 4x between cells/runs for kernels that mix accumulate precisions (the
    # FA2 kernel mixes f16-dst and f32-dst HMMA since #673) — %-roofline
    # jumped with the classification, not with the kernel. Reduces to the
    # single-dtype peak for pure kernels; dtypes without a configured rate
    # fall back to the dominant dtype's (keeps old behavior for SASS keys).
    fpc_table = gpu_cfg["flop_per_cycle"]
    fpc_dom = fpc_table[dtype]
    if ach_flops > 0 and len(rates) > 1:
        denom = sum(fl / fpc_table.get(d, fpc_dom) for d, fl in rates.items())
        fpc = ach_flops / denom if denom > 0 else fpc_dom
    else:
        fpc = fpc_dom
    peak_flops_boost = fpc * gpu_cfg["nominal_boost_ghz"] * 1e9
    peak_flops_clk = fpc * clk_hz
    ridge = peak_flops_boost / peak_bw
    bound = "memory" if ai < ridge else "compute"
    pct = 100.0 * (ach_bw / peak_bw if bound == "memory"
                   else (ach_flops / peak_flops_clk if peak_flops_clk else 0.0))
    return {
        "class": class_name, "dtype": dtype, "time_ns": t * 1e9,
        "n_kernels": len(members),
        "n_launches": sum(r.get("n_launches", 0) for r in members),
        "ai": ai, "achieved_gflops": ach_flops / 1e9, "achieved_gbs": ach_bw / 1e9,
        "flop_rates": {k: v / 1e9 for k, v in rates.items()},
        "flops_estimated_from_r0": flops_estimated,
        "ridge": ridge, "bound_by": bound, "pct_roofline": pct,
        "occupancy_pct": occ, "l1_hit_pct": l1_hit, "sm_clock_mhz": clk_hz / 1e6,
        "peak_gflops_at_clock": peak_flops_clk / 1e9,
        "peak_gbs": gpu_cfg["dram_peak_gbs"],
    }


def build_rows(run, cfg):
    gpu_cfg = cfg.get("gpu", cfg)
    ncu_cfg = cfg.get("ncu", {})
    """-> list of rows: one per (cell, class) with min/med/max over restarts."""
    rows = []
    for cell_key, restarts in sorted(run["cells"].items()):
        model, shape = cell_key.split("|")
        valid = [c for c in restarts if c.get("ncu_groups")]
        if not valid:
            continue
        classes = sorted({r["class"]
                          for c in valid
                          for g in c["ncu_groups"].values()
                          for r in g["kernels"].values()})
        for cls in classes:
            # restart 0 (or the first with TC/SASS groups) anchors the FLOP rates
            anchor_rates = None
            for c in valid:
                rt = class_flop_rates(c["ncu_groups"], cls, ncu_cfg)
                if rt:
                    anchor_rates = rt
                    break
            per_restart = [a for c in valid
                           if (a := class_aggregate(c["ncu_groups"], cls, gpu_cfg, ncu_cfg,
                                                    fallback_flop_rates=anchor_rates))]
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
    """Annotate rows with % of captured-window time within their cell (fallback
    when no nsys share is available; rl_report overrides with nsys shares)."""
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["model"], r["shape"]), []).append(r)
    for cell_rows in by_cell.values():
        total = sum(r["time_ns"] for r in cell_rows) or 1
        for r in cell_rows:
            r["time_share_pct"] = 100.0 * r["time_ns"] / total
    return rows
