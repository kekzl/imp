"""Kernel classification + roofline math (AI, achieved FLOPS/BW, %-roofline). stdlib only."""
import re

# tensor ops-path counter -> dtype peak key
TC_OPS_METRICS = {
    "sm__ops_path_tensor_src_fp4_dst_fp32.sum": "tc_fp4",
    "sm__ops_path_tensor_src_fp4_fp6_dst_fp16.sum": "tc_fp4",
    "sm__ops_path_tensor_src_fp4_fp6_dst_fp32.sum": "tc_fp4",
    "sm__ops_path_tensor_src_fp8_dst_fp16.sum": "tc_fp8",
    "sm__ops_path_tensor_src_fp8_dst_fp32.sum": "tc_fp8",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum": "tc_fp16",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum": "tc_fp16",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum": "tc_bf16",
    "sm__ops_path_tensor_src_int8.sum": "tc_int8",
}
# SASS thread-inst counters -> (flops per inst, dtype key). HFMA2/packed-half executes
# 2 FMAs per thread-inst; we count hfma as packed (x4 flops) — calibrated against
# known-shape GEMV, see tools/roofline/README.md "FLOP counting".
SASS_FLOPS = {
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum": (2.0, "cuda_fp32"),
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum": (1.0, "cuda_fp32"),
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum": (1.0, "cuda_fp32"),
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum": (4.0, "cuda_fp16"),
    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum": (2.0, "cuda_fp16"),
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum": (2.0, "cuda_fp16"),
}


def build_classifier(cfg):
    rules = [(c["name"], c["group"], re.compile(c["regex"])) for c in cfg["kernel_classes"]]

    def classify(kernel_name):
        for name, group, rx in rules:
            if rx.search(kernel_name):
                return name, group
        return "unclassified", "unclassified"

    return classify


def kernel_flops(metrics):
    """Return (total_flops, dominant_dtype). TC ops counters count FLOPs directly
    (1 op = 1 multiply or add); SASS half/float inst are converted per SASS_FLOPS."""
    by_dtype = {}
    for m, key in TC_OPS_METRICS.items():
        v = metrics.get(m, 0.0) or 0.0
        if v:
            by_dtype[key] = by_dtype.get(key, 0.0) + v
    for m, (factor, key) in SASS_FLOPS.items():
        v = metrics.get(m, 0.0) or 0.0
        if v:
            by_dtype[key] = by_dtype.get(key, 0.0) + v * factor
    total = sum(by_dtype.values())
    dominant = max(by_dtype, key=by_dtype.get) if by_dtype else "cuda_fp32"
    return total, dominant


def roofline_point(metrics, gpu_cfg):
    """Per-launch (or aggregated) metrics dict -> roofline record."""
    t_s = metrics.get("gpu__time_duration.sum", 0.0)  # SI (seconds) post-parser
    t_ns = t_s * 1e9
    dram_bytes = metrics.get("dram__bytes.sum", 0.0)
    sm_hz = metrics.get("gpc__cycles_elapsed.avg.per_second", 0.0)
    flops, dtype = kernel_flops(metrics)

    ai = (flops / dram_bytes) if dram_bytes > 0 else 0.0
    achieved_flops = flops / t_s if t_s > 0 else 0.0
    achieved_bw = dram_bytes / t_s if t_s > 0 else 0.0

    fpc = gpu_cfg["flop_per_cycle"][dtype]
    peak_flops_at_clock = fpc * sm_hz                       # at the measured (locked) clock
    peak_flops_boost = fpc * gpu_cfg["nominal_boost_ghz"] * 1e9
    peak_bw = gpu_cfg["dram_peak_gbs"] * 1e9
    ridge = peak_flops_boost / peak_bw

    bound = "memory" if ai < ridge else "compute"
    if bound == "memory":
        pct = 100.0 * achieved_bw / peak_bw
    else:
        pct = 100.0 * achieved_flops / peak_flops_at_clock if peak_flops_at_clock > 0 else 0.0

    return {
        "time_ns": t_ns,
        "flops": flops,
        "dram_bytes": dram_bytes,
        "dtype": dtype,
        "ai_flop_per_byte": ai,
        "achieved_gflops": achieved_flops / 1e9,
        "achieved_gbs": achieved_bw / 1e9,
        "sm_clock_mhz": sm_hz / 1e6,
        "ridge_flop_per_byte": ridge,
        "bound_by": bound,
        "pct_roofline": pct,
        "sm_pct": metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", 0.0),
        "dram_pct": metrics.get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", 0.0),
        "occupancy_pct": metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0),
        "l1tex_pct": metrics.get("l1tex__throughput.avg.pct_of_peak_sustained_elapsed", 0.0),
        "l1_hit_pct": metrics.get("l1tex__t_sector_hit_rate.pct", 0.0),
    }


def aggregate_launches(launches, gpu_cfg):
    """Sum counters over all captured launches of one kernel, then compute the
    roofline point on the aggregate (time-weighted by construction). Percent/rate
    metrics are time-weighted-averaged."""
    if not launches:
        return None
    summed, weighted = {}, {}
    total_t = sum(l.get("gpu__time_duration.sum", 0.0) for l in launches) or 1.0
    for l in launches:
        t = l.get("gpu__time_duration.sum", 0.0)
        for k, v in l.items():
            if v is None or not isinstance(v, (int, float)):
                continue
            if k.endswith(".sum"):
                summed[k] = summed.get(k, 0.0) + v
            else:
                weighted[k] = weighted.get(k, 0.0) + v * t
    agg = dict(summed)
    for k, v in weighted.items():
        agg[k] = v / total_t
    rec = roofline_point(agg, gpu_cfg)
    rec["n_launches"] = len(launches)
    return rec
