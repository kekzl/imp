"""Kernel classification + plain per-kernel metric aggregation. The roofline
math itself lives in rl_table (it joins multiple ncu metric groups). stdlib only."""
import re

# tensor ops-path counter -> dtype peak key.
# dst_fp32 maps to a *_f32acc peak: GeForce sm_120 runs FP16/FP8 tensor cores
# with FP32 accumulate at 1/4 of the f16-accumulate rate (measured 2026-06-07,
# saturated mma.sync microbench: fp16 f16acc 1956 vs f32acc 253 TFLOPS,
# fp8 f32acc 496 TOPS — see issue #595/#596 calibration).
TC_OPS_METRICS = {
    "sm__ops_path_tensor_src_fp4_dst_fp32.sum": "tc_fp4",
    "sm__ops_path_tensor_src_fp4_fp6_dst_fp16.sum": "tc_fp4",
    "sm__ops_path_tensor_src_fp4_fp6_dst_fp32.sum": "tc_fp4",
    "sm__ops_path_tensor_src_fp8_dst_fp16.sum": "tc_fp8",
    "sm__ops_path_tensor_src_fp8_dst_fp32.sum": "tc_fp8_f32acc",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum": "tc_fp16",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum": "tc_fp16_f32acc",
    "sm__ops_path_tensor_src_bf16_dst_fp32.sum": "tc_bf16_f32acc",
    "sm__ops_path_tensor_src_int8.sum": "tc_int8",
}
# SASS thread-inst counters -> (flops per inst, dtype key). HFMA2/packed-half
# executes 2 FMAs per thread-inst; hfma is counted as packed (x4 flops) — an
# upper bound, flagged in the report methodology.
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


def aggregate_metrics(launches):
    """Sum .sum counters over the captured launches of one kernel; time-weight
    everything else (rates/percentages). Values are SI (post rl_ncu parser)."""
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
    agg["n_launches"] = len(launches)
    return agg


def flops_by_dtype(metrics):
    """{dtype: flops} from whatever TC/SASS counters are present in `metrics`."""
    out = {}
    for m, key in TC_OPS_METRICS.items():
        v = metrics.get(m, 0.0) or 0.0
        if v:
            out[key] = out.get(key, 0.0) + v
    for m, (factor, key) in SASS_FLOPS.items():
        v = metrics.get(m, 0.0) or 0.0
        if v:
            out[key] = out.get(key, 0.0) + v * factor
    return out
