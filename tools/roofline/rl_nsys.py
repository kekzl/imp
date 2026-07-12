"""nsys capture + sqlite extraction: kernel time shares, legacy-attention
attribution (marker adjacency), phase split (init/prefill/decode), and ncu
launch-skip calibration. stdlib only.

Legacy-attention attribution: the materialized path launches
build_attn_ptr_arrays -> cuBLAS QK^T -> causal_softmax -> cuBLAS PV, so a
cuBLAS GEMM kernel launch-adjacent (+-1 in launch order) to one of those marker
kernels belongs to attention; all other cuBLAS GEMMs are dense layer GEMMs.
(The nsys cublas API trace produces no sqlite table in this setup, so
correlation-id attribution is not available — validated empirically: the
attributed count must be ~2x the causal_softmax count.)"""
import os
import re
import sqlite3
import subprocess

PREFILL_ATTN_RX = re.compile(
    r"fmha|flash_attention|causal_softmax|softcap_fp16|build_attn_ptr")
MARKER_RX = re.compile(r"causal_softmax|softcap_fp16|build_attn_ptr")


def docker_nsys_cmd(cfg, out_host_dir, out_base_name, imp_cli_args, extra_env=None):
    d = cfg["docker"]
    env_flags = []
    for k, v in {**d.get("env", {}), **(extra_env or {})}.items():
        env_flags += ["-e", f"{k}={v}"]
    return [
        "docker", "run", "--rm", "--gpus", "all",
        "-u", f"{os.getuid()}:{os.getgid()}", "-w", "/tmp",
        "-v", f"{d['models_mount']}:/models",
        "-v", f"{out_host_dir}:/out",
        "-v", "/opt/nvidia:/opt/nvidia:ro",
        *env_flags,
        "--entrypoint", cfg["nsys"]["binary"],
        d["image"],
        "profile",
        f"--trace={cfg['nsys']['trace']}",
        # NO --cuda-memory-usage: nothing in the pipeline reads the memory
        # tables, and on this WSL2 driver the flag makes the nsys injection
        # SIGSEGV gpt-oss-20b at model teardown (valid cudaFreeAsync burst +
        # mempool trims; app dies before the CUPTI flush -> rep has no kernel
        # rows). Kernel/API/NVTX timing is unaffected. Isolated 2026-07-13.
        "--force-overwrite=true",
        "--stats=false",
        "--sample=none", "--cpuctxsw=none", "--backtrace=none",
        "-o", f"/out/{out_base_name}",
        d["imp_cli"], *imp_cli_args,
    ]


def export_sqlite(nsys_bin, rep_path):
    sqlite_path = rep_path.replace(".nsys-rep", ".sqlite")
    out = subprocess.run(
        [nsys_bin, "export", "--type", "sqlite", "--force-overwrite", "true",
         "--output", sqlite_path, rep_path],
        capture_output=True, text=True)
    if out.returncode != 0 or not os.path.exists(sqlite_path):
        raise RuntimeError(f"nsys export failed: {out.stderr[-500:]}")
    return sqlite_path


def _kernel_rows(con):
    """[(start, end, name)] for all kernel executions, launch-ordered."""
    cur = con.cursor()
    cols = {r[1] for r in cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")}
    name_col = "demangledName" if "demangledName" in cols else "shortName"
    return cur.execute(
        f"SELECT k.start, k.end, s.value "
        f"FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.{name_col} = s.id "
        f"ORDER BY k.start").fetchall()


def extract(sqlite_path, classify):
    """Per-kernel time totals (class + legacy-attention attribution) and phase
    split. Phases: init (before first prefill-attention kernel), prefill_window
    (up to the END of the last prefill-attention kernel — for decode-shape runs
    this covers warmup + pp-bench + the tg prefill), post_prefill (pure decode
    for decode-shape runs). Returns a JSON-serializable dict."""
    con = sqlite3.connect(sqlite_path)
    kernels = _kernel_rows(con)
    con.close()
    if not kernels:
        return {"error": "no kernels in trace"}

    names = [k[2] for k in kernels]
    classes = [classify(n) for n in names]
    is_marker = [bool(MARKER_RX.search(n)) for n in names]
    attn_starts = [s for (s, e, n) in kernels if PREFILL_ATTN_RX.search(n)]
    first_attn = min(attn_starts) if attn_starts else kernels[0][0]
    last_attn_end = max((e for (s, e, n) in kernels if PREFILL_ATTN_RX.search(n)),
                        default=kernels[0][0])

    per_kernel = {}
    n_attn_gemm = 0
    for i, (s, e, name) in enumerate(kernels):
        kcls, group = classes[i]
        if s < first_attn:
            phase = "init"
        elif s >= last_attn_end:
            phase = "post_prefill"
        else:
            phase = "prefill_window"
        rec = per_kernel.setdefault(name, {
            "class": kcls, "group": group, "count": 0, "time_ns": 0,
            "phase_time_ns": {}, "attn_gemm_time_ns": 0,
            "attn_gemm_phase_time_ns": {}})
        t = e - s
        rec["count"] += 1
        rec["time_ns"] += t
        rec["phase_time_ns"][phase] = rec["phase_time_ns"].get(phase, 0) + t
        if kcls == "gemm_cublas" and (
                (i > 0 and is_marker[i - 1]) or
                (i + 1 < len(kernels) and is_marker[i + 1])):
            n_attn_gemm += 1
            rec["attn_gemm_time_ns"] += t
            rec["attn_gemm_phase_time_ns"][phase] = \
                rec["attn_gemm_phase_time_ns"].get(phase, 0) + t

    n_softmax = sum(1 for i, n in enumerate(names)
                    if classes[i][0] == "attn_legacy_softmax" and "causal_softmax" in n)
    return {
        "n_kernels_total": len(kernels),
        "first_attn_ts": first_attn,
        "last_attn_end_ts": last_attn_end,
        "n_attn_gemm_attributed": n_attn_gemm,
        "n_causal_softmax": n_softmax,
        "per_kernel": per_kernel,
    }


def matched_launches_after_init(extract_result, capture_rx):
    """How many post-init kernel launches match the ncu capture regex —
    used to derive --launch-skip so the ncu window sits in steady state."""
    rx = re.compile(capture_rx)
    total = 0
    for name, rec in extract_result["per_kernel"].items():
        if rx.search(name):
            total += max(rec["count"] - _phase_count_estimate(rec, "init"), 0)
    return total


def _phase_count_estimate(rec, phase):
    # counts are not tracked per phase; estimate via time share (init kernels are
    # conversion bursts, close enough for skip calibration)
    t_total = rec["time_ns"] or 1
    return int(round(rec["count"] * (rec["phase_time_ns"].get(phase, 0) / t_total)))


def matched_init_launches(extract_result, capture_rx):
    rx = re.compile(capture_rx)
    return sum(_phase_count_estimate(rec, "init")
               for name, rec in extract_result["per_kernel"].items()
               if rx.search(name))
