"""Markdown report: Module-1 roofline table + Module-2 coverage matrix + lever
list. Renders purely from history (run JSON + nsys extracts). stdlib only."""
import json
import os
import re
import statistics

import rl_history
import rl_table

BATCHED_RX = re.compile(r"[Bb]atched")


def load_nsys_extracts(run):
    """{cell_key: [extract_dict per restart]}"""
    raw = os.path.join(rl_history.RAW_DIR, run["run_id"])
    out = {}
    for cell_key, restarts in run["cells"].items():
        for c in restarts:
            f = c.get("nsys_extract_file")
            if not f:
                continue
            p = os.path.join(raw, f)
            if os.path.exists(p):
                with open(p) as fh:
                    out.setdefault(cell_key, []).append(json.load(fh))
    return out


def coverage_for_extract(ex, phases):
    """Per-class time shares within the given phases + attention-path split.
    Legacy attention = causal_softmax/softcap/ptr-array kernels + the cuBLAS
    GEMM kernels launch-adjacent to those markers (QK^T/PV) — see rl_nsys."""
    per_class = {}
    attn = {"legacy_softmax": 0, "legacy_cublas_qkpv": 0, "fa2": 0,
            "fmha_fp8": 0, "fmha_wmma": 0, "wmma_fallback": 0, "decode_paged": 0}
    total = 0
    for name, rec in ex["per_kernel"].items():
        t = sum(rec["phase_time_ns"].get(p, 0) for p in phases)
        if t <= 0:
            continue
        total += t
        per_class[rec["class"]] = per_class.get(rec["class"], 0) + t
        attn_gemm = sum(rec.get("attn_gemm_phase_time_ns", {}).get(p, 0) for p in phases)
        if rec["class"] == "attn_legacy_softmax":
            attn["legacy_softmax"] += t
        elif rec["class"] == "gemm_cublas" and attn_gemm:
            attn["legacy_cublas_qkpv"] += attn_gemm
        elif rec["class"] == "attn_fa2":
            attn["fa2"] += t
        elif rec["class"] == "attn_fmha_fp8":
            attn["fmha_fp8"] += t
        elif rec["class"] in ("attn_fmha_wmma", "attn_fmha_mxfp4"):
            attn["fmha_wmma"] += t
        elif rec["class"] == "attn_wmma_fallback":
            attn["wmma_fallback"] += t
        elif rec["class"] == "attn_decode_paged":
            attn["decode_paged"] += t
    legacy = attn["legacy_softmax"] + attn["legacy_cublas_qkpv"]
    attn_total = sum(attn.values())
    return {
        "total_ns": total,
        "class_share_pct": {k: 100.0 * v / total for k, v in sorted(per_class.items())},
        "attn_ns": attn,
        "legacy_attn_share_of_total_pct": 100.0 * legacy / total if total else 0.0,
        "legacy_share_of_attention_pct": 100.0 * legacy / attn_total if attn_total else 0.0,
        "attribution_sanity": {
            "n_attn_gemm_attributed": ex.get("n_attn_gemm_attributed", 0),
            "n_causal_softmax": ex.get("n_causal_softmax", 0),
        },
    }


def coverage_matrix(run, cfg):
    """{cell_key: {metric: [per-restart values]}} summarized min/med/max.
    Prefill cells use the prefill_window phase; decode cells the post_prefill
    phase (pure decode — see rl_nsys.extract docstring)."""
    extracts = load_nsys_extracts(run)
    out = {}
    for cell_key, exs in sorted(extracts.items()):
        shape_key = cell_key.split("|")[1]
        phase = cfg["shapes"][shape_key]["phase"]
        phases = ("prefill_window",) if phase == "prefill" else ("post_prefill",)
        covs = [coverage_for_extract(e, phases) for e in exs if "per_kernel" in e]
        if not covs:
            continue
        agg = {"n_restarts": len(covs)}
        for metric in ("legacy_attn_share_of_total_pct", "legacy_share_of_attention_pct"):
            vals = [c[metric] for c in covs]
            agg[metric] = {"min": min(vals), "med": statistics.median(vals), "max": max(vals)}
        # median-restart class shares
        med_cov = sorted(covs, key=lambda c: c["legacy_attn_share_of_total_pct"])[len(covs) // 2]
        agg["class_share_pct_med_restart"] = med_cov["class_share_pct"]
        out[cell_key] = agg
    return out


def apply_nsys_time_shares(rows, cov):
    """Replace the ncu-capture-window time share (oversampling bias) with the
    true per-phase class share from the nsys timeline where available."""
    for r in rows:
        cell_key = f"{r['model']}|{r['shape']}"
        share = cov.get(cell_key, {}).get("class_share_pct_med_restart", {}).get(r["class"])
        if share is not None:
            r["time_share_pct"] = share
            r["time_share_source"] = "nsys"
        else:
            r["time_share_source"] = "ncu-window"
    return rows


def fmt_num(x, digits=1):
    if x >= 1000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def module1_table(rows, min_share_pct=1.0):
    hdr = ("| Kernel class | Model/Shape | Time share % (nsys) | AI (FLOP/B) | achieved "
           "(med) | Peak | %-roofline (min/med/max) | bound-by | dtype | L1-hit% | Occ % |\n"
           "|---|---|---|---|---|---|---|---|---|---|---|\n")
    lines = []
    for r in sorted(rows, key=lambda r: (r["model"], r["shape"], -r.get("time_share_pct", 0))):
        if r.get("time_share_pct", 0) < min_share_pct:
            continue
        if r["bound_by"] == "memory":
            ach = f"{fmt_num(r['achieved_med'])} GB/s"
            peak = f"{fmt_num(r['peak_gbs'])} GB/s"
        else:
            ach = f"{fmt_num(r['achieved_med'])} GFLOPS"
            peak = f"{fmt_num(r['peak_gflops_at_clock'])} GFLOPS @{r['sm_clock_mhz']:.0f}MHz"
        share = f"{r['time_share_pct']:.1f}"
        if r.get("time_share_source") == "ncu-window":
            share += "*"
        lines.append(
            f"| {r['class']} | {r['model']}/{r['shape']} | {share} "
            f"| {r['ai']:.2f} | {ach} | {peak} "
            f"| {r['pct_roofline_min']:.1f} / {r['pct_roofline_med']:.1f} / {r['pct_roofline_max']:.1f} "
            f"| {r['bound_by']} | {r['dtype']} | {r.get('l1_hit_pct', 0):.0f} | {r['occupancy_pct']:.0f} |")
    return hdr + "\n".join(lines) + "\n(*) = time share from the ncu window (nsys share missing)\n"


def module2_table(cov):
    hdr = ("| Cell | Legacy-attn %-share of window (min/med/max) | Legacy share of "
           "attention % (min/med/max) | Restarts |\n|---|---|---|---|\n")
    lines = []
    for cell, agg in cov.items():
        a = agg["legacy_attn_share_of_total_pct"]
        b = agg["legacy_share_of_attention_pct"]
        lines.append(f"| {cell} | {a['min']:.1f} / {a['med']:.1f} / {a['max']:.1f} "
                     f"| {b['min']:.1f} / {b['med']:.1f} / {b['max']:.1f} | {agg['n_restarts']} |")
    return hdr + "\n".join(lines) + "\n"


def levers(rows, cov, targets):
    """Estimated levers, ranked: time_share x roofline headroom (Amdahl).
    All gains are ESTIMATES derived from measured shares + measured %-roofline."""
    out = []
    for r in rows:
        share = r.get("time_share_pct", 0)
        if share < 2.0:
            continue
        target = targets["memory_bound_pct_target" if r["bound_by"] == "memory"
                         else "compute_bound_pct_target"]
        pct = r["pct_roofline_med"]
        if pct >= target:
            continue
        # speedup of this class to target => saved fraction of window time
        saved_pct = share * (1.0 - pct / target)
        out.append({
            "kind": "roofline_headroom",
            "class": r["class"], "cell": f"{r['model']}/{r['shape']}",
            "time_share_pct": share, "pct_roofline_med": pct, "target_pct": target,
            "est_gain_pct": saved_pct,
            "bound_by": r["bound_by"], "dtype": r["dtype"],
        })
    for cell, agg in cov.items():
        med = agg["legacy_attn_share_of_total_pct"]["med"]
        if med >= 2.0:
            out.append({
                "kind": "legacy_attention_fallback", "class": "attn_legacy_softmax+cublas",
                "cell": cell, "time_share_pct": med,
                "est_gain_pct": med * 0.5,
                "note": "assumes FA2-class path is ~2x the materialized path (estimate)",
            })
    out.sort(key=lambda l: -l["est_gain_pct"])
    return out


def render(run, cfg, ab_results=None):
    gpu = cfg["gpu"]
    rows = rl_table.time_share_within_cell(rl_table.build_rows(run, cfg))
    cov = coverage_matrix(run, cfg)
    rows = apply_nsys_time_shares(rows, cov)
    lv = levers(rows, cov, cfg["roofline"])
    meta = run["meta"]
    md = []
    md.append(f"# imp Roofline Audit — Run `{run['run_id']}`\n")
    md.append(f"- Commit: `{meta['git']['commit']}`{' (dirty)' if meta['git']['dirty'] else ''}"
              f" · Timestamp: {meta['timestamp']} · config_version: {meta['config_version']}")
    env = meta.get("env", {})
    md.append(f"- GPU: {env.get('gpu','?')} · Driver: {env.get('driver','?')} · "
              f"CUDA: {env.get('cuda','?')} · ncu: {env.get('ncu','?')}")
    md.append(f"- Methodology: ncu `--clock-control base` (clocks locked), pinned counters "
              f"(config_version {meta['config_version']}), {meta['restarts']} container restarts, "
              f"AI from measured `dram__bytes.sum`, FLOPs from `sm__ops_path_tensor_*`/SASS counters. "
              f"Raw exports: `tools/roofline/history/raw/{run['run_id']}/`.\n")
    md.append("## Module 1 — Roofline per kernel class\n")
    md.append("Time share = class share of the cell's nsys phase timeline "
              "(prefill_window resp. post_prefill=decode). %-roofline/AI from the "
              "ncu steady-state window. Peak compute normalized to the measured (locked) "
              "SM clock; ridge to the boost clock.\n")
    md.append(module1_table(rows))
    md.append("\n## Module 2 — Coverage / legacy fallback (from the nsys timeline)\n")
    md.append(module2_table(cov))
    if ab_results:
        md.append("\n### A/B fallback deltas (measured, unprofiled)\n")
        md.append("```json\n" + json.dumps(ab_results, indent=1) + "\n```\n")
    md.append("\n## Lever list (prioritized, **all gains = estimate** via Amdahl "
              "from measured time share × roofline headroom)\n")
    md.append("\n> **Graphs caveat (2026-07-13):** these shares come from "
              "`--no-cuda-graphs` profiles; under the shipped graphs+PDL decode "
              "loop, grid-(1,1,1)/launch-latency classes (moe_routing, rmsnorm, "
              "rope, kv_write, elementwise, split-K reduce) largely overlap away "
              "— on Qwen3-30B the no-graphs kernel-time sum is ~1.8× the real "
              "graphs-ON step. Fusing the router chain (6.9% share) moved e2e 0%; "
              "capping decode split-K regressed −21…−35%. Validate any lever from "
              "these classes with a graphs-ON e2e A/B before building it; only "
              "byte-holding / critical-path classes translate.\n")
    for i, l in enumerate(lv[:15], 1):
        md.append(f"{i}. **{l['class']}** @ {l['cell']} — est. window gain "
                  f"~{l['est_gain_pct']:.1f}% (time share {l['time_share_pct']:.1f}%, "
                  f"{'%-roofline med ' + format(l.get('pct_roofline_med', 0), '.1f') + ' vs target ' + format(l.get('target_pct', 0), '.0f') if l['kind']=='roofline_headroom' else l.get('note','')})")
    md.append("\n*(Every number traceable: run_id = commit_timestamp; raw ncu CSV + "
              "nsys extract are append-only under history/raw/<run_id>/.)*\n")
    return "\n".join(md)
