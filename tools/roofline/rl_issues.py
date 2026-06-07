"""Deliverable 5: GitHub issues from audit findings (English), idempotent via
audit-key marker. Dry-run by default; real creation only with --create. stdlib only."""
import json
import re
import subprocess

import rl_report
import rl_table

AUDIT_KEY_RX = re.compile(r"<!-- audit-key: (\S+) -->")


# Classes whose low %-roofline at the measured shapes is a workload artifact or
# noise floor, not an actionable lever — kept out of issues (documented in the
# audit report's "Known artifacts" section): tiny-ctx decode attention,
# launch-bound micro-kernels at tg256, routing overhead, unclassified leftovers.
SKIP_CLASSES = {"unclassified", "moe_routing", "kv_write", "kv_gather",
                "elementwise", "sampling", "rope"}
SKIP_CELL_CLASS = {("tg256", "attn_decode_paged"), ("tg256", "rmsnorm")}


def build_issues(run, cfg, ab_results=None):
    rows = rl_table.time_share_within_cell(rl_table.build_rows(run, cfg))
    cov = rl_report.coverage_matrix(run, cfg)
    rows = rl_report.apply_nsys_time_shares(rows, cov)
    lv = rl_report.levers(rows, cov, cfg["roofline"])
    min_gain = cfg["issues"]["min_lever_pct_estimate"]
    min_share = cfg["issues"].get("min_time_share_pct", 8.0)

    # consolidate: ONE issue per kernel class (cells listed inside), one for
    # the legacy-attention coverage finding — no per-cell spam.
    by_class = {}
    legacy = []
    for l in lv:
        if l["est_gain_pct"] < min_gain:
            continue
        if l["kind"] == "legacy_attention_fallback":
            legacy.append(l)
            continue
        shape = l["cell"].split("/")[1]
        if l["class"] in SKIP_CLASSES or (shape, l["class"]) in SKIP_CELL_CLASS:
            continue
        if l["time_share_pct"] < min_share:
            continue
        by_class.setdefault(l["class"], []).append(l)

    candidates = []
    for cls, ls in by_class.items():
        ls.sort(key=lambda x: -x["est_gain_pct"])
        top = ls[0]
        candidates.append({"kind": "roofline_headroom", "class": cls,
                           "top": top, "cells": ls,
                           "est_gain_pct": top["est_gain_pct"]})
    if legacy:
        legacy.sort(key=lambda x: -x["est_gain_pct"])
        candidates.append({"kind": "legacy_attention_fallback",
                           "class": "attn_legacy_softmax+cublas",
                           "top": legacy[0], "cells": legacy,
                           "est_gain_pct": legacy[0]["est_gain_pct"]})
    candidates.sort(key=lambda c: -c["est_gain_pct"])

    issues = []
    for rank, c in enumerate(candidates, 1):
        prio = "P0" if rank <= 2 else ("P1" if rank <= 4 else "P2")
        top = c["top"]
        if c["kind"] == "legacy_attention_fallback":
            key = "roofline:coverage:legacy_attention"
            title = (f"Prefill attention coverage: hd!=128 models still run the "
                     f"materialized causal_softmax+cuBLAS path "
                     f"({top['time_share_pct']:.1f}% of {top['cell']} window)")
            label_kind = "coverage"
        else:
            key = f"roofline:class:{c['class']}"
            title = (f"{c['class']}: {top['pct_roofline_med']:.0f}% roofline "
                     f"({top['bound_by']}-bound) at {top['time_share_pct']:.0f}% "
                     f"window time on {top['cell']}")
            label_kind = "roofline"
        body = render_body_consolidated(c, run, key)
        issues.append({"key": key, "title": title, "body": body,
                       "labels": cfg["issues"]["labels"] + [label_kind, prio.lower()],
                       "prio": prio})
    return issues


def render_body_consolidated(c, run, key):
    meta = run["meta"]
    top = c["top"]
    lines = [f"<!-- audit-key: {key} -->", "## Finding"]
    if c["kind"] == "roofline_headroom":
        lines += [
            f"- Kernel class **{c['class']}** reaches only "
            f"**{top['pct_roofline_med']:.1f}%** of the applicable roofline "
            f"({top['bound_by']}-bound, dtype `{top['dtype']}`) on its "
            f"heaviest cell **{top['cell']}** ({top['time_share_pct']:.1f}% of "
            f"the measured window).",
            "- Affected cells (time share / %-roofline median):",
        ]
        for l in c["cells"]:
            lines.append(f"  - {l['cell']}: {l['time_share_pct']:.1f}% / "
                         f"{l['pct_roofline_med']:.1f}%")
    else:
        lines += [
            f"- The materialized attention path (cuBLAS QK^T -> causal_softmax "
            f"-> cuBLAS PV) is **0.0%** on all hd=128 models, but hd!=128 "
            f"models (gemma-3, hd=256) still spend "
            f"**{top['time_share_pct']:.1f}%** of the prefill window there "
            f"(= 92-99% of their attention time) — the whole FA2 family "
            f"declines hd!=128.",
        ]
    lines += [
        "",
        "## Evidence",
        f"- History run: `{run['run_id']}` (commit `{meta['git']['commit']}`, "
        f"{meta['timestamp']}), raw exports: "
        f"`tools/roofline/history/raw/{run['run_id']}/`",
        f"- Report: `docs/audit/roofline_2026_06_07.md` · Reproduce: "
        f"`tools/roofline/roofline report --run {run['run_id']}`",
        "",
        "## Expected lever",
        f"- **Estimate:** up to ~{c['est_gain_pct']:.0f}% of the affected window "
        "(Amdahl: time share x roofline headroom). This is an estimate, not a "
        "measurement; structural limits (LSU-bound attention, small-M GEMM) may "
        "cap it well below that.",
        "",
        "## Acceptance criteria",
    ]
    if c["kind"] == "roofline_headroom":
        lines += [f"- `{c['class']}` reaches > "
                  f"{top.get('target_pct', 70):.0f}% roofline on {top['cell']} "
                  f"(or a documented structural-ceiling analysis), verified via "
                  f"`roofline regress --baseline {run['run_id']}`."]
    else:
        lines += ["- Legacy prefill attention share < 2% on hd=256 models "
                  "(FA2-family hd!=128 support or an equivalent tiled path), "
                  "verified via the coverage matrix of a new roofline run."]
    return "\n".join(lines)


def render_body(l, run, key):
    meta = run["meta"]
    lines = [
        f"<!-- audit-key: {key} -->",
        "## Finding",
    ]
    if l["kind"] == "roofline_headroom":
        lines += [
            f"- Kernel class **{l['class']}** on **{l['cell']}** reaches "
            f"**{l['pct_roofline_med']:.1f}%** of the applicable roofline "
            f"({l['bound_by']}-bound, dtype `{l['dtype']}`); target is {l['target_pct']:.0f}%.",
            f"- Measured time share of the profiled window: **{l['time_share_pct']:.1f}%**.",
        ]
    else:
        lines += [
            f"- **{l['time_share_pct']:.1f}%** (median over restarts) of the "
            f"{l['cell']} prefill window still runs on the legacy materialized "
            f"attention path (causal_softmax kernels + batched cuBLAS QK^T/PV).",
        ]
    lines += [
        "",
        "## Evidence",
        f"- History run: `{run['run_id']}` (commit `{meta['git']['commit']}`"
        f"{', dirty' if meta['git']['dirty'] else ''}, {meta['timestamp']})",
        f"- Raw exports: `tools/roofline/history/raw/{run['run_id']}/`",
        f"- Reproduce: `tools/roofline/roofline report --run {run['run_id']}`",
        "",
        "## Expected lever",
        f"- **Estimate:** ~{l['est_gain_pct']:.1f}% of the affected window "
        "(Amdahl: time share x roofline headroom). This is an estimate, not a measurement.",
        "",
        "## Acceptance criteria",
    ]
    if l["kind"] == "roofline_headroom":
        lines += [f"- `{l['class']}` reaches > {l['target_pct']:.0f}% roofline on "
                  f"{l['cell']} (verify: `roofline regress --baseline {run['run_id']}`).",]
    else:
        lines += ["- Legacy prefill attention path share < 5% on the affected cell "
                  f"(verify: `roofline report --run <new>` coverage matrix vs `{run['run_id']}`).",]
    return "\n".join(lines)


def existing_audit_issues():
    out = subprocess.run(
        ["gh", "issue", "list", "--label", "audit", "--state", "all",
         "--json", "number,title,body", "--limit", "200"],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"gh issue list failed: {out.stderr[-300:]}")
    by_key = {}
    for iss in json.loads(out.stdout or "[]"):
        m = AUDIT_KEY_RX.search(iss.get("body") or "")
        if m:
            by_key[m.group(1)] = iss
    return by_key


def ensure_labels(labels):
    for lb in labels:
        subprocess.run(["gh", "label", "create", lb, "--force"],
                       capture_output=True, text=True)


def create_or_update(issues, dry_run=True):
    existing = existing_audit_issues() if not dry_run else {}
    results = []
    for iss in issues:
        if dry_run:
            results.append(("DRY-RUN", iss["title"]))
            continue
        ensure_labels(iss["labels"])
        if iss["key"] in existing:
            num = str(existing[iss["key"]]["number"])
            subprocess.run(["gh", "issue", "comment", num, "--body",
                            "Updated by new audit run:\n\n" + iss["body"]],
                           capture_output=True, text=True)
            results.append(("updated #" + num, iss["title"]))
        else:
            out = subprocess.run(
                ["gh", "issue", "create", "--title", iss["title"],
                 "--body", iss["body"]]
                + sum((["--label", lb] for lb in iss["labels"]), []),
                capture_output=True, text=True)
            results.append(("created " + out.stdout.strip(), iss["title"]))
    return results
