"""Deliverable 5: GitHub issues from audit findings (English), idempotent via
audit-key marker. Dry-run by default; real creation only with --create. stdlib only."""
import json
import re
import subprocess

import rl_report
import rl_table

AUDIT_KEY_RX = re.compile(r"<!-- audit-key: (\S+) -->")


def build_issues(run, cfg, ab_results=None):
    gpu = cfg["gpu"]
    rows = rl_table.time_share_within_cell(rl_table.build_rows(run, gpu))
    cov = rl_report.coverage_matrix(run, cfg)
    rows = rl_report.apply_nsys_time_shares(rows, cov)
    lv = rl_report.levers(rows, cov, cfg["roofline"])
    min_gain = cfg["issues"]["min_lever_pct_estimate"]
    issues = []
    for rank, l in enumerate([x for x in lv if x["est_gain_pct"] >= min_gain], 1):
        prio = "P0" if rank <= 2 else ("P1" if rank <= 5 else "P2")
        if l["kind"] == "legacy_attention_fallback":
            key = f"roofline:legacy_attn:{l['cell'].replace('/', ':')}"
            title = (f"Prefill attention coverage: {l['time_share_pct']:.0f}% of "
                     f"{l['cell']} window on materialized causal_softmax+cuBLAS path")
            label_kind = "coverage"
        else:
            key = f"roofline:{l['class']}:{l['cell'].replace('/', ':')}"
            title = (f"{l['class']} at {l['pct_roofline_med']:.0f}% roofline "
                     f"({l['bound_by']}-bound) on {l['cell']} — "
                     f"{l['time_share_pct']:.0f}% of window time")
            label_kind = "roofline"
        body = render_body(l, run, key)
        issues.append({"key": key, "title": title, "body": body,
                       "labels": cfg["issues"]["labels"] + [label_kind, prio.lower()],
                       "prio": prio})
    return issues


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
