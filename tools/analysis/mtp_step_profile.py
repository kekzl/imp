#!/usr/bin/env python3
"""mtp_step_profile.py - per-step kernel time of the decode phase of each client window (imp-server, nsys).

Splits an nsys SQLite export (imp-server under `--cuda-graph-trace=node`) into windows at device idle gaps
> --gap-s (the long-context client tokenizes between contexts), drops each window's prefill (everything up to
the last --prefill-pat kernel), then prints per window: decode wall, steps (launches of --anchor divided by
--per-step), wall / busy union / kernel sum per step, and the top kernels in ms per step.

Usage: mtp_step_profile.py profile.sqlite [--gap-s 1.5] [--anchor REGEX] [--per-step 1] [--top 25]
--anchor default: the bound head scan (one launch per draft pass: 1 per step at mtp_batch_verify_k=1, 2 at 2).
"""
import argparse
import re
import sqlite3

CLASSES = [
    ("paged_attn", r"paged_attention|attn_decode|splitk_combine|paged_.*combine"),
    ("mtp_scan", r"mtp_attn_kv_scan|mtp_attn_kv_combine"),
    ("mtp_other", r"^mtp_|fp8_head|head_cpasync|mtp_topw"),
    ("gdn", r"gdn_|ssm_conv"),
    ("gemm_gemv", r"gemm|gemv|smallm|cublas|cutlass|sm\d+_xmma"),
    ("norm_quant_elt", r"rmsnorm|quantize|elementwise|gated_|silu|rope|add_|sigmoid"),
    ("kv_write", r"write_kv|kv_append|kv_chunk"),
    ("argmax_sample", r"argmax|topk|sample|penalt"),
]


def cls(name):
    for c, rx in CLASSES:
        if re.search(rx, name):
            return c
    return "other"


def union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--gap-s", type=float, default=1.5)
    ap.add_argument("--min-window-ms", type=float, default=500.0)
    ap.add_argument("--prefill-pat", default="fmha_sm120")
    ap.add_argument("--anchor", default=r"^mtp_attn_kv_scan_rows(_split)?_kernel$")
    ap.add_argument("--per-step", type=float, default=1.0)
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    rows = sqlite3.connect(a.sqlite).execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = k.shortName ORDER BY k.start").fetchall()
    if not rows:
        print("no kernels")
        return
    wins, cur, last_end = [], [rows[0]], rows[0][1]
    for r in rows[1:]:
        if r[0] - last_end > a.gap_s * 1e9:
            wins.append(cur)
            cur = []
        cur.append(r)
        last_end = max(last_end, r[1])
    wins.append(cur)
    t0 = rows[0][0]
    for i, w in enumerate(wins):
        if (w[-1][1] - w[0][0]) / 1e6 < a.min_window_ms:
            continue
        pf = [k for k, r in enumerate(w) if a.prefill_pat in r[2]]
        dec = w[pf[-1] + 1:] if pf else w
        if not dec:
            continue
        lo, hi = dec[0][0], max(r[1] for r in dec)
        wall = hi - lo
        ksum = sum(e - s for s, e, _ in dec)
        busy = union([(s, e) for s, e, _ in dec])
        cnt, dur = {}, {}
        for s, e, n in dec:
            cnt[n] = cnt.get(n, 0) + 1
            dur[n] = dur.get(n, 0) + (e - s)
        n_anchor = sum(c for n, c in cnt.items() if re.search(a.anchor, n))
        steps = n_anchor / a.per_step
        print(f"window {i}: t={(w[0][0] - t0) / 1e9:.1f}s prefill kernels {len(pf)}, decode wall "
              f"{wall / 1e6:.1f} ms, {len(dec)} kernels, anchor {a.anchor} x{n_anchor} / {a.per_step} "
              f"= {steps:.0f} steps")
        if steps <= 0:
            continue
        ps = 1e6 * steps
        print(f"  per step: wall {wall / ps:.3f} ms, busy union {busy / ps:.3f} ms, idle {(wall - busy) / ps:.3f} "
              f"ms, kernel sum {ksum / ps:.3f} ms")
        cd = {}
        for n, d in dur.items():
            c = cls(n)
            cd[c] = cd.get(c, 0) + d
        for c, d in sorted(cd.items(), key=lambda kv: -kv[1]):
            print(f"  class {c:16s} {d / ps:7.3f} ms/step {100.0 * d / max(ksum, 1):6.2f} %")
        for n, d in sorted(dur.items(), key=lambda kv: -kv[1])[:a.top]:
            print(f"    {d / ps:7.3f} ms/step {cnt[n] / steps:7.2f} n/step {d / cnt[n] / 1e3:9.2f} us/call  {n[:90]}")


if __name__ == "__main__":
    main()
