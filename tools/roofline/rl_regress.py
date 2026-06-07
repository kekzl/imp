"""Regression gate: compare a run against a pinned baseline run. Exit != 0 if a
hot-path kernel class drops more than threshold below baseline. stdlib only."""
import rl_table


def compare(baseline_run, current_run, cfg, threshold_pct):
    gpu = cfg["gpu"]
    min_share = cfg["regress"]["min_time_share_pct"]
    base = {(r["model"], r["shape"], r["class"]): r
            for r in rl_table.time_share_within_cell(rl_table.build_rows(baseline_run, cfg))}
    cur = {(r["model"], r["shape"], r["class"]): r
           for r in rl_table.time_share_within_cell(rl_table.build_rows(current_run, cfg))}
    failures, infos = [], []
    for key in sorted(set(base) & set(cur)):
        b, c = base[key], cur[key]
        if b["time_share_pct"] < min_share:
            continue
        if b["pct_roofline_med"] <= 0:
            continue
        delta_rel = 100.0 * (c["pct_roofline_med"] - b["pct_roofline_med"]) / b["pct_roofline_med"]
        line = (f"{key[2]} @ {key[0]}/{key[1]}: %roofline med "
                f"{b['pct_roofline_med']:.1f} -> {c['pct_roofline_med']:.1f} ({delta_rel:+.1f}%)")
        if delta_rel < -threshold_pct:
            # restart-variance guard: only fail if the current MAX is also below
            # the baseline MIN (i.e. no overlap of restart ranges)
            if c["pct_roofline_max"] < b["pct_roofline_min"]:
                failures.append(line + "  [ranges disjoint -> REGRESSION]")
            else:
                infos.append(line + "  [within restart variance, not gated]")
        else:
            infos.append(line)
    return failures, infos
