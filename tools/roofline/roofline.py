#!/usr/bin/env python3
"""roofline — reproducible roofline + kernel-coverage pipeline for imp.

Subcommands:
  measure   ncu+nsys sweep over the shape set (>=3 restarts), new history entry
  ab        unprofiled A/B bench for a fallback knob (e.g. fa2)
  plot      render roofline.png / trend / compare from history (no re-measure)
  report    markdown report (Modul 1 + Modul 2 + levers) from history
  regress   exit!=0 if a kernel class fell > threshold below a baseline run
  issues    generate GitHub issues from findings (dry-run unless --create)
  list      list history runs
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rl_config
import rl_history

PLOT_IMAGE = "imp-roofline-plot"


def cmd_measure(args):
    cfg = rl_config.load_config(args.config)
    import rl_measure
    model_keys = args.models.split(",") if args.models else list(cfg["models"])
    shape_keys = args.shapes.split(",") if args.shapes else list(cfg["shapes"])
    for mk in model_keys:
        if mk not in cfg["models"]:
            raise SystemExit(f"unknown model key {mk}")
    for sk in shape_keys:
        if sk not in cfg["shapes"]:
            raise SystemExit(f"unknown shape key {sk}")
    run = rl_measure.measure(cfg, args.restarts, model_keys, shape_keys,
                             note=args.note, dry_run=args.dry_run)
    if run:
        bad = [k for k, cells in run["cells"].items() if any("error" in c for c in cells)]
        if bad:
            print(f"WARNING: cells with errors: {bad}", file=sys.stderr)
        print(run["run_id"])
    return 0


def cmd_ab(args):
    cfg = rl_config.load_config(args.config)
    import rl_measure
    knobs = {
        "fa2": {"IMP_FMHA_FA2": "0"},
    }
    if args.knob not in knobs:
        raise SystemExit(f"unknown knob {args.knob}; have {list(knobs)}")
    model_keys = args.models.split(",") if args.models else list(cfg["models"])
    shape_keys = args.shapes.split(",") if args.shapes else ["pp512", "pp2048", "pp4096"]
    res = rl_measure.ab_test(cfg, model_keys, shape_keys, args.knob,
                             knobs[args.knob], restarts=args.restarts)
    out = args.out or os.path.join(rl_config.HISTORY_DIR,
                                   f"ab_{args.knob}_{rl_config.new_run_id()}.json")
    with open(out, "w") as f:
        json.dump({"knob": args.knob, "git": rl_config.git_meta(),
                   "timestamp": rl_config.now_ts(), "results": res}, f, indent=1)
    print(out)
    return 0


def _ensure_plot_image():
    have = subprocess.run(["docker", "image", "inspect", PLOT_IMAGE],
                          capture_output=True).returncode == 0
    if not have:
        subprocess.run(["docker", "build", "-t", PLOT_IMAGE, "-f",
                        os.path.join(rl_config.TOOL_DIR, "Dockerfile.plot"),
                        rl_config.TOOL_DIR], check=True)


def cmd_plot(args):
    _ensure_plot_image()
    repo = rl_config.REPO_ROOT
    out_dir = args.out_dir or os.path.join(rl_config.TOOL_DIR, "plots")
    os.makedirs(out_dir, exist_ok=True)
    rel_tool = os.path.relpath(rl_config.TOOL_DIR, repo)

    def run_plot(mode, out_name, run_ref="latest", run_b=None):
        cmd = ["docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
               "-e", "MPLCONFIGDIR=/tmp",
               "-v", f"{repo}:/repo", "-w", f"/repo/{rel_tool}",
               PLOT_IMAGE, "python3", "rl_plot.py", "--mode", mode,
               "--run", run_ref, "--out",
               os.path.join("/repo", os.path.relpath(out_dir, repo), out_name)]
        if run_b:
            cmd += ["--run-b", run_b]
        subprocess.run(cmd, check=True)

    if args.compare:
        run_plot("compare", "roofline_compare.png", args.compare[0], args.compare[1])
    else:
        ref = args.commit or args.run or "latest"
        run_plot("roofline", "roofline.png", ref)
        run_plot("trend", "roofline_trend.png", ref)
    print(f"plots in {out_dir}")
    return 0


def cmd_report(args):
    cfg = rl_config.load_config(args.config)
    import rl_report
    run = rl_history.load_run(args.run)
    ab = None
    if args.ab_file and os.path.exists(args.ab_file):
        with open(args.ab_file) as f:
            ab = json.load(f)
    md = rl_report.render(run, cfg, ab_results=ab)
    if args.out:
        with open(args.out, "w") as f:
            f.write(md)
        print(args.out)
    else:
        print(md)
    return 0


def cmd_regress(args):
    cfg = rl_config.load_config(args.config)
    import rl_regress
    base = rl_history.load_run(args.baseline)
    cur = rl_history.load_run(args.run)
    thr = args.threshold if args.threshold is not None else cfg["regress"]["default_threshold_pct"]
    failures, infos = rl_regress.compare(base, cur, cfg, thr)
    for l in infos:
        print("  " + l)
    if failures:
        print(f"\nREGRESSIONS (> {thr}% below baseline {base['run_id']}):")
        for l in failures:
            print("  FAIL " + l)
        return 1
    print(f"\nOK: no kernel class > {thr}% below baseline {base['run_id']}")
    return 0


def cmd_issues(args):
    cfg = rl_config.load_config(args.config)
    import rl_issues
    run = rl_history.load_run(args.run)
    issues = rl_issues.build_issues(run, cfg)
    if not issues:
        print("no findings above issue threshold")
        return 0
    for st, title in rl_issues.create_or_update(issues, dry_run=not args.create):
        print(f"[{st}] {title}")
    if not args.create:
        print(f"\n({len(issues)} issues planned — re-run with --create to create/update them)")
    return 0


def cmd_list(args):
    for r in rl_history.list_runs():
        print(f"{r['run_id']}  cells={len(r['cells'])}  note={r.get('note','')}")
    return 0


def main():
    ap = argparse.ArgumentParser(prog="roofline", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure")
    m.add_argument("--restarts", type=int, default=3)
    m.add_argument("--models", help="comma-separated model keys (default: all)")
    m.add_argument("--shapes", help="comma-separated shape keys (default: all)")
    m.add_argument("--note", default="")
    m.add_argument("--dry-run", action="store_true")
    m.set_defaults(fn=cmd_measure)

    a = sub.add_parser("ab")
    a.add_argument("--knob", required=True)
    a.add_argument("--models")
    a.add_argument("--shapes")
    a.add_argument("--restarts", type=int, default=3)
    a.add_argument("--out")
    a.set_defaults(fn=cmd_ab)

    p = sub.add_parser("plot")
    p.add_argument("--run")
    p.add_argument("--commit")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
    p.add_argument("--out-dir")
    p.set_defaults(fn=cmd_plot)

    r = sub.add_parser("report")
    r.add_argument("--run", default="latest")
    r.add_argument("--ab-file")
    r.add_argument("-o", "--out")
    r.set_defaults(fn=cmd_report)

    g = sub.add_parser("regress")
    g.add_argument("--baseline", required=True)
    g.add_argument("--run", default="latest")
    g.add_argument("--threshold", type=float, help="percent, e.g. 5")
    g.set_defaults(fn=cmd_regress)

    i = sub.add_parser("issues")
    i.add_argument("--run", default="latest")
    i.add_argument("--create", action="store_true")
    i.set_defaults(fn=cmd_issues)

    l = sub.add_parser("list")
    l.set_defaults(fn=cmd_list)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
