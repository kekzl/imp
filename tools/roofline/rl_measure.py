"""Measurement orchestration: per (model, shape, restart) run nsys (coverage +
skip calibration) then ncu (pinned counters). Each docker run is a fresh
container => restart variance is captured by construction. stdlib only."""
import json
import os
import re
import subprocess
import sys
import time

import rl_config
import rl_ncu
import rl_nsys
from rl_classify import build_classifier, aggregate_launches

BENCH_RX = re.compile(r"^(pp|tg)\s+(\d+) tokens\s+avg\s+([\d.]+) ms\s+\(([\d.]+) tok/s\)", re.M)


def _log(msg):
    print(f"[roofline] {msg}", flush=True)
    sys.stdout.flush()


def _run_cmd(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    return p, dt


def bench_args(model_path, shape, no_graphs=True):
    args = ["--model", model_path, "--bench",
            "--bench-pp", str(shape["bench_pp"]),
            "--bench-reps", str(shape["bench_reps"]),
            "--max-tokens", str(shape["max_tokens"]),
            "--temperature", "0", "--seed", "42"]
    if no_graphs:
        args.append("--no-cuda-graphs")
    return args


def parse_bench_tps(text):
    out = {}
    for kind, n, _ms, tps in BENCH_RX.findall(text):
        out[f"{kind}{n}"] = float(tps)
    return out


def measure_cell(cfg, classify, raw_dir, model_key, shape_key, restart, dry_run=False):
    model = cfg["models"][model_key]
    shape = cfg["shapes"][shape_key]
    base = f"{model_key}_{shape_key}_r{restart}"
    cell = {"model": model_key, "shape": shape_key, "restart": restart}
    args = bench_args(model["path"], shape)

    # --- pass 1: nsys (fast, full timeline) ---
    nsys_cmd = rl_nsys.docker_nsys_cmd(cfg, raw_dir, base, args)
    if dry_run:
        _log("DRY nsys: " + " ".join(nsys_cmd))
        return cell
    _log(f"{base}: nsys pass")
    p, dt = _run_cmd(nsys_cmd)
    if p.returncode != 0:
        cell["error"] = f"nsys failed rc={p.returncode}: {p.stderr[-800:]}"
        _log(cell["error"])
        return cell
    cell["bench_tps_under_nsys"] = parse_bench_tps(p.stdout + p.stderr)
    cell["nsys_wall_s"] = round(dt, 1)

    rep = os.path.join(raw_dir, base + ".nsys-rep")
    sqlite_path = rl_nsys.export_sqlite(cfg["nsys"]["binary"], rep)
    extract = rl_nsys.extract(sqlite_path, classify)
    with open(os.path.join(raw_dir, base + ".nsys_extract.json"), "w") as f:
        json.dump(extract, f, indent=1, sort_keys=True)
    cell["nsys_extract_file"] = base + ".nsys_extract.json"

    # --- pass 2: ncu with calibrated skip ---
    capture_rx = cfg["capture_regex"][shape["phase"]]
    n_init = rl_nsys.matched_init_launches(extract, capture_rx)
    n_steady = rl_nsys.matched_launches_after_init(extract, capture_rx)
    if n_steady == 0:
        cell["error"] = "no kernels matched capture regex post-init"
        return cell
    launch_count = min(cfg["ncu"]["launch_count"], max(n_steady // 2, 1))
    skip = n_init + int(n_steady * cfg["ncu"]["skip_fraction"])
    if skip + launch_count > n_init + n_steady:
        skip = max(n_init + n_steady - launch_count, 0)
    cell["ncu_launch_skip"] = skip
    cell["ncu_launch_count"] = launch_count

    ncu_cmd = rl_ncu.docker_ncu_cmd(cfg, capture_rx, skip, launch_count, raw_dir, base, args)
    _log(f"{base}: ncu pass (skip={skip} count={launch_count}, init={n_init}, steady={n_steady})")
    p, dt = _run_cmd(ncu_cmd, timeout=3600)
    rep_path = os.path.join(raw_dir, base + ".ncu-rep")
    if p.returncode != 0 or not os.path.exists(rep_path):
        cell["error"] = f"ncu failed rc={p.returncode}: {(p.stderr or p.stdout)[-800:]}"
        _log(cell["error"])
        return cell
    cell["ncu_wall_s"] = round(dt, 1)

    csv_gz = os.path.join(raw_dir, base + ".ncu_raw.csv.gz")
    rl_ncu.export_csv(cfg["ncu"]["binary"], rep_path, csv_gz)
    cell["ncu_csv_file"] = os.path.basename(csv_gz)

    launches = rl_ncu.parse_csv_gz(csv_gz, cfg["ncu"]["metrics"])
    cell["ncu_kernels"] = aggregate_by_kernel(cfg, classify, launches)
    return cell


def aggregate_by_kernel(cfg, classify, launches):
    by_kernel = {}
    for l in launches:
        by_kernel.setdefault(l["kernel_name"], []).append(l)
    out = {}
    for name, ls in by_kernel.items():
        kcls, group = classify(name)
        rec = aggregate_launches(ls, cfg["gpu"])
        rec["class"] = kcls
        rec["group"] = group
        out[name] = rec
    return out


def measure(cfg, restarts, model_keys, shape_keys, note="", dry_run=False):
    classify = build_classifier(cfg)
    run_id = rl_config.new_run_id()
    import rl_history
    raw_dir = rl_history.raw_dir_for(run_id)
    meta = {
        "git": rl_config.git_meta(),
        "timestamp": rl_config.now_ts(),
        "config_version": cfg["config_version"],
        "ncu_metrics": cfg["ncu"]["metrics"],
        "note": note,
        "restarts": restarts,
    }
    if not dry_run:
        meta["env"] = rl_config.env_meta(cfg)
    run = {"run_id": run_id, "meta": meta, "cells": {}}
    _log(f"run {run_id}: {len(model_keys)} models x {len(shape_keys)} shapes x {restarts} restarts")

    for restart in range(restarts):
        for mk in model_keys:
            for sk in shape_keys:
                allowed = cfg["models"][mk].get("shapes")
                if allowed and sk not in allowed:
                    continue
                key = f"{mk}|{sk}"
                cell = measure_cell(cfg, classify, raw_dir, mk, sk, restart, dry_run)
                run["cells"].setdefault(key, []).append(cell)
                # checkpoint after every cell so a crash loses nothing
                if not dry_run:
                    with open(os.path.join(raw_dir, "run_partial.json"), "w") as f:
                        json.dump(run, f, indent=1, sort_keys=True)
    if dry_run:
        return None
    import rl_history
    path = rl_history.write_run(run)
    _log(f"run written: {path}")
    return run


def ab_test(cfg, model_keys, shape_keys, knob, env_overrides, restarts=3):
    """Plain (unprofiled) bench A/B: baseline env vs override env, paired per
    restart so cuBLAS restart variance hits both arms equally."""
    results = []
    for restart in range(restarts):
        for mk in model_keys:
            for sk in shape_keys:
                model, shape = cfg["models"][mk], cfg["shapes"][sk]
                args = bench_args(model["path"], shape, no_graphs=False)
                pair = {"model": mk, "shape": sk, "restart": restart, "knob": knob}
                for arm, env in (("baseline", None), ("variant", env_overrides)):
                    cmd = _plain_bench_cmd(cfg, args, env)
                    _log(f"ab {knob} {mk}|{sk} r{restart} {arm}")
                    p, _ = _run_cmd(cmd)
                    pair[arm] = parse_bench_tps(p.stdout + p.stderr)
                    if p.returncode != 0:
                        pair[arm + "_error"] = (p.stderr or "")[-400:]
                results.append(pair)
    return results


def _plain_bench_cmd(cfg, imp_cli_args, extra_env=None):
    d = cfg["docker"]
    env_flags = []
    for k, v in {**d.get("env", {}), **(extra_env or {})}.items():
        env_flags += ["-e", f"{k}={v}"]
    return ["docker", "run", "--rm", "--gpus", "all",
            "-u", f"{os.getuid()}:{os.getgid()}",
            "-v", f"{d['models_mount']}:/models",
            *env_flags,
            "--entrypoint", d["imp_cli"], d["image"], *imp_cli_args]
