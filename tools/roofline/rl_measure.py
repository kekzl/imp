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
from rl_classify import build_classifier, aggregate_metrics

BENCH_RX = re.compile(r"^(pp|tg)\s+(\d+) tokens\s+avg\s+([\d.]+) ms\s+\(([\d.]+) tok/s\)", re.M)


def _log(msg):
    print(f"[roofline] {msg}", flush=True)
    sys.stdout.flush()


def _run_cmd(cmd, timeout=1800):
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    dt = time.time() - t0
    return p, dt


def bench_args(model_path, shape, no_graphs=True, cap_seq=False):
    args = ["--model", model_path, "--bench",
            "--bench-pp", str(shape["bench_pp"]),
            "--bench-reps", str(shape["bench_reps"]),
            "--max-tokens", str(shape["max_tokens"]),
            "--temperature", "0", "--seed", "42"]
    if cap_seq:
        # imp auto-sizes the KV cache to fill VRAM; under ncu that collides with
        # the profiler's own GPU buffers (StoragePlanner "vram budget
        # insufficient", driver "resource unavailable" on 12B+ models). Cap the
        # context to the bench needs + margin — kernel behavior in the measured
        # window is unchanged, only the KV pool shrinks.
        args += ["--max-seq-len", str(shape["bench_pp"] + shape["max_tokens"] + 512)]
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
    args = bench_args(model["path"], shape, cap_seq=True)

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

    # --- pass 2..N: ncu, calibrated skip, SINGLE-PASS metric groups for prefill
    # (multi-pass kernel replay dies on TMA kernels on this WSL2 driver — see
    # config.json ncu.replay_comment), one multi-pass invocation for decode.
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

    if shape["phase"] == "decode":
        groups = {"full": cfg["ncu"]["decode_metrics"]}
    else:
        groups = {k: v for k, v in cfg["ncu"]["prefill_groups"].items()
                  if not k.endswith("_comment")}
        if restart == 0:
            # TC ops + SASS FLOP counts are workload-deterministic — restart 0 only.
            for m in cfg["ncu"]["tc_groups_by_family"][model["family"]]:
                if "tensor_src_" in m:
                    gname = "tc_" + m.split("tensor_src_")[1].rsplit(".", 1)[0]
                else:
                    gname = "tc_pipe"
                groups[gname] = cfg["ncu"]["tc_group_base"] + [m]
        else:
            groups = {"base": groups["base"]}

    cell["ncu_groups"] = {}
    for gname, metrics in groups.items():
        gbase = f"{base}_{gname}"
        ncu_cmd = rl_ncu.docker_ncu_cmd(cfg, capture_rx, skip, launch_count,
                                        raw_dir, gbase, args, metrics)
        _log(f"{base}: ncu {gname} (skip={skip} count={launch_count})")
        rep_path = os.path.join(raw_dir, gbase + ".ncu-rep")
        # The WSL2 driver intermittently reports "a driver resource was
        # unavailable" when ncu sessions run back-to-back — transient, recovers
        # after a pause. Retry with backoff before declaring the group failed.
        for attempt in range(3):
            p, dt = _run_cmd(ncu_cmd, timeout=3600)
            ok = p.returncode == 0 and os.path.exists(rep_path)
            transient = "resource was unavailable" in (p.stdout + p.stderr) \
                or p.returncode == 9
            if ok or not transient:
                break
            _log(f"{gbase}: transient driver-resource error, retry {attempt + 1}/2 in 45s")
            time.sleep(45)
        if p.returncode != 0 or not os.path.exists(rep_path):
            cell.setdefault("group_errors", {})[gname] = \
                f"rc={p.returncode}: {(p.stderr or p.stdout)[-500:]}"
            _log(f"{gbase}: FAILED {cell['group_errors'][gname][:120]}")
            continue
        time.sleep(3)  # breathing room between profiler sessions
        csv_gz = os.path.join(raw_dir, gbase + ".ncu_raw.csv.gz")
        rl_ncu.export_csv(cfg["ncu"]["binary"], rep_path, csv_gz)
        launches = rl_ncu.parse_csv_gz(csv_gz, metrics)
        cell["ncu_groups"][gname] = {
            "csv": os.path.basename(csv_gz),
            "wall_s": round(dt, 1),
            "kernels": aggregate_by_kernel(classify, launches),
        }
    if not cell["ncu_groups"].get("base") and not cell["ncu_groups"].get("full"):
        cell["error"] = "ncu base group failed: " + \
            cell.get("group_errors", {}).get("base", "?")[:300]
    return cell


def aggregate_by_kernel(classify, launches):
    by_kernel = {}
    for l in launches:
        by_kernel.setdefault(l["kernel_name"], []).append(l)
    out = {}
    for name, ls in by_kernel.items():
        kcls, group = classify(name)
        rec = aggregate_metrics(ls)
        rec["class"] = kcls
        rec["group"] = group
        out[name] = rec
    return out


def check_gpu_free():
    """Abort when another process holds a CUDA context — a concurrent context
    breaks ncu sessions on big models ('counter availability image'
    ResourceUnavailable) and skews every benchmark number. Found the hard way:
    an idle imp-server container made all 12B+ cells fail."""
    p = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name",
                        "--format=csv,noheader"], capture_output=True, text=True)
    procs = [l for l in p.stdout.splitlines() if l.strip()]
    # WSL2 nvidia-smi often lists no compute apps even when containers hold a
    # CUDA context — additionally flag likely-GPU containers by image name.
    q = subprocess.run(["docker", "ps", "--format", "{{.Names}}\t{{.Image}}"],
                       capture_output=True, text=True)
    gpu_containers = [l for l in q.stdout.splitlines()
                      if l.strip() and any(s in l for s in
                                           ("imp", "vllm", "llama", "cuda"))]
    if procs or gpu_containers:
        raise SystemExit(
            "GPU not free — close these before measuring:\n  "
            + "\n  ".join(procs + gpu_containers))


def check_profilers(cfg):
    """Abort when a configured profiler binary is missing. The paths are pinned
    to an exact toolkit version, so a host-side Nsight upgrade silently breaks
    them: the binary is bind-mounted into the container, a missing source path
    makes docker fail the container with rc=127, and every ncu pass dies while
    the nsys passes keep succeeding. That failure only surfaces as a
    'cells with errors' line after the whole sweep (hours), and the run is
    written with no ncu data at all — found the hard way when ncu 2026.2.0 ->
    2026.2.1 left `roofline measure` producing nsys-only runs."""
    missing = [f"{tool}: {cfg[tool]['binary']}"
               for tool in ("ncu", "nsys")
               if not os.path.exists(cfg[tool]["binary"])]
    if missing:
        import glob
        hints = []
        for tool, root in (("ncu", "/opt/nvidia/nsight-compute"),
                           ("nsys", "/opt/nvidia/nsight-systems")):
            found = sorted(glob.glob(f"{root}/*"))
            if found:
                hints.append(f"  installed {tool} versions: {', '.join(os.path.basename(f) for f in found)}")
        raise SystemExit("profiler binary missing (fix tools/roofline/config.json):\n  "
                         + "\n  ".join(missing) + ("\n" + "\n".join(hints) if hints else ""))


def measure(cfg, restarts, model_keys, shape_keys, note="", dry_run=False):
    check_profilers(cfg)
    if not dry_run:
        check_gpu_free()
    classify = build_classifier(cfg)
    run_id = rl_config.new_run_id()
    import rl_history
    raw_dir = rl_history.raw_dir_for(run_id)
    meta = {
        "git": rl_config.git_meta(),
        "timestamp": rl_config.now_ts(),
        "config_version": cfg["config_version"],
        "ncu_config": {k: v for k, v in cfg["ncu"].items() if k != "binary"},
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
