"""ncu invocation + raw-CSV ingest. The parser reads ncu's own --csv --page raw
export — no hand-entered numbers. stdlib only."""
import csv
import gzip
import io
import os
import re
import subprocess


def export_csv(ncu_bin, ncu_rep_path, csv_gz_path):
    """ncu-rep -> raw-page CSV (gzipped). Runs on the host ncu (no GPU needed)."""
    out = subprocess.run(
        [ncu_bin, "--import", ncu_rep_path, "--csv", "--page", "raw"],
        capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"ncu import failed for {ncu_rep_path}: {out.stderr[-500:]}")
    with gzip.open(csv_gz_path, "wt") as f:
        f.write(out.stdout)
    return csv_gz_path


def _to_float(s):
    if s is None or s == "":
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


# ncu raw-page units -> SI factor (seconds, bytes, Hz; counts/percent stay).
UNIT_SCALE = {
    "": 1.0, "inst": 1.0, "%": 1.0, "cycle": 1.0, "sector": 1.0, "warp": 1.0,
    "byte": 1.0, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "Tbyte": 1e12,
    "ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6,
    "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0,
    "hz": 1.0, "Hz": 1.0, "Khz": 1e3, "Mhz": 1e6, "Ghz": 1e9,
}


def parse_csv_gz(csv_gz_path, metric_names):
    """Yield {kernel_name, block, grid, <metric>: float-in-SI} per launch.
    Row 1 = column names, row 2 = per-column units (normalized via UNIT_SCALE)."""
    with gzip.open(csv_gz_path, "rt") as f:
        text = f.read()
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 3:
        return []
    header, units = rows[0], rows[1]
    idx = {name: header.index(name) for name in header}
    scale = {}
    for m in metric_names:
        if m in idx:
            u = units[idx[m]]
            if u not in UNIT_SCALE:
                raise RuntimeError(f"unknown ncu unit '{u}' for {m} — extend UNIT_SCALE")
            scale[m] = UNIT_SCALE[u]
    launches = []
    for row in rows[2:]:
        if len(row) != len(header):
            continue
        rec = {"kernel_name": row[idx.get("Kernel Name", 0)]}
        for opt in ("Block Size", "Grid Size"):
            if opt in idx:
                rec[opt.lower().replace(" ", "_")] = row[idx[opt]]
        for m, sc in scale.items():
            v = _to_float(row[idx[m]])
            rec[m] = v * sc if v is not None else None
        launches.append(rec)
    return launches


def container_name(out_base_name):
    """Deterministic container name per ncu group, so a timeout can clean up."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", out_base_name)
    return f"roofline_ncu_{safe}"


def docker_ncu_cmd(cfg, kernel_regex, launch_skip, launch_count, out_host_dir,
                   out_base_name, imp_cli_args, metrics, extra_env=None):
    d = cfg["docker"]
    n = cfg["ncu"]
    env_flags = []
    for k, v in {**d.get("env", {}), **(extra_env or {})}.items():
        env_flags += ["-e", f"{k}={v}"]
    return [
        "docker", "run", "--rm", "--gpus", "all", "--privileged",
        # Named so a wedged pass can be force-removed on timeout (see
        # rl_measure._run_cmd) — an orphaned container keeps the GPU.
        "--name", container_name(out_base_name),
        "-u", f"{os.getuid()}:{os.getgid()}", "-w", "/tmp",
        "-v", f"{d['models_mount']}:/models",
        "-v", f"{out_host_dir}:/out",
        "-v", "/opt/nvidia:/opt/nvidia:ro",
        *env_flags,
        "--entrypoint", n["binary"],
        d["image"],
        "--target-processes", "all",
        "--kernel-name", f"regex:{kernel_regex}",
        "--launch-skip", str(launch_skip),
        "--launch-count", str(launch_count),
        "--metrics", ",".join(metrics),
        "--replay-mode", "kernel",
        "--clock-control", n.get("clock_control", "base"),
        "--force-overwrite",
        "-o", f"/out/{out_base_name}",
        d["imp_cli"], *imp_cli_args,
    ]
