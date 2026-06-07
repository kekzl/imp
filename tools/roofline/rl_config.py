"""Config + run-metadata helpers for the roofline pipeline. stdlib only."""
import json
import os
import subprocess
import datetime

TOOL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TOOL_DIR, "..", ".."))
HISTORY_DIR = os.path.join(TOOL_DIR, "history")
RUNS_DIR = os.path.join(HISTORY_DIR, "runs")
RAW_DIR = os.path.join(HISTORY_DIR, "raw")
INDEX_PATH = os.path.join(HISTORY_DIR, "index.jsonl")


def load_config(path=None):
    with open(path or os.path.join(TOOL_DIR, "config.json")) as f:
        return json.load(f)


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw).stdout.strip()


def git_meta():
    sha = _run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"])
    dirty = bool(_run(["git", "-C", REPO_ROOT, "status", "--porcelain",
                       "--", "src", "include", "tools", "CMakeLists.txt"]))
    return {"commit": sha, "dirty": dirty}


def env_meta(cfg):
    drv = _run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"])
    driver, _, gpu_name = drv.partition(", ")
    cuda = _run(["bash", "-c",
                 "nvidia-smi -q | grep -m1 'CUDA Version' | awk -F': ' '{print $2}'"])
    ncu_ver = _run([cfg["ncu"]["binary"], "--version"]).splitlines()[-1] if os.path.exists(cfg["ncu"]["binary"]) else "missing"
    return {"driver": driver, "gpu": gpu_name, "cuda": cuda, "ncu": ncu_ver}


def now_ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def new_run_id():
    g = git_meta()
    return "{}{}_{}".format(g["commit"], "-dirty" if g["dirty"] else "", now_ts())
