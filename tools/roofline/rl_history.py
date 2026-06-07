"""Append-only run history: history/runs/<run_id>.json + index.jsonl. stdlib only."""
import json
import os

from rl_config import RUNS_DIR, RAW_DIR, INDEX_PATH


def write_run(run):
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, run["run_id"] + ".json")
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing run {path} (append-only history)")
    with open(path, "w") as f:
        json.dump(run, f, indent=1, sort_keys=True)
    summary = {
        "run_id": run["run_id"],
        "commit": run["meta"]["git"]["commit"],
        "dirty": run["meta"]["git"]["dirty"],
        "timestamp": run["meta"]["timestamp"],
        "config_version": run["meta"]["config_version"],
        "cells": sorted(run["cells"].keys()),
        "note": run["meta"].get("note", ""),
    }
    with open(INDEX_PATH, "a") as f:
        f.write(json.dumps(summary, sort_keys=True) + "\n")
    return path


def list_runs():
    if not os.path.exists(INDEX_PATH):
        return []
    with open(INDEX_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def resolve_run_id(ref):
    """ref: 'latest', exact run_id, or commit-SHA prefix (newest match wins)."""
    runs = list_runs()
    if not runs:
        raise SystemExit("no runs in history")
    if ref in (None, "latest"):
        return runs[-1]["run_id"]
    matches = [r for r in runs if r["run_id"] == ref] or \
              [r for r in runs if r["run_id"].startswith(ref) or r["commit"].startswith(ref)]
    if not matches:
        raise SystemExit(f"no run matching '{ref}' in history")
    return matches[-1]["run_id"]


def load_run(ref):
    run_id = resolve_run_id(ref)
    with open(os.path.join(RUNS_DIR, run_id + ".json")) as f:
        return json.load(f)


def raw_dir_for(run_id):
    d = os.path.join(RAW_DIR, run_id)
    os.makedirs(d, exist_ok=True)
    return d
