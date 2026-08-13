#!/usr/bin/env python3
"""Regenerate the <!-- PERF:BEGIN --> blocks from the pinned baseline.

The dispatch specifies `benchmarks/results/*.json` as the source. That directory
does not exist in this tree; the pinned, CI-defended figures live in
`tests/perf_baseline.json`, which is also what `make verify-fast` compares
against. Generating from anywhere else would let the README and the gate drift
apart, which is the exact failure the generated block is meant to prevent.

Usage:  python3 scripts/sync_docs.py [--check]
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
BASELINE = ROOT / "tests" / "perf_baseline.json"
TARGETS = ["README.md", "docs/PERF.md"]

BEGIN = "<!-- PERF:BEGIN -->"
END = "<!-- PERF:END -->"


def render(data: dict) -> str:
    pre = data["metrics"]["prefill_tps"]
    dec = data["metrics"]["decode_tps"]
    mem = data["metrics"]["memory_mb"]
    th = data["thresholds"]
    model = data["model"].replace(".gguf", "")
    date = data["timestamp"][:10]

    rows = [
        ("decode tg128", f"**{dec['tg128']} tok/s**", f"{th['decode_regression_pct']} %"),
        ("prefill pp128", f"{pre['pp128']} tok/s", f"{th['prefill_regression_pct']} %"),
        ("prefill pp512", f"**{pre['pp512']} tok/s**", f"{th['prefill_regression_pct']} %"),
        ("prefill pp4096", f"{pre['pp4096']} tok/s", f"{th['prefill_regression_pct']} %"),
        ("peak VRAM (own)", f"{mem['own_peak_mb']} MiB", f"{th['vram_increase_pct']} %"),
    ]

    out = ["| metric | value | threshold |", "|---|---|---|"]
    out += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    out.append("")
    out.append(
        f"[PROV: commit=1e4fad60 date={date} hw=RTX5090 model={model} quant=Q8_0\n"
        f"       cuda=13.3 path=gguf-dp4a cmd=`make verify-fast` "
        f"n={data['n_trials']}x{data['reps']}]"
    )
    return "\n".join(out)


def main() -> int:
    check = "--check" in sys.argv
    data = json.loads(BASELINE.read_text(encoding="utf-8"))
    block = render(data)
    drift = False

    for rel in TARGETS:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        try:
            head, rest = text.split(BEGIN, 1)
            _, tail = rest.split(END, 1)
        except ValueError:
            print(f"FAIL {rel}: no {BEGIN}/{END} markers")
            return 1
        new = f"{head}{BEGIN}\n{block}\n{END}{tail}"
        if new == text:
            continue
        drift = True
        if check:
            print(f"DRIFT {rel}: generated block is out of date, run scripts/sync_docs.py")
        else:
            path.write_text(new, encoding="utf-8")
            print(f"updated {rel}")

    if check and drift:
        return 1
    if not drift:
        print("sync_docs: up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
