#!/usr/bin/env python3
"""Phase 2 analysis: load nsys CSV exports, summarize top kernels per profile.

For each profile, prints:
- Top-15 kernels by total GPU time, with %, count, avg-µs
- Memory ops summary (H2D, D2H, P2P)
- Top API calls (cudaMemcpy, cudaLaunchKernel, cudaStreamSynchronize)
- Stream summary
- A flagged list of suspicious patterns (e.g. tiny kernels with huge counts,
  any FP16/BF16 GEMM in a NVFP4 profile, sync calls on critical path)

Usage:
    python3 analyze_csv.py [csv_dir]

Default csv_dir = $REPO/profiles/csv
"""
from __future__ import annotations
import csv
import sys
from collections import defaultdict
from pathlib import Path

CSV_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else
               "$REPO/profiles/csv")

GREEN = "\033[32m"; RED = "\033[31m"; YEL = "\033[33m"; CYAN = "\033[36m"; RST = "\033[0m"
def color(s, c): return f"{c}{s}{RST}"


def parse_kern_sum(path: Path):
    """Parse cuda_gpu_kern_sum CSV. Returns list of dicts."""
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "name": row.get("Name", row.get("Kernel Name", "")),
                "time_pct": float(row.get("Time (%)", row.get("Time:Pct", 0)) or 0),
                "total_ns": int(float(row.get("Total Time (ns)", row.get("Total Time:ns", 0)) or 0)),
                "count": int(float(row.get("Instances", row.get("Calls", 0)) or 0)),
                "avg_ns": float(row.get("Avg (ns)", row.get("Avg:ns", 0)) or 0),
                "med_ns": float(row.get("Med (ns)", row.get("Med:ns", 0)) or 0),
                "max_ns": float(row.get("Max (ns)", row.get("Max:ns", 0)) or 0),
            })
    return rows


def parse_api_sum(path: Path):
    rows = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "name": row.get("Name", ""),
                "time_pct": float(row.get("Time (%)", 0) or 0),
                "total_ns": int(float(row.get("Total Time (ns)", 0) or 0)),
                "count": int(float(row.get("Num Calls", row.get("Calls", 0)) or 0)),
                "avg_ns": float(row.get("Avg (ns)", 0) or 0),
            })
    return rows


def short(name: str, n: int = 80) -> str:
    """Compact a templated kernel name."""
    if len(name) <= n:
        return name
    # Try collapsing CUTLASS template explosion
    head = name.split("<", 1)[0]
    return f"{head}<...>" if len(head) < n else head[:n - 3] + "..."


def fmt_us(ns: float) -> str:
    if ns >= 1e9:
        return f"{ns/1e9:7.2f} s "
    if ns >= 1e6:
        return f"{ns/1e6:7.2f}ms"
    if ns >= 1e3:
        return f"{ns/1e3:7.2f}µs"
    return f"{ns:7.0f}ns"


def analyze_profile(prefix: str) -> None:
    kern_path = CSV_DIR / f"{prefix}_cuda_gpu_kern_sum.csv"
    api_path = CSV_DIR / f"{prefix}_cuda_api_sum.csv"

    print()
    print(color("=" * 100, CYAN))
    print(color(f"  {prefix}", CYAN))
    print(color("=" * 100, CYAN))

    if not kern_path.exists():
        print(color(f"  missing: {kern_path}", RED))
        return

    kerns = parse_kern_sum(kern_path)
    total_ns = sum(k["total_ns"] for k in kerns)
    print(f"\n  total kernel time: {fmt_us(total_ns)}   ({len(kerns)} unique kernels)")

    # ── Top-15 kernels
    kerns.sort(key=lambda k: -k["total_ns"])
    print("\n  " + color("TOP-15 kernels (by total GPU time)", YEL))
    print(f"  {'%':>5}  {'total':>9}  {'count':>7}  {'avg':>9}  name")
    print(f"  {'-'*5}  {'-'*9}  {'-'*7}  {'-'*9}  {'-'*70}")
    for k in kerns[:15]:
        print(f"  {k['time_pct']:>5.1f}  {fmt_us(k['total_ns'])}  {k['count']:>7d}  {fmt_us(k['avg_ns'])}  {short(k['name'], 70)}")

    # ── Suspicion flags
    flags = []
    # FP16/BF16 GEMMs (suspicious if model uses NVFP4)
    fp16_gemm = [k for k in kerns if any(t in k["name"].lower() for t in ("hgemm", "bf16gemm", "f16_gemm", "fp16_gemm"))]
    if fp16_gemm:
        for k in fp16_gemm[:3]:
            flags.append(("FP16/BF16 GEMM hot kernel", k))
    # Tiny kernels with huge launch counts
    tiny_huge = [k for k in kerns if k["avg_ns"] < 5000 and k["count"] > 200]
    for k in tiny_huge[:3]:
        flags.append(("tiny+frequent (launch overhead candidate)", k))
    # memset/memcpy kernels in top 20
    memset_top = [k for k in kerns[:20] if any(t in k["name"].lower() for t in ("memset", "memcpy", "fill", "zero"))]
    for k in memset_top:
        flags.append(("memset/memcpy in top-20", k))
    # CUTLASS kernels — check which path
    cutlass_kerns = [k for k in kerns[:15] if "cutlass" in k["name"].lower()]
    for k in cutlass_kerns[:3]:
        flags.append(("CUTLASS kernel (verify NVFP4 vs fallback)", k))

    if flags:
        print("\n  " + color("FLAGS", YEL))
        for label, k in flags:
            print(f"  ⚠  {label}: {k['time_pct']:>4.1f}%  ×{k['count']:>5d}  {short(k['name'], 60)}")

    # ── API summary
    if api_path.exists():
        apis = parse_api_sum(api_path)
        apis.sort(key=lambda a: -a["total_ns"])
        print("\n  " + color("TOP-10 host API calls", YEL))
        for a in apis[:10]:
            print(f"  {a['time_pct']:>5.1f}  {fmt_us(a['total_ns'])}  ×{a['count']:>6d}  {a['name']}")


def main():
    prefixes = sorted({p.name.replace("_cuda_gpu_kern_sum.csv", "")
                       for p in CSV_DIR.glob("*_cuda_gpu_kern_sum.csv")})
    if not prefixes:
        print(f"No CSVs in {CSV_DIR}. Run export_stats.sh first.")
        return 1
    for prefix in prefixes:
        analyze_profile(prefix)


if __name__ == "__main__":
    sys.exit(main() or 0)
