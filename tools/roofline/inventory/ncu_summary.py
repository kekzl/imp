"""Summarises an ncu_cell.sh CSV against the measured peaks (tools/roofline/peaks/PEAKS.md).
Usage: python ncu_summary.py <csv> [peaks.json]
Per kernel (median over launches): time, DRAM/L2 GB/s and % of measured peak, issue-slot use,
achieved occupancy, top-5 stall reasons (cycles per issued instruction).
"""

import csv
import json
import statistics
import sys

DRAM_PEAK = 1695.0  # GB/s, peaks_20260923T044423Z/044701Z median of read runs
L2_PEAK = 7100.0


def _num(s):
    try:
        return float(s.replace(",", ""))
    except (ValueError, AttributeError):
        return None


def main():
    lines = [ln for ln in open(sys.argv[1], errors="replace") if ln.startswith('"')]
    rows = list(csv.reader(lines))
    if len(rows) < 3:
        print("no kernel rows")
        return
    head, units, data = rows[0], rows[1], rows[2:]
    col = {h: i for i, h in enumerate(head)}
    by_kernel = {}
    for r in data:
        by_kernel.setdefault(r[col["Kernel Name"]], []).append(r)

    def med(rs, name):
        if name not in col:
            return None
        vals = [_num(r[col[name]]) for r in rs]
        vals = [v for v in vals if v is not None]
        return statistics.median(vals) if vals else None

    for kname, rs in by_kernel.items():
        t_unit = units[col["gpu__time_duration.sum"]]
        t = med(rs, "gpu__time_duration.sum")
        t_s = t * {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(t_unit, 1e-9)
        dram = (med(rs, "dram__bytes_read.sum") or 0) + (med(rs, "dram__bytes_write.sum") or 0)
        l2 = med(rs, "lts__t_bytes.sum") or 0
        clk = med(rs, "sm__cycles_elapsed.avg.per_second")
        print(f"== {kname[:120]}  (n={len(rs)})")
        print(f"   time {t_s * 1e6:.1f} us  clock {clk / 1e6 if clk else 0:.0f} MHz")
        print(f"   DRAM {dram / t_s / 1e9:.0f} GB/s = {100 * dram / t_s / 1e9 / DRAM_PEAK:.1f} % of {DRAM_PEAK:.0f}"
              f"   L2 {l2 / t_s / 1e9:.0f} GB/s = {100 * l2 / t_s / 1e9 / L2_PEAK:.1f} % of {L2_PEAK:.0f}")
        for label, m in (
            ("issue slots busy", "sm__inst_executed.avg.pct_of_peak_sustained_elapsed"),
            ("SM throughput", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
            ("mem throughput", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
            ("achieved occupancy", "sm__warps_active.avg.pct_of_peak_sustained_active"),
            ("theoretical occupancy", "sm__maximum_warps_per_active_cycle_pct"),
            ("regs/thread", "launch__registers_per_thread"),
            ("grid", "launch__grid_size"),
            ("block", "launch__block_size"),
            ("waves/SM", "launch__waves_per_multiprocessor"),
            ("local ld bytes (spill)", "l1tex__t_bytes_pipe_lsu_mem_local_op_ld.sum"),
            ("smem bank conflicts", "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"),
            ("tensor pipe insts", "smsp__inst_executed_pipe_tensor.sum"),
        ):
            v = med(rs, m)
            if v is not None:
                print(f"   {label:24s} {v:g}")
        stalls = []
        for h in head:
            if h.startswith("smsp__average_warp_latency_issue_stalled_") and h.endswith(".ratio"):
                v = med(rs, h)
                if v:
                    stalls.append((v, h[len("smsp__average_warp_latency_issue_stalled_"):-len(".ratio")]))
            elif h.startswith("smsp__average_warps_issue_stalled_") and h.endswith("_per_issue_active.ratio"):
                v = med(rs, h)
                if v:
                    stalls.append((v, h[len("smsp__average_warps_issue_stalled_"):-len("_per_issue_active.ratio")]))
        stalls.sort(reverse=True)
        if stalls:
            print("   stalls (cycles/issue): " + ", ".join(f"{n} {v:.2f}" for v, n in stalls[:6]))


if __name__ == "__main__":
    main()
