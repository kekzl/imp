"""Roofline plots (matplotlib). Runs inside the plot container (see
Dockerfile.plot) — invoked by `roofline plot`, renders purely from history."""
import argparse
import math
import sys

import rl_config
import rl_history
import rl_table

GROUP_COLOR = {
    "attention": "tab:blue", "attention_legacy": "tab:red", "gemm": "tab:green",
    "gemv": "tab:purple", "dequant": "tab:orange", "norm": "tab:brown",
    "rope": "tab:pink", "sampling": "tab:gray", "misc": "tab:olive",
    "unclassified": "black",
}


def _import_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _roofs(gpu):
    peak_bw = gpu["dram_peak_gbs"]  # GB/s
    roofs = []
    for dtype, label in (("tc_fp4", "FP4 TC"), ("tc_fp8", "FP8 TC"), ("tc_fp16", "FP16 TC"),
                         ("cuda_fp32", "FP32 CUDA")):
        peak = gpu["flop_per_cycle"][dtype] * gpu["nominal_boost_ghz"]  # GFLOPS
        roofs.append((dtype, label, peak, peak / peak_bw))
    return peak_bw, roofs


def plot_roofline(run, out_path, cfg, title_suffix=""):
    plt = _import_mpl()
    gpu = cfg["gpu"]
    rows = rl_table.time_share_within_cell(rl_table.build_rows(run, cfg))
    peak_bw, roofs = _roofs(gpu)

    fig, ax = plt.subplots(figsize=(13, 9))
    ai_axis = [2 ** e for e in range(-4, 13)]
    for dtype, label, peak, ridge in roofs:
        ys = [min(ai * peak_bw, peak) for ai in ai_axis]
        ax.plot(ai_axis, ys, lw=1, alpha=0.6, color="gray")
        ax.annotate(f"{label} {peak/1000:.0f} TFLOPS", xy=(ai_axis[-1], peak),
                    fontsize=8, ha="right", va="bottom", color="gray")
        ax.axvline(ridge, color="lightgray", ls=":", lw=0.7)

    classes_seen = {}
    for r in rows:
        if r["time_share_pct"] < 1.0 or r["ai"] <= 0:
            continue
        color = GROUP_COLOR.get(_group_of(cfg, r["class"]), "black")
        marker = "o" if r["bound_by"] == "memory" else "^"
        size = 20 + 4 * r["time_share_pct"]
        ax.scatter(r["ai"], r["achieved_gflops"], s=size, color=color, marker=marker,
                   alpha=0.75, edgecolors="white", linewidths=0.5)
        lbl = f"{r['class']} ({r['model']}/{r['shape']})"
        ax.annotate(lbl, xy=(r["ai"], r["achieved_gflops"]), fontsize=5.5,
                    xytext=(3, 3), textcoords="offset points")
        classes_seen[r["class"]] = color

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Arithmetic intensity [FLOP/byte], from measured dram__bytes")
    ax.set_ylabel("Achieved [GFLOPS]")
    ax.set_title(f"imp roofline — {gpu['name']}\nrun {run['run_id']}{title_suffix} "
                 f"(o = memory-bound, ^ = compute-bound; size ~ time share)")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def plot_trend(runs, out_path, cfg, last_n=20):
    plt = _import_mpl()
    gpu = cfg["gpu"]
    runs = runs[-last_n:]
    series = {}   # (cell,class) -> [(idx, pct_med)]
    labels = []
    for i, run in enumerate(runs):
        labels.append(run["run_id"][:18])
        rows = rl_table.time_share_within_cell(rl_table.build_rows(run, cfg))
        for r in rows:
            if r["time_share_pct"] < 3.0:
                continue
            series.setdefault(f"{r['class']} {r['model']}/{r['shape']}", []).append(
                (i, r["pct_roofline_med"]))
    fig, ax = plt.subplots(figsize=(13, 7))
    for name, pts in sorted(series.items()):
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3, lw=1, label=name, alpha=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("% roofline (median over restarts)")
    ax.set_title("imp roofline trend — hot-path kernel classes (>3% cell time)")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=5, ncol=2, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def plot_compare(run_a, run_b, out_path, cfg):
    plt = _import_mpl()
    gpu = cfg["gpu"]
    peak_bw, roofs = _roofs(gpu)
    fig, ax = plt.subplots(figsize=(13, 9))
    ai_axis = [2 ** e for e in range(-4, 13)]
    for dtype, label, peak, ridge in roofs:
        ax.plot(ai_axis, [min(ai * peak_bw, peak) for ai in ai_axis],
                lw=1, alpha=0.5, color="gray")

    def keyed(run):
        rows = rl_table.time_share_within_cell(rl_table.build_rows(run, cfg))
        return {(r["model"], r["shape"], r["class"]): r for r in rows
                if r["time_share_pct"] >= 1.0 and r["ai"] > 0}

    ra, rb = keyed(run_a), keyed(run_b)
    for key in sorted(set(ra) & set(rb)):
        a, b = ra[key], rb[key]
        color = GROUP_COLOR.get(_group_of(cfg, key[2]), "black")
        ax.annotate("", xy=(b["ai"], b["achieved_gflops"]),
                    xytext=(a["ai"], a["achieved_gflops"]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2, alpha=0.8))
        ax.scatter([a["ai"]], [a["achieved_gflops"]], s=18, color=color, alpha=0.4)
        ax.scatter([b["ai"]], [b["achieved_gflops"]], s=26, color=color, alpha=0.9)
        ax.annotate(f"{key[2]} ({key[0]}/{key[1]})", xy=(b["ai"], b["achieved_gflops"]),
                    fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("Arithmetic intensity [FLOP/byte]")
    ax.set_ylabel("Achieved [GFLOPS]")
    ax.set_title(f"imp roofline compare\nA={run_a['run_id']} (faint) -> B={run_b['run_id']} (solid)")
    ax.grid(True, which="both", alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f"wrote {out_path}")


def _group_of(cfg, class_name):
    for c in cfg["kernel_classes"]:
        if c["name"] == class_name:
            return c["group"]
    return "unclassified"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["roofline", "trend", "compare"], required=True)
    ap.add_argument("--run", default="latest")
    ap.add_argument("--run-b")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = rl_config.load_config()
    if args.mode == "roofline":
        plot_roofline(rl_history.load_run(args.run), args.out, cfg)
    elif args.mode == "trend":
        runs = [rl_history.load_run(r["run_id"]) for r in rl_history.list_runs()]
        plot_trend(runs, args.out, cfg)
    else:
        plot_compare(rl_history.load_run(args.run), rl_history.load_run(args.run_b),
                     args.out, cfg)


if __name__ == "__main__":
    sys.exit(main())
