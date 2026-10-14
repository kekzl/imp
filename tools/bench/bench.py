#!/usr/bin/env python3
"""imp benchmark suite — MBU, MFU, TTFT, TBT measurements.

Runs imp-cli --bench for various model/prompt configurations,
computes hardware efficiency metrics, and optionally compares
against llama.cpp baseline results.

Usage:
    python bench.py                          # Run all configs
    python bench.py --model qwen3-8b         # Single model
    python bench.py --compare llamacpp       # Compare with llama.cpp results
    python bench.py --imp-bin /path/imp-cli  # Custom binary path

Requires: imp-cli binary (built with --bench support)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

# ── RTX 5090 Hardware Specs ──────────────────────────────────────────────

RTX5090 = {
    "name": "RTX 5090",
    "mem_bw_gbps": 1792,          # GB/s GDDR7 peak
    "bf16_tflops": 838,           # BF16 dense tensor core
    "fp16_tflops": 838,           # FP16 dense tensor core
    "fp8_tflops": 1677,           # FP8 dense tensor core
    "fp4_tflops": 3354,           # NVFP4 dense tensor core
    "int8_tops": 1677,            # INT8 dense tensor core
    "sms": 170,
    "vram_gb": 32,
}

# ── Model Registry ──────────────────────────────────────────────────────

# model_id -> (gguf_filename, param_count_billions, quant_type, bytes_per_param)
MODEL_INFO = {
    "qwen3-8b": {
        "file": "Qwen3-8B-Q8_0.gguf",
        "params_b": 8.19,
        "quant": "Q8_0",
        "bytes_per_param": 1.0625,  # 8-bit + scales
        "hidden_dim": 4096,
        "n_layers": 36,
    },
    "qwen3-4b": {
        "file": "Qwen3-4B-Instruct-2507-Q8_0.gguf",
        "params_b": 4.02,
        "quant": "Q8_0",
        "bytes_per_param": 1.0625,
        "hidden_dim": 2560,
        "n_layers": 36,
    },
    "gemma3-12b": {
        "file": "gemma-3-12b-it-Q8_0.gguf",
        "params_b": 12.0,
        "quant": "Q8_0",
        "bytes_per_param": 1.0625,
        "hidden_dim": 3840,
        "n_layers": 48,
    },
    "deepseek-r1-7b": {
        "file": "DeepSeek-R1-Distill-Qwen-7B-Q8_0.gguf",
        "params_b": 7.62,
        "quant": "Q8_0",
        "bytes_per_param": 1.0625,
        "hidden_dim": 3584,
        "n_layers": 28,
    },
    "phi4-mini": {
        "file": "Phi-4-mini-instruct.Q8_0.gguf",
        "params_b": 3.84,
        "quant": "Q8_0",
        "bytes_per_param": 1.0625,
        "hidden_dim": 3072,
        "n_layers": 32,
    },
}

# ── Benchmark Configurations ────────────────────────────────────────────

BENCH_CONFIGS = [
    # (model_id,           pp,    tg,  description)
    ("qwen3-8b",          128,   128, "short chat"),
    ("qwen3-8b",         2048,   256, "long context"),
    ("qwen3-8b",         8192,   128, "very long context"),
    ("qwen3-4b",          128,   128, "small model short"),
    ("gemma3-12b",        128,   128, "medium model short"),
    ("deepseek-r1-7b",    128,   128, "deepseek short"),
    ("phi4-mini",         128,   128, "phi4 short"),
]


@dataclass
class BenchResult:
    model_id: str
    model_file: str
    quant: str
    pp_tokens: int
    tg_tokens: int
    description: str
    # Raw measurements
    pp_avg_ms: float = 0.0
    tg_avg_ms: float = 0.0
    pp_toks_per_sec: float = 0.0
    tg_toks_per_sec: float = 0.0
    # Derived metrics
    ttft_ms: float = 0.0          # Time to first token (≈ pp_avg_ms)
    tbt_ms: float = 0.0           # Time between tokens
    mbu_pct: float = 0.0          # Model Bandwidth Utilization %
    mfu_pct: float = 0.0          # Model FLOPS Utilization %
    achieved_bw_gbps: float = 0.0  # Achieved memory bandwidth
    achieved_tflops: float = 0.0   # Achieved TFLOPS
    model_size_gb: float = 0.0
    error: str = ""


def compute_model_size_bytes(info: dict) -> int:
    """Estimate model weight size in bytes."""
    return int(info["params_b"] * 1e9 * info["bytes_per_param"])


def compute_mbu(model_size_bytes: int, tg_toks_per_sec: float, hw_bw_gbps: float) -> tuple:
    """Compute Model Bandwidth Utilization for decode.

    During decode, each token requires reading all model weights once.
    MBU = achieved_bandwidth / peak_bandwidth
    achieved_bandwidth = model_size_bytes * tokens_per_second
    """
    if tg_toks_per_sec <= 0:
        return 0.0, 0.0
    achieved_bw = model_size_bytes * tg_toks_per_sec  # bytes/sec
    achieved_bw_gbps = achieved_bw / 1e9
    mbu = (achieved_bw_gbps / hw_bw_gbps) * 100.0
    return mbu, achieved_bw_gbps


def compute_mfu(params_b: float, seq_len: int, pp_toks_per_sec: float,
                peak_tflops: float) -> tuple:
    """Compute Model FLOPS Utilization for prefill.

    Approximate FLOPS for a transformer forward pass:
    flops_per_token ≈ 2 * num_params (multiply-accumulate)
    achieved_tflops = flops_per_token * tokens_per_second
    """
    if pp_toks_per_sec <= 0:
        return 0.0, 0.0
    flops_per_token = 2 * params_b * 1e9
    achieved_flops = flops_per_token * pp_toks_per_sec  # FLOPS
    achieved_tflops = achieved_flops / 1e12
    mfu = (achieved_tflops / peak_tflops) * 100.0
    return mfu, achieved_tflops


def run_imp_bench(imp_bin: str, model_path: str, pp: int, tg: int,
                  reps: int = 3, extra_args: list = None) -> tuple:
    """Run imp-cli --bench and parse output.

    Returns (pp_avg_ms, pp_toks, tg_avg_ms, tg_toks, error_str).
    """
    cmd = [
        imp_bin,
        "--model", model_path,
        "--bench",
        "--bench-pp", str(pp),
        "--bench-reps", str(reps),
        "--max-tokens", str(tg),
        "--temperature", "0",
    ]
    if extra_args:
        cmd.extend(extra_args)

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
        output = result.stderr + result.stdout

        # Parse: "pp  128 tokens  avg   12.34 ms  ( 1234.56 tok/s)  [3 reps]"
        pp_match = re.search(
            r"pp\s+(\d+)\s+tokens\s+avg\s+([\d.]+)\s+ms\s+\(\s*([\d.]+)\s+tok/s\)",
            output
        )
        tg_match = re.search(
            r"tg\s+(\d+)\s+tokens\s+avg\s+([\d.]+)\s+ms\s+\(\s*([\d.]+)\s+tok/s\)",
            output
        )

        if not pp_match or not tg_match:
            return 0, 0, 0, 0, f"Parse error. Output:\n{output[-500:]}"

        pp_ms = float(pp_match.group(2))
        pp_toks = float(pp_match.group(3))
        tg_ms = float(tg_match.group(2))
        tg_toks = float(tg_match.group(3))

        return pp_ms, pp_toks, tg_ms, tg_toks, ""

    except subprocess.TimeoutExpired:
        return 0, 0, 0, 0, "Timeout (600s)"
    except FileNotFoundError:
        return 0, 0, 0, 0, f"Binary not found: {imp_bin}"
    except Exception as e:
        return 0, 0, 0, 0, str(e)


def run_benchmarks(imp_bin: str, models_dir: str, configs: list,
                   reps: int = 3, hw: dict = None) -> list:
    """Run all benchmark configurations and compute metrics."""
    if hw is None:
        hw = RTX5090

    results = []
    for model_id, pp, tg, desc in configs:
        info = MODEL_INFO.get(model_id)
        if not info:
            print(f"  SKIP {model_id}: not in MODEL_INFO registry")
            continue

        model_path = os.path.join(models_dir, info["file"])
        if not os.path.exists(model_path):
            print(f"  SKIP {model_id}: {info['file']} not found in {models_dir}")
            continue

        print(f"  Running: {model_id} pp={pp} tg={tg} ({desc})...")
        sys.stdout.flush()

        pp_ms, pp_toks, tg_ms, tg_toks, error = run_imp_bench(
            imp_bin, model_path, pp, tg, reps
        )

        r = BenchResult(
            model_id=model_id,
            model_file=info["file"],
            quant=info["quant"],
            pp_tokens=pp,
            tg_tokens=tg,
            description=desc,
            pp_avg_ms=pp_ms,
            tg_avg_ms=tg_ms,
            pp_toks_per_sec=pp_toks,
            tg_toks_per_sec=tg_toks,
            error=error,
        )

        if not error:
            model_bytes = compute_model_size_bytes(info)
            r.model_size_gb = model_bytes / 1e9
            r.ttft_ms = pp_ms
            r.tbt_ms = (tg_ms / tg) if tg > 0 else 0

            # MBU (decode — memory-bound)
            r.mbu_pct, r.achieved_bw_gbps = compute_mbu(
                model_bytes, tg_toks, hw["mem_bw_gbps"]
            )

            # MFU (prefill — compute-bound)
            # Use FP16 peak for Q8_0 models (cuBLAS uses FP16 tensor cores)
            peak = hw["fp16_tflops"]
            r.mfu_pct, r.achieved_tflops = compute_mfu(
                info["params_b"], pp, pp_toks, peak
            )

        results.append(r)

    return results


def print_results_table(results: list, title: str = "imp Benchmark Results"):
    """Print results as formatted table."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Header
    print(f"{'Model':<16} {'Config':<12} {'PP tok/s':>10} {'TG tok/s':>10} "
          f"{'TTFT ms':>9} {'TBT ms':>8} {'MBU %':>7} {'MFU %':>7} "
          f"{'BW GB/s':>9}")
    print("-" * 100)

    for r in results:
        if r.error:
            print(f"{r.model_id:<16} pp{r.pp_tokens}/tg{r.tg_tokens:<5} "
                  f"{'ERROR':>10} {r.error}")
            continue

        config = f"pp{r.pp_tokens}/tg{r.tg_tokens}"
        print(f"{r.model_id:<16} {config:<12} {r.pp_toks_per_sec:>10.1f} "
              f"{r.tg_toks_per_sec:>10.1f} {r.ttft_ms:>9.2f} {r.tbt_ms:>8.2f} "
              f"{r.mbu_pct:>6.1f}% {r.mfu_pct:>6.1f}% {r.achieved_bw_gbps:>9.1f}")

    print("-" * 100)

    # Summary
    valid = [r for r in results if not r.error]
    if valid:
        avg_mbu = sum(r.mbu_pct for r in valid) / len(valid)
        avg_mfu = sum(r.mfu_pct for r in valid) / len(valid)
        max_tg = max(r.tg_toks_per_sec for r in valid)
        print(f"\n  Avg MBU: {avg_mbu:.1f}%  |  Avg MFU: {avg_mfu:.1f}%  "
              f"|  Peak decode: {max_tg:.1f} tok/s")
        print(f"  Target: MBU > 85% ({RTX5090['mem_bw_gbps']} GB/s)  |  "
              f"MFU > 50% ({RTX5090['fp16_tflops']} TFLOPS FP16)")


def compare_with_baseline(imp_results: list, baseline_path: str):
    """Compare imp results with llama.cpp (or other) baseline."""
    if not os.path.exists(baseline_path):
        print(f"\n  No baseline found at {baseline_path}")
        print(f"  Run: python bench.py --engine llamacpp --output {baseline_path}")
        return

    with open(baseline_path) as f:
        baseline = json.load(f)

    baseline_map = {}
    for r in baseline.get("results", []):
        key = (r["model_id"], r["pp_tokens"], r["tg_tokens"])
        baseline_map[key] = r

    print(f"\n{'=' * 110}")
    print(f"  Gap Analysis: imp vs llama.cpp")
    print(f"{'=' * 110}")
    print(f"{'Model':<16} {'Config':<12} {'imp TG':>9} {'lcp TG':>9} "
          f"{'Gap':>7} {'imp PP':>9} {'lcp PP':>9} {'Gap':>7} {'Winner':<8}")
    print("-" * 110)

    for r in imp_results:
        if r.error:
            continue
        key = (r.model_id, r.pp_tokens, r.tg_tokens)
        b = baseline_map.get(key)
        if not b:
            continue

        tg_gap = r.tg_toks_per_sec / b["tg_toks_per_sec"] if b["tg_toks_per_sec"] > 0 else 0
        pp_gap = r.pp_toks_per_sec / b["pp_toks_per_sec"] if b["pp_toks_per_sec"] > 0 else 0

        tg_winner = "imp" if tg_gap >= 1.0 else "llamacpp"
        pp_winner = "imp" if pp_gap >= 1.0 else "llamacpp"
        overall = "imp" if (tg_gap + pp_gap) >= 2.0 else "llamacpp"

        config = f"pp{r.pp_tokens}/tg{r.tg_tokens}"
        print(f"{r.model_id:<16} {config:<12} "
              f"{r.tg_toks_per_sec:>8.1f} {b['tg_toks_per_sec']:>8.1f} "
              f"{tg_gap:>6.2f}x "
              f"{r.pp_toks_per_sec:>8.1f} {b['pp_toks_per_sec']:>8.1f} "
              f"{pp_gap:>6.2f}x {overall:<8}")

    print("-" * 110)


def save_results(results: list, output_path: str, engine: str = "imp"):
    """Save results as JSON."""
    data = {
        "engine": engine,
        "hardware": RTX5090,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Results saved to: {output_path}")


def run_llamacpp_bench(llamacpp_bin: str, model_path: str, pp: int, tg: int,
                       reps: int = 3) -> tuple:
    """Run llama-bench and parse output.

    llama-bench output format:
    | model | ... | pp ... | tg ... | pp t/s | tg t/s |
    """
    cmd = [
        llamacpp_bin,
        "-m", model_path,
        "-p", str(pp),
        "-n", str(tg),
        "-r", str(reps),
        "-ngl", "999",
        "-fa",
        "-t", "1",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        # llama-bench outputs a markdown table with pp t/s and tg t/s columns
        # Look for lines with actual numbers
        pp_toks = 0
        tg_toks = 0
        for line in output.split("\n"):
            if "|" in line and "model" not in line.lower() and "---" not in line:
                cols = [c.strip() for c in line.split("|")]
                # Last two numeric columns are typically pp t/s and tg t/s
                nums = []
                for c in reversed(cols):
                    try:
                        nums.append(float(c.replace(",", "")))
                    except ValueError:
                        continue
                    if len(nums) >= 2:
                        break
                if len(nums) >= 2:
                    tg_toks = nums[0]
                    pp_toks = nums[1]

        if pp_toks > 0 and tg_toks > 0:
            pp_ms = (pp / pp_toks) * 1000.0
            tg_ms = (tg / tg_toks) * 1000.0
            return pp_ms, pp_toks, tg_ms, tg_toks, ""

        return 0, 0, 0, 0, f"Parse error. Output:\n{output[-500:]}"

    except subprocess.TimeoutExpired:
        return 0, 0, 0, 0, "Timeout (600s)"
    except FileNotFoundError:
        return 0, 0, 0, 0, f"Binary not found: {llamacpp_bin}"
    except Exception as e:
        return 0, 0, 0, 0, str(e)


def main():
    parser = argparse.ArgumentParser(description="imp benchmark suite")
    parser.add_argument("--imp-bin", default="./build/imp-cli",
                        help="Path to imp-cli binary")
    parser.add_argument("--llamacpp-bin", default="llama-bench",
                        help="Path to llama-bench binary")
    parser.add_argument("--models-dir", default="./models",
                        help="Directory containing model files")
    parser.add_argument("--model", default=None,
                        help="Run only this model ID")
    parser.add_argument("--reps", type=int, default=3,
                        help="Benchmark repetitions")
    parser.add_argument("--engine", choices=["imp", "llamacpp"], default="imp",
                        help="Which engine to benchmark")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (auto-generated if not set)")
    parser.add_argument("--compare", default=None,
                        help="Path to baseline JSON to compare against")

    args = parser.parse_args()

    # Filter configs by model if requested
    configs = BENCH_CONFIGS
    if args.model:
        configs = [c for c in configs if c[0] == args.model]
        if not configs:
            print(f"Error: no configs for model '{args.model}'")
            print(f"Available: {sorted(set(c[0] for c in BENCH_CONFIGS))}")
            sys.exit(1)

    # Default output paths
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if args.output is None:
        args.output = os.path.join(results_dir, f"{args.engine}_baseline.json")

    print(f"imp Benchmark Suite")
    print(f"  Engine:     {args.engine}")
    print(f"  Models dir: {args.models_dir}")
    print(f"  Configs:    {len(configs)}")
    print(f"  Reps:       {args.reps}")
    print()

    if args.engine == "imp":
        results = run_benchmarks(args.imp_bin, args.models_dir, configs, args.reps)
    else:
        # llama.cpp benchmarks
        results = []
        hw = RTX5090
        for model_id, pp, tg, desc in configs:
            info = MODEL_INFO.get(model_id)
            if not info:
                continue
            model_path = os.path.join(args.models_dir, info["file"])
            if not os.path.exists(model_path):
                print(f"  SKIP {model_id}: {info['file']} not found")
                continue

            print(f"  Running: {model_id} pp={pp} tg={tg} ({desc})...")
            pp_ms, pp_toks, tg_ms, tg_toks, error = run_llamacpp_bench(
                args.llamacpp_bin, model_path, pp, tg, args.reps
            )

            r = BenchResult(
                model_id=model_id, model_file=info["file"], quant=info["quant"],
                pp_tokens=pp, tg_tokens=tg, description=desc,
                pp_avg_ms=pp_ms, tg_avg_ms=tg_ms,
                pp_toks_per_sec=pp_toks, tg_toks_per_sec=tg_toks, error=error,
            )
            if not error:
                model_bytes = compute_model_size_bytes(info)
                r.model_size_gb = model_bytes / 1e9
                r.ttft_ms = pp_ms
                r.tbt_ms = (tg_ms / tg) if tg > 0 else 0
                r.mbu_pct, r.achieved_bw_gbps = compute_mbu(
                    model_bytes, tg_toks, hw["mem_bw_gbps"]
                )
                r.mfu_pct, r.achieved_tflops = compute_mfu(
                    info["params_b"], pp, pp_toks, hw["fp16_tflops"]
                )
            results.append(r)

    # Output
    print_results_table(results, f"{args.engine} Benchmark Results")
    save_results(results, args.output, args.engine)

    # Compare if requested
    if args.compare:
        compare_with_baseline(results, args.compare)
    elif args.engine == "imp":
        baseline_path = os.path.join(results_dir, "llamacpp_baseline.json")
        if os.path.exists(baseline_path):
            compare_with_baseline(results, baseline_path)


if __name__ == "__main__":
    main()
