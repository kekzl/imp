#!/usr/bin/env python3
"""Säule 4 of Track E gating bench.

Compute the per-(model × KV dtype) maximum context length, and the additional
context unlocked by freeing the 1 GiB cuBLAS S-matrix workspace.

Inputs are hardcoded from the supported-models table + roadmap. Output is a
markdown table. No GPU required.

Usage: dev python scripts/analyze_attention_workspace_savings.py
"""

from __future__ import annotations

VRAM_TOTAL_GIB = 32.0          # RTX 5090 GDDR7
S_MATRIX_GIB = 1.0             # cuBLAS S-matrix workspace freed by Track E
RUNTIME_OVERHEAD_GIB = 2.0     # rough: workspaces, activations, KV-tile buffers,
                               # FP8 calibration scratch, MoE prequant cache, etc.


def model_kv_bytes_per_token(n_kv_heads: int, head_dim: int, n_layers: int,
                             kv_dtype: str) -> int:
    """KV-cache bytes per token across all layers, both K and V."""
    bytes_per_elem = {
        "FP16": 2,
        "FP8":  1,
        "NVFP4": 0.5 + 0.0625,  # 4-bit data + UE8M0 per-16 scale ≈ 0.5625 B/elem
        "INT4": 0.5 + 0.0625,
        "MXFP4": 0.5 + 0.0625,
    }[kv_dtype]
    return int(round(2 * n_kv_heads * head_dim * n_layers * bytes_per_elem))


def max_ctx(weight_gib: float, kv_dtype: str, model: dict) -> int:
    avail_gib = VRAM_TOTAL_GIB - weight_gib - RUNTIME_OVERHEAD_GIB
    if avail_gib <= 0:
        return 0
    avail_bytes = int(avail_gib * 1024**3)
    bytes_per_tok = model_kv_bytes_per_token(
        model["n_kv_heads"], model["head_dim"], model["n_layers"], kv_dtype)
    if bytes_per_tok <= 0:
        return 0
    return avail_bytes // bytes_per_tok


# Production model table. Weights GiB from supported-models doc + perf baselines.
MODELS = [
    {"name": "Qwen3-4B Q8_0",       "weight_gib":  4.4, "n_kv_heads":  8, "head_dim": 128, "n_layers": 36},
    {"name": "Qwen3-8B Q8_0",       "weight_gib":  8.2, "n_kv_heads":  8, "head_dim": 128, "n_layers": 36},
    {"name": "Qwen3-8B NVFP4",      "weight_gib":  6.2, "n_kv_heads":  8, "head_dim": 128, "n_layers": 36},
    {"name": "Llama-3.2-3B Q8_0",   "weight_gib":  3.4, "n_kv_heads":  8, "head_dim": 128, "n_layers": 28},
    {"name": "Qwen3.5-9B GDN Q8_0", "weight_gib":  9.1, "n_kv_heads":  8, "head_dim": 128, "n_layers": 36},
    {"name": "Qwen3.6-35B Q4_K_M",  "weight_gib": 18.5, "n_kv_heads":  8, "head_dim": 128, "n_layers": 48},
    {"name": "Qwen3.6-35B NVFP4",   "weight_gib": 14.2, "n_kv_heads":  8, "head_dim": 128, "n_layers": 48},
    {"name": "Gemma-4-26B Q4_K_M",  "weight_gib": 14.8, "n_kv_heads": 16, "head_dim": 256, "n_layers": 32},
    {"name": "Gemma-4-26B NVFP4",   "weight_gib": 11.5, "n_kv_heads": 16, "head_dim": 256, "n_layers": 32},
]

KV_DTYPES = ["FP16", "FP8", "NVFP4"]


def main() -> None:
    print("# Säule 4: Attention workspace savings analysis\n")
    print(f"VRAM total: **{VRAM_TOTAL_GIB} GiB**, runtime overhead: "
          f"**{RUNTIME_OVERHEAD_GIB} GiB**, freed by Track E: "
          f"**{S_MATRIX_GIB} GiB**\n")
    print("Per-model maximum context length before vs after freeing the "
          "1 GiB S-matrix workspace (workspace stays for cuBLAS path, freed "
          "for tiled-streaming path).\n")
    print("| Model | KV dtype | ctx (with 1 GiB ws) | ctx (freed) | Δ ctx | Δ % |")
    print("|---|---|---:|---:|---:|---:|")
    for m in MODELS:
        for kv in KV_DTYPES:
            ctx_with_ws = max_ctx(m["weight_gib"] + S_MATRIX_GIB, kv, m)
            ctx_freed = max_ctx(m["weight_gib"], kv, m)
            delta = ctx_freed - ctx_with_ws
            pct = (delta / ctx_with_ws * 100.0) if ctx_with_ws > 0 else 0.0
            print(f"| {m['name']} | {kv} | {ctx_with_ws:>7,} | {ctx_freed:>7,} | "
                  f"+{delta:,} | +{pct:.1f}% |")

    print("\n## Aggregate")
    rows = []
    for m in MODELS:
        for kv in KV_DTYPES:
            ctx_with_ws = max_ctx(m["weight_gib"] + S_MATRIX_GIB, kv, m)
            ctx_freed = max_ctx(m["weight_gib"], kv, m)
            if ctx_with_ws > 0:
                pct = (ctx_freed - ctx_with_ws) / ctx_with_ws * 100.0
                rows.append(pct)
    if rows:
        rows.sort()
        median = rows[len(rows) // 2]
        print(f"- median Δ context: **+{median:.1f}%**")
        print(f"- min Δ context:    +{rows[0]:.1f}%")
        print(f"- max Δ context:    +{rows[-1]:.1f}%")
        print(f"- n configs:        {len(rows)}")

    print("\n## Decision input\n")
    print("Per spec decision matrix: if **median Δ context ≤ 5%**, the "
          "workspace saving alone does not justify multi-week Track E work — "
          "defer unless Säule 3 ceiling also shows ≥2× headroom.")


if __name__ == "__main__":
    main()
