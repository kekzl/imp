# Memory-Traffic Reduction Catalog (imp, RTX 5090 sm_120f)

Brainstormed 2026-04-24. RTX 5090 is **memory-bound at batch=1 decode**
(28:1 memory:compute ratio per `docs/SM120_OPTIMIZATION_STATUS.md`).
Further perf comes from reducing bytes moved, not from faster MMA. This
document catalogs options across weight, KV, activation, and MoE
traffic, with tradeoff labels and implementation status in imp.

## Legend

- **Gain** — realistic, measured where possible. Not marketing numbers.
- **VRAM** — impact on GPU memory footprint.
- **Quality** — PPL / top-1 deltas; `~0` means undetectable.
- **Effort** — `klein` (≤1 day), `mittel` (2-7 days), `hoch` (≥1 week or per-model work).
- **Status** — `vorhanden` (shipped), `teilweise` (partial/stubs), `nicht da` (missing).

## Decode @ batch=1 — Weight-Traffic

Weight loads dominate decode bandwidth at batch=1. Every byte shaved off
weights scales linearly with decode throughput.

| # | Option | Gain | VRAM | Qualität | Aufwand | Status |
|---|---|---|---|---|---|---|
| W1 | 2-bit Weight-Quant (AQLM, QuIP#, HQQ-2bit) | +30-60% decode | -50% | -0.5 bis -2 PPL | hoch (Calibration-Pipeline) | nicht da |
| W2 | Speculative Decoding mit EAGLE-3 | +1.5-2.5× bei Accept >70% | +400MB (draft head) | gleich (lossless) | mittel (lib existiert) | Dead-End historisch, worth revisit |
| W3 | Medusa Heads (multi-token predict) | +1.3-2× | +300MB | gleich (verify lossless) | mittel | nicht da |
| W4 | PLE / Prompt-Lookup (strukturierter Kontext) | 2-5× bei JSON/Code | 0 | gleich | klein | N-gram-Spec ist Variante davon |
| W5 | Batch>1 Coalescing (continuous batching) | 2-8× Throughput | 0 | gleich | teilweise vorhanden | teilweise |
| W6 | Weight-Tying zwischen Layers | +5-10% bei tied layers | -5-15% VRAM | ~0 per model | hoch (per-model Arbeit) | nicht da |

## KV-Cache — lange Kontexte

KV traffic overtakes weights around ctx=4-8K. At ctx=32K+ it dominates
entirely.

| # | Option | Gain | VRAM | Qualität | Aufwand | Status |
|---|---|---|---|---|---|---|
| K1 | INT4 KV-Cache | 50% KV-Traffic | -50% KV | -0.1 bis -0.3 PPL | mittel | teilweise (Stubs) |
| K2 | Multi-Head Latent Attention (MLA, DeepSeek) | -90% KV-VRAM | -90% | gleich per model | hoch (per-model impl) | nicht da |
| K3 | YOCO / Cross-Layer KV-Sharing | -50% KV traffic | -50% | per model unterschiedlich | hoch | nicht da |
| K4 | Attention Sinks + Sliding Window (StreamingLLM) | konstanter KV, unbounded ctx | fix | minimal (long-range weg) | klein | vorhanden |
| K5 | Token-Eviction per Attention-Score (H2O, Scissorhands) | -50-70% KV bei langem ctx | dynamisch | -0.2 PPL | mittel | nicht da |
| K6 | TurboQuant Lite (QJL-Sketch für K) | ~3 bits/elem avg | -60% K | -0.1 PPL | mittel | vorhanden |
| K7 | Chunked Prefill + KV-Compression between chunks | Prefill-Burst-Traffic reduzieren | 0 | gleich | mittel | nicht da |
| K8 | KV Offload to CPU pinned (async prefetch) | ermöglicht 100K+ ctx | +RAM | gleich | mittel | nicht da |
| K9 | SnapKV / PyramidKV (nicht-uniformes Layer-Budget) | -30-50% KV | variable | -0.1 PPL | mittel | nicht da |

## Activation-Traffic

Intermediate tensors (QKV projections, FFN gate/up, attention output)
between kernels. Fusion removes round-trips to global memory.

| # | Option | Gain | VRAM | Qualität | Aufwand | Status |
|---|---|---|---|---|---|---|
| A1 | QKV-Fused Projection | -2 GEMM-Calls/layer | 0 | gleich | klein | vorhanden |
| A2 | Gate+Up Fused FFN (SwiGLU) | -1 GEMM-Call/layer | 0 | gleich | klein | vorhanden |
| A3 | In-place RMSNorm+Residual | -1 Alloc/layer | marginal | gleich | klein | vorhanden |
| A4 | FP16-Accum für Logits (statt FP32) | -50% Logits-Write | 0 | -0.1% top-1 | klein | teilweise |
| A5 | Fused Attention-Out + Residual + RMSNorm | -2 Kernel-Launches | 0 | gleich | mittel | teilweise |
| A6 | Fused MoE Routing (alles ohne D2H) | Graph-Safe MoE | 0 | gleich | mittel | teilweise (Gemma-4 Fast-Path) |

## MoE-spezifisch

Expert-Swap zwischen Host und Device dominiert MoE-Traffic wenn Experts
nicht alle in VRAM passen.

| # | Option | Gain | VRAM | Qualität | Aufwand | Status |
|---|---|---|---|---|---|---|
| M1 | Expert-Prefetch basierend auf Router-Top-K | -20-40% Expert-Swap-Wait | 0 | gleich | mittel | nicht da |
| M2 | Shared-Expert-Pinning (Gemma-4) | -Traffic für shared FFN | marginal | gleich | klein | vorhanden |
| M3 | Expert-LRU-Cache (VRAM-Pool) | -60% Swap bei Offload | Cache-Budget | gleich | mittel | vorhanden |
| M4 | Top-K-Stickiness (Reuse-Boost im Router) | +10-20% Cache-Hit | 0 | -0.05 PPL | mittel | nicht da |
| M5 | Expert-Token-Sort statt Scatter/Gather | -Traffic in Dispatch | 0 | gleich | mittel | vorhanden |

## Cross-Cutting

Nicht "Traffic-Reduktion" im engen Sinn, aber Pipeline-Overlap reduziert
effektive Wartezeit auf Traffic.

| # | Option | Gain | VRAM | Qualität | Aufwand | Status |
|---|---|---|---|---|---|---|
| X1 | L2 Persist für KV (75% L2 reserved) | +2-4% decode | 0 | gleich | klein | vorhanden |
| X2 | PDL (Programmatic Dep Launch) | Tail/Head-Overlap | 0 | gleich | klein | vorhanden |
| X3 | Green-Contexts (Prefill/Decode SM-Split) | bessere Pipeline-Overlap | 0 | gleich | klein | vorhanden |
| X4 | CUDA-Graphs (launch-Overhead) | -300 Launches/token | 0 | gleich | mittel | vorhanden (Verify-Pool + Vision + ExecUpdate 2026-04-24) |

## Top-Kandidaten — ROI-sortiert, noch nicht geshippt

1. **K1 INT4 KV-Cache fertigstellen** — Stubs existieren, größter
   einzelner ROI für lange Kontexte, ~2-3 Tage. Ohne Per-Model-Arbeit.
2. **W2 EAGLE-3 Revisit** — dead_ends.md markiert alte Version gescheitert,
   aber active development bei EAGLE-Repo. Bei Accept-Rate >70%: 1.5-2×
   decode.
3. **K5 Token-Eviction (H2O)** — lange Kontexte, moderate Qualitätsverluste,
   ~1 Woche Arbeit. Orthogonal zu K4/K6.
4. **M1 Expert-Prefetch** — MoE-Modelle mit Host-Experts. Pipeline-Parallelism
   zwischen Router-Output und Expert-Load.
5. **W3 Medusa Heads** — lossless, orthogonal zu EAGLE, kleinere VRAM-Kosten.

## Was bereits gewonnen wurde (Referenz)

- FP8 Prefill Weight-Cache: +40-60% prefill (Q8_0 Modelle)
- NVFP4 Decode Weight-Cache: +4.7-16% decode (prmt-LUT)
- NVFP4 Prequant (Model Optimizer): 38 tok/s auf Qwen3-Coder-30B-A3B
- FP8 KV Cache: 50% KV-Traffic (stabil mit deterministic GEMM workaround)
- MXFP4 FMHA Prefill: +7-18% über FP8 (seq>=1024)
- MoE Persistent Work-Queue: -38% TC kernel time
- CUDA Graphs (Verify + Vision + ExecUpdate): Launch-Overhead-Reduktion

## Referenzen

- `docs/SM120_OPTIMIZATION_STATUS.md` — Bottleneck-Analyse Decode/Prefill
- `docs/cuda131-optimization-plan.md` — historischer Plan (teilweise realisiert)
- Memory: `perf_optimizations.md`, `dead_ends.md`, `turbodraft.md`,
  `nvfp4_prequant_status.md`, `gdn_debug_status.md`
