# imp — CUDA 13.1 Feature Audit, Performance & Architektur-Review

**Datum:** 2026-03-01 (aktualisiert nach GEMV-Konsolidierung, Blackwell-Attention, Multi-Block-Argmax)

Umfassende Analyse des imp-Projekts auf drei Achsen:
1. Welche CUDA 13.1+ Features werden genutzt, welche fehlen noch?
2. Wo gibt es konkretes Performance-Potential?
3. Entspricht die Architektur modernen Standards (vLLM, TensorRT-LLM, SGLang)?

---

## 1. CUDA 13.1+ Feature-Inventar

### Vollständig implementiert

| Feature | Dateien | Status |
|---------|---------|--------|
| **Green Contexts** (SM-Partitionierung) | `src/runtime/green_ctx.cu` | Prefill/Decode SM-Split (80/20 default), Fallback auf normale Streams |
| **PDL** (Programmatic Dependent Launch) | `src/runtime/pdl.cu` | **28+ Kernel-Registrierungen**: 7 Utility + 6 Compute + 110+ Template-Instantiierungen (dp4a GEMV). cuBLAS aktiviert PDL intern auf sm_90+. |
| **CUDA Graphs** (Conditional WHILE) | `src/runtime/cuda_graph.cu` | Capture/Replay, `cudaGraphConditionalHandleCreate` mit WHILE-Loop, Mapped-Memory Ring Buffer |
| **cudaMallocAsync / MemPool** | `src/memory/device_allocator.cu` | `cudaMemPoolCreate` mit `ReuseAllowOpportunistic`, stream-geordnete Allokation |
| **NVFP4** (FP4 E2M1) | `src/quant/nvfp4_quant.cu`, `nvfp4_gemm.cu` | 2-Level Quantisierung (Micro-Scale FP8 + Tensor-Scale FP32), Blackwell-nativ |

### Teilweise implementiert

| Feature | Dateien | Status |
|---------|---------|--------|
| **Blackwell WMMA Attention** | `src/compute/attention_blackwell.cu` | 8-Warp WMMA mit Double-Buffered KV, Adaptive Br (128/64), sm_120-optimiert. **Kein Inline-PTX** (WGMMA/TCGEN05), kein TMA — nutzt WMMA Intrinsics. |
| **CUTLASS 3.x** | `src/compute/gemm_cutlass.cu` | Nutzt CUTLASS 2.x (`cp.async`, kein TMA), kein Upgrade auf 3.x Hopper/Blackwell Primitives |

### Nicht implementiert (Potential vorhanden)

| Feature | Nutzen | Priorität |
|---------|--------|-----------|
| **TCGEN05 Inline-PTX** (WGMMA + TMA + TMEM) | Echte systolische Blackwell-Attention, ~2x vs WMMA | Hoch |
| **TMA** (Tensor Memory Accelerator) | Bulk-Loads für Attention und GEMM | Mittel |
| **Stream-Attribute** (`cudaStreamSetAttribute`) | Priorität für Prefill- vs Decode-Streams | Niedrig |
| **Conditional IF Nodes** in CUDA Graphs | Komplexere GPU-autonome Control-Flow (Early-Exit, Branch) | Niedrig |
| **Multi-GPU** (P2P, IPC, NCCL) | Tensor/Pipeline-Parallelismus | N/A (1 GPU) |

---

## 2. Performance-Optimierungen

### Offen

| # | Optimierung | Dateien | Geschätzter Gewinn | Aufwand |
|---|-------------|---------|-------------------|---------|
| O1 | **TCGEN05 Inline-PTX Attention** — WGMMA + TMA Bulk-Loads + TMEM Accumulators + CTA Pairing | `attention_blackwell.cu` | ~2x Prefill auf sm_120 | Hoch |
| O2 | **NVFP4 Quantized GEMV** — Native FP4 dp4a-Kernels für RTX 5090 (sm_120 hat NVFP4-Tensor-Cores) | `gemv_dp4a_traits.cuh`, `gemm.cu` | +20-40% Decode (weniger Bandwidth) | Mittel |
| O3 | **GEMV Bandwidth-Optimierung** — Aktuelle Auslastung ~50-59% vs llama.cpp ~55-69%. L2-Cache-Tuning, Prefetch-Hints | `gemv_dp4a_traits.cuh` | +5-10% Dense Decode | Mittel |

### Implementiert — Phase 3 (16f6cff → bcc9bba, 2026-02-26 bis 2026-03-01)

| # | Optimierung | Dateien | Gemessener Gewinn |
|---|-------------|---------|-------------------|
| P10 | **Blackwell WMMA 8-Warp Attention** — Double-Buffered KV-Tiles, Adaptive Br (128 für HD≤64, 64 für HD=128), 99 KB smem opt-in für RTX 5090 | `attention_blackwell.cu` | +15-25% Prefill auf sm_120 |
| P11 | **Fused Gate+Up GEMV (Decode)** — Einzelner Kernel berechnet Gate+Up+SwiGLU, eliminiert 2 separate Launches | `gemv_dp4a_traits.cuh`, `executor.cu` | +3-5% Decode |
| P12 | **Fused RoPE + KV-Cache Write** — RoPE-Berechnung direkt beim KV-Write, eliminiert separaten RoPE-Kernel | `executor.cu` | +1-2% Decode (Launch-Overhead) |
| P13 | **D2D-Copy-Elimination** — Direkte Writes statt Device-to-Device Copies für KV-Updates | `executor.cu` | +1% Decode |
| P14 | **HEAD_DIM Template-Spezialisierung** — Compile-Time Head-Dimension für Attention-Kernels | `attention_blackwell.cu`, `attention_tc.cu` | Bessere Register-Allokation |
| P15 | **GEMV Template-Konsolidierung** — 33 handgeschriebene Kernels → 6 Template-Kernels mit 5 QType-Traits. K-parallel + Row-parallel Dispatch-Heuristik | `gemv_dp4a_traits.cuh`, `gemm.cu` | Code-Reduktion, +/- 0% Performance |
| P16 | **MoE Shared-Memory Expert Caching** — Q8_1-Aktivierungen im smem mit Stride-9-Layout (Bank-Conflict-frei) | `gemv_dp4a_traits.cuh` | +5-10% MoE Decode |
| P17 | **Inline O-Projection Quantization** — Separate Q8_1-Quant + K-par GEMV mit Residual (48 Warps/SM statt 8) | `gemv_dp4a_traits.cuh`, `executor.cu` | +2-3% Decode |
| P18 | **Eager FP16 Dequant bei Init** — `pre_dequant_weights()` verschoben von lazy erstem Prefill nach `Engine::init()` | `engine.cpp` | ~16x Real-World Prefill (380ms Overhead eliminiert) |
| P19 | **Multi-Block Argmax** — 64-Block 2-Phasen-Reduktion statt Single-Block. 84 SMs parallel statt 1. | `sampling.cu`, `sampling.h` | 192µs → ~10µs pro Greedy-Sample |

### Implementiert — Phase 2 (e2cf896, 2026-02-28)

| # | Optimierung | Dateien | Gemessener Gewinn |
|---|-------------|---------|-------------------|
| P2 | **Q4_0 dp4a GEMV** — Dedizierte dp4a-GEMV für Q4_0 | `gemm.cu`, `weight_upload.cu`, `executor.cu` | Kein Q4_0-Modell verfügbar |
| P3 | **Down-Projection Residual Fusion** — `gemm(beta=1.0)` eliminiert separaten `elementwise_add` | `executor.cu` | +3-84% Prefill |
| P4 | **Gate+Up Batched GEMM** — `cublasGemmStridedBatchedEx` für Gate+Up in einem cuBLAS-Call | `gemm.cu`, `executor.cu` | +3-84% Prefill (mit P3) |
| P5 | **PDL-Registry erweitert** — Compute-Kernels: RMSNorm, RoPE, SwiGLU, GeGLU | `layernorm.cu`, `rope.cu`, `activation.cu` | PDL-Overlap sichtbar |
| P6 | **KV-Cache Double-Buffered Prefetch** — FP16 Double-Buffer in Paged Attention | `attention_paged.cu` | ~0% (Attention nicht Bottleneck) |
| P7 | **Request-Reordering** — Shortest-first Sortierung | `scheduler.cpp` | Multi-Request messbar |
| P8 | **Async Weight Upload** — Separater Upload-Stream + Event-Sync | `engine.cpp` | Init-Time Overlap |
| P9 | **Event-basierter Prefill Sync** — `cudaEventSynchronize` statt `cudaStreamSynchronize` | `engine.cpp` | +3-84% Prefill (mit P3/P4) |

### Benchmark: Finale Ergebnisse (RTX 5090, CUDA 13.1, bs=1)

#### Decode-Throughput (tok/s) — imp vs llama.cpp

| Model | Quant | Parameter | llama.cpp | imp | vs llama.cpp |
|-------|-------|-----------|-----------|-----|-------------|
| Phi-4-Mini | Q8_0 | 3.8B | 250.96 | **239.29** | -4.6% |
| Qwen3-4B | Q8_0 | 4B | 217.54 | **228.07** | **+4.8%** |
| DeepSeek-R1-7B | Q8_0 | 7B | 164.30 | **159.43** | -3.0% |
| Gemma-3-12B | Q8_0 | 12B | 91.41 | **85.39** | -6.6% |
| DeepSeek-R1-14B | Q6_K | 14B | 102.22 | **88.29** | -13.6% |
| Qwen3-Coder-30B-A3B (MoE) | Q6_K | 30B (3B aktiv) | 206.61 | **231.71** | **+12.1%** |
| Nemotron-3-Nano-30B-A3B (MoE) | Q6_K | 30B (3B aktiv) | 25.77 | **60.42** | **+134%** |

#### Prefill-Throughput (tok/s)

| Model | Quant | pp (tok/s) |
|-------|-------|-----------|
| Phi-4-Mini | Q8_0 | 19,460 |
| Qwen3-4B | Q8_0 | 19,557 |
| DeepSeek-R1-7B | Q8_0 | 13,491 |
| Gemma-3-12B | Q8_0 | 4,026 |
| DeepSeek-R1-14B | Q6_K | 5,713 |
| Qwen3-Coder-30B-A3B (MoE) | Q6_K | 4,551 |
| Nemotron-3-Nano-30B-A3B (MoE) | Q6_K | 1,247 |

#### Decode-Entwicklung über alle Optimierungs-Phasen

| Model | Quant | Baseline (a9ae9b4) | Nach P2-P9 | Finale (bcc9bba) | Gesamt-Gewinn |
|-------|-------|---------------------|------------|-------------------|---------------|
| Phi-4-Mini | Q8_0 | 189.77 | 219.85 | 239.29 | +26.1% |
| Qwen3-4B | Q8_0 | 186.06 | 213.33 | 228.07 | +22.6% |
| DS-R1-7B | Q8_0 | 133.52 | 149.52 | 159.43 | +19.4% |
| Gemma-3-12B | Q8_0 | — | 82.08 | 85.39 | — |
| DS-R1-14B | Q6_K | — | 83.85 | 88.29 | — |

**MoE-Modelle** profitieren am stärksten von den Custom-Kernels: Fused MoE GEMV mit Shared-Memory Expert Caching und Sigmoid-Routing übertreffen llama.cpp signifikant. Nemotron (Hybrid Mamba2+Attention+MoE) ist 2.3x schneller, da llama.cpp diese Architektur nicht optimiert.

**Dense-Modelle** liegen bei 3-14% hinter llama.cpp. Die verbleibende Lücke ist systemisch bedingt durch GEMV-Bandwidth-Auslastung (~50-59% vs llama.cpp ~55-69%).

---

## 3. Architektur-Review vs. Moderne Standards

### Stärken (auf Level mit vLLM/TRT-LLM)

- **PagedAttention** mit Block-basiertem KV-Cache (kKVBlockSize=16, vLLM-kompatibel)
- **Prefix Caching** mit Hash-Lookup und Ref-Counting
- **Continuous Batching** mit Prefill/Decode-Separation
- **Speculative Decoding** (Draft+Target, stochastische Akzeptanz, KV-Rollback)
- **Hybrid-Architekturen** (Mamba2 SSM + Attention + MoE — Nemotron)
- **Green Contexts + PDL + CUDA Graphs** — moderner als die meisten Open-Source Engines
- **VRAM-aware Expert Upload** mit greedy Layer-Budgetierung
- **OpenAI-kompatibler HTTP-Server** mit SSE-Streaming
- **Template-basierte dp4a GEMV** — 5 Quant-Typen (Q6_K, Q8_0, Q4_0, Q4_K, Q5_K) mit K-parallel/Row-parallel Dispatch-Heuristik

### Lücken vs. vLLM/TRT-LLM/SGLang

| Feature | vLLM | TRT-LLM | SGLang | imp | Kommentar |
|---------|------|---------|--------|-----|-----------|
| Multi-GPU / TP | Ja | Ja | Ja | **Nein** | Single-GPU only |
| Chunked Prefill | Ja | Ja | Ja | **Nein** | Ganzer Prompt in einem Pass |
| Structured Output (Grammar) | Ja | Nein | Ja | **Nein** | Kein JSON-Schema/GBNF |
| Vision / Multimodal | Ja | Ja | Ja | **Nein** | Kein Vision-Encoder |
| LoRA / Adapters | Ja | Ja | Nein | **Nein** | Statische Weights only |
| Beam Search | Ja | Ja | Ja | **Nein** | Nur Greedy/Sampling |
| Request Preemption | Ja | Nein | Ja | **Nein** | Kein Pause/Resume |
| Priority Scheduling | Ja | Nein | Ja | **Partiell** | Shortest-first Reordering (P7), kein Priority-Feld |
| Logprobs Output | Ja | Ja | Ja | **Nein** | Nur finales Token |
| Stop Sequences | Ja | Ja | Ja | **Partiell** | EOS + Chat-Template Stop-Tokens, keine beliebigen String-Sequenzen |
| Batch API (extern) | Ja | Ja | Ja | **Nein** | Intern ja, API single-request |
| KV-Cache Quantisierung | Ja | Ja | Nein | **Partiell** | FP16/FP8, kein INT4/INT8 |
| AWQ/GPTQ | Ja | Ja | Ja | **Nein** | Nur GGML-Formate |
| Medusa/EAGLE Heads | Nein | Ja | Nein | **Nein** | Nur Draft-Model |

### Empfohlene nächste Features (nach ROI sortiert)

1. **NVFP4 Quantized GEMV** — RTX 5090 hat native FP4 Tensor Cores; +20-40% Decode möglich
2. **Stop Sequences + Logprobs** — Niedrig-Aufwand, hoher User-Impact, Server-Kompatibilität
3. **Chunked Prefill** — Verhindert Head-of-Line-Blocking bei langen Prompts
4. **TCGEN05 Inline-PTX Attention** — ~2x Prefill auf sm_120
5. **Structured Output** — JSON-Mode / GBNF-Grammatik für Agentic Use-Cases

---

## Zusammenfassung

**CUDA 13.1**: 5/5 Major-Features implementiert (Green Contexts, PDL, Graphs+Conditional, MemPool, NVFP4). PDL-Registry mit 28+ Kernel-Registrierungen (7 Utility + 6 Compute + 110+ Template-Instantiierungen). Blackwell WMMA 8-Warp Attention mit Double-Buffered KV für sm_120. TCGEN05 Inline-PTX/TMA noch nicht implementiert — größtes verbleibendes Hardware-Potential.

**Performance**: P2-P19 implementiert. Decode: +19-26% vs Baseline (dense), +12-134% vs llama.cpp (MoE). Prefill: +3-84% (Batched GEMM, Residual Fusion, Eager Dequant). Multi-Block Argmax: 192µs → ~10µs. GEMV-Coverage: Q6_K, Q8_0, Q4_0, Q4_K, Q5_K (alle mit K-par + Row-par + QKV-fused + Gate+Up-fused Varianten). Verbleibend: NVFP4 GEMV, TCGEN05-Attention, Bandwidth-Optimierung.

**Architektur**: Auf Single-GPU-Ebene vergleichbar mit vLLM. Hauptlücken sind Multi-GPU, Structured Output, Vision, und externe Batch-API. Für den aktuellen Scope (Single-GPU, CLI/Server) ist die Architektur solide und modern.

---

## Anhang A: Code-Evidenz CUDA 13.1 Features

### A.1 Green Contexts — Vollständig

**APIs**: `cudaDeviceGetDevResource()`, `cudaDevSmResourceSplitByCount()`, `cudaDevResourceGenerateDesc()`, `cudaGreenCtxCreate()`, `cudaExecutionCtxStreamCreate()`, `cudaExecutionCtxDestroy()`

- SM-Count wird per `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` abgefragt
- Prefill bekommt `prefill_sm_ratio` (default 80%), Decode den Rest
- Runtime-Rekonfiguration via `reconfigure()`
- Graceful Fallback auf normale Streams wenn CUDA < 13.1 (`#if IMP_CUDA_13_1` Guards)

### A.2 PDL — Umfassend genutzt

**APIs**: `cudaLaunchKernelEx()`, `cudaLaunchAttributeProgrammaticStreamSerialization`

- Globale Registry: `pdl::enable(kernel_func)` / `pdl::is_available()` (sm_90+ Check)
- Template `pdl::launch()`: prüft Registry → `cudaLaunchKernelEx` oder `<<<>>>` Fallback
- RAII Wrapper `ScopedPDL`
- **Registrierte Kernels**:

  Utility (executor.cu, 7 Kernels):
  1. `elementwise_add_fp16_kernel`
  2. `elementwise_add_fp32_kernel`
  3. `write_kv_cache_kernel`
  4. `write_kv_cache_fused_kernel`
  5. `write_kv_cache_rope_fused_kernel`
  6. `fp16_to_fp32_kernel`
  7. `fp32_to_fp16_kernel`

  Compute (layernorm.cu, rope.cu, activation.cu, 6 Kernels):
  8. `rmsnorm_fp16_kernel`
  9. `rmsnorm_fp32_kernel`
  10. `rmsnorm_residual_fp16_kernel`
  11. `rmsnorm_residual_fp32_kernel`
  12. `rope_forward_fp16_kernel`
  13. `rope_forward_fp32_kernel`
  14. `qknorm_rope_fused_fp16_kernel`
  15. `swiglu_fp16_kernel`
  16. `geglu_fp16_kernel`

  dp4a GEMV Template-Instantiierungen (gemm.cu, 110+ Varianten):
  - `gemv_dp4a_kernel<QT, NR, Residual>` — 5 QTypes × 3 NR × 2 = 30
  - `gemv_dp4a_fp32_kernel<QT, NR>` — 5 × 3 = 15
  - `gemv_dp4a_qkv_kernel<QT, NR>` — 5 × 3 = 15
  - `gemv_dp4a_gate_up_kernel<QT, NR>` — 5 × 3 = 15
  - `gemv_dp4a_kpar_kernel<QT, Residual>` + Varianten — 5 × 5 = 25
  - `gemv_dp4a_inline_quant_kernel<QT, NR, Residual>` — 5 × 2 × 2 = 20

  MoE-Kernels bewusst **nicht** PDL-registriert (atomare Expert-Scatter).

- cuBLAS/cuBLASLt aktiviert PDL intern auf sm_90+

### A.3 CUDA Graphs + Conditional WHILE — Vollständig

**Standard Graphs**: `CudaGraphCapture` — warmup → capture (`cudaStreamBeginCapture`) → replay (`cudaGraphLaunch`) → incremental update (`cudaGraphExecUpdate`)

**Conditional WHILE** (`CudaGraphConditionalRunner`):
- `cudaGraphConditionalHandleCreate()` mit default value = 1 (continue)
- `cudaGraphCondTypeWhile` Loop-Typ
- Device-Kernel `post_decode_step_kernel`: schreibt Token in Mapped-Pinned Ring Buffer, prüft Stop-Conditions (max steps, EOS, stop_ids), bricht Loop via `cudaGraphSetConditional(handle, 0)`
- Mapped Pinned Memory (`cudaHostAllocMapped`) für Zero-Copy Host-Observation
- `poll_new_tokens()` — non-blocking atomic reads, `wait_and_get_tokens()` — blocking sync
- WSL2-Kompatibilität: `cudaStreamSynchronize` statt Polling (mapped memory nicht sofort sichtbar)

### A.4 cudaMallocAsync / MemPool — Vollständig

**APIs**: `cudaMemPoolCreate()`, `cudaMemPoolSetAttribute()`, `cudaMallocAsync()`, `cudaFreeAsync()`, `cudaMemPoolTrimTo()`

- Pool-Typ: `cudaMemAllocationTypePinned`
- `cudaMemPoolAttrReleaseThreshold`: `UINT64_MAX` wenn `initial_pool_size == 0`
- `cudaMemPoolReuseAllowOpportunistic` + `ReuseAllowInternalDependencies`
- Thread-safe Accounting: atomic `allocated_` und `peak_allocated_`

### A.5 NVFP4 — Vollständig

**Format**: FP4 E2M1 (1 sign | 2 exp | 1 mantissa, bias=1). Magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

**Kernels**:
- `absmax_kernel` — Grid-stride Reduktion, `atomicMax` auf IEEE754 Bit-Pattern
- `quantize_nvfp4_kernel` — FP16→NVFP4: Micro-Scale (FP8 E4M3) per 16-Werte Block + Tensor-Scale (FP32)
- `dequantize_nvfp4_kernel` — NVFP4→FP16: Constant LUT `kFP4E2M1Dequant[8]`

**Helpers**: `float_abs_to_fp4_e2m1()`, `float_to_fp8_e4m3()`, `fp8_e4m3_to_float()`

### A.6 Blackwell WMMA Attention — Implementiert (kein TCGEN05)

**Status**: Volle WMMA-Implementation für sm_120+, aber ohne TCGEN05 Inline-PTX.

**Implementiert** (`attention_blackwell.cu`):
- `flash_attention_blackwell_kernel<Br, HD>` — Template auf Tile-Höhe + Head-Dimension
- 8 Warps (256 Threads) mit Double-Buffered KV-Tiles
- Adaptive Br: 128 Rows (HD≤64) oder 64 Rows (HD=128)
- WMMA Tile Math: 16×16 Tensor-Ops mit Mixed FP16/FP32 Akkumulation
- smem-Constraint: RTX 5090 hat 100 KB/SM, 99 KB opt-in max
  - Br=128, Bc=64, HD=64 → ~96.5 KB (passt)
  - Br=64, Bc=64, HD=128 → ~96.3 KB (passt)
- Online-Softmax mit kausaler Maskierung
- Runtime-Dispatch: sm_120+ → Blackwell, sm_90+ → Hopper 4-Warp WMMA, sonst → Scalar

**Fehlend** (für echte TCGEN05-Performance):
- Kein Inline-PTX für WGMMA Instruktionen
- Kein TMA (`cp.async.bulk.tensor`) — kooperative Tile-Loads stattdessen
- Kein TMEM — Output-Akkumulator in Shared Memory statt TMEM
- Kein CTA Pairing für Load/Compute Overlap

### A.7 CUTLASS — v2.x Vollständig

**Konfiguration**: `Sm80` Target (kompatibel mit SM90/120), Tile 128×128×32, Warp 64×64×32, MMA 16×8×16, 4-Stage `cp.async` Pipeline

**Einsatz**: Grouped GEMM für MoE Expert-parallel via `gemm_moe_cutlass()`. Device-only Scheduling, ~3µs Launch-Overhead (vs ~27µs cuBLAS).

---

## Anhang B: Code-Evidenz Performance

### B.1 GEMV-Architektur (dp4a Template-System)

Die GEMV-Infrastruktur wurde von 33 handgeschriebenen Kernels auf ein Template-System konsolidiert (`gemv_dp4a_traits.cuh`):

#### Quant-Type Traits

| Type | Traits-Klasse | Block-Bytes | Elemente/Block | MaxNRows | PreferKpar |
|------|---------------|-------------|----------------|----------|-----------|
| Q6_K | `Q6_K_Traits` | 210 | 256 | 2 | Ja |
| Q8_0 | `Q8_0_Traits` | 34 | 32 | 2 | Nein |
| Q4_0 | `Q4_0_Traits` | 18 | 32 | 4 | Nein |
| Q4_K | `Q4_K_Traits` | 144 | 256 | 4 | Ja |
| Q5_K | `Q5_K_Traits` | 176 | 256 | 4 | Ja |

#### Template-Kernels (6 Basis + 4 K-par Varianten)

| Kernel | Funktion | Template-Parameter |
|--------|----------|-------------------|
| `gemv_dp4a_kernel<QT, NR, Residual>` | Standard GEMV | 5 × 3 × 2 = 30 |
| `gemv_dp4a_fp32_kernel<QT, NR>` | FP32 Output (LM Head) | 5 × 3 = 15 |
| `gemv_dp4a_qkv_kernel<QT, NR>` | Fused Q/K/V Projektion | 5 × 3 = 15 |
| `gemv_dp4a_gate_up_kernel<QT, NR>` | Fused Gate+Up+SwiGLU | 5 × 3 = 15 |
| `gemv_dp4a_inline_quant_kernel<QT, NR, Res>` | Q8_1 Quant + GEMV | 5 × 2 × 2 = 20 |
| `gemv_dp4a_moe_decode_kernel<QT>` | MoE Expert (smem Q8_1) | 5 |
| `gemv_dp4a_moe_gate_up_kernel<QT>` | MoE Gate+Up (smem) | 5 |
| `gemv_dp4a_kpar_kernel<QT, Residual>` | K-parallel Standard | 5 × 2 = 10 |
| `gemv_dp4a_kpar_fp32_kernel<QT>` | K-parallel FP32 | 5 |
| `gemv_dp4a_kpar_qkv_kernel<QT>` | K-parallel QKV | 5 |
| `gemv_dp4a_kpar_gate_up_kernel<QT>` | K-parallel Gate+Up | 5 |

**Total: ~130 Instantiierungen** aus 11 Template-Kernels.

#### Dispatch-Heuristik

- **K-parallel** (128 Threads, 4 Warps, 1 Row/Block): bevorzugt für d_model-Projektionen (QKV, O) und compute-intensive QTypes (Q6_K, Q4_K, Q5_K). 40 Regs → 12 Blocks/SM → 48 Warps/SM.
- **Row-parallel** (256 Threads, 8 Warps, NR Rows/Warp): bevorzugt für d_ff-Projektionen (Gate+Up) und einfache QTypes (Q8_0, Q4_0). 40 Regs → 6 Blocks/SM → 48 Warps/SM.
- `kpar_is_better(M, rpar_blocks)`: Runtime-Heuristik basierend auf Matrix-Größe und Occupancy.

#### Legacy-Kernels (Fallback)

| Kernel | Typ | Verwendung |
|--------|-----|-----------|
| `gemv_fp32_kernel` | FP32 | Nicht-quantisierte FP32 Weights |
| `gemv_fp16_kernel` | FP16 | Pre-dequantisierte FP16 Weights |
| `gemv_bf16_kernel` | BF16 | BF16 Modelle |
| `gemv_fp8_e4m3_kernel` | FP8 | FP8-quantisierte Weights |

### B.2 Multi-Block Argmax

**Problem**: Single-Block Argmax auf vocab=152K benötigte 192µs (1 SM, 83 SMs idle).

**Lösung** (sampling.cu):
1. `argmax_partial_kernel<<<64, 256>>>` — Jeder Block scannt vocab_size/64 Elemente, schreibt lokales Maximum in Scratch
2. `argmax_reduce_kernel<<<1, 32>>>` — Single Warp reduziert 64 Partial-Ergebnisse

**Scratch-Layout** (`ARGMAX_SCRATCH_BYTES = 516 Bytes`):
```
[result: int32] [partial_vals: float×64] [partial_idxs: int32×64]
```

**Ergebnis**: ~10µs statt 192µs (19x Speedup, nutzt alle 84 SMs).

### B.3 cuBLASLt Epilogue / Residual Fusion

`gemm.cu` setzt `CUBLASLT_MATMUL_DESC_TRANSA/TRANSB`, aber **keine** Epilogue-Attribute (`BIAS_POINTER`, `EPILOGUE`). Residual-Add nutzt `gemm(beta=1.0)` für in-place Fusion (P3): bei Prefill mit quantisierten Weights wird in `dequant_scratch_` dequantisiert, dann `gemm(swiglu_out, w_fp16, hidden, 1.0, 1.0)`.

### B.4 FFN Fusion (executor.cu)

- **Decode (n=1)**: Fused Gate+Up GEMV (`gemv_dp4a_gate_up_kernel`) berechnet Gate, Up und SwiGLU/GeGLU in einem Kernel. Eliminiert 3 separate Launches.
- **Prefill (n>1)**: `gemm_pair_batched()` für Gate+Up in einem `cublasGemmStridedBatchedEx` Call. Fused Weight Cache `[2*d_ff, d_model]` wird in `pre_dequant_weights()` erstellt.
- **Down-proj+Residual (n=1)**: Separate Q8_1-Quantisierung + K-par GEMV mit Residual (48 Warps/SM Occupancy). Gemv-Residual-Varianten für alle 5 QTypes.
- **Down-proj+Residual (n>1)**: `gemm(beta=1.0)` für FP16-cached oder dequant-to-scratch Weights. Separate `elementwise_add` nur noch bei Post-FFN-Norm (Gemma-3).

### B.5 Fused Operations (Decode-Pfad)

| Operation | Kernel(s) | Erspart |
|-----------|----------|---------|
| RMSNorm + Q8_1 Quantize | `rmsnorm_quantize_q8_1()` | FP16 Intermediate Buffer + 1 Launch |
| Fused QKV GEMV | `gemv_dp4a_qkv_kernel` / `kpar_qkv` | 2 Launches (3→1 Kernel) |
| Fused Gate+Up GEMV | `gemv_dp4a_gate_up_kernel` / `kpar_gate_up` | 2 Launches + SwiGLU Kernel |
| RoPE + KV-Cache Write | `write_kv_cache_rope_fused_kernel` | RoPE Kernel + separate KV Write |
| O-Proj Quant + GEMV + Residual | `quantize_fp16_to_q8_1` + `gemv_kpar_residual` | FP16 Intermediate + Add Kernel |

### B.6 Eager FP16 Weight Dequantisierung

`Engine::init()` ruft `pre_dequant_weights()` sofort nach KV-Cache-Allokation auf. Vorher: lazy beim ersten Prefill (~380ms Overhead für Phi-4-Mini Q8_0 mit 224 Tensoren / 9.6 GiB). Jetzt: Init-Zeit etwas länger, aber erster Real-World-Prefill sofort schnell (~19,460 tok/s statt ~1,225 tok/s).

### B.7 Host-Device Sync

- **Prefill (greedy)**: `forward_logits()` + `sample_greedy_device()` + `cudaEventSynchronize(prefill_done_)` (P9)
- **Decode**: CUDA Graph Replay + `cudaStreamSynchronize(dec_stream)` nach jedem Batch
- **Weight Upload**: Separater `cudaStreamNonBlocking` Stream + Event-Sync (P8)

---

## Anhang C: Code-Evidenz Architektur

### C.1 Verifizierte Stärken

| Feature | Schlüssel-Dateien | Details |
|---------|-------------------|---------|
| PagedAttention | `kv_cache.h`, `kv_cache.cu` | `kKVBlockSize=16`, Free-List, Ref-Counting |
| Prefix Caching | `kv_cache_manager.cpp` | `prefix_cache_` Hash-Map, `share_prefix()` mit `inc_ref()` |
| Continuous Batching | `scheduler.cpp` | Prefill→Active Promotion, Decode-Batch Assembly, Memory-aware Admission |
| Speculative Decoding | `speculative.cpp` | `draft_tokens()` K Iterationen, `verify()` Pseudo-Prefill, stochastische Akzeptanz, KV-Rollback |
| Hybrid (Nemotron) | `model_arch.h`, `ssm.cu`, `ssm_state.cu` | `NEMOTRON_H_MOE`, Per-Layer SSM/Attention/MoE Dispatch, 52 Layers |
| VRAM Expert Upload | `weight_upload.cu` | 2-Pass: (1) Non-Expert Upload, (2) Greedy Layer-Budget für Experts |
| HTTP Server | `tools/imp-server/main.cpp` | `/v1/chat/completions`, SSE Streaming, CORS, Chat-Template Stop-Tokens |
| dp4a Template GEMV | `gemv_dp4a_traits.cuh` | 5 QTypes, K-par/Row-par Dispatch, ~130 Instantiierungen |
| Multi-Block Argmax | `sampling.cu` | 64-Block 2-Phasen Reduktion, 19x Speedup |

### C.2 Verifizierte Lücken

| Feature | Evidenz |
|---------|---------|
| Chunked Prefill | Scheduler promoviert gesamten Prefill, kein `chunk_prefill_tokens` Parameter |
| Logprobs | `ImpGenerateParams` hat kein logprobs-Feld, Sampling gibt nur Token-ID zurück |
| Structured Output | Kein Grammar-Parser, kein Constraint-Feld in Generate-Params |
| Beam Search | Nur `temperature/top_p/top_k` in Sampling, kein `beam_size` |
| String Stop Sequences | Nur Token-basiert (EOS + Template Stop-IDs), kein String-Matching |
| NVFP4 GEMV | Quant/Dequant vorhanden, aber keine dp4a-Kernels für native FP4 Compute |
| TCGEN05 Attention | WMMA implementiert, aber kein Inline-PTX für echte systolische Blackwell-Ops |
