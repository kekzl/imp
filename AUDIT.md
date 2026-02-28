# imp — CUDA 13.1 Feature Audit, Performance & Architektur-Review

**Datum:** 2026-02-28 (updated after P2-P9 optimizations)

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
| **PDL** (Programmatic Dependent Launch) | `src/runtime/pdl.cu` | Infrastruktur vorhanden (`cudaLaunchKernelEx` + Registry). **12 Kernels registriert**: 6 Utility (elementwise_add, write_kv_cache, type-cast) + 6 Compute (RMSNorm FP16/FP32, RMSNorm-Residual FP16/FP32, RoPE FP16/FP32, QKNorm-RoPE-fused, SwiGLU, GeGLU). cuBLAS aktiviert PDL intern auf sm_90+. |
| **CUDA Graphs** (Conditional WHILE) | `src/runtime/cuda_graph.cu` | Capture/Replay, `cudaGraphConditionalHandleCreate` mit WHILE-Loop, Mapped-Memory Ring Buffer |
| **cudaMallocAsync / MemPool** | `src/memory/device_allocator.cu` | `cudaMemPoolCreate` mit `ReuseAllowOpportunistic`, stream-geordnete Allokation |
| **NVFP4** (FP4 E2M1) | `src/quant/nvfp4_quant.cu`, `nvfp4_gemm.cu` | 2-Level Quantisierung (Micro-Scale FP8 + Tensor-Scale FP32), Blackwell-nativ |

### Scaffolded / unvollständig

| Feature | Dateien | Status |
|---------|---------|--------|
| **TCGEN05 Systolic** (Blackwell Attention) | `src/compute/attention_blackwell.cu` | Tile-Sizes definiert (128x128), Kernel-Skeleton vorhanden, aber **kein Inline-PTX** — fällt auf skalare Referenzimpl. zurück |
| **TMA** (Tensor Memory Accelerator) | `src/compute/attention_blackwell.cu` | Kommentare referenzieren `cp.async.bulk.tensor`, aber keine tatsächliche Implementation |
| **CUTLASS 3.x** | `src/compute/gemm_cutlass.cu` | Nutzt CUTLASS 2.x (`cp.async`, kein TMA), kein Upgrade auf 3.x Hopper/Blackwell Primitives |

### Nicht implementiert (Potential vorhanden)

| Feature | Nutzen | Priorität |
|---------|--------|-----------|
| **Stream-Attribute** (`cudaStreamSetAttribute`) | Priorität für Prefill- vs Decode-Streams | Niedrig |
| **Conditional IF Nodes** in CUDA Graphs | Komplexere GPU-autonome Control-Flow (Early-Exit, Branch) | Niedrig |
| **Cluster Launch** (`cudaLaunchConfig_t.clusterDim`) | SM-Cluster für tiled MoE Scatter | Niedrig |
| **Multi-GPU** (P2P, IPC, NCCL) | Tensor/Pipeline-Parallelismus | N/A (1 GPU) |

---

## 2. Performance-Optimierungen

### Offen

| # | Optimierung | Dateien | Geschätzter Gewinn | Aufwand |
|---|-------------|---------|-------------------|---------|
| P1 | **TCGEN05 Attention fertigstellen** — Inline-PTX für WGMMA + TMA Bulk-Loads + TMEM Accumulators | `attention_blackwell.cu` | 2x auf sm_120 | Hoch |

### Implementiert (e2cf896, 2026-02-28)

| # | Optimierung | Dateien | Gemessener Gewinn |
|---|-------------|---------|-------------------|
| P2 | **Q4_0 dp4a GEMV** — Dedizierte dp4a-GEMV für Q4_0: standard, residual-fused, FP32 (LM head), fused QKV. Raw Q4_0 Upload für Decode-Path. | `gemm.cu`, `gemm.h`, `weight_upload.cu`, `executor.cu` | Kein Q4_0-Modell zum Messen verfügbar |
| P3 | **Down-Projection Residual Fusion** — Dequant in Scratch + `gemm(beta=1.0)` eliminiert separaten `elementwise_add` Kernel bei Prefill | `executor.cu` | +3-84% Prefill (modellabhängig) |
| P4 | **Gate+Up Batched GEMM** — `cublasGemmStridedBatchedEx` für Gate+Up in einem cuBLAS-Call. Fused Weight Cache `[2*d_ff, d_model]` | `gemm.cu`, `gemm.h`, `executor.cu`, `executor.h` | +3-84% Prefill (zusammen mit P3) |
| P5 | **PDL-Registry erweitert** — 6 Compute-Kernels registriert: RMSNorm (FP16/FP32 + Residual), RoPE (FP16/FP32), QKNorm-RoPE-fused, SwiGLU, GeGLU. Launches via `pdl::launch()` | `layernorm.cu/.h`, `rope.cu/.h`, `activation.cu/.h`, `executor.cu` | In CUDA-Graph-Capture sichtbar (389-437 PDL edges) |
| P6 | **KV-Cache Double-Buffered Prefetch** — GQA Paged Attention nutzt FP16 Double-Buffer Shared Memory (gleiche smem-Größe wie FP32 Single-Buffer). Overlap next-block load mit current-block compute | `attention_paged.cu` | ~0% Decode (Attention nicht bottleneck bei Q8_0) |
| P7 | **Request-Reordering** — `std::deque` + `std::sort` nach aufsteigender Token-Anzahl (shortest-first) vor Admission-Loop | `scheduler.cpp`, `scheduler.h` | Nur bei Multi-Request Batching messbar |
| P8 | **Async Weight Upload** — Separater `cudaStreamNonBlocking` Upload-Stream + Event-Sync vor erstem Weight-Zugriff | `engine.cpp` | Init-Time Overlap (nicht in tok/s messbar) |
| P9 | **Event-basierter Prefill Sync** — `forward_logits()` + `sample_greedy_device()` + `cudaEventSynchronize` statt `cudaStreamSynchronize`. `cudaFreeAsync` vor Event-Sync für Overlap | `engine.cpp`, `engine.h` | +3-84% Prefill (zusammen mit P3/P4) |

### Benchmark: Vorher/Nachher (a9ae9b4 → e2cf896)

| Model | Quant | pp tok/s (alt) | pp tok/s (neu) | tg tok/s (alt) | tg tok/s (neu) |
|-------|-------|----------------|----------------|----------------|----------------|
| Phi-4-Mini-Instruct | Q8_0 | 57.69 | 105.95 (+84%) | 189.77 | 189.84 (+0%) |
| Qwen3-4B-Instruct | Q8_0 | 130.85 | 165.65 (+27%) | 186.06 | 183.75 (-1%) |
| DeepSeek-R1-Distill-Qwen-7B | Q8_0 | 107.22 | 110.60 (+3%) | 133.52 | 133.15 (0%) |

Decode flat (CUDA Graphs eliminieren Launch-Overhead). Prefill-Gewinn durch P3/P4 (batched GEMM, fused residual).

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

1. **Stop Sequences + Logprobs** — Niedrig-Aufwand, hoher User-Impact, Server-Kompatibilität
2. **Chunked Prefill** — Verhindert Head-of-Line-Blocking bei langen Prompts
3. **Batch API** — Externe Batch-Endpoint für Throughput-Workloads
4. **Structured Output** — JSON-Mode / GBNF-Grammatik für Agentic Use-Cases
5. **TCGEN05 Blackwell** — Hardware-Level Performance auf RTX 5090

---

## Zusammenfassung

**CUDA 13.1**: 5/5 Major-Features implementiert (Green Contexts, PDL, Graphs+Conditional, MemPool, NVFP4). PDL-Infrastruktur steht, aber nur 6 Utility-Kernels registriert — Kern-Compute profitiert via cuBLAS-internes PDL. Blackwell TCGEN05/TMA sind scaffolded aber nicht funktional — größtes ungenutztes Hardware-Potential.

**Performance**: P2-P9 implementiert. Prefill +3-84% (Gate+Up Batched GEMM, Down-Proj Residual Fusion, Event-Sync). Decode unverändert (CUDA Graphs dominieren). Verbleibend: TCGEN05-Attention (2x sm_120), GEMV für Q4_K_M/INT-Typen. GEMV-Coverage jetzt: FP32/FP16/BF16/FP8/Q6_K/Q8_0/Q4_0.

**Architektur**: Auf Single-GPU-Ebene vergleichbar mit vLLM. Hauptlücken sind Multi-GPU, Structured Output, Vision, und externe Batch-API. Für den aktuellen Scope (Single-GPU, CLI/Server) ist die Architektur solide und modern.

---

## Anhang A: Code-Evidenz CUDA 13.1 Features

### A.1 Green Contexts — Vollständig

**APIs**: `cudaDeviceGetDevResource()`, `cudaDevSmResourceSplitByCount()`, `cudaDevResourceGenerateDesc()`, `cudaGreenCtxCreate()`, `cudaExecutionCtxStreamCreate()`, `cudaExecutionCtxDestroy()`

- SM-Count wird per `cudaDeviceGetAttribute(cudaDevAttrMultiProcessorCount)` abgefragt
- Prefill bekommt `prefill_sm_ratio` (default 80%), Decode den Rest
- Runtime-Rekonfiguration via `reconfigure()`
- Graceful Fallback auf normale Streams wenn CUDA < 13.1 (`#if IMP_CUDA_13_1` Guards)

### A.2 PDL — Infrastruktur vorhanden, partiell genutzt

**APIs**: `cudaLaunchKernelEx()`, `cudaLaunchAttributeProgrammaticStreamSerialization`

- Globale Registry: `pdl::enable(kernel_func)` / `pdl::is_available()` (sm_90+ Check)
- Template `pdl::launch()`: prüft Registry → `cudaLaunchKernelEx` oder `<<<>>>` Fallback
- RAII Wrapper `ScopedPDL`
- **Registrierte Kernels** (executor.cu + compute modules):
  Utility (executor.cu):
  1. `elementwise_add_fp16_kernel`
  2. `elementwise_add_fp32_kernel`
  3. `write_kv_cache_kernel`
  4. `write_kv_cache_fused_kernel`
  5. `fp16_to_fp32_kernel`
  6. `fp32_to_fp16_kernel`
  Compute (P5 — layernorm.cu, rope.cu, activation.cu):
  7. `rmsnorm_fp16_kernel`
  8. `rmsnorm_fp32_kernel`
  9. `rmsnorm_residual_fp16_kernel`
  10. `rmsnorm_residual_fp32_kernel`
  11. `rope_forward_fp16_kernel`
  12. `rope_forward_fp32_kernel`
  13. `qknorm_rope_fused_fp16_kernel`
  14. `swiglu_fp16_kernel`
  15. `geglu_fp16_kernel`
- cuBLAS/cuBLASLt aktiviert PDL intern auf sm_90+

### A.3 CUDA Graphs + Conditional WHILE — Vollständig

**Standard Graphs**: `CudaGraphCapture` — warmup → capture (`cudaStreamBeginCapture`) → replay (`cudaGraphLaunch`) → incremental update (`cudaGraphExecUpdate`)

**Conditional WHILE** (`CudaGraphConditionalRunner`):
- `cudaGraphConditionalHandleCreate()` mit default value = 1 (continue)
- `cudaGraphCondTypeWhile` Loop-Typ
- Device-Kernel `post_decode_step_kernel`: schreibt Token in Mapped-Pinned Ring Buffer, prüft Stop-Conditions (max steps, EOS, stop_ids), bricht Loop via `cudaGraphSetConditional(handle, 0)`
- Mapped Pinned Memory (`cudaHostAllocMapped`) für Zero-Copy Host-Observation
- `poll_new_tokens()` — non-blocking atomic reads, `wait_and_get_tokens()` — blocking sync
- WSL2-Kompatibilität: atomic acquire/release Semantik für GPU→Host Sichtbarkeit

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

### A.6 TCGEN05 Blackwell Attention — Scaffold

**Vorhanden**: Kernel-Signatur für sm_120+, Shared-Memory Layout (Q/KV Tiles 128×128), Online-Softmax State, kausale Maskierung

**Fehlend**:
- Kein Inline-PTX für WGMMA Instruktionen
- Kein TMA (`cp.async.bulk.tensor`) — skalare kooperative Tile-Loads
- Kein TMEM — Output-Akkumulator in Shared Memory statt TMEM
- Kein CTA Pairing für Load/Compute Overlap
- Fallback auf Hopper WMMA (`flash_attention_prefill_tc`) für sm_90/100/120

### A.7 CUTLASS — v2.x Vollständig

**Konfiguration**: `Sm80` Target (kompatibel mit SM90/120), Tile 128×128×32, Warp 64×64×32, MMA 16×8×16, 4-Stage `cp.async` Pipeline

**Einsatz**: Grouped GEMM für MoE Expert-parallel via `gemm_moe_cutlass()`. Device-only Scheduling, ~3µs Launch-Overhead (vs ~27µs cuBLAS).

---

## Anhang B: Code-Evidenz Performance

### B.1 GEMV-Coverage

Dedizierte GEMV-Kernels in `src/compute/gemm.cu`:

| Kernel | Quant-Typ | Zeile |
|--------|-----------|-------|
| `gemv_fp32_kernel` | FP32 | ~358 |
| `gemv_fp16_kernel` | FP16 | ~402 |
| `gemv_bf16_kernel` | BF16 | ~508 |
| `gemv_fp8_e4m3_kernel` | FP8 E4M3 | ~711 |
| `gemv_q6k_kernel` | Q6_K | ~813 |
| `gemv_q8_0_kernel` | Q8_0 | ~890 |
| `gemv_q6k_q8_1_kernel` | Q6_K + Q8_1 aktiviert | ~1518 |
| `gemv_q8_0_q8_1_kernel` | Q8_0 + Q8_1 aktiviert | ~1769 |
| `gemv_qkv_fused_q6k_q8_1_kernel` | QKV-fused Q6_K | ~1880 |
| `gemv_qkv_fused_q8_0_q8_1_kernel` | QKV-fused Q8_0 | ~1993 |
| `gemv_q4_0_q8_1_kernel` | Q4_0 + Q8_1 dp4a | ~2107 |
| `gemv_q4_0_q8_1_fp32_kernel` | Q4_0 FP32 output (LM head) | ~2230 |
| `gemv_qkv_fused_q4_0_q8_1_kernel` | QKV-fused Q4_0 | ~2300 |
| MoE Gate/Decode Varianten | Diverse | ~935+ |

**Fehlend**: Q4_K_M, INT4, INT8 → fallen auf `gemm_dispatch()` (cuBLAS) zurück.

### B.2 cuBLASLt Epilogue / Residual Fusion

`gemm.cu` setzt `CUBLASLT_MATMUL_DESC_TRANSA/TRANSB`, aber **keine** Epilogue-Attribute (`BIAS_POINTER`, `EPILOGUE`). Residual-Add nutzt stattdessen `gemm(beta=1.0)` für in-place Fusion (P3): bei Prefill mit quantisierten Weights wird in `dequant_scratch_` dequantisiert, dann `gemm(swiglu_out, w_fp16, hidden, 1.0, 1.0)` — eliminiert separaten `elementwise_add` Kernel.

### B.3 FFN Fusion (executor.cu)

- **Decode (n=1)**: Gate/Up nutzen pre-quantisierte Q8_1 Buffers → separate `geglu()` / `swiglu()` Kernel
- **Prefill (n>1)**: `gemm_pair_batched()` für Gate+Up in einem `cublasGemmStridedBatchedEx` Call (P4). Fused Weight Cache `[2*d_ff, d_model]` wird in `pre_dequant_weights()` erstellt. Fallback auf separate `gemm_dispatch()` wenn kein Cache
- **Down-proj+Residual (n=1)**: Fusioniert via `gemv_q6k_q8_1_residual()` / `gemv_q8_0_q8_1_residual()` / `gemv_q4_0_q8_1_residual()`
- **Down-proj+Residual (n>1)**: `gemm(beta=1.0)` für FP16-cached oder dequant-to-scratch Weights (P3). Separate `elementwise_add` nur noch bei Post-FFN-Norm (Gemma-3)

### B.4 Scheduler

`scheduler.h`: `std::deque<std::shared_ptr<Request>> pending_` — shortest-first Sortierung (P7). `std::sort` nach aufsteigender `input_tokens.size()` vor Admission-Loop. Memory-aware Admission via `kv_manager_->can_allocate()`.

### B.5 Host-Device Sync

- **Prefill (greedy)**: `forward_logits()` + `sample_greedy_device()` + `cudaEventSynchronize(prefill_done_)` (P9). `cudaFreeAsync` calls vor Event-Sync für Overlap. Fallback auf `cudaStreamSynchronize` für non-greedy Sampling
- **Decode**: `cudaStreamSynchronize(dec_stream)` nach jedem Decode-Batch (unverändert)
- **Weight Upload**: Separater `cudaStreamNonBlocking` Stream + Event-Sync (P8)

---

## Anhang C: Code-Evidenz Architektur

### C.1 Verifizierte Stärken

| Feature | Schlüssel-Dateien | Details |
|---------|-------------------|---------|
| PagedAttention | `kv_cache.h:10`, `kv_cache.cu:36` | `kKVBlockSize=16`, Free-List, Ref-Counting |
| Prefix Caching | `kv_cache_manager.cpp:132-168` | `std::unordered_map<size_t, vector<int>> prefix_cache_`, `share_prefix()` mit `inc_ref()` |
| Continuous Batching | `scheduler.cpp:29-61` | Prefill→Active Promotion, Decode-Batch Assembly, Memory-aware Admission |
| Speculative Decoding | `speculative.cpp:65-470` | `draft_tokens()` K Iterationen, `verify()` Pseudo-Prefill, stochastische Akzeptanz, KV-Rollback |
| Hybrid (Nemotron) | `model_arch.h:12`, `model.h:39-47`, `ssm.cu`, `ssm_state.cu` | `NEMOTRON_H_MOE`, Per-Layer SSM/Attention/MoE Dispatch |
| VRAM Expert Upload | `weight_upload.cu:593-857` | 2-Pass: (1) Non-Expert Upload, (2) Greedy Layer-Budget für Experts |
| HTTP Server | `tools/imp-server/main.cpp` | `/v1/chat/completions`, SSE Streaming, CORS, Chat-Template Stop-Tokens |

### C.2 Verifizierte Lücken

| Feature | Evidenz |
|---------|---------|
| Chunked Prefill | Scheduler promoviert gesamten Prefill, kein `chunk_prefill_tokens` Parameter |
| Logprobs | `ImpGenerateParams` hat kein logprobs-Feld, Sampling gibt nur Token-ID zurück |
| Structured Output | Kein Grammar-Parser, kein Constraint-Feld in Generate-Params |
| Beam Search | Nur `temperature/top_p/top_k` in Sampling, kein `beam_size` |
| String Stop Sequences | Nur Token-basiert (EOS + Template Stop-IDs), kein String-Matching |
