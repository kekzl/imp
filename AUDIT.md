# imp — CUDA 13.1 Feature Audit, Performance & Architektur-Review

**Datum:** 2026-03-02 (aktualisiert nach Stop Sequences, Logprobs, JSON Mode, CUTLASS FMHA, NVFP4-Decode-Cache, Decode-Attention-Optimierungen)

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
| **NVFP4** (FP4 E2M1) | `src/quant/nvfp4_quant.cu`, `nvfp4_gemm.cu` | 2-Level Quantisierung (Micro-Scale FP8 + Tensor-Scale FP32), Blackwell-nativ. **NVFP4 Decode Weight Cache**: Init-Time FP16→NVFP4 Quantisierung, K-parallel dp4a GEMV für Decode. `--decode-nvfp4` / `--decode-nvfp4-only` Flags. |

### Teilweise implementiert

| Feature | Dateien | Status |
|---------|---------|--------|
| **Blackwell WMMA Attention** | `src/compute/attention_blackwell.cu` | 8-Warp WMMA mit Double-Buffered KV, Adaptive Br (128/64), sm_120-optimiert. **Kein Inline-PTX** (WGMMA/TCGEN05), kein TMA — nutzt WMMA Intrinsics. |
| **FP8 Prefill Weight Cache** | `src/graph/executor.cu`, `src/quant/fp8_quant.cu` | Hybrid FP16+FP8: FP16-Cache zuerst (inkl. Fused KV/Gate+Up), FP8 E4M3 für Overflow-Weights. `--prefill-fp8` Flag. Per-Tensor Device-Scales, cuBLASLt FP8×FP8→FP16 GEMM. |
| **CUTLASS 4.4.1 FMHA** | `src/compute/gemm_cutlass.cu`, `attention_cutlass_fmha.cu` | MoE GEMM: CUTLASS 2.x API (cp.async). **Prefill Attention**: CUTLASS Hopper FMHA (WGMMA + TMA, Example 88) auf sm_90+, Fallback auf WMMA wenn deaktiviert (`IMP_NO_CUTLASS_FMHA=1`). |

### Nicht implementiert (Potential vorhanden)

| Feature | Nutzen | Priorität |
|---------|--------|-----------|
| **TCGEN05 Inline-PTX** (WGMMA + TMA + TMEM) | Prefill: bereits via CUTLASS FMHA. Decode: WGMMA erfordert M≥64, Decode hat M=1 — nicht anwendbar. Verbleibender Nutzen nur für Custom-Prefill-Kernels ohne CUTLASS. | Niedrig |
| **TMA** (Tensor Memory Accelerator) | Prefill: bereits via CUTLASS FMHA. Decode: cp.async reicht für M=1 Vektor-Loads (Phase 7). TMA-Bulk-Loads lohnen sich erst bei größeren Tiles. | Niedrig |
| **Stream-Attribute** (`cudaStreamSetAttribute`) | Priorität für Prefill- vs Decode-Streams | Niedrig |
| **Conditional IF Nodes** in CUDA Graphs | Komplexere GPU-autonome Control-Flow (Early-Exit, Branch) | Niedrig |
| **Multi-GPU** (P2P, IPC, NCCL) | Tensor/Pipeline-Parallelismus | N/A (1 GPU) |

---

## 2. Performance-Optimierungen

### Offen

| # | Optimierung | Dateien | Geschätzter Gewinn | Aufwand |
|---|-------------|---------|-------------------|---------|
| O1 | **TCGEN05 Inline-PTX Decode Attention** — WGMMA + TMA für Paged Attention Decode. **Hinweis**: WGMMA erfordert M≥64, Decode hat M=1 — nicht direkt anwendbar auf Decode-Vektoren. Nutzen primär für Prefill (bereits via CUTLASS FMHA). Decode-Attention wurde stattdessen über cp.async Pipelining + Vektorisierung optimiert (Phase 7). | `attention_blackwell.cu` | ~15-30% Decode-Attention Kernel (nicht ~2x — M=1 inkompatibel mit WGMMA) | Niedrig |
| O3 | **GEMV Bandwidth-Optimierung** — Aktuelle Auslastung ~50-59% vs llama.cpp ~55-69%. L2-Cache-Tuning, Prefetch-Hints | `gemv_dp4a_traits.cuh` | +5-10% Dense Decode | Mittel |

### Implementiert — Phase 6: Server-Features (2026-03-02)

| # | Feature | Dateien | Details |
|---|---------|---------|---------|
| F1 | **Stop Sequences** — `stop` Parameter (String oder Array, max 4). Text-Level Matching mit Buffered Streaming Output (Partial-Match-Sicherheit). CLI: `--stop` Flag. | `imp-server/main.cpp`, `imp-cli/main.cpp`, `imp-cli/args.h/cpp` | `finish_reason: "stop"` bei Match, kein Engine-Eingriff nötig |
| F2 | **Logprobs** — Per-Token Log-Softmax + Top-N Alternativen (0-20). CPU-Side Berechnung via D2H Logits-Copy in Pinned Buffer (~0.3ms für 152K Vocab). OpenAI-kompatibles Response-Format. | `sampling.h/cu`, `executor.h/cu`, `engine.cpp`, `request.h`, `imp.h`, `imp_api.cpp`, `imp-server/main.cpp` | CUDA-Graphs deaktiviert wenn Logprobs aktiv (D2H Copy außerhalb Graph) |
| F3 | **JSON Mode** — Stack-basierte JSON-FSM mit Token-Klassifikation via Bitfield-Kategorien. GPU Logit-Masking Kernel setzt ungültige Tokens auf -FLT_MAX. Lazy Init beim ersten `json_mode` Request. | `json_constrain.h/cu`, `executor.cu`, `engine.h/cpp`, `request.h`, `imp.h`, `imp_api.cpp`, `imp-server/main.cpp` | `response_format: {"type": "json_object"}`, CUDA-Graphs deaktiviert wenn aktiv |

### Implementiert — Phase 7: Decode-Attention-Optimierungen (2026-03-02)

| # | Optimierung | Dateien | Details |
|---|-------------|---------|---------|
| P26 | **Dynamic Split-K SM Heuristic** — Hardcoded `target_blocks=340` (nur RTX 5090 korrekt) ersetzt durch `2 * kpar_n_sms()` (gecachte SM-Count-Abfrage). Split-K Aktivierungsschwelle von `< 128` auf `< 2 * num_sms` angehoben. | `attention_paged.cu` | Portabilität: korrekte Occupancy auf allen GPU-Größen |
| P27 | **Templated + Vectorized Fallback Kernels** — MHA, FP8 und INT8 Decode-Kernels auf `HEAD_DIM` Template-Parameter umgestellt. Contiguous Lane-Mapping (`lane_offset = lane_id * ELEMS`) + half2/uint32_t vektorisierte Loads statt strided Scalar-Loads. Generic Fallback für nicht-standard head_dim (Tests). | `attention_paged.cu` | Compile-Time Unrolling, Coalesced Memory Access |
| P28 | **cp.async Pipelined Split-K** — Neue `paged_attention_splitk_pipeline_kernel` (FP16) und `paged_attention_splitk_fp8_pipeline_kernel` (FP8). Per-Warp smem mit Double-Buffered K + V-Buffer. `cp.async.ca` überlappt V[t]+K[t+1] Loads mit K[t] Dot-Product. Dispatch auf sm_90+, Fallback auf Standard-Split-K. | `attention_paged.cu` | Memory/Compute Overlap: ~50ns statt ~196ns pro Token (ideal) |
| P29 | **Paged Attention Decode Benchmark** — `bench_paged_attention()` in imp-bench: 4 Head-Konfigurationen (MHA-32h, GQA-32/8, GQA-32/4, GQA-28/4) × 6 Context-Längen (64–32K). Reports Latency, Bandwidth, Kernel-Path. | `bench_attention.cu`, `main.cpp` | Messbare Decode-Attention Baseline + Regressions-Erkennung |

**cp.async für GQA/Cluster** (Step 4 des Plans) wurde evaluiert und **nicht implementiert**: GQA/Cluster-Kernels nutzen strided Global→Smem Copies (Slot-Stride über KV-Heads), was inkompatibel mit cp.async (kontiguöse Quell-Adressen) ist. Double-Buffered Tile-Loads bieten bereits Compute/Memory Overlap.

### Implementiert — Phase 5 (2ebb2c4, 2026-03-02)

| # | Optimierung | Dateien | Gemessener Gewinn |
|---|-------------|---------|-------------------|
| P25 | **CUTLASS Hopper FMHA** — WGMMA + TMA Prefill-Attention auf sm_90+ via CUTLASS v4.4.1 Example 88. Causal, GQA, HD=64/128. VRAM-Budget-integrierte Workspace-Allokation (LSE + Kernel). Fallback auf WMMA via `IMP_NO_CUTLASS_FMHA=1`. | `attention_cutlass_fmha.cu/h`, `attention_dispatch.cu`, `executor.cu`, `CMakeLists.txt` | ~1.0x auf sm_120 (WMMA 8-Warp bereits stark), erwartet >1.5x auf sm_90 (Hopper) |

### Implementiert — Phase 4 (0d3be11 → 81b1bea, 2026-03-02)

| # | Optimierung | Dateien | Gemessener Gewinn |
|---|-------------|---------|-------------------|
| P21 | **NVFP4 Decode Weight Cache** — Init-Time FP16→NVFP4 Quantisierung aller dense Weights. K-parallel dp4a GEMV mit On-the-fly FP4→INT8 Promotion. `--decode-nvfp4` (additiv: FP16 Prefill + NVFP4 Decode) und `--decode-nvfp4-only` (ersetzt FP16). | `config.h`, `engine.h/cpp`, `executor.h/cu`, `nvfp4_gemm.h/cu`, `args.h/cpp` | +25-39% Decode (Phi-4-mini: 241→315, Qwen3-4B: 186→259, DS-R1-14B: 86→116 tok/s) |
| P22 | **Zentralisiertes VRAM-Budget** — Engine berechnet einmalig `effective_free_vram() - 1 GiB Reserve`, gemeinsame `remaining_budget` Variable über alle 3 Weight-Cache-Phasen (FP16, FP8, NVFP4). Entfernt 2x redundante `cudaMemGetInfo` Aufrufe. | `engine.cpp`, `executor.cu` | Korrekte VRAM-Budgetierung, verhindert WSL2-Overcommit |
| P23 | **FFN-Budget-Reallokation** — Bei aktivem NVFP4: dense FFN (gate/up/down) aus FP16-Cache auslassen, wenn NVFP4-tauglich. Phase 3 nutzt `dequant_scratch_` als transientes FP16-Staging (GGUF→scratch→NVFP4), spart cudaMalloc/Free pro Tensor. | `executor.cu` | DS-R1-7B: 116→196/196 NVFP4 Tensors, +24% Decode. DS-R1-14B: 0→336 NVFP4 Tensors, +34% Decode |
| P24 | **Temperature-0 CUDA-Graph Fix** — `d_token_id_` in `CudaGraphConditionalRunner` von 4B auf `ARGMAX_SCRATCH_BYTES` (516B) vergrößert. Multi-Block-Argmax Scratch überschrieb `d_position_`/`d_context_len_`/`d_step_counter_`. | `cuda_graph.cu` | Greedy Sampling in CUDA-Graph-Loop funktioniert jetzt korrekt |

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
| P20 | **FP8 Prefill Weight Cache** — Hybrid FP16+FP8 E4M3: FP16-Cache mit Fused KV/Gate+Up zuerst, dann FP8-Overflow für restliche Weights. `--prefill-fp8` Flag. Device-side Scales, amortisierte Activation-Quantisierung (4 pro Layer). | `config.h`, `engine.h/cpp`, `executor.h/cu`, `args.h/cpp` | 1.79x Prefill (DS-R1-7B), 0% Regression auf kleine Modelle, 50% weniger VRAM für Overflow-Weights |

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

#### Decode-Throughput (tok/s) — imp vs imp+NVFP4 vs llama.cpp

| Model | Quant | Parameter | llama.cpp | imp | imp+NVFP4 | NVFP4 vs llama |
|-------|-------|-----------|-----------|-----|-----------|----------------|
| Phi-4-Mini | Q8_0 | 3.8B | 250.96 | 239.29 | **315** | **+25.5%** |
| Qwen3-4B | Q8_0 | 4B | 217.54 | 228.07 | **259** | **+19.1%** |
| DeepSeek-R1-7B | Q8_0 | 7B | 164.30 | 159.43 | **179** | **+8.9%** |
| Gemma-3-12B | Q8_0 | 12B | 91.41 | 85.39 | — | -6.6% |
| DeepSeek-R1-14B | Q6_K | 14B | 102.22 | 88.29 | **116** | **+13.5%** |
| Qwen3-Coder-30B-A3B (MoE) | Q6_K | 30B (3B aktiv) | 206.61 | 231.71 | — | **+12.1%** |
| Nemotron-3-Nano-30B-A3B (MoE) | Q6_K | 30B (3B aktiv) | 25.77 | 60.42 | — | **+134%** |

NVFP4-Decode-Cache (`--decode-nvfp4`) quantisiert Weights bei Init von FP16 auf FP4 E2M1 und nutzt einen K-parallelen dp4a GEMV für Decode (M=1). Der ~4x kleinere Weight-Footprint erhöht den effektiven Bandwidth erheblich. Bei MoE-Modellen und Gemma-3 fehlt NVFP4 noch (MoE: sparse Expert-Routing nicht kompatibel, Gemma-3: Post-Norm FP32-Accumulator).

#### Prefill-Throughput (tok/s)

| Model | Quant | pp FP16 | pp FP8 (`--prefill-fp8`) | Speedup |
|-------|-------|---------|--------------------------|---------|
| Phi-4-Mini | Q8_0 | 15,575 | 16,055 | 1.03x |
| Qwen3-4B | Q8_0 | 15,189 | — (all FP16) | 1.00x |
| DeepSeek-R1-7B | Q8_0 | 7,382 | 13,196 | **1.79x** |
| Gemma-3-12B | Q8_0 | 6,386 | — (all FP16) | 1.00x |
| DeepSeek-R1-14B | Q6_K | 5,587 | — (all FP16) | 1.00x |
| Qwen3-Coder-30B-A3B (MoE) | Q6_K | 4,446 | — (all FP16) | 1.00x |
| Nemotron-3-Nano-30B-A3B (MoE) | Q6_K | 1,153 | 1,160 | 1.01x |

FP8 Prefill nutzt eine hybride Strategie: FP16-Cache mit Fused KV/Gate+Up wird immer zuerst gebaut. Nur Weights die nicht in den FP16-VRAM-Budget passen, werden als FP8 E4M3 gecacht (50% kleiner). Bei kleinen Modellen passt alles in FP16 → kein FP8-Overhead, keine Regression. Bei großen Modellen (DS-R1-7B: 19.4 GiB FP16-Cache) profitieren die Overflow-Weights von FP8-Tensor-Cores (419 TFLOPS auf sm_120).

#### Decode-Entwicklung über alle Optimierungs-Phasen

| Model | Quant | Baseline (a9ae9b4) | Nach P2-P9 | P10-P20 (bcc9bba) | +NVFP4 (0d3be11) | Gesamt-Gewinn |
|-------|-------|---------------------|------------|---------------------|-------------------|---------------|
| Phi-4-Mini | Q8_0 | 189.77 | 219.85 | 239.29 | **315** | **+66.0%** |
| Qwen3-4B | Q8_0 | 186.06 | 213.33 | 228.07 | **259** | **+39.2%** |
| DS-R1-7B | Q8_0 | 133.52 | 149.52 | 159.43 | **179** | **+34.1%** |
| Gemma-3-12B | Q8_0 | — | 82.08 | 85.39 | — | — |
| DS-R1-14B | Q6_K | — | 83.85 | 88.29 | **116** | — |

**Mit NVFP4**: Dense-Modelle übertreffen llama.cpp jetzt um +9-26% statt hinter llama.cpp zu liegen. Die verbleibende Lücke bei Baseline (ohne NVFP4) ist systemisch bedingt durch GEMV-Bandwidth-Auslastung (~50-59% vs llama.cpp ~55-69%). NVFP4 umgeht dies durch ~4x kleineren Weight-Footprint.

**MoE-Modelle** profitieren am stärksten von den Custom-Kernels: Fused MoE GEMV mit Shared-Memory Expert Caching und Sigmoid-Routing übertreffen llama.cpp signifikant. Nemotron (Hybrid Mamba2+Attention+MoE) ist 2.3x schneller, da llama.cpp diese Architektur nicht optimiert. NVFP4 für MoE steht noch aus (sparse Expert-Routing).

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
| Chunked Prefill | Ja | Ja | Ja | **Ja** | `--prefill-chunk-size`, chunked Prefill mit State-Tracking |
| Structured Output (Grammar) | Ja | Nein | Ja | **Partiell** | JSON Mode (`response_format: {"type": "json_object"}`), kein JSON-Schema/GBNF |
| Vision / Multimodal | Ja | Ja | Ja | **Ja** | Gemma-3 SigLIP (single-image, 896×896, mmproj.gguf) |
| LoRA / Adapters | Ja | Ja | Nein | **Nein** | Statische Weights only |
| Beam Search | Ja | Ja | Ja | **Nein** | Nur Greedy/Sampling |
| Request Preemption | Ja | Nein | Ja | **Nein** | Kein Pause/Resume |
| Priority Scheduling | Ja | Nein | Ja | **Partiell** | Shortest-first Reordering (P7), kein Priority-Feld |
| Logprobs Output | Ja | Ja | Ja | **Ja** | Per-Token Log-Softmax + Top-N Alternativen, OpenAI-kompatibles Format |
| Stop Sequences | Ja | Ja | Ja | **Ja** | Bis zu 4 String-Sequenzen, Streaming-safe mit Buffered Output |
| Batch API (extern) | Ja | Ja | Ja | **Nein** | Intern ja, API single-request |
| KV-Cache Quantisierung | Ja | Ja | Nein | **Ja** | FP16, FP8 E4M3 (`--kv-fp8`), INT8 dp4a (`--kv-int8`) |
| AWQ/GPTQ | Ja | Ja | Ja | **Nein** | Nur GGML-Formate |
| Medusa/EAGLE Heads | Nein | Ja | Nein | **Nein** | Nur Draft-Model |

### Empfohlene nächste Features (nach ROI sortiert)

1. **JSON Schema / GBNF Grammar** — Erweiterung von JSON Mode zu Schema-Validierung und GBNF-Grammatik
2. **GEMV Bandwidth-Optimierung** — L2-Tuning, Prefetch-Hints, +5-10% Dense Decode
3. **NVFP4 cuBLASLt Prefill** — RTX 5090 native NVFP4 TensorCore Support für Prefill-Throughput
4. **Multi-Image Vision / Pan-and-Scan** — Erweiterung der Gemma-3 SigLIP-Unterstützung auf mehrere Bilder und variable Auflösung

---

## Zusammenfassung

**CUDA 13.1**: 5/5 Major-Features implementiert (Green Contexts, PDL, Graphs+Conditional, MemPool, NVFP4). PDL-Registry mit 28+ Kernel-Registrierungen (7 Utility + 6 Compute + 110+ Template-Instantiierungen). Blackwell WMMA 8-Warp Attention mit Double-Buffered KV für sm_120. TCGEN05 Inline-PTX/TMA noch nicht implementiert — größtes verbleibendes Hardware-Potential.

**Performance**: P2-P29 implementiert. CUTLASS Hopper FMHA (P25) für Prefill-Attention auf sm_90+ (WGMMA + TMA). Decode mit NVFP4: +34-66% vs Baseline, +9-26% vs llama.cpp (dense). MoE: +12-134% vs llama.cpp (ohne NVFP4). Prefill: +3-84% (Batched GEMM, Residual Fusion, Eager Dequant), +79% mit FP8-Overflow (P20, DS-R1-7B). Zentralisiertes VRAM-Budget (P22) mit FFN-Reallokation (P23) ermöglicht volle NVFP4-Coverage auch bei großen Modellen. Multi-Block Argmax: 192µs → ~10µs. Temperature-0 CUDA-Graph-Bug gefixt (P24). Decode-Attention: cp.async Pipelining, vektorisierte Fallback-Kernels, dynamische Split-K SM-Heuristik (P26-P29). TCGEN05/WGMMA für Decode-Attention als unpraktisch eingestuft (M=1 erfordert min. M=64 für WGMMA Tiles). Verbleibend: Bandwidth-Optimierung, NVFP4 für MoE, NVFP4 cuBLASLt Prefill.

**Architektur**: Auf Single-GPU-Ebene vergleichbar mit vLLM. Chunked Prefill, KV-Cache-Quantisierung (FP16/FP8/INT8), Logprobs, Stop Sequences, JSON Mode und Vision (Gemma-3 SigLIP) sind implementiert. Hauptlücken sind Multi-GPU, JSON Schema/GBNF Grammar, Multi-Image Vision und externe Batch-API. Für den aktuellen Scope (Single-GPU, CLI/Server) ist die Architektur solide und modern.

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

### A.5 NVFP4 — Vollständig (Quant + Decode GEMV)

**Format**: FP4 E2M1 (1 sign | 2 exp | 1 mantissa, bias=1). Magnitudes: {0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}

**Quantisierungs-Kernels** (`nvfp4_quant.cu`):
- `absmax_kernel` — Grid-stride Reduktion, `atomicMax` auf IEEE754 Bit-Pattern
- `quantize_nvfp4_kernel` — FP16→NVFP4: Micro-Scale (FP8 E4M3) per 16-Werte Block + Tensor-Scale (FP32)
- `dequantize_nvfp4_kernel` — NVFP4→FP16: Constant LUT `kFP4E2M1Dequant[8]`

**GEMV-Kernels** (`nvfp4_gemm.cu`):
- `gemv_nvfp4_kpar_kernel` — K-parallel GEMV (128 Threads, 4 Warps, 1 Row/Block). Jeder Thread lädt 32 FP4-Nibbles (16 Bytes), promoted On-the-fly zu INT8, dann dp4a gegen Q8_1-quantisierte Aktivierung. 2-Level Dequant: `tensor_scale * micro_scale * dp4a_sum`.
- `gemv_nvfp4_kpar` — Host-Launcher mit Input Q8_1-Quantisierung, Grid = N Rows.
- Init-Time Weight Cache: `pre_dequant_weights()` Phase 3 quantisiert FP16→NVFP4 mit zentralem VRAM-Budget. Nutzt `fp16_cache_` oder `dequant_scratch_` als FP16-Source.

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

### A.7 CUTLASS — v4.4.1 (MoE GEMM + Hopper FMHA)

**MoE GEMM** (CUTLASS 2.x API):
- `Sm80` Target (kompatibel mit SM90/120), Tile 128×128×32, Warp 64×64×32, MMA 16×8×16, 4-Stage `cp.async` Pipeline
- Grouped GEMM für MoE Expert-parallel via `gemm_moe_cutlass()`. Device-only Scheduling, ~3µs Launch-Overhead (vs ~27µs cuBLAS).

**Hopper FMHA** (CUTLASS v4.4.1 Example 88, `attention_cutlass_fmha.cu`):
- WGMMA (asynchronous warpgroup MMA) + TMA (Tensor Memory Accelerator)
- `KernelTmaWarpSpecializedCooperative` Scheduler
- FP16 Input, FP32 Accumulator, CausalFusion / DefaultFusion
- Tile-Konfigurationen: HD=128 → Shape<128, 128, 128>, HD=64 → Shape<128, 64, 64>
- GQA via stride tricks: B_eff = batch × n_kv_heads, H_eff = groups
- VRAM-integrierte Workspace-Allokation: LSE-Buffer + Kernel-Workspace (pre-allokiert in `allocate_auxiliary_buffers()`)
- Dispatch: `attention_dispatch.cu` versucht CUTLASS FMHA zuerst (sm_90+, kein Softcap/Sliding Window), Fallback auf WMMA
- Deaktivierbar via `IMP_NO_CUTLASS_FMHA=1` Env-Variable

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

### B.7 FP8 Prefill Weight Cache

**Problem**: Prefill ist compute-bound (großes M). FP16-Tensor-Cores liefern 209 TFLOPS, FP8-Tensor-Cores 419 TFLOPS auf RTX 5090 (sm_120). Naive FP8-Umstellung verursacht aber Regression auf kleine Modelle (0.64-0.71x), weil Fused Batched GEMMs (KV, Gate+Up) verloren gehen.

**Lösung**: Hybrid FP16+FP8 in `pre_dequant_weights()`:
1. **Phase 1 (immer)**: FP16-Cache mit Fused KV (`gemm_kv_batched`) und Fused Gate+Up (`gemm_pair_batched`). Identisch mit Default-Verhalten.
2. **Phase 2 (nur `--prefill-fp8`)**: Re-Query VRAM, FP8 E4M3 für Weights die nicht in FP16 passten. GGUF→FP16(tmp)→`calibrate_fp8_scale`→`quantize_fp16_to_fp8_e4m3_scaled`→FP16 tmp freigeben.

**Runtime-Dispatch** (`run_attention`, `run_ffn`, `gemm_dispatch`):
- FP8-Cache prüfen → wenn Hit: `quantize_fp16_to_fp8_e4m3(activation)` + `gemm_cublaslt(FP8×FP8→FP16, aScale, bScale)`
- Sonst: Fused KV/Gate+Up → FP16-Cache → On-the-fly Dequant (bestehende Fallback-Kette)

**Key Design**: Device-side Scale Pointers (kein Host-Sync), amortisierte Activation-Quantisierung (4× pro Layer statt 7×), kein Weight in beiden Caches gleichzeitig.

**Dateien**: `config.h`, `imp_api.cpp`, `engine.h/cpp`, `executor.h/cu`, `args.h/cpp`, `main.cpp`

### B.8 NVFP4 Decode Weight Cache

**Problem**: Decode ist bandwidth-bound (M=1 GEMV). GGML Q8_0/Q6_K Weights sind 8/6.5 Bits pro Element — NVFP4 ist ~4 Bits (~2x weniger Bandwidth).

**Lösung**: 3-Phasen Weight-Cache mit zentralem VRAM-Budget:
1. **Phase 1 (FP16)**: Prefill-Weight-Cache (Fused KV, Gate+Up). Bei aktivem NVFP4: dense FFN (gate/up/down) auslassen falls NVFP4-tauglich.
2. **Phase 2 (FP8)**: Overflow für `--prefill-fp8`.
3. **Phase 3 (NVFP4)**: `--decode-nvfp4` — quantisiert FP16→NVFP4 bei Init. Source: `fp16_cache_` (wenn vorhanden) oder `dequant_scratch_` (GGUF→FP16 transient→NVFP4). Budget-Check verhindert VRAM-Überallokation.

**VRAM-Budget**: `effective_free_vram() - 1 GiB Reserve`, geteilt über alle 3 Phasen. `remaining_budget` wird nach jeder Phase dekrementiert. Löst WSL2-Overcommit-Problem (cudaMalloc succeeded beyond physical VRAM, spills to system RAM).

**FFN-Reallokation**: Bei `use_nvfp4_decode > 0` und NVFP4-tauglichem QType (Q8_0, Q8_K, Q6_K, Q5_K) werden dense FFN Weights aus Phase 1 ausgelassen. Da FP16 ~3.5x größer als NVFP4 ist, gibt dies genug Budget für volle NVFP4-Coverage. DS-R1-7B: 116/196 → 196/196 NVFP4 Tensors.

**Runtime-Dispatch** (`gemm_dispatch`): `nvfp4_cache_` Lookup → wenn Hit und M=1: `gemv_nvfp4_kpar()`. Sonst: FP16/FP8/dequant Fallback-Kette.

### B.9 Host-Device Sync

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
| HTTP Server | `tools/imp-server/main.cpp` | `/v1/chat/completions`, SSE Streaming, CORS, Stop Sequences, Logprobs, JSON Mode, Vision (base64 images) |
| Stop Sequences | `tools/imp-server/main.cpp`, `tools/imp-cli/main.cpp` | Bis zu 4 String-Sequenzen (`stop` Parameter), Streaming-safe mit Buffered Output, `finish_reason: "stop"` |
| Logprobs | `sampling.cu`, `executor.cu`, `engine.cpp`, `request.h` | CPU-side Log-Softmax + Top-N Min-Heap, D2H Logits via Pinned Buffer, OpenAI-kompatibles Format (`logprobs` + `top_logprobs` 0-20) |
| JSON Mode | `json_constrain.h/cu`, `executor.cu`, `engine.cpp` | Stack-basierte JSON-FSM, Token-Klassifikation via Bitfield-Kategorien, GPU Logit-Masking Kernel, `response_format: {"type": "json_object"}` |
| dp4a Template GEMV | `gemv_dp4a_traits.cuh` | 5 QTypes, K-par/Row-par Dispatch, ~130 Instantiierungen |
| Multi-Block Argmax | `sampling.cu` | 64-Block 2-Phasen Reduktion, 19x Speedup |
| FP8 Prefill Weight Cache | `executor.cu`, `fp8_quant.cu` | Hybrid FP16+FP8, `--prefill-fp8`, Device-Scales, 1.79x Prefill (DS-R1-7B) |
| NVFP4 Decode Weight Cache | `executor.cu`, `nvfp4_gemm.cu` | FP16→NVFP4 Init-Time Quant, K-par dp4a GEMV, `--decode-nvfp4`, +25-39% Decode |
| Chunked Prefill | `engine.cpp` | `--prefill-chunk-size`, State-Tracking über Chunks, verhindert Head-of-Line-Blocking |
| INT8 KV Cache | `kv_cache.cu`, `attention_paged.cu` | `--kv-int8`, dp4a Attention mit Per-Token INT8 Quantisierung |
| Vision (Gemma-3 SigLIP) | `src/vision/vision_encoder.cu`, `vision_loader.cpp`, `image_processor.cpp` | 27-Layer SigLIP ViT (896×896, 256 Tokens), mmproj.gguf Loader, stb_image Preprocessing, cuBLAS Batched Attention, `--mmproj`/`--image` CLI, OpenAI multimodal API |

### C.2 Verifizierte Lücken

| Feature | Evidenz |
|---------|---------|
| Beam Search | Nur `temperature/top_p/top_k` in Sampling, kein `beam_size` |
| JSON Schema / GBNF | JSON Mode (syntaktische Validität) implementiert, aber kein Schema-Constraining oder GBNF-Grammatik |
| TCGEN05 Decode Attention | Prefill nutzt CUTLASS FMHA (WGMMA+TMA). Decode nutzt WMMA + cp.async Pipelining (Phase 7). WGMMA (min. M=64) ist inkompatibel mit M=1 Decode — kein weiterer Gewinn durch TCGEN05 auf Decode-Pfad. |

---

## 4. Deep Architecture Audit (2026-03-15)

Systematische Analyse der gesamten Engine-Architektur auf wiederkehrende Anti-Patterns, Ressourcen-Allokation, Dispatch-Entscheidungen, Datenfluss, Fehlerbehandlung und Scheduling. Fünf parallele Audit-Achsen:

1. **VRAM & Ressourcen-Allokation** — Wurde das VRAM-Budget-Problem (First-Come-First-Serve) auch anderswo gemacht?
2. **Compute Dispatch** — Sind Kernel-Auswahl und Konfiguration modell-bewusst?
3. **Datenfluss & Synchronisation** — Unnötige Host-Device Syncs, redundante Kopien, Pipeline-Stalls?
4. **Fehlerbehandlung & Robustheit** — Silent Failures, State Corruption, Resource Leaks, Concurrency?
5. **Scheduling & Batching** — Suboptimale Batching-Entscheidungen, Lifecycle-Issues?

### Wiederkehrende Anti-Patterns

Vier strukturelle Probleme ziehen sich durch die gesamte Codebase:

**Pattern 1: "First Come First Serve" Allokation** (5 Instanzen)
VRAM wird ad-hoc allokiert ohne globale Planung. Wer zuerst `cudaMalloc` ruft, bekommt den Platz — unabhängig davon, ob eine andere Komponente den VRAM besser nutzen könnte.
- *Behoben*: FP8/NVFP4/KV Budget → `VRAMBudget` Planner
- *Offen*: Expert LRU Cache, Attention S-Matrix, weight_upload Reserve, workspace_estimate

**Pattern 2: "Hardcoded Magic Number" statt Modell-Awareness** (8 Instanzen)
Konstanten die für ein Modell/GPU getunt wurden, aber bei anderen versagen.
- 4096 max_tokens, 75% L2, 1 GiB Reserve, 256 MiB weight Reserve, 64 MiB auxiliary, kKVBlockSize=16, MoE TC Threshold `ne*24`, cuBLAS attention 256 MiB cap

**Pattern 3: "Sync in Loop" im Hot Path** (4 Instanzen)
Wiederholte Host-Device Synchronisation oder Allokation im Decode-Loop, wo pre-allokierte Buffers und append-only Updates reichen würden.
- Penalty-Token re-upload, MoE FP8 batch 6×sync/layer, DRY malloc/free/O(n²), cuBLAS benchmark bei neuem M

**Pattern 4: "Silent Degradation"** (6 Instanzen)
Fehler die nicht crashen, sondern leise die Performance oder Korrektheit ruinieren — ohne dass der User es merkt.
- Workspace nullptr → crash, Green Context failure → kein Log, Dequant scratch fail → leise langsam, CUTLASS FMHA fallback → kein Log, Schema Constrainer shared state, stale CUDA errors geschluckt

---

### 4.1 Kritisch — Sofort fixen

#### F1: Workspace-Allokation: Crash bei cudaMalloc-Fehler

**Dateien:** `executor_workspace.cu:320-325` (persistent), `executor_workspace.cu:371-377` (shared)
**Problem:** `allocate_persistent_workspace()` und `allocate_shared_workspace()` loggen bei `cudaMalloc`-Fehler und returnen, aber der Caller `allocate_workspaces()` gibt `true` zurück. `hidden_`, `residual_`, `norm_out_`, `logits_` zeigen auf nullptr. Jeder Folge-Kernel crasht mit illegal memory access.
**Fix:** Return bool propagieren bis `Engine::init()`.

#### F2: JSON/Schema Constrainer ist Singleton — concurrent Requests korrumpieren FSM

**Datei:** `engine.cpp:1179-1219`
**Problem:** `json_constrainer_` und `schema_constrainer_` sind einzelne Instanzen auf der Engine. Im Server-Mode mit Continuous Batching teilen sich parallele JSON-mode Requests denselben FSM-State. Interleaved Tokens korrumpieren die Grammatik-Position.
**Fix:** Constrainer pro Request statt pro Engine. Oder: JSON Mode auf Single-Sequence Batch limitieren.

#### F3: ServerState ohne Lock — Race zwischen Auto-Load und Requests

**Datei:** `handlers.cpp:508-552`
**Problem:** `state.model_loaded()`, `state.tok`, `state.chat_tpl` werden aus HTTP-Handler-Threads ohne Lock gelesen. Wenn ein Request Auto-Load triggert während ein anderer tokenisiert, wird ein dangling Pointer dereferenziert.
**Fix:** Shared Lock für Reads, oder atomarer Pointer für Tokenizer mit Read-Copy-Update.

#### F4: MoE Non-dp4a Fallback: Q4_K Weights an Q8_0 Kernel

**Datei:** `executor_forward.cu:1504-1529`
**Problem:** Wenn `q8_1_buf_` null ist, dispatcht der non-dp4a Fallback nur `gemv_q6k_moe_decode` oder `gemv_q8_0_moe_decode`. Q4_K, Q5_K, Q2_K, Q3_K Weights werden fälschlich an den Q8_0 Kernel gegeben — falsches Block-Layout → Garbage Output.
**Fix:** Q4_K/Q5_K/Q2_K/Q3_K Dispatch-Varianten analog zum dp4a-Pfad hinzufügen.

#### F5: Expert LRU Cache umgeht VRAMBudget — selbes Anti-Pattern

**Datei:** `executor_workspace.cu:631-645`
**Problem:** Expert LRU Cache allokiert bis 2 GiB mit eigenem `cudaMemGetInfo` + 128 MiB Reserve — komplett unabhängig vom VRAMBudget. Selbes "First Come First Serve" wie der ursprüngliche VRAM-Bug.
**Fix:** Expert Cache Budget als Feld in `VRAMBudget`. `plan_vram_budget()` entscheidet die Aufteilung.

---

### 4.2 Medium — Performance-Architektur

#### F6: workspace_estimate() nutzt flat 64 MiB für Auxiliary

**Datei:** `executor_workspace.cu:230`
**Problem:** Die Schätzung für Expert-Upload-Reserve nutzt pauschal 64 MiB für alle Hilfs-Buffer. Die tatsächliche Auxiliary-Allokation (dequant scratch, sampling, MMVQ, split-K, S-Matrix, MoE, FP8, CUTLASS) kann deutlich mehr sein. Expert Upload über-committed VRAM, Features degradieren still.
**Fix:** Echte Summe der Einzelpuffer berechnen statt flat 64 MiB.

#### F7: weight_upload.cu hat eigene 256 MiB Reserve zusätzlich zu Engine's 1 GiB

**Datei:** `weight_upload.cu:24`
**Problem:** `kWeightReserveMiB = 256` wird auf Engine's `expert_reserve` (enthält bereits 768 MiB Safety) draufgerechnet. Kumulativ 1.75+ GiB Reserves auf 32 GB Karte. Letzte Dense-Layers fallen unnötig auf Host zurück.
**Fix:** Engine übergibt berechneten Reserve-Wert direkt, `kWeightReserveMiB` eliminieren.

#### F8: VRAMBudget Reserve ist flat 1 GiB

**Datei:** `engine.cpp` (plan_vram_budget)
**Problem:** 1 GiB Reserve ist nicht proportional zu GPU/Modell. 5% einer 32 GB Karte, 11% bei 16 GB Budget. Kleine Modelle ohne CUDA Graphs / Speculative brauchen weniger.
**Fix:** Reserve berechnen basierend auf aktiven Features (CUDA Graphs: +256 MiB, cuBLAS: +128 MiB, etc.).

#### F9: Scheduler bricht bei erster Admission-Failure ab statt zu skippen

**Datei:** `scheduler.cpp:42-47`
**Problem:** Wenn `kv_manager_->can_allocate(blocks_needed)` für eine Request fehlschlägt, macht der Scheduler `break` — blockiert ALLE nachfolgenden (möglicherweise kleineren) Requests. Eviction wird nicht versucht.
**Fix:** `break` → `continue`. Oder: Eviction im Scheduler statt nur in Engine.

#### F10: Scheduler sortiert std::deque jeden step(), keine Fairness

**Datei:** `scheduler.cpp:29-34`
**Problem:** O(n log n) Sort auf non-contiguous Memory pro Token-Generierung. Kein Aging — lange Prompts werden endlos von kürzeren verdrängt.
**Fix:** Nur bei neuen Requests sortieren (dirty flag). Aging-Mechanismus für Starvation-Prevention.

#### F11: Penalty-Tokens werden komplett re-uploaded pro Step

**Datei:** `engine.cpp:1592-1616`
**Problem:** Gesamter `output_tokens` Vektor wird jeden Decode-Step per `cudaMemcpyAsync` hochgeladen (wachsend). Bei 4096 Tokens: 16 KB/Step. Bei Realloc: synchroner `cudaMalloc` + `cudaFree` im Hot Path.
**Fix:** Pre-allocate auf `max_tokens`, append-only (nur neues Token pro Step).

#### F12: MoE TC/Scalar Threshold-Gap für Short Prefill

**Datei:** `executor_forward.cu:1628-1629`
**Problem:** TC-Pfad ab `expanded > ne*24`, Scalar ab `expanded <= ne*12`. Dazwischen (`ne*12 < expanded <= ne*24`) fällt alles auf FP16-Batch mit D2H Sync. Für Qwen3-Coder (128 Experts): 192-384 Tokens (top_k=8) treffen die Gap. Agentic Workloads haben typisch 100-300 Token Prefills.
**Fix:** TC-Threshold senken — nach persistent work-queue Optimierung sollte der TC-Kernel auch bei kleinen Batches performen.

#### F13: MoE FP8 Batch: 6 cudaStreamSynchronize pro Layer

**Datei:** `executor_forward.cu:1906-1916`
**Problem:** FP8 Batch-Path macht pro Projektion 2 D2H Syncs (Offsets + Scales). Bei 3 Projektionen pro Layer: 6 Syncs pro Layer, 192 für 32-Layer MoE Modell → ~1 ms Pipeline-Bubbles.
**Fix:** Device-side Grouped GEMM API nutzen (existiert als `gemm_moe_device_grouped`), oder Scale-Arrays device-seitig halten.

#### F14: GEMM Cache-Key mit exaktem M — Benchmark-Storm bei Variable-Length Prefill

**Datei:** `gemm.cu:148-158`
**Problem:** Jede neue Prefill-Länge (M) erzeugt Cold-Start mit cuBLASLt Benchmark (5 Iterationen, `cudaEventSynchronize`). Server mit diversen Prompt-Längen zahlt Multi-ms Stall pro neuem M.
**Fix:** M auf Potenzen von 2 oder Vielfache von 64 bucketen.

#### F15: FP8 GEMV: 16 skalare Operationen pro Vektor-Load

**Datei:** `gemm.cu:822-851`
**Problem:** `gemv_fp8_e4m3_kernel` lädt 16 FP8 Bytes via float4, verarbeitet sie aber in einer Schleife mit Per-Element Branching und skalarer Akkumulation. Kein vektorisiertes half2/float2 FMA.
**Fix:** Unroll in 2×8 Gruppen, Branch eliminieren, half2-FMA wie im FP16 GEMV.

#### F16: cuBLAS Attention ohne Crossover-Threshold

**Datei:** `executor_forward.cu:570-599`
**Problem:** cuBLAS QK^T Materialisierung wird bis zur Buffer-Kapazität genutzt (bis 2896 Tokens). Bei n=2048, nh=32 materialisiert das 256 MiB S-Matrix. Flash Attention braucht O(1) extra Memory und ist ab ~512-1024 Tokens schneller.
**Fix:** Heuristik-Threshold (z.B. n <= 512 oder n <= 1024) statt Buffer-Kapazität.

#### F17: INT4 GEMM Kernel ist rein skalar

**Datei:** `quant_gemm.cu:41-169`
**Problem:** Q4_1 GEMM nutzt 64×64×32 Tiles mit skalarem FP32 FMA. Kein WMMA/Tensor Core. Auf Blackwell mit 680 Tensor Cores ist das 8-16× langsamer als nötig.
**Fix:** Q4_1 durch dp4a routen (Block-Reformat) oder WMMA 16×16×16 mit on-the-fly Dequant.

#### F18: KV allocate_block_with_eviction() versucht nur 1 Block

**Datei:** `kv_cache_manager.cpp:441-452`
**Problem:** Bei leerem Free-Pool wird genau 1 Cached Block reclaimed. Wenn der pinned ist → sofortige Aufgabe. Engine's Decode-Path macht nur einen Eviction-Versuch (Zeile 1441-1444). Premature Request-Cancel bei KV-Druck mit Prefix Caching + Pinning.
**Fix:** Loop über Cached Blocks. Retry-Loop im Decode-Path analog zum Prefill-Path.

#### F19: apply_dry_penalty macht cudaMalloc/Free pro Token

**Datei:** `sampling.cu:876-895`
**Problem:** Jeder DRY-Penalty-Aufruf: cudaMalloc für d_tokens + d_values, Upload, Kernel, cudaFreeAsync. CPU Suffix-Matching ist O(n²) in Token-History.
**Fix:** Pre-allocate Penalty-Buffer bei Init. Suffix-Matching cappen oder auf GPU verschieben.

#### F20: KV Cache Save/Load: 6400 synchrone cudaMemcpy

**Datei:** `kv_cache_manager.cpp:631-639, 718-724`
**Problem:** Pro Block: n_layers × 2 synchrone `cudaMemcpy`. Bei 32 Layers, 100 Blocks: 6400 Calls.
**Fix:** `cudaMemcpyAsync` mit Pipeline, oder Batch-Copy pro Block statt pro Layer.

#### F21: Green Context SM-Split statisch, nie angepasst

**Datei:** `engine.h:33`, `green_ctx.cu:35-37`
**Problem:** 80/20 Prefill/Decode Split wird bei Init gesetzt und nie geändert. `reconfigure()` existiert aber wird nicht gerufen. Wenn kein Prefill anliegt: 80% der SMs idle.
**Fix:** Dynamische Anpassung in `step()` basierend auf Queue-Depths.

#### F22: kKVBlockSize=16 hardcoded

**Datei:** `kv_cache.h:10`
**Problem:** Compile-Time Konstante. Für GQA Modelle mit wenig KV Heads (Qwen3: 4 Heads) sind Blocks klein (16 KB/Layer) → hoher Block-Table Overhead, schlechte Coalescing. Prefix Cache Granularität (16 Tokens exact match) ist zu grob für kurze System-Prompts, zu fein für lange Prefixe.
**Fix:** Konfigurierbar bei Init-Time. 16 für viele KV Heads, 32/64 für wenige.

#### F23: Stale CUDA Errors still geschluckt

**Datei:** `executor_forward.cu:2709-2710`
**Problem:** `cudaGetLastError()` vor Forward Pass löscht Fehler auf DEBUG Level. Ein früherer Kernel-Fehler (z.B. aus Workspace-Allokation) wird verschluckt, Forward läuft auf korrupten Daten weiter.
**Fix:** Mindestens WARN Level. Bei stale Error: Forward abbrechen.

#### F24: Green Context Init-Failure: Kein Log

**Datei:** `engine.cpp:597-599`
**Problem:** `green_ctx_.init()` Failure hat Kommentar "Non-fatal" aber keinen Log. User aktiviert `--green-contexts` und bekommt keine Rückmeldung dass es nicht funktioniert.
**Fix:** WARN Log bei Failure.

#### F25: Batched Decode nutzt Request[0]'s Penalty-Tokens für alle

**Datei:** `engine.cpp:1569-1589`
**Problem:** `InferenceState` wird aus `valid_decode[0]` befüllt. `host_penalty_tokens` zeigt auf Request 0's Daten. `sample_per_request` Lambda overridet pro Request, aber Penalties bleiben von Request 0. Andere Requests bekommen falsche Penalty-Tokens.
**Fix:** Penalty-Tokens per Request im Sampling-Loop behandeln.

#### F26: KV Cache k_ptr/v_ptr ohne Bounds-Check

**Datei:** `kv_cache.cu:134-146`
**Problem:** `k_ptr(layer, block_id)` und `v_ptr(layer, block_id)` rechnen Offsets ohne Validierung von `layer < n_layers_` oder `block_id < max_blocks_`. Out-of-range block_id schreibt an beliebige GPU-Adresse.
**Fix:** Bounds-Assert in Debug Builds. Release: block_id gegen max_blocks_ validieren.

#### F27: Unchecked cudaMemcpy/fwrite in Prefix Cache Save/Load

**Datei:** `kv_cache_manager.cpp:616-641, 718-724`
**Problem:** `cudaMemcpy` und `fwrite` Return-Werte werden nicht geprüft. Volle Disk → truncated File. Device-Lost → korrupte Daten im Cache.
**Fix:** Return-Werte prüfen, bei Fehler Save/Load abbrechen.

#### F28: NVFP4 GEMM Fallback: statischer Dequant-Buffer, Thread-unsafe

**Datei:** `nvfp4_gemm.cu:1286-1317`
**Problem:** Wenn CUTLASS/cuBLASLt nicht verfügbar: Dequant ganzer Weight-Matrix [N, K] in statischen Buffer (z.B. 96 MiB für Qwen3-8B Gate/Up). Buffer per `cudaMalloc` bei Erstaufruf, nie freigegeben. Kein Mutex → Thread-unsafe bei Multi-Stream.
**Fix:** Pre-allokierten `dequant_scratch_` aus Executor nutzen. Warn-Log bei Fallback.

---

### 4.3 Low — Edge Cases und bekannte Dead Ends

| # | Bereich | Problem | Notiz |
|---|---------|---------|-------|
| F29 | Speculative | 8 sync mallocs + 15 sync memcpys + Host-Logits D2H pro Step | Dead End auf Single-GPU (MEMORY.md) |
| F30 | Speculative | Approx. Draft-Probabilities (1/top_k), keine gecachten Logits | Reduziert Acceptance Rate bei non-greedy |
| F31 | Speculative | Kein KV Rollback nach Rejection → stale Blocks belegt | `kv_manager_->rollback()` existiert, wird nie gerufen |
| F32 | Vision | cudaMalloc/Free für Pixel-Buffer pro `set_image()` | Pre-allocate bei Init (4.6 MB fix) |
| F33 | Activation | GeGLU FP32 nicht vektorisiert (kein vec4 wie SwiGLU) | FP32 Path selten genutzt |
| F34 | Activation | FP16 blockDim=256 für alle Sizes → 12 Blocks auf 170 SMs bei MoE | ~1 μs Kernel, nicht auf Critical Path |
| F35 | PDL | `gelu()` und FP32 Activation-Kernels nicht PDL-registriert | PDL hat 0% Impact (MEMORY.md) |
| F36 | GEMM | FP8 Scale-Pointer Mutation auf cached opDesc ohne Lock (latent) | Nur bei Multi-Stream (Green Ctx) relevant |
| F37 | GEMM | Redundante Q8_1 Quantisierung bei fallback QKV (3× statt 1×) | Fused QKV Path deckt Common Case ab |
| F38 | Attention | CUTLASS FMHA Fallback auf WMMA ohne Log | Diagnostic-Lücke, kein Perf-Impact |
| F39 | NVFP4 | Multi-row Threshold `n_mb<=512` nicht SM-adaptiv | Getunt für RTX 5090, ok für aktuelles Target |
| F40 | Split-K | Scratch für max_splits=32 unabhängig von Context | ~17 MiB pro Batch, nicht KV-proportional |
| F41 | Sampling | Fallback-Overloads ohne pre-allocted d_result machen sync malloc | Nur wenn d_sample_result_ null (Init-Fehler) |
| F42 | Scheduler | Chunked Prefill: Linear Search für already-queued Requests | O(active × prefill_batch), negligible |
| F43 | CUDA Graph | Invalidierung bei Batch-Size-Änderung → 2 Steps non-graphed | ~0.6 ms pro Transition |
| F44 | Profiling | CUDA Events in Forward-Pass bei IMP_PROFILE=1 pro Call erstellt, nie RAII | Leak nur im Profiling-Mode |
| F45 | L2 Cache | 75% L2 Reserve nicht model-aware (GQA mit wenig KV Heads braucht weniger) | Akzeptabel für RTX 5090 |
| F46 | Mirostat | mu-Update erfordert Host-Sync pro Token | Fundamentales Feedback-Loop |
| F47 | LongRoPE | Frequency-Buffer Allokation unchecked | Crash nur bei Phi-4 + OOM |
| F48 | Prefix Cache | Legacy `register_prefix()` Block-IDs werden nie bereinigt → stale Pointers | Nur legacy API, content-addressed ist separat |

---

### 4.4 Priorisierte Reihenfolge

**Phase 1 — Korrektheit (F1-F5)**
Workspace nullptr-Crash, JSON Constrainer Race, ServerState Lock, MoE Q4_K Dispatch, Expert Cache Budget. Alle können Crashes oder korrupte Outputs verursachen.

**Phase 2 — Hot-Path Performance (F11, F14, F9, F12, F13)**
Penalty re-upload, GEMM benchmark storm, Scheduler break→continue, MoE threshold gap, MoE FP8 syncs. Direkt messbar in tok/s.

**Phase 3 — VRAM Effizienz (F6, F7, F8)**
workspace_estimate, weight_upload Reserve, Budget Reserve. Zusammen ~500 MiB+ verschwendet auf 32 GB.

**Phase 4 — Server Robustheit (F10, F18, F23, F24, F25, F26, F27)**
Scheduler Fairness, KV Eviction, Error Handling, Logging, Bounds Checks. Wichtig für Production Server.

**Phase 5 — Kernel Optimierung (F15, F16, F17, F19, F28)**
FP8 GEMV, cuBLAS Attention Crossover, INT4 TC, DRY Penalty, NVFP4 Fallback. Modell-spezifische Gewinne.
