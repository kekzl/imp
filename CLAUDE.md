# CLAUDE.md

## Project Overview

**imp** is a high-performance LLM inference engine written in C++20 and CUDA, targeting exclusively the NVIDIA RTX 5090 (GB202, Blackwell, sm_120f). It requires CUDA 13.2+ and leverages Blackwell-specific features: Green Contexts for SM partitioning, Programmatic Dependent Launch (PDL), CUDA Graphs, packed FP8 E4M3 conversion (cvt.e4m3x2), and MXFP4 tensor core attention. No support for older architectures — sm_120 only. The engine supports GGUF and SafeTensors model formats, multiple quantization schemes (FP8, INT8, INT4, NVFP4, MXFP4), and architectures including LLaMA, Mistral, Mixtral, DeepSeek, Qwen3, Qwen3.5 (Gated DeltaNet), Gemma-3 (text + vision), and Nemotron-H. Vision support uses a SigLIP encoder for Gemma-3 multimodal via separate mmproj.gguf files.

## Repository Structure

```
imp/
├── include/imp/          # Public C API headers
│   ├── imp.h             # Main API: model load, context, generate, tokenize
│   ├── config.h          # ImpConfig struct and defaults
│   ├── types.h           # Enums: DType, ModelArch, QuantType, ModelFormat
│   └── error.h           # Error codes and imp_error_string()
├── src/
│   ├── core/             # Tensor, Buffer, Allocator, Logging, Threading
│   ├── compute/          # CUDA kernels: GEMM, attention, RoPE, LayerNorm,
│   │                     #   activation, embedding, sampling, softmax, MoE routing
│   ├── memory/           # GPU/pinned allocators, KV cache (block-based), KV cache manager
│   ├── model/            # Model loading (GGUF/SafeTensors), weight upload, tokenizer
│   ├── quant/            # Quantization: FP8, INT8, FP16 dequant, NVFP4, quant GEMM
│   ├── graph/            # Compute graph DAG (Op, Graph, GraphExecutor)
│   ├── runtime/          # Engine, Scheduler, Request, Batch, Green Contexts,
│   │                     #   CUDA Graphs, PDL, Speculative Decoding
│   ├── vision/           # SigLIP vision encoder, mmproj GGUF loader, image preprocessing
│   └── api/              # C API implementation (imp_api.cpp)
├── tools/
│   ├── imp-cli/          # CLI tool: interactive and single-prompt inference
│   ├── imp-server/       # OpenAI-compatible HTTP server (SSE streaming)
│   └── imp-bench/        # Benchmark tool: GEMM, attention, end-to-end
├── third_party/stb/      # stb_image headers (image loading for vision)
├── tests/                # Google Test suite (63 test files, ~700 tests)
├── scripts/              # verify.sh, gen_perf_baseline.sh, pre-push.hook, imp-pull.py
├── docs/                 # SM120 status, MXFP4, Qwen3.6 roadmap, gemv plan, layer-diff dumps
├── cmake/                # Custom CMake modules (CompilerFlags, FindCUDAToolkit131)
├── CMakeLists.txt        # Build configuration
├── Makefile              # Docker-based test/bench/verify targets (canonical workflow)
├── Dockerfile            # CUDA 13.2 build image
├── docker-compose.yml    # imp-server + open-webui stack
└── .gitignore
```

## Build System

CMake 3.25+ with C++20 host code and CUDA 13.2+ device code. The project builds as a static library (`imp`) plus optional tools and tests.

### Build Commands

```bash
# Configure (out-of-source build)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build build -j$(nproc)

# Build with specific options
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
  -DIMP_BUILD_TESTS=ON \
  -DIMP_BUILD_TOOLS=ON \
  -DIMP_BUILD_BENCH=ON

# Release with debug info
cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build with sanitizers (host code only, for debugging)
cmake -B build-asan -DIMP_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
```

### Build Options

| Option | Default | Description |
|---|---|---|
| `IMP_BUILD_TESTS` | ON | Build the Google Test suite |
| `IMP_BUILD_TOOLS` | ON | Build imp-cli and imp-bench |
| `IMP_BUILD_BENCH` | ON | Build benchmark tool |
| `IMP_BUILD_SERVER` | ON | Build imp-server (OpenAI-compatible HTTP server) |
| `IMP_SANITIZERS` | OFF | Enable ASAN + UBSAN (host C++ code only) |
| `CMAKE_CUDA_ARCHITECTURES` | `sm_120f` (hardcoded) | Target GPU architecture (RTX 5090 only) |

### Dependencies

- **CUDA Toolkit 13.2+** (required) — cudart, cuda_driver, cublas, cublasLt
- **CUTLASS v4.4.2** (fetched via FetchContent) — SM120 FMHA (FP16/FP8/MXFP4), NVFP4/MXFP4 GEMM, MoE Grouped GEMM
- **Google Test v1.14.0** (fetched via FetchContent when tests enabled)
- **stb_image / stb_image_resize2** (vendored in `third_party/stb/`) — image loading for vision
- **pthread** (linked privately)

### Target GPU: NVIDIA RTX 5090 (GB202, Blackwell)

| Spec | Value |
|---|---|
| Compute Capability | sm_120a |
| SMs | 170 |
| CUDA Cores | 21,760 (128/SM) |
| Tensor Cores | 680 (5th gen, 4/SM) |
| Boost Clock | 2,407 MHz |
| VRAM | 32 GB GDDR7, 512-bit bus |
| Memory Bandwidth | 1,792 GB/s (28 Gbps/pin) |
| TDP | 575 W |

**Cache Hierarchy:**

| Level | Size | Notes |
|---|---|---|
| L0 Instruction Cache | 32 KB/SM | |
| L1 Data Cache / Shared Memory | 128 KB/SM | Configurable split (e.g. 64/64, 100/28, 28/100) |
| L2 Cache | 96 MB | Unified, shared across all SMs |
| L3 Cache | n/a | L3 only on data center Blackwell (B200/B300) |

**Tensor Core Throughput (at boost clock):**

| Precision | Dense | 2:4 Sparse |
|---|---|---|
| FP4 (NVFP4 E2M1) | 3,354 TOPS | 6,708 TOPS |
| FP8 (E4M3/E5M2) | 1,677 TFLOPS | 3,354 TFLOPS |
| FP16 / BF16 | 838 TFLOPS | 1,677 TFLOPS |
| INT8 (dp4a) | 1,677 TOPS | 3,354 TOPS |
| FP32 (CUDA Cores) | 105 TFLOPS | — |

**Key for imp kernel tuning:**
- L2 is large enough to cache full KV blocks for moderate context lengths
- 128 KB configurable L1/SMEM per SM — attention kernels use high SMEM configs
- NVFP4 tensor cores give 2x FP8 throughput — decode GEMV is still memory-bound
- 170 SMs → split-K paged attention targets ~340 blocks (2 blocks/SM occupancy)

### Hardware Constraints

Only one GPU is available. **Always test models sequentially** — never run multiple model instances in parallel.

## Running Tests

The canonical workflow uses the Makefile (Docker, GPU passthrough). Host builds also work if CUDA 13.2+ is installed on the host.

```bash
# Docker workflow (primary)
make build               # docker build → imp:test
make test-unit           # CPU-only filter (~5s)
make test-gpu            # full CUDA suite (~30s)
make test-e2e            # real-model E2E (Qwen3-4B, Qwen3.5-4B GDN, Gemma-4)
make bench               # full benchmark suite across baseline models
make verify-fast         # build + filtered tests + perf baseline + 1 smoke prompt (~90s)
make verify              # full pre-merge gate (~5min)
make install-hooks       # install pre-push hook → runs verify-fast on src/include/tools/tests changes

# Host build (no Docker)
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
./build/imp-tests                                  # ~700 tests across 63 files
./build/imp-tests --gtest_filter="TensorTest.*"    # specific suite
./build/imp-tests --gtest_filter="LlmCompressorE2E.*"   # NVFP4 SafeTensors prequant (Mistral, Gemma-4, Qwen-Coder-30B, Qwen3.6)
```

Test files live in `tests/` (Google Test). Most CUDA tests require sm_120; CPU-only tests are filtered by `make test-unit`.

## Tools

### imp-cli

Interactive and single-shot LLM inference. Supports both GGUF files and SafeTensors directories.

```bash
./build/imp-cli --model path/to/model.gguf --prompt "Hello world"
./build/imp-cli --model path/to/Qwen3-Coder-30B-A3B-FP4/ --prompt "Hello"
./build/imp-cli --model path/to/model.gguf --interactive
```

Options: `--model`, `--prompt`, `--max-tokens`, `--temperature`, `--top-p`, `--top-k`, `--seed`, `--interactive`, `--device`, `--mmproj`, `--image`, `--chat-template`, `--bench`.

### imp-server

OpenAI-compatible HTTP server with SSE streaming. Runs in Docker via `docker compose up imp-server` (pairs with Open WebUI on port 3000) or directly: `./build/imp-server --model path/to/model.gguf --port 8080`. Configuration via env vars — see "Environment Variables" below.

### imp-bench

Benchmarks for GEMM, attention, and end-to-end inference.

```bash
./build/imp-bench
```

## Benchmarks

Live baselines: `tests/perf_baseline.json` (consumed by the verify gate). Refresh after intentional perf changes via `scripts/gen_perf_baseline.sh`. Historical numbers: `BENCHMARKS.md`. llama.cpp comparison harness: `bench_compare.sh`.

**Notes:**
- Decode (tg256) is stable and the reliable A/B signal. Prefill (pp512) has up to 2.6× variance from cuBLAS autotuning across container restarts — do not gate on it.
- GDN models (Qwen3.5/3.6) use FP16 prefill instead of FP8 (~8% slower but eliminates multi-turn state collapse).
- imp uses NVFP4 decode cache + FP8 prefill (non-GDN) / FP16 prefill (GDN) by default.

## Code Conventions

### Language and Standards
- **C++20** for host code, **CUDA C++20** for device code
- Public API is C-compatible (`extern "C"` in `include/imp/`)
- Internal code uses the `imp` namespace

### Naming
- Classes and structs: `PascalCase` (`GraphExecutor`, `KVCacheManager`)
- Functions and methods: `snake_case` (`forward_batch`, `allocate_blocks`)
- Member variables: `trailing_underscore_` (`model_`, `config_`)
- Constants: `kPascalCase` (`kMaxDims`, `kKVBlockSize`)
- Enums: `PascalCase` values (`DType::FP16`, `OpType::ATTENTION_PREFILL`)
- C API: `imp_` prefix with `snake_case` (`imp_model_load`, `imp_context_create`)
- Macros: `IMP_UPPER_CASE` (`IMP_LOG_ERROR`, `IMP_CUDA_CHECK_LOG`)

### File Organization
- Headers (`.h`) and implementations (`.cpp` / `.cu`) are co-located in `src/` subdirectories
- Public headers live in `include/imp/` and use `#pragma once`
- CUDA files use `.cu` extension; pure C++ uses `.cpp`
- Each `src/` subdirectory corresponds to a logical module (core, compute, memory, model, quant, graph, runtime, vision, api)

### Error Handling
- C API returns `ImpError` codes (negative values indicate errors, 0 = success)
- Internal C++ code uses `bool` return values (true = success) with logging
- Logging uses `IMP_LOG_DEBUG/INFO/WARN/ERROR/FATAL` macros (defined in `src/core/logging.h`)
- CUDA errors are checked and logged (not thrown as exceptions)

### Memory Management
- GPU memory: `device_allocator.cu` with stream-ordered `cudaMallocAsync`/`cudaFreeAsync` + `cudaMemPool`
- Pinned host memory: `pinned_allocator.cpp`
- KV cache: block-based allocation with configurable block size (`kKVBlockSize = 16` tokens)
- Model weights: mmap'd from disk, then uploaded/dequantized to GPU
- `std::unique_ptr` and `std::shared_ptr` for ownership; raw pointers for non-owning references

### Compiler Flags
- C++: `-Wall -Wextra -Wpedantic`
- CUDA: `--expt-relaxed-constexpr --extended-lambda`
- Debug builds define `IMP_DEBUG=1`
- Release builds use `-O3 --use_fast_math`

## Architecture Notes

### Inference Pipeline
1. **Model Loading** — GGUF or SafeTensors parsed and weights mmap'd (`src/model/`). SafeTensors BF16 weights auto-converted to FP16. NVFP4 prequant scales (weight_scale, weight_scale_2) uploaded separately.
2. **Weight Upload** — Weights dequantized/converted and uploaded to GPU (`weight_upload.cu`). BF16→FP16 conversion for SafeTensors non-quantized weights.
3. **Graph Construction** — Transformer DAG built for visualization/debug (`src/graph/`)
4. **Execution** — `GraphExecutor` runs a hardcoded forward pass (no graph walking at runtime)
5. **Scheduling** — `Scheduler` manages continuous batching with prefill/decode separation
6. **KV Cache** — Paged block allocation with LRU eviction and prefix caching
7. **Sampling** — Temperature, top-p, top-k sampling from logits

### Attention Dispatch
Runtime dispatch (SM120 only, no architecture checks):
- **Prefill**: MXFP4 FMHA (`attention_fmha_mxfp4_sm120.cu`, if enabled) → FP8 FMHA (`attention_fmha_sm120.cu`) → FP16 FMHA → Blackwell WMMA 128x64 (`attention_blackwell.cu`)
- **Decode**: Paged attention with split-K (`attention_paged.cu`, `attention_paged_fp8.cu`)
- Overrides via `imp.conf`: `attention.mxfp4 = "always"` (force MXFP4 prefill), `attention.fp8_fmha = "never"` (force FP16), `attention.fmha_sm120 = "never"` (force WMMA fallback). Per-run via `--set` CLI flag. Legacy `IMP_MXFP4_ATTENTION` / `IMP_NO_FP8_FMHA` / `IMP_NO_FMHA_SM120` env vars still work as dev escape hatches in `attention_dispatch.cu` but are no longer the supported interface.

### Quantization Support
- **FP8 E4M3**: Per-tensor scale, FP8 GEMM via cuBLAS
- **INT8**: Per-channel dequantization
- **INT4 (Q4_0, Q4_K_M)**: GGML-compatible block formats
- **NVFP4 (FP4_E2M1)**: Blackwell-native, two-level micro-scale + tensor-scale
- **NVFP4 Prequant (Model Optimizer)**: SafeTensors models with calibrated NVFP4 weights (AWQ/SmoothQuant). Loaded directly — no re-quantization. BF16 non-quantized weights (norms, router, embeddings) auto-converted to FP16.
  - **Shape convention**: SafeTensors NVFP4 `weight` arrives as `U8` (loader → INT8 → Phase-0 promote → NVFP4) with `shape[1] = K_logical/2` (two FP4 nibbles per byte). Phase-0 promote in `executor_pre_dequant.cu` only changes qtype + populates `.scales` / `.tensor_scale` sidecars — it does **not** change shape. Existing dispatch in `executor_attention.cu` / `executor_ffn.cu` recovers logical K via `tmp.K = hw->shape[1] * 2`.
  - **MoE path**: For SafeTensors prequant the loader writes per-expert tensors only (`expert_w_*[e]`); `expert_*_packed.data` is null. `cache_moe_native_nvfp4` (`executor_pre_dequant.cu`) builds one contiguous `[ne, N, K_packed]` packed buffer per layer per projection by D2D-memcpy from the per-expert tensors, populates `wcache_.nvfp4_moe`, and frees the per-expert allocations inline (the legacy fallback can't reach a layer where `nvfp4_moe_*_ptr` is non-null, and keeping both copies wouldn't fit in 32 GiB on Qwen3.6-35B-A3B). Without this path the legacy FP16 dequant + cuBLAS sm_80 WMMA fallback fires per layer per token, killing CUDA Graphs.

### Gated DeltaNet (GDN) — Qwen3.5
Hybrid architecture: 24 GDN layers (recurrent) + 8 attention layers + 32 dense FFN layers.
- **Delta rule scan**: Recurrent state `H[n_heads, state_size, head_dim]` updated via `H = g*H + k*(v - g*H@k)*beta`. State cached in registers during fused multi-token kernel.
- **Fused scan kernel**: Single launch processes all prefill tokens. Register-cached state eliminates per-token global memory round-trips (125x less state traffic).
- **Fused RMSNormGated+SiLU**: `y = rmsnorm(y) * silu(gate)` in one kernel for all tokens × heads.
- **Attention output gate**: Q+Gate interleaved projection, de-interleaved before attention, sigmoid gate applied before Wo projection.
- **Partial RoPE**: Only first `rope_dim` (64) of `head_dim` (256) dimensions get rotary encoding.
- **CUDA Graphs**: Enabled for GDN decode (recurrent state updated in-place with fixed pointers).
- **Norm handling**: Qwen3.5 uses `post_attn_norm` as FFN input norm (NOT sandwich norm like Gemma-3). The `has_post_attn_norm` flag in `run_attention` requires BOTH `post_attn_norm` AND `ffn_norm` to distinguish true sandwich norm from FFN-input-norm pattern.
- **V-head tiling**: GGUF converter reorders V heads to tiled order. Kernel uses `h % n_groups` for group mapping. All tensors (V, gate, alpha, beta, A_log, dt_bias, conv1d, ssm_out) are consistently tiled.

### CUDA 13.2 Features
- **Green Contexts**: SM partitioning for concurrent prefill/decode (`green_ctx.cu`)
- **PDL (Programmatic Dependent Launch)**: Overlaps kernel tails with next kernel heads (`pdl.cu`)
- **CUDA Graphs**: Captured decode iterations for reduced launch overhead (`cuda_graph.cu`). NVFP4-prequant MoE models (Qwen3.6, Gemma-4 llm-compressor NVFP4) capture cleanly — `cache_moe_native_nvfp4` builds the contiguous per-layer expert buffer and the decode fast-path runs entirely device-side (no D2H expert-offsets sync). GGUF MoE decode still falls through the legacy expert-routing path with a D2H sync per layer per token and is graph-incompatible. Prefill is never captured (variable n).

### Supported Model Architectures
- LLaMA (dense transformer)
- Mistral (GQA variant)
- Mixtral (Mixture-of-Experts)
- DeepSeek (MoE)
- Qwen3 / Qwen3-MoE
- Qwen3.5 / Qwen3.5-MoE (Gated DeltaNet hybrid — GDN + Attention + dense FFN)
- Qwen3.6 (35B-A3B GDN+MoE hybrid)
- Gemma-3 (text + vision via SigLIP encoder)
- Gemma-4 (26B-A4B MoE; FP32 router, host gate_up split, decode fast-path supports CUDA Graphs)
- Nemotron-H (Mamba2 + Attention + MoE hybrid)
- Generic fallback

### Runtime Configuration

Runtime knobs are read from `imp.conf` (TOML-subset). All previous
`IMP_*`-prefixed environment variables (~50 of them) have been replaced
by sectioned keys in this file. See `imp.conf.example` in the repo root
for the full schema with defaults and inline comments.

**Loading precedence** (first non-empty wins):
1. `--config <path>` CLI flag
2. `$IMP_CONFIG` environment variable
3. `./imp.conf` (working directory)
4. `~/.config/imp/imp.conf`
5. embedded defaults (no file)

**Per-run overrides** on top of the loaded config:

```bash
imp-cli --set kv_cache.dtype=fp8 --set runtime.cuda_graphs=never \
        --model X.gguf --prompt "..."
```

**Common keys** (excerpt — see `imp.conf.example` for the rest):

| Section / Key | Default | Effect |
|---|---|---|
| `runtime.cuda_graphs` | `"auto"` | `auto` / `always` / `never` |
| `runtime.deterministic_gemm` | `false` | Pin cuBLAS algo for byte-stable runs |
| `runtime.warmup` | `true` | Run warmup forward at engine init |
| `runtime.no_pdl` | `false` | Disable Programmatic Dependent Launch |
| `kv_cache.dtype` | `"fp16"` | `fp16` / `fp8` / `int8` / `int4` / `nvfp4` |
| `attention.fp8_prefill` | `"auto"` | `auto` / `never` |
| `attention.fp8_fmha` | `"auto"` | `auto` / `never` |
| `attention.fmha_sm120` | `"auto"` | `auto` / `never` |
| `attention.mxfp4` | `"auto"` | `auto` / `always` (MXFP4 prefill FMHA) |
| `moe.expert_overhead_pct` | `10` | 10 = aggressive, 30 = conservative |
| `moe.force_host_experts` | `0` | Force last N MoE layers to host |
| `gdn.fp32_scan` | `false` | FP32 GDN scan (slower, higher precision) |
| `gemma4.no_graphs` | `false` | Bypass Gemma-4 CUDA graph capture |
| `generation.think_budget` | `0` | Max tokens spent in `<think>...</think>` |
| `paths.mmproj` | `""` | mmproj.gguf for vision (Gemma-3) |
| `diagnostics.dump_hidden_dir` | `""` | Per-layer hidden-state .npy dumps |
| `diagnostics.exit_layer` | `-1` | Stop forward at layer N |

**Build-time only env vars** (kept as ENV because they shape the build):

| Variable | Default | Effect |
|---|---|---|
| `IMP_BUILD_TESTS` | ON | Build the GTest suite |
| `IMP_BUILD_TOOLS` | ON | Build imp-cli and imp-bench |
| `IMP_BUILD_BENCH` | ON | Build benchmark tool |
| `IMP_BUILD_SERVER` | ON | Build imp-server |
| `IMP_SANITIZERS` | OFF | ASAN + UBSAN (host C++ code only) |
| `IMP_CONFIG` | unset | Path to imp.conf (overrides search-path) |

### Vision (Multimodal)
Gemma-3 vision uses a frozen 400M-parameter SigLIP ViT that produces 256 image tokens per image, projected into the LLM's embedding space. The vision encoder weights ship as a separate `mmproj.gguf` file. The pipeline: load image → resize 896x896 → normalize → extract 14x14 patches → 27 SigLIP transformer layers → 4x4 avg pool → RMSNorm + linear projection → replace `<image_soft_token>` embeddings before LLM prefill.

### Speculative Decoding
Draft model generates K candidate tokens, target model verifies in a single pass. Uses stochastic acceptance for non-greedy sampling. KV cache manager supports rollback for rejected tokens.

## Verification Before Commit

**Every change MUST be verified in this order before `git add`, `git commit`, and `git push`:**

1. **Tests** — `make test-gpu` (or `./build/imp-tests`). All 606 tests must pass.
2. **Performance** — `make verify-fast` runs the perf baseline gate (`tests/perf_baseline.json`, 3% decode / 5% prefill thresholds). Refresh the baseline after intentional perf changes via `scripts/gen_perf_baseline.sh`.
3. **Real prompts** — `--prompt "..."` on at least 2-3 affected models to confirm coherent output (degeneration detector covers this in `verify-fast` smoke step).

`make install-hooks` wires `verify-fast` into the pre-push hook so this gate runs automatically on `src/`, `include/`, `tools/`, `tests/` changes.
