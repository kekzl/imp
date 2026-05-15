# cuDNN NVFP4 MoE on SM120 — feasibility audit (2026-05-10)

## Question 1: Does cuDNN expose NVFP4 grouped MoE on SM120?

**Partial — documented support in release notes, but practically broken on SM120 and architecturally
mismatched. The cuDNN MoE grouped matmul was designed for SM100 (B200 datacenter), not SM120 (RTX
5090 consumer Blackwell).**

Evidence:

- **cuDNN 9.7.0** (release notes): *"Native matmul and convolution fusion support with TF32, BF16,
  FP16, FP8, MXFP8, NVFP4 input and output tensors has been added for compute capability 10.0 and
  12.0."* — This is dense matmul only, not grouped.
- **cuDNN 9.13.0** (release notes): *"A new MoE grouped matmul operation has been introduced to
  support grouped GEMM for Mixture-of-Experts workloads in the runtime fusion engine."* — FP4
  datatype is **not mentioned** in the 9.13.0 entry.
- **cudnn-frontend v1.21.0–v1.23.0** (the open-source C++/Python wrapper): `GroupedGemmQuantSm100`
  (note: explicitly named Sm100) supports `float4_e2m1fn_x2` (NVF4). The compute capability guard
  is `if compute_capability < 100: raise RuntimeError(...)`. SM120 passes this gate (120 >= 100).
- **Practical reality on SM120**: FlashInfer issue #2577 (March 2026) reports that the cuDNN
  backend returns `cudnnGraphNotSupportedError: No execution plans support the graph` on SM120.
  Tested backends were cuDNN, CUTLASS, and TRT-LLM — all failed.

**Bottom line**: cuDNN 9.13.0+ nominally covers SM120 (≥ SM100 check passes), and the
`GroupedGemmQuant` kernel does support `float4_e2m1fn_x2` (NVFP4). But in practice, as of May
2026, the cuDNN execution engine finds no valid execution plans on SM120 for NVFP4 grouped GEMM.
The kernel was validated on SM100 (B200) hardware, not SM120.

---

## Question 2: API surface (if exposed)

- **cuDNN version required**: ≥ 9.13.0 for MoE grouped matmul; ≥ 9.7.0 for NVFP4 dense matmul
- **cudnn-frontend version**: ≥ v1.21.0 (`GroupedGemmQuantSm100` / `grouped_gemm_quant_wrapper_sm100`)
- **API path**: cudnn-frontend high-level Python/C++ wrapper (not the legacy C frontend API, not
  pure backend API). Uses `cute.compile()` internally — this is a **CUTLASS-based kernel
  compiled at runtime via cuTile/cute DSL**, not a hand-optimized cuDNN graph plan.
- **Function signature (Python wrapper)**:

  ```python
  GroupedGemmQuantSm100(
      sample_a,        # torch.float4_e2m1fn_x2 or uint8 or fp8
      sample_sfa,      # block scales for A
      sample_padded_offsets,  # expert token offsets
      sample_alpha,    # per-expert scalar
      sample_d,        # output tensor (fp16 / bf16 / fp32 / fp8 / fp4)
      sample_b=None,   # weight tensor (optional for pre-built)
      sample_sfb=None, # block scales for B
      num_experts=None,
      b_shape=None,
      b_dtype=None,
  )

  # Then: instance.compile(); instance.execute(a, sfa, offsets, alpha, d, b, sfb)
  ```

- **PyTorch operator wrapper** (`cudnn.moe_grouped_matmul`):

  ```python
  result = moe_grouped_matmul(
      token,               # input activations
      weight,              # expert weights
      first_token_offset,  # routing offsets
      token_index=None,
      token_ks=None,
      mode="none",
      top_k=1,
  )
  ```
  Requires `cudnn.backend_version() >= 91800` (cuDNN 9.18.0+). Tested datatypes in NVIDIA's own
  test suite: **FP16 only** — no FP4 test as of v1.23.0.

- **Reference docs**:
  - https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/release-notes.html
  - https://docs.nvidia.com/deeplearning/cudnn/frontend/v1.22.0/fe-oss-apis/gemm_fusions/grouped_gemm_quant_unified.html
  - https://github.com/NVIDIA/cudnn-frontend/blob/main/python/cudnn/grouped_gemm/grouped_gemm_quant/api.py
  - https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_moe_grouped_matmul_op.py

---

## Question 3: Benchmarks on RTX 5090 / SM120

**Not measured publicly for cuDNN MoE NVFP4 on SM120.** The benchmark landscape on SM120:

| Backend | NVFP4 MoE prefill (SM120 RTX 5090) | Status |
|---|---|---|
| cuDNN `GroupedGemmQuantSm100` | Not measured | `cudnnGraphNotSupportedError` in practice |
| CUTLASS grouped 3.x (imp, vLLM) | ~14k tok/s pp512 (imp current) | Functional via `compute_120f` patch |
| FlashInfer CUTLASS SM120 patch | ~39 tok/s native FP4 (CUTLASS issue #3096) | Functional but slow at inference decode |
| Marlin W4A16 (FP4→FP16 dequant) | 80–100 tok/s decode (vLLM community) | Best decode fallback on SM120 |
| vLLM FlashInfer CUTLASS | ~25513 tok/s pp512 (cited in imp bench notes) | Likely via cuBLAS autotuned fallback path, not native FP4 |

The vLLM 25513 tok/s pp512 figure cited in imp's bench notes likely reflects vLLM using a **cuBLAS
grouped GEMM dequant path** (weights dequanted to BF16, cuBLAS does the GEMM), not native NVFP4
compute. This is consistent with `cutlass_nvfp4_sm120_nondeterministic` memo: "FlashInfer-CUTLASS
NVFP4 39 tok/s vs Marlin 46–49 tok/s" — native FP4 is *slower*, not faster, on SM120.

---

## Question 4: Linkage in imp:test

- **cuDNN headers in image**: No
- **libcudnn.so in image**: No
- **Base image**: `nvidia/cuda:13.2.1-devel-ubuntu24.04` — this is the `devel` (not `cudnn-devel`)
  variant. The Dockerfile's apt upgrade loop includes `libcudnn` patterns, but only upgrades
  packages that are *already installed*; the base devel image has none.
- **Verified**: `docker run --rm nvidia/cuda:13.2.1-devel-ubuntu24.04 bash -c 'dpkg -l | grep -i
  cudnn'` → no output.
- **If needed, install path**:
  - Switch base to `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` (cuDNN bundled), or
  - `apt-get install -y libcudnn9-dev` from NVIDIA's apt repo (ships cuDNN 9.x, version depends on
    repo snapshot).
  - cudnn-frontend C++ wrapper: header-only from https://github.com/NVIDIA/cudnn-frontend (no
    separate binary, links against libcudnn.so at runtime).

---

## Question 5: Open-source examples

**No open-source project uses cuDNN MoE NVFP4 on SM120 successfully.** Here is what each uses
instead:

**vLLM (current, as of v0.19.1)**:
- MoE NVFP4 on SM120: `FLASHINFER_CUTLASS` backend via FlashInfer's SM120-patched kernels
  (compute_120f workaround, merged after vLLM PR #29242 and related issues #31085, #33333).
- cuDNN is **not in vLLM's MoE backend list** (`CutlassExpertsFp4`, `FlashInferExperts`,
  `TrtLlmVariants` — no `CuDNNExperts`).
- Source: `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py` — NvFp4MoeBackend selection.

**FlashInfer**:
- Issue #2577 (March 2026) explicitly tested cuDNN backend for NVFP4 GEMM on SM120 → failed with
  `cudnnGraphNotSupportedError`. No fix shipped.

**cudnn-frontend itself**:
- `GroupedGemmQuantSm100` with NVF4 (`float4_e2m1fn_x2`) exists in Python wrapper.
- Compute capability check: `>= 100` (passes SM120 gate).
- But: `test_moe_grouped_matmul_op.py` only tests FP16, not FP4. No evidence this is validated
  on SM120 hardware. Named "Sm100" for a reason.

**Relevant files**:
- `https://github.com/NVIDIA/cudnn-frontend/blob/main/python/cudnn/grouped_gemm/grouped_gemm_quant/api.py`
- `https://github.com/NVIDIA/cudnn-frontend/blob/main/test/python/test_moe_grouped_matmul_op.py`

---

## Recommendation

**Do not pursue cuDNN as the path to close the vLLM prefill gap. Pivot analysis to understanding
what vLLM actually does for high-throughput NVFP4 prefill on SM120.**

Concrete reasoning:

1. cuDNN's `GroupedGemmQuantSm100` is targeted at SM100 (B200 datacenter). The "SM100+" check is
   permissive but the execution plans were tuned and validated on SM100. SM120 gets
   `cudnnGraphNotSupportedError` in practice.

2. The vLLM 25513 tok/s pp512 figure is almost certainly **not** from native NVFP4 grouped GEMM.
   At low-batch prefill on SM120, native FP4 is *slower* than dequant+BF16 GEMM (FlashInfer
   native FP4 = 39 tok/s vs Marlin dequant = 46–49 tok/s on decode; same pattern applies to
   prefill). The high vLLM prefill throughput likely comes from cuBLAS autotuned grouped GEMM on
   BF16 dequanted weights — which is exactly the `gemm_cutlass_grouped_3x.cu` path imp already
   uses.

3. The prefill gap (imp 14k vs vLLM 25k tok/s pp512) is most likely explained by:
   - vLLM using TRT-LLM's `fp4_gemm` with heavy autotuning across many tactics (per the
     vllm_comparison_2026_05_10 memo: "TRT-LLM fp4_gemm autotuned via flashinfer")
   - OR vLLM's chunked/pipelined prefill implementation handling larger batch sizes per chunk
   - This is multi-week CUDA kernel work, not a cuDNN API call away.

**Next concrete step**: Profile what vLLM actually dispatches for a pp512 prefill request on
Qwen3-Coder-30B-A3B-NVFP4 using `nsys`. The kernel names will tell us whether it's cuBLAS,
CUTLASS, or TRT-LLM's fp4_gemm. That profile, not this audit, is the right input for a gap-closing
plan.

---

## Effort estimate

- **cuDNN integration attempt**: Not recommended. Even if SM120 execution plans are eventually
  added by NVIDIA, `GroupedGemmQuantSm100` is a CUTLASS-based JIT kernel internally — it competes
  with what imp already does via `gemm_cutlass_grouped_3x.cu`. Net expected speedup: 0–5%.
- **If pursued anyway**: 1–2 days to add cuDNN dependency + wire the Python/C++ frontend API.
  High risk of hitting the `cudnnGraphNotSupportedError` wall.
- **Risk factors**:
  - cuDNN not in imp:test image (requires image rebuild)
  - No validated SM120 FP4 grouped execution plans in cuDNN ≤ 9.18.1
  - cudnn-frontend `GroupedGemmQuantSm100` uses cute.compile() internally — same CUTLASS paths imp
    already takes
  - cuDNN is a large runtime dependency (libcudnn.so ~300–500 MB); conflicts with imp's "no new
    third-party deps" rule
