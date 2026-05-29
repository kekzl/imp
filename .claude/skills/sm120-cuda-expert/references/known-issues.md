# sm_120a Known Issues, Dead Ends, Root-Cause Reference

Heavy reference for the `sm120-cuda-expert` skill. Things that don't work, things that *used* to not work but now do, and historical bugs whose fixes are load-bearing for current hot-path code.

---

## Pre-flight before non-trivial kernel work

1. Check the dead-ends list below — many "obvious" optimizations are proven failures on sm_120.
2. If the installed CUDA version differs from when a dead end was tested (check `nvcc --version`), see "Version-dependent dead ends" — some are worth retrying.

For small edits (parameter tweak, kernel-signature change, fusing two existing kernels), skip pre-flight and go straight to the patterns.

---

## Version-dependent dead ends (worth retrying when CUDA version changes)

> **CUDA 13.3 re-test (2026-05-29, PTX ISA 9.3): NO new sm_120a capability.** Full
> `ptx_survey_all.sh` at `compute_120a` under 13.2 vs 13.3 = **0 of 247 instructions
> flipped** (none unlocked, none regressed). The "retry on CUDA 13.3+" rows below were
> re-probed and stay ❌. sm_120's ISA surface is silicon-fixed; toolkit bumps don't add
> tcgen05/wgmma/TMA. Baselines: `docs/ptx-status-2026-05-29-cuda13{2,3}-sm120a.md`.
> 13.3's value is tooling (CUDA Tile C++, CompileIQ) + cuBLAS perf, not instructions.

| Dead end | Blocked by | Retry on |
|----------|-----------|----------|
| cuBLASLt grouped layout sm_120 | Zero algorithms for consumer Blackwell | New cuBLAS release (check algorithm count) |
| CUTLASS TC GEMM at M=1 | Activation quant + TMA overhead | CUTLASS 4.5+ |
| `cp.async.bulk` with `.ignore_oob` | Requires TMA descriptor rewrite | ~~CUDA 13.3+~~ still ❌ on 13.3 — TMA not on sm_120; next major |
| `st.async .b128` to global | PTX 9.2 only targets `shared::cluster` | ~~New PTX ISA~~ still ❌ on PTX ISA 9.3 (13.3) |
| CUTLASS NVFP4 sm_120 graph-determinism | Universally non-deterministic for `cudaGraphExecUpdate` re-capture (verified 2026-05-05) | Future CUTLASS NVFP4 deterministic mode |
| Native FP4 GEMM faster than dequant→cuBLAS on sm_120 | Marlin-style dequant→cuBLAS measured *faster* than FlashInfer-CUTLASS-NVFP4 on consumer Blackwell. Storage-format wins (4× VRAM) but compute-speed parity is open. | Future custom kernel or upstream CUTLASS fix |

---

## Resolved (no longer dead ends)

- **PTX `cvt.rn.satfinite.e2m1x2.{f32,f16x2,bf16x2}`** and reverse direction work on both `sm_120f` and `sm_120a` under CUDA 13.2.1 (re-verified 2026-05-04). Correct usage routes the FP4 byte through a `.b8` register — see `references/ptx-patterns.md` "FP4 ↔ FP16/FP32/BF16 packed conversion". SASS confirms hardware emission: `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C`.

- **Build target `sm_120a`** (was historically blocked by a `ptxas` C7600 bug on `120f` that needed the `f` workaround). As of CUDA 13.2.1 the `a` arch suffix is the correct target — superset of `120f`, adds `mma.sync.kind::mxf4nvf4.block_scale` and TMA-WS-Grouped-GEMM. Switched 2026-05-04 (commit `6568652`).

- **CUDA Graphs + prequant-NVFP4 MoE.** Earlier "non-Gemma-4 MoE blocks graph capture" claim was stale. The MoE decode fast-path (`executor_forward_moe.cu:524`) is fully device-side, no D2H sync — graph-safe. Verified 2026-05-07 across Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 (all +193%–234% decode vs `--no-cuda-graphs`). GGUF MoE prefill paths still use D2H sync, but prefill isn't graph-captured anyway. Hybrid Mamba2 (Nemotron-H) does NOT benefit yet — SSM layers don't fast-path.

- **Lever 1 SSM dispatch (commit `5b2c5db`).** Registered `ssm_in`/`ssm_out` in the `cutlass_nvfp4_cache` so GDN/SSM weights hit the fast NVFP4 GEMM path. Showed +95–376% decode on Qwen3.5/3.6 GDN families on 2026-05-04 — but the gain came from CUDA Graph capture *enabled by* the faster GEMM, not the GEMM speedup itself. **Always re-bench graphs ON after a hot-path kernel change.**

---

## Load-bearing root-cause fixes (don't regress these)

These bugs were diagnosed at high cost. The current kernels assume the fix is in place.

| Fix | Symptom if regressed | Where |
|-----|----------------------|-------|
| **FP8 FMHA S_tile pointer advance** | Long-context cliff at prompt > 1024 tokens | `attention_fmha_sm120.cu` — pointer must advance with `sizeof(half)`. Regression test in tree. |
| **Qwen3.5/3.6 GDN `__launch_bounds__(HD,1)` not `(HD,2)`** | HD=128 GDN miscompile, garbage output | GDN kernel — keep `(HD,1)`. |
| **Qwen3.5 partial RoPE pair offset `+ rope_pairs` (not `+ head_dim/2`)** | Sister bug to launch_bounds; partial-RoPE corruption | RoPE kernel. |
| **Qwen3.5 Q8 α/β qtype consistency** | Pre-dequanted Q8→FP16 without updating qtype → dispatcher mis-interprets bytes → state collapse | `upload_weight` path — keep qtype tag in sync with stored bytes. |
| **Qwen 3.6 h_state FP32 gate + PyTorch L2 norm** | NaN at L38 in GDN | h_state must be FP32, not FP16/BF16. |
| **Gemma-4 per-layer `rope_freqs` for non-SWA layers, `n_rot=hd`** | L13/L14 drift 11–15% (was) → <2% (fixed) | Pass per-layer rope_freqs through. |
| **MoE expert-offload auto-probe at 10% before falling back to 30%** | Qwen3-Coder-30B Q6_K decode 234 → 77 tok/s | `executor_moe.cu` — keep the 10% probe. |
| **L2 access-policy window `num_bytes` clamp to `cudaDevAttrMaxAccessPolicyWindowSize`** | Silent CUDA error / IMA on 5090 (128 MiB max) | `set_l2_streaming` / `set_l2_persist_kv` in `runtime/`. |
| **NVFP4 dequant graph-safe fallback (PR #121)** | `cudaMallocAsync` inside captured graph crash | `set_nvfp4_dequant_workspace()` + capture-guard in `ensure_dequant_buffer`. |

---

## Performance-relevant scaling rules

| Rule | Source |
|------|--------|
| Decode at batch=1: launch overhead first, memory second (post Lever 1) | Three Laws #1 in main SKILL.md |
| `__launch_bounds__` cost on regular paths: -4.5% to -20% | Repeated benchmarks 2026-04 to 2026-05 |
| `mxf4nvf4.block_scale` raw MMA: 2.60× over f8f6f4 | `mxf4nvf4_mma_bench` 2026-04-25 |
| CUDA Graph decode on prequant NVFP4 MoE: +193% to +234% | Qwen3-Coder, Qwen3.6, Gemma-4 NVFP4 — verified 2026-05-07 |
| pp512 cuBLAS-autotune variance: up to 2.6× across container restarts | Use `tg256` for A/B; see `benchmark-cuda` skill |

---

## Negative results (don't repeat)

- **Generic `compute_120` PTX fallback.** Lacks FP8 MMA + block-scale. Always pin `compute_120a/sm_120a`.
- **Async `wgmma` / `tcgen05` / TMEM on consumer Blackwell.** Not available — SM100 (B200) exclusives. sm_120 peak path is register `mma.sync`. (Note: the *synchronous* `nvcuda::wmma` API *does* compile on sm_120 but lowers to **HMMA** — it is not async wgmma and not the peak path; it costs extra smem traffic and a smem round-trip vs hand-written `mma.sync` with register-resident fragments.)
- **Materializing the attention score tile (S/P) in shared memory.** A FA-style kernel that writes S to smem, runs softmax over smem, then reads P back for the PV MMA becomes **barrier- / L1-TEX-bound** (tensor cores idle, compute util in the teens) — the smem round-trip + `__syncthreads` dominate, not the MMAs. True FA2 keeps row max/sum and the S/P fragments **register-resident** and fuses softmax into the QK→PV handoff. Don't trust a kernel header that *claims* register-based softmax — verify against the code (some in-tree kernels are mislabeled).
- **`__noinline__` on device inner-loop helpers.** Spills to Local Memory (DRAM). Use `__forceinline__`.
- **`reinterpret_cast` on Q8_0 blocks.** 34-byte blocks NOT 4-aligned. Use `memcpy()`.
- **Skipping graph re-bench after a hot-path patch.** Compute speedup alone often shows ~0% in tok/s — the win is graph-replay-mediated. Always re-bench graphs ON.
- **Increasing SMEM beyond `cudaDeviceProp::sharedMemPerBlockOptin`** assuming H100's 228 KB. RTX 5090 max is ~99 KB.
