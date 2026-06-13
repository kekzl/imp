# DISPATCH — FlashAttention-2 via CUDA 13.3 Tile (C++)

Progress log for the autonomous Tile-FA2 mandate (see `GOAL.md`). A fresh run
continues from "Next step" at the bottom. Newest entry last.

## Mission (short)
Build a production Tile-C++ FA2 attention path (prefill + paged-decode) that meets
or beats the native hand-tuned path, gated behind `--attn-backend=tile` (default
`native`) until proven faster at equal correctness. Anti-degeneration contract (§4)
is absolute: no metric regresses, native path is never deleted.

## Environment (verified 2026-05-29)
- Toolkit **CUDA 13.3** (nvcc V13.3.33, PTX ISA 9.3); host driver **610.47** (≫ R590). §11 preconditions OK.
- Build: Docker `imp:builder-133` (cuda-toolkit-13-3). Dev container `impdev` (host `/home/kekz/models` mounted).
- imp already switched to 13.3 (Dockerfile builder+runtime); full GPU suite green, PR #477 (hand-written FA2) open.

## Phase 0 — Investigation (§5) — DONE, decision = GO
CUDA Tile C++ structural viability for FA2 on sm_120a, evidence-based:
- API: header `cuda_tile.h` → `crt/cuda_tile.h`, namespace `cuda::tiles` (alias `ct`). Kernel attr
  `__tile_global__`. Ops: `ct::tensor_span{ptr, ct::extents{...}}`, `ct::partition_view{span, ct::shape{...}}`,
  `.load_masked(blockIdx...)` / `.store_masked(tile, blockIdx...)`, `ct::mma(A,B,acc)`, `ct::matmul(A,B)`,
  `ct::full<tile<E,shape<...>>>(v)`, `ct::bid()`, `ct::irange()`, literals `_ic`. Compile `nvcc -std=c++20 --enable-tile -arch=sm_120a`.
- **Arch:** NVIDIA blog confirms cuTile supports "Blackwell (compute capabilities 10.x and 12.x)" → **CC 12.x = sm_120 = RTX 5090 officially supported.**
- **SASS proof (`tools/analysis/tile_probe_mma.cu`, fp16 64×64×64, AOT via `--tilecubin`):
  lowers to `HMMA.16816.F16` (= mma.sync m16n8k16). NO tcgen05/UTCMMA.** sm_120-runnable. §11 NOT triggered.
- **dtypes (matmul):** int8, fp8 e4m3/e5m2, fp16, bf16, tf32, float, double. **No e2m1/NVFP4.** Irrelevant for
  attention (KV is FP16/FP8/INT8); NVFP4 is GEMM-weight-only. Documented gap (§5.6), not a blocker.
- **⚠ Runtime JIT unavailable in WSL2 container** ("PTX JIT compiler library not found" — driver dlopen fails even
  with caps=all + WSL ptxjitcompiler on path). Tile IR is JIT'd at launch by default. **→ imp MUST AOT-compile tile
  code** (tileiras → SASS/cubin embedded), which is required anyway (§10, no runtime deps). AOT confirmed working.
- Probes: `tools/analysis/tile_probe.cu` (fp32 tiny → scalar FMA, ran but WSL-JIT no-op), `tile_probe_mma.cu`
  (fp16 → HMMA, AOT SASS inspected). Both committed as investigation artifacts.

### BREAKTHROUGH — Tile kernel RUNS CORRECTLY on sm_120a (2026-05-29)
Tile matmul executed on the RTX 5090, `max_abs_err=0`, `launch err: no error`. The earlier
"PTX JIT compiler library not found" was a **WSL2 driver-injection mismatch**: `--gpus all` injected a
STALE WSL driver path (`nv_dispi.inf_amd64_c8bc842500fab35b`) while the live host driver is
`...b7cca8360c0d57e9`. Fix = mount the live WSL driver tree + point the loader at it.

**WORKING DEV RUN RECIPE (Tile JIT path, for build/test loop):**
```
CUR=/usr/lib/wsl/drivers/nv_dispi.inf_amd64_b7cca8360c0d57e9   # = `ls -d /usr/lib/wsl/drivers/nv_dispi*` (live one)
docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v /usr/lib/wsl:/usr/lib/wsl:ro -e LD_LIBRARY_PATH="$CUR:/usr/lib/wsl/lib" \
  -w /src impdev:ncu  ./build/<binary>
```
(Note: the inf hash can change across host driver updates — recompute `CUR` each session.)
- Compile: `nvcc -std=c++20 --enable-tile -arch=sm_120a` (image `imp:builder-133`, toolkit 13.3.33).
- Host toolkit is only 13.2 (no 13.3) → cannot compile tile on host; Docker `imp:builder-133` is the only tile-capable toolchain.

### Open integration questions (resolve in Phase 1)
1. **AOT vs JIT for production:** dev/test works via JIT (recipe above). For shipping, imp wants AOT (§10) —
   `--tilecubin` produces a device-only cubin with AOT HMMA SASS (`.text._Z..`, 25 KB smem, `.note.nv.tkinfo`).
   Decide: ship JIT (embed tile IR, requires the driver JIT lib at runtime — fine on a correctly-provisioned box)
   vs AOT (separate tile-cubin compile + `cuModuleLoad`, or find the nvcc flag that embeds AOT tile SASS in the
   linked fatbin). Resolve before any perf/CI claim; CMake must build tile TUs for `sm_120a` (+`sm_120f`).
2. **Tile execution model:** `<<<grid,1>>>` launch; the tile compiler rewrites the launch + maps the tile across
   threads/warps (SASS shows SHFL.BFLY + 25 KB smem for the 64×64 fp16 tile). Confirm block/warp mapping + smem
   budget for an attention tile (Bq×Bkv scores + online softmax).
3. **Online softmax in Tile:** FA2 needs running max/sum + rescale across KV tiles. Determine the cuTile idiom
   (tile-axis reductions, `ct::` exp/max/reduce ops) — official "Tuning Flash Attention ... CUDA Tile" blog + TileGym.

## FA2 algorithm → C++ cuda::tiles op mapping (Phase 2 design, grounded in official FA-tuning blog + header)
Online softmax (FA2), per KV tile j, all ops confirmed present in `crt/cuda_tile.h` C++ API:
```
acc  = ct::full<tile<float, shape<TILE_M, TILE_D>>>(0);     // O accumulator
m_i  = ct::full<tile<float, shape<TILE_M,1>>>(-INF);        // running row max
l_i  = ct::full<tile<float, shape<TILE_M,1>>>(0);           // running row sum
q    = qView.load_masked(...);                              // Q tile (once)
for j in ct::irange(0, Tc):
  k   = kView.load_masked(..., order transpose);            // K^T tile
  qk  = ct::mma(q, k, ct::full<...>(0));                    // QK^T  (HMMA.16816 on sm_120)
  // mask: ct::select(cond_from_ct::iota(), qk, -INF)       // causal/SWA
  mij = ct::max(m_i, ct::reduce_max(qk, axis=last));        // reduce_max (line 2304) + broadcast
  alpha = ct::exp2(m_i - mij);                              // exp2 (line 2869)
  acc = acc * ct::broadcast(alpha);                         // rescale (mutual_broadcast operator*)
  p   = ct::exp2(qk - ct::broadcast(mij));
  l_i = l_i * alpha + reduce_sum(p, axis=last);             // reduction-family
  v   = vView.load_masked(...);
  acc = ct::mma(p, v, acc);                                 // P·V
out = acc / ct::broadcast(l_i);                             // ct::div / operator/
```
Confirmed C++ builtins: `mma`, `matmul`, `reduce_max`, reduction_result_t family (sum), `exp2`/`exp`, `tanh`
(softcap), `rsqrt`/`sqrt`, `select`, `iota`, `broadcast`, `reshape`, `extract`, `full`/`ones`, `div`,
`fma`, `abs`, mutual-broadcast arithmetic operators, `load`/`load_masked`/`store`/`atomic_*`, `partition_view`.
Tile sizes from blog: 64×64 baseline, 256×128 optimized (autotuned per seqlen). CC 8.x/10.x/11.x/12.x (sm_120 ✓).

## Status: Phase 0 COMPLETE + Phase 2 PROTOTYPE CORRECT (non-causal).
- §11 viability GO (HMMA on sm_120, runs correctly) + C++ API completeness GO.
- **`tools/analysis/tile_fa2_probe.cu`: standalone CUDA Tile C++ FA2 prefill kernel, fp16, S=128 D=64,
  RUNS CORRECTLY on sm_120a (NON-CAUSAL **and** CAUSAL), `max_rel_err=0.00000` vs CPU oracle.** The full
  online-softmax op-mapping works: `mma → causal-mask(iota/`/`,`%`/select) → reduce_max(1_ic) → select(m>rmax) → exp → auto-broadcast (acc*alpha, qk-mij, acc/l) → mma(P·V)`. Causal mask: `ct::iota<tile<int,shape>>()` flat idx → row=idx/TN col=idx%TN, `select(gcol>grow, -inf, qk)`. Arithmetic auto-broadcasts [TM,D] op [TM,1]; binary max via
  `ct::select` (no binary-max builtin); fp32→fp16 via explicit `ct::tile<__half,shape>(p)` ctor; view
  dims need `_ic` literals. Built with `nvcc -std=c++20 --enable-tile -arch=sm_120a`, run via WSL recipe.
- Standalone proto — NOT yet integrated into imp (zero degeneration risk; native path untouched).

## IMMEDIATE NEXT STEP (decisive experiment — fresh context)
**Autotuned cuTile perf ceiling** — the one experiment that decides whether Tile is worth productionizing:
write the FA2 in **Python cuTile** (`import cuda.tile as ct`; `@ct.kernel`; `ct.load/ct.mma/ct.exp2/ct.store`;
native `ct.Array`, no PyTorch needed — `cuda.tile.tune` is a toolkit module, TileGym/torch optional) and run
`cuda.tile.tune` over tile sizes (64×64 … 256×128) for S∈{2048,4096} D=128 causal. Measure best eff-TFLOPS,
run via the WSL-driver recipe.
- If tuned cuTile ≥ hand-written native → productionize (port configs to C++ via ct::shape<> + hints; build
  AOT CMake integration; `--attn-backend=tile` then default-switch).
- If tuned cuTile < native (likely per Yadav 0.53×) → Tile stays a documented optional backend / shelved;
  redirect effort to CompileIQ on the native hand-FA2 (ptxas ACF, the real near-term win).
Entry points: `cuda.tile` submodules `tune`, `compilation`, `_execution`, `_load_libcuda`, `jax`.

## Next step (Phase 1 → 2)
Investigation + design grounding done. Remaining (multi-week, in order):
1. **Phase 1 baselines (§4.1):** freeze `bench/baseline.json` (decode/prefill/long-ctx: Qwen3.6-35B-A3B-NVFP4
   decode, Qwen3-Coder MoE prefill, Qwen3-14B-NVFP4 pp4096, plus a dense + MoE), capture golden logits +
   perplexity reference. (`baseline/fa2-tile-pre` tag already placed.)
2. **Decide JIT-vs-AOT deploy + CMake recipe** (open Q1) — a standalone host harness that builds a tile
   attention kernel TU and launches it on GPU (dev recipe in BREAKTHROUGH section), verified correct.
3. **Phase 2 — Tile FA2 prefill kernel** (`compute/attention_tile_fa2.cu`, new `--attn-backend=tile` /
   `attention.attn_backend` flag, default `native`): implement the op-mapping above for fp16 first, numeric
   parity vs the CPU oracle (reuse `tests/test_fmha_fp8.cu` harness) + vs native; gate on §4.2.
4. **Phase 3 — paged-decode** path; **Phase 4** — fp8/bf16 dtypes; **Phase 5** — autotune/CompileIQ; report.
The hand-written register-resident FA2 (PR #477, +20% pp4096, HMMA.16816) is the native baseline the Tile
path must beat — same HW instruction, so the contest is cuTile codegen/scheduling vs hand-tuned.

## Perf datapoint (2026-05-29) — naive Tile FA2 ≈ 24 eff-TFLOPS (un-tuned)
`tools/analysis/tile_fa2_bench.cu`: causal fp16 S=2048 D=128, 32 heads, device memory, cudaEvent.
- **1.41 ms/iter, 24.4 eff-TFLOPS (2.9% of 838 FP16 roofline). Correctness OK (max_rel_err=0.0000).**
- ⚠ **Bench lesson:** first run with `cudaMallocManaged` reported a bogus 0.44 TFLOPS (200× too slow) —
  WSL2 page-migration artifact. Use device `cudaMalloc`+memcpy for all Tile benches (benchmark-cuda skill).
- 24 TFLOPS is NAIVE (TM=TN=64, `ct::exp` not exp2, no autotuned tile sizes, no fast-math, no smem-pipeline
  hints via `__applicable_tile_hints__`). Blog reports 256×128 tiles + fast-math + autotune = much higher.
- **Open: head-to-head vs the hand-written FA2 (PR #477) at identical shape** — needs the hand kernel
  standalone or both wired in imp. THIS decides default-switch vs `--attn-backend=tile`-only. Next: tile-size
  sweep (exp2/fast-math, 128×128 / 256×128) to find naive-tuned ceiling, then the head-to-head.

### Tuning probes (2026-05-29) — naive knobs don't lift it; needs the autotuner
- `exp2` + `--use_fast_math` (vs `ct::exp`): **flat** (24.1 vs 24.4 TFLOPS) → softmax math is NOT the bottleneck.
- TN 64→256 (bigger KV tile): **worse** (14.6 TFLOPS) → larger tiles raise smem pressure / cut occupancy.
- ⇒ The ~24 TFLOPS naive ceiling is structural (occupancy / smem / scheduling), not knob-tunable by hand.
  Reaching competitiveness needs the cuTile **autotuner (TileGym)** + **CompileIQ**, i.e. the real Phase-5
  tuning effort — consistent with the prior datapoint (Yadav: cuTile ≈ 0.53× FA2 on sm_120). 
- **Provisional steer:** Tile likely lands as optional `--attn-backend=tile` (NOT default) unless autotuning
  closes the gap to the hand-written FA2 (PR #477). Exactly the conditional GOAL §1/§4.4 anticipated.

### CompileIQ applicability (2026-05-29) — ptxas/nvcc only, NOT tileiras
Installed `compileiq` v1.0.0 (NVIDIA, public PyPI). Its compiler search spaces are
`compiler: Literal["ptxas", "nvcc"]` (`search_spaces/compilers.py`: `PtxasSearchSpace`,
`NvccSearchSpace`) — **no `tileiras` provider.** It emits an ACF consumed by
`ptxas --apply-controls` / `nvcc --apply-controls`.
- **⇒ CompileIQ CANNOT tune the Tile FA2 kernel** — that codegen is done by `tileiras`, which
  CompileIQ has no search space for. The ~24 TFLOPS Tile ceiling needs the cuTile autotuner
  (TileGym), NOT CompileIQ. (Structural gap, documented per GOAL §5.6.)
- **CompileIQ IS applicable to the hand-written FA2 (PR #477)** and imp's other nvcc→ptxas hot
  kernels (NVFP4 GEMV/GEMM, etc.) — standard ptxas path. That is the productive CompileIQ target:
  last-mile ptxas tuning of the +20% hand-written kernel (Worker = build imp attn TU with candidate
  ACF → imp-cli pp4096 tok/s as score). Deferred to a focused session (per-candidate imp rebuild ~13s).
- API: `compileiq.ciq.Search` (pydantic) + `.start()`; `Worker` ABC = build+run+score; search spaces
  are GitHub-release-backed per (compiler, version) — verify an sm_120/13.3 ptxas space exists first.

### Tuning ecosystem (2026-05-29) — autotuners are Python; C++ path is manual
- **TileGym 1.3.0** (pip): the cuTile autotuner — but **requires PyTorch** and tunes the **Python** cuTile
  (`@ct.kernel` / `cuda.tile`) path, NOT C++ `cuda::tiles`. `cuda.tile` Python DSL is present (has `.tune`).
- **C++ `cuda::tiles` has no shipping autotuner.** Tuning knobs there: (a) tile shapes via `ct::shape<>`
  (manual; my 64×64→64×256 sweep got WORSE), (b) `__applicable_tile_hints__("latency"|...)` string hints
  (not tile sizes). Manual C++ exploration is the only in-mandate lever and was limited (~24 TFLOPS ceiling).
- **Bridge for competitive C++ perf (Phase 5):** use Python cuTile + TileGym/`cuda.tile.tune` to *discover*
  optimal (tile sizes, hints) for the FA2 shapes, then hand-port those configs into the C++ kernel. Requires
  PyTorch at dev time (not shipped — OK), plus a port step. Substantial + uncertain to beat hand-tuned native.

### STRATEGIC CONCLUSION (honest, per GOAL §1/§4.4/§6)
Tile FA2 in C++ is **viable + correct** but **naive perf (~24 TFLOPS, 2.9% roofline) does not beat the
hand-written FA2** (PR #477, +20%, same HMMA.16816), and the in-C++ tuning levers are weak. Reaching
competitive perf needs a Python-autotune→C++-port workflow (Phase 5), uncertain to win. Therefore the
realistic landing is **`--attn-backend=tile` as an OPTIONAL, non-default path with the native fallback
preserved** — exactly the GOAL §4.4 conditional. CompileIQ tunes the *native* ptxas kernels (incl. hand-FA2),
not Tile. No production code touched; native path intact. DoD remains multi-week (autotune-bridge + integration).

## DECISIVE EXPERIMENT DONE (2026-05-29) — autotuned cuTile ceiling = 26.5 TFLOPS → SHELVE
Ran the pinned experiment: a real Python cuTile FA2 (causal fp16 S=2048 D=128) + `cuda.tile.tune.exhaustive_search`.
- **Correctness:** max_rel_err=0.0 vs CPU causal oracle. cuTile DSL FA2 works on sm_120a (cupy arrays, live-driver mount).
- **Autotuned best = TM32/TN64 @ 40.5 µs/head → 26.5 eff-TFLOPS = 3.2% of the 838 FP16 roofline** (naive 64×64 = 59.5 µs ≈ 18 TFLOPS; autotune lift ~1.1–1.5×; 128-row tiles 5–7× worse). Tile-size is the only real lever, as predicted.
- **DECISION: SHELVE the Tile backend.** 3.2% roofline is ~order-of-magnitude below competitive and far below the native hand-FA2 (PR #477). Confirms Yadav (0.53× FA2 on sm_120a, same arch) + the naive 24-TFLOPS datapoint. The GOAL §4.4 conditional resolves to the negative branch.
- **Multi-week Tile FA2 integration (Phase 2-5): NOT BUILT — intentionally.** A production Tile prefill/decode backend would be strictly slower than native; building it would violate the "only if ≥ parity" contract. Tile remains investigated-and-shelved; the standalone prototype + this autotune ceiling are the deliverables. Native path untouched throughout.
- Harness: `tools/analysis/Dockerfile.cutile` (→ imp:cutile), `cutile_fa2.py`, `cutile_smoke.py`. Memory: `cutile_autotune_ceiling_shelve_2026_05_29`.
