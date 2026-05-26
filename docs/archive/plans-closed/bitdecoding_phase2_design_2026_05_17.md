# BitDecoding Phase 2 design memo — 2026-05-17

**TL;DR — Phase 2 is already shipped.** Both the Q·K and P·V GEMMs of the
NVFP4 paged attention decode kernel run on Tensor Cores in current `main`.
This memo is a status check that ends in "no Phase 2 work owed", with the
design space recorded for any future revisit of the same lever.

## Table of contents

1. [Status check — is Phase 2 still open?](#1-status-check)
2. [Current Phase 1 + Phase 2 architecture](#2-current-architecture)
3. [Phase 2 design (as actually shipped)](#3-phase-2-design-as-shipped)
4. [Risks & open questions](#4-risks--open-questions)
5. [Implementation plan (retrospective + revisit roadmap)](#5-implementation-plan)
6. [Decision recommendation](#6-decision-recommendation)

---

## 1. Status check

**Phase 2 (P·V GEMM on Tensor Cores) shipped via PR #145, commit `f03961c`,
2026-05-09.**

Git evidence:

```
f03961c perf(nvfp4): Phase 2 BitDecoding production — block-softmax + WMMA V (#145)
21d3feb BitDecoding Phase 3 — multi-seq + spill fix + splitk path + graph-safe (parity) (#147)
```

Source evidence in `src/compute/attention_paged_nvfp4_tc.cu` (current main):

- Q·K WMMA: lines 217–265 (`a_frag`, `b_frag` col_major, `c_frag` FP16 accumulator).
- P·V WMMA: lines 367–411 (`v_frag` FP32 accumulator, `b_frag_v` row_major).
- Residual P·V WMMA (Phase 3 path): lines 447–590, same shape.

Memo evidence:

- [[bitdecoding_phase2_v_tc_bug_2026_05_09]] (archived 2026-05-14) was the
  earlier "Phase 2 has a V-on-TC bug" entry. It is **archived/superseded**:
  the body explicitly states "no V on TC bug exists to fix" and that PR #145
  is the resolution.
- [[bitdecoding_phase3_continuation_2026_05_09]] documents Phase 3 building
  on a working Phase 2.
- [[bitdecoding_long_context_eval_2026_05_14]] benches Phase 1+2+3 together
  and confirms the kernel routes through `paged_attention_decode_nvfp4_tc_kernel`.

**The task prompt's premise is stale by ~8 days.** "Phase 2 = P·V on TC,
still owed" was true on 2026-05-08 and false from 2026-05-09 onward. The
last remaining work item the prompt could mean — "make P·V TC dispatch
worth flipping ON by default" — is a different problem (E2E perf, not
kernel structure). Section 6 below recommends what to do given that.

---

## 2. Current architecture

Anchor file: `src/compute/attention_paged_nvfp4_tc.cu` (1216 lines).

### Kernel template parameters

- `template <int HEAD_DIM>` — instantiated for 64 / 128 / 256 / 512 at the
  launcher switch (line 1114–1130 splitk path, line 1195–1211 non-splitk path).
- `WARP_SIZE = 32`, `NUM_WARPS = 8` (256 threads per CTA), `BLOCK_THREADS`
  set in `attention_paged_common.cuh`.
- `block_size` is the paged-cache block size (KV `block_size` field, 16 today).
  Phase 1's design comment at line 204 calls out the dependency: `block_size <= 16`
  maps cleanly to a 16×16 WMMA tile; the per-warp loop iterates one paged
  block at a time, dequantizes 16 tokens × HEAD_DIM into shared mem, then
  runs `K_TILES = HEAD_DIM / 16` WMMA tiles to cover the full head dim.

### Q·K WMMA fragment layout

Fragment declarations (line 217–219):

- `a_frag`: matrix_a, `<16,16,16>`, `__half`, **row_major** — holds replicated Q.
- `b_frag`: matrix_b, `<16,16,16>`, `__half`, **col_major** — holds 16 tokens × 16 hd_chunk of K.
- `c_frag`: accumulator, `<16,16,16>`, **`__half`** — output dots (FP16).

The col_major declaration on `b_frag` is intentional: `sK_w` is stored
row-major in shared memory (`sK_w[tok*16 + hd]`), so reading it as col_major
produces an effective transpose (`B[hd, tok]` instead of `B[tok, hd]`).
This is the standard FA2-style trick for Q·K^T without an explicit
transpose kernel.

The Q row is **replicated 16×** across the m-dim (line 230–233) because each
warp services one (batch, head, paged_block) combination — the WMMA M dim
gives 16 redundant copies of the same Q·K row, which is wasteful but
hidden behind the high TC throughput. Row 0 of `c_frag` after `mma_sync`
is the 16-token dot vector.

### Softmax / causal mask between Q·K and P·V

After Q·K (line 265 stores `c_frag` to `sK_w[0..16]`), block-softmax runs
on the 16 dot values (line 268–336):

- Lane `lane_id < 16` reads its own `dot = sK_w[lane_id] * scale`, applies
  softcap (line 300–302).
- Warp-shuffle reduction (lines 307–310, 322–326) gives `m_local` and `l_local`.
- Online-softmax update integrates with the running `(m_w, l_w, o_reg)` via
  the standard `m_new = max(m_w, m_local)`, `exp_diff = exp(m_w - m_new)`,
  `l_new = exp_diff * l_w + l_local`, `rescale_norm = exp_diff * l_w / l_new`.
- `o_reg` is **already normalized by `l_w`**; the rescale multiplier folds
  the previous block's contribution into the new block's denominator. This
  invariant is critical and the source comment at line 268–281 calls it out.

Causal mask is applied via `tok_end / first_tok / effective_start` clamping
in the dequant step (lines 242–252): tokens past the causal cutoff get zeroed
in `sK_w`, so their dot is zero and softmax weight is zero. No separate
masking pass.

### How P is currently passed to P·V

Not via a separate "P-tensor" — instead the 16 normalized weights live
**in shared memory as `weights_smem[0..15]`** (line 319), one per active
token in the current paged block. To feed the WMMA P·V, lines 359–362
replicate those 16 weights into all 16 rows of the `sQ_w` buffer:

```text
A[m, k] = weights_smem[k] * l_inv
       (l_inv = 1 / l_new, normalizes on the fly into the running average)
```

So at WMMA time, the "A operand" for P·V is **the same fragment type as
the QK A operand** (`<16,16,16>` half row_major), reusing `a_frag`. The
trick is that the m-dim is broadcast (16 redundant copies of the weights
row); only one m-row's output is useful per WMMA call.

### Where V dequant from NVFP4 happens

In the same WMMA inner loop, just before each P·V `mma_sync` (line 372–391):

- For each `kt = 0..K_TILES-1`, dequantize V[16 tokens × 16 hd_chunk] from
  packed NVFP4 + UE4M3 scales into `sK_w[token, hd_local]` (shared mem).
- The dequant uses the PTX intrinsic `cvt.rn.f16x2.e2m1x2` (lines 42–47,
  single instruction replaces a ~12-op LUT path).
- `b_frag_v` is loaded from `sK_w` as **row_major** (different from the QK
  phase's col_major). That makes `B[k=token, n=hd_local]` correct without
  another transpose.
- FP32 accumulator (`v_frag`) — comment at line 365–366 explains FP16 accum
  loses precision after a few decode steps.

After `mma_sync` (line 397), `v_frag` is stored to `sFV_w[0..16]` and the
**per-lane scatter** (line 403–410) only commits the contributions whose
hd-chunk matches the lane's ownership window — only one of `K_TILES`
iterations lands in any given lane's `o_reg`. The remaining `K_TILES - 1`
WMMAs are pure waste for that lane (but the warp as a whole keeps the
Tensor Core busy).

### Phase 3 residual pass

Same WMMA structure replayed on FP16 K/V from the residual ring (lines
414–615). No FP4 dequant, no scale fold; just `ldmatrix`-equivalent loads
into `sQ_r` / `sK_r` and the same QK → block-softmax → P·V dance. Fragments
declared **inside** the loop (line 447–451) so warps that skip the loop
don't pay register pressure.

### Crosswarp reduction

`crosswarp_reduce_and_write<HEAD_DIM>` (line 607) — defined in
`attention_paged_common.cuh`. Folds 8 warps' partial `(m_w, l_w, o_reg)`
into one final normalized output. Same code as the FP16 baseline.

### Split-K vs non-splitk

The launcher (line 1100–1213) picks between three paths:

- Non-splitk: one warp per (batch, head), full decode in one CTA.
- Splitk: `paged_attention_splitk_nvfp4_tc_kernel` writes per-split partials
  to scratch, then `paged_attention_launch_reduce` folds them.
- Splitk + residual: `paged_attention_residual_reduce_kernel` (line 782+)
  folds both the per-split partials AND the FP16 residual ring into the
  final output.

`compute_splitk_splits` (line 1082) picks split count based on grid
occupancy. The residual path forces splitk on (line 1097–1098) because
embedding residual into the non-splitk kernel was 12× slower per the
nsys profile (Phase 3 root-cause).

---

## 3. Phase 2 design (as shipped)

This section is what a "design memo before code" would have produced.
The shipped code matches each item, with the file:line refs in section 2.

### WMMA P·V fragment layout

- Reuse Phase 1's `<16,16,16>` `__half` fragments.
- Reuse `a_frag` for the P operand (broadcast 16 rows of weights).
- New `b_frag_v` with **row_major** declaration (vs Phase 1's col_major for
  `b_frag`), so V[token, hd] reads naturally without a separate transpose.
- New `v_frag` accumulator declared as **FP32** (`<16,16,16,float>`) — the
  decisive design choice. FP16 accumulator drifts within ~3 decode steps
  on entropy-rich prompts.

### Where to dequant V

In shared memory, just before the WMMA load. Same per-warp `sK_w` buffer
the QK phase used (reusable because Phase 2 happens after the QK
`store_matrix_sync` at line 265 and `sK_w` is unused between then and the
P·V dequant). No additional shared memory budget vs Phase 1.

### Register-pressure budget vs Phase 1

Shipped kernel has `STACK:0` (no spills) per cuobjdump. The decisive moves:

1. **Phase 2-only fragments declared inside loop body** (line 447–451 in
   the residual path; the main paged path declares them once because they
   reuse Phase 1's `a_frag`).
2. **Per-thread float arrays removed** — earlier prototypes used
   `float dots[16]` / `float weights[16]` on the stack. Replaced with
   `dots_smem` / `weights_smem` (line 294–295) backed by the same `sFV_w`
   region the FP32 accumulator stores into. Saves 128 B/lane of stack.

Register count: not specifically measured for this memo, but the SASS audit
([[bitdecoding_sass_audit_2026_05_09]]) found 24 HMMA ops at HD=128 with
zero `STACK:N` lines.

### Softmax → P fusion

Not fused into a TC store. The shipped design holds normalized weights in
`weights_smem` and replicates them into `sQ_w` at line 359–362 (16 rows ×
16 cols = 256 `__half2float` ops, hidden behind the WMMA's HMMA latency).
A fused softmax-with-TC-store of P was considered but would not have helped
— the bottleneck is bandwidth on V dequant, not softmax bookkeeping.

### Reduction across V-tiles

The K dim of P·V is the paged block size (≤ 16). One WMMA call covers it.
The "tile" loop in the shipped kernel iterates the **N dim** instead
(`K_TILES = HEAD_DIM / 16`) — for HEAD_DIM=128 that's 8 WMMA calls per
paged block, producing 8 disjoint hd-chunks of output. Per-lane scatter
(line 403–410) selects which chunk each lane commits to `o_reg`.

This is slightly wasteful (each lane runs 7 of 8 WMMAs whose output it
discards), but the warp-level Tensor Core is fully occupied, which is the
relevant resource. The alternative — letting different lanes own different
hd-chunks within one WMMA — would require fragment-element layout games
that PTX `mma.sync` exposes but the WMMA C++ API doesn't.

---

## 4. Risks & open questions

### 4.1 The 0% long-ctx gain finding

[[bitdecoding_long_context_eval_2026_05_14]] A/B'd Phase 1+2+3 against
NVFP4 KV baseline at pp ∈ {512, 4096, 8192, 16384} on Qwen3-4B Q8,
Qwen3-8B Q8, and Gemma-4-NVFP4. **Result: 0% gain (±0.6%) at every
tested config.** The +25% kernel-microbench claim does not survive at E2E.

Root cause per the same memo: decode at consumer-Blackwell scale is
bandwidth-bound on weight loads, not attention math. The dequant work
that BitDecoding kernel-ifies is <5% of decode wall-time. There's no
phase-shift in dispatch shape large enough to break this.

**Measurement scenario that would justify a Phase 2 revisit:**

- Either a model+context combination where attention dominates
  wall-time (e.g. very wide attention, small FFN, or 128K+ context where
  KV bandwidth dwarfs weight bandwidth on smaller models),
- Or a multi-batch decode (`batch_size > 1` with shared prefix) where the
  per-warp KV-decode cost stays constant but the FFN cost amortizes — KV
  decode then dominates again.

Neither is in imp's hot-path scope today. Both would need to be
prototyped before committing port-time.

### 4.2 Quality risk — V from FP4 in TC

V is dequanted from NVFP4 (1×PTX cvt) and the WMMA P·V uses FP16 operands
with FP32 accumulator. The shipped code uses FP32 accumulator specifically
because the comment at line 365–366 says FP16 accum drifts; that was
verified during PR #145 debug.

SageAttention3-style two-level accumulator (FP16 inner-MMA + FP32 outer-rescale)
is NOT used. The current single-level FP32 accumulator is sufficient at
≤ 16K context per [[bitdecoding_long_context_eval_2026_05_14]] (parity output
on Qwen3-4B/8B). At 64K+ context this should be re-verified.

### 4.3 sm_120 register-pressure ceiling

Current main has `STACK:0` per cuobjdump — no spills. The 255 regs/thread
ceiling is not threatening. Phase 2 added ~30 regs vs Phase 1 (FP32
accumulator + extra fragment), and Phase 3's residual pass scoped the
extra fragments inside the loop body to avoid double-counting. Headroom
remains.

If a future port adds tcgen05 / wgmma paths (not on sm_120; see
[[sm120_real_perf_levers_2026_05_04]]), the register accounting changes
entirely — those instructions allocate from TMEM, not RF.

### 4.4 HD=128 vs HD=256

The launcher instantiates HD ∈ {64, 128, 256, 512} (line 1115–1129).
Qwen3.5 GDN (HD=128) and Gemma-4 (HD=256 globals, HD=128 SWA) both
exercise these paths. The 2026-05-14 sweep covered HD=128 (Qwen3-4B,
Qwen3-8B) and HD=256 (Gemma-4). Both showed 0% gain.

The HD=256 instantiation has 2× the `K_TILES` count (16 vs 8 at HD=128),
which means 2× more wasted WMMAs per lane in the per-lane scatter pattern.
That's the highest-leverage place to look if a future port wants
fragment-element-level layout games to recoup the waste.

---

## 5. Implementation plan

### Retrospective (what shipped)

| Step | PR | Commit | LoC est. | Risk realized |
|---|---|---|---|---|
| 1. Phase 0 microbench harness | #141 | `40a2df5` | ~300 | None |
| 2. Phase 1 WMMA QK dispatch (opt-in env var) | #142 | `8c6128e` | ~400 | None — opt-in shielded prod |
| 3. Phase 2 production WMMA V + block-softmax | #145 | `f03961c` | ~500 | V-TC bug surfaced + fixed (FP32 accumulator + per-lane scatter) |
| 4. Phase 3a residual buffer infrastructure | #146 | `a04c5ae` | ~300 | None |
| 5. Phase 3b/3c residual reads + writes | #147 | `21d3feb` | ~600 | Two-step debug: per-layer advance bug, splitk path bug |
| 6. Final A/B at long context | n/a | (eval-only) | 0 | **Refuted +25% claim → 0% E2E gain** |

### Revisit roadmap (if someone re-investigates Phase 2)

Anything below is **gated on a measurement scenario from §4.1** showing
non-trivial E2E upside. Do not start without that gate.

| Step | LoC est. | Risk | Justifies start by |
|---|---|---|---|
| R1. Profile decode at 64K+ context on Qwen3-4B + `--kv-nvfp4` | 0 (eval) | low | Confirms attention > 30% of decode wall-time |
| R2. Per-lane scatter → fragment-element layout (PTX mma.sync) | ~400 | high — PTX-level rewrite | R1 shows hd-chunk waste > 20% of WMMA time |
| R3. SageAttention3 two-level accumulator | ~150 | mid — quality risk | R1 shows FP32 accumulator overhead > 5% of WMMA time |
| R4. Multi-warp along N-dim (BitDecoding `Wn` knob) | ~500 | mid — softmax sync via shared sTMP | R1 shows < 50% TC occupancy on the existing single-warp-N path |
| R5. Software-pipelined dequant ⇄ MMA (next slice's `ldmatrix` overlapped with current `mma`) | ~600 | high — `cp.async` interactions with WMMA | R1 shows dequant stalls > 15% of warp time |
| R6. Final A/B at long context + Qwen3.5 (HD=128 hybrid) + Gemma-4 (HD=256) | 0 (eval) | low — same protocol as 2026-05-14 |

**Total estimate to retry Phase 2 with all five levers**: ~2.5k LoC + 1–2
weeks debug + 1 week A/B = 3-4 weeks. Same order of magnitude as the
original port. **Not justified without R1's gate.**

---

## 6. Decision recommendation

**Defer Phase 2.** Specifically:

- Phase 2 is **already shipped** in current main; there is nothing to
  build right now.
- The 2026-05-14 E2E eval showed 0% gain. The kernel is faster but the
  attention math isn't the bottleneck on imp's hot path at the contexts
  we test today.
- The R-track in §5 is multi-week work that requires a measurement gate
  (§4.1) imp doesn't have a hot-path scenario for.

**One-sentence justification**: kernel-level work is done and verified
correct; the remaining gap to a "ship by default" decision is a workload
that makes attention math dominate decode wall-time, and imp doesn't run
one today.

### Concrete follow-ups (small, can be done at any time)

These do not need a Phase 2 revisit; they are housekeeping:

1. **Stale comment fix** — `src/compute/attention_paged_nvfp4_tc.cu:204–207`
   says "V accumulation remains per-token scalar in Phase 1 (Phase 2 will
   TC the PV path)". The PV path is now TC. Update the comment to reflect
   shipped reality. *Would also need to change this file — flagged here,
   not done in this memo per the no-edit constraint.*
2. **No change needed to `kv_cache.bitdecoding_qk` default** — keep `false`.
   The flag still gates the TC dispatch and the 0% gain means flipping it
   on doesn't help.
3. **No change needed to `kv_cache.bitdecoding_residual_tokens` default**
   — keep `0`. Same reason; also costs ~1.1 MiB/seq.

---

## Cross-references

- Phase-stack memos:
  - [[bitdecoding_phase3_continuation_2026_05_09]] — Phase 3 done @ parity
  - [[bitdecoding_phase2_v_tc_bug_2026_05_09]] — archived; "no bug exists"
  - [[bitdecoding_sass_audit_2026_05_09]] — 346 scalar ops → 24 HMMA after Phase 1
  - [[bitdecoding_long_context_eval_2026_05_14]] — 0% E2E gain at every tested config
  - [[kv_research_grade_eval_2026_05_09]] — original ROI evaluation
- Architectural context:
  - [[sm120_real_perf_levers_2026_05_04]] — no tcgen05/TMEM on consumer Blackwell
  - [[lever2_nvfp4_kv_implemented_2026_05_07]] — the underlying NVFP4 KV plumbing
  - [[hw_capability_audit_complete_2026_05_10]] — full HW capability survey
- PRs: #141, #142, #145, #146, #147
- Paper: [BitDecoding, arxiv:2503.18773](https://arxiv.org/abs/2503.18773) (HPCA 2026)
- Reference impl: [OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding)
