# PTX feature acceptance survey for sm_120f

Generated: 2026-05-29T11:10:09Z
Toolkit:   V13.2.78
Arch:      compute_120f / sm_120 (RTX 5090 GB202)

Each section is a separate ptxas-acceptance test for one PTX instruction
family. ✅ = ptxas accepts on sm_120f (instruction is callable; runtime
behavior may still differ). ❌ = ptxas rejects with the cited reason.

Re-run `tools/analysis/ptx_survey_all.sh` after every CUDA toolkit
upgrade — newly-supported instructions surface here as ✅ flips.


---

## PTX cvt survey — sm_120f / imp:builder

### `e2m1` (FP4, 8-bit packed pair)

Status | Instruction | Reason
---|---|---
✅ | `cvt.rn.satfinite.e2m1x2.f32` | OK
❌ | `cvt.rn.e2m1x2.f32 (no .satfinite)` | '.satfinite' modifier required for instruction 'cvt' with destination type '.e2m1x2'
✅ | `cvt.rn.satfinite.e2m1x2.f16x2` | OK
✅ | `cvt.rn.satfinite.e2m1x2.bf16x2` | OK
✅ | `cvt.rn.f16x2.e2m1x2` | OK
❌ | `cvt.f16x2.e2m1x2 (no .rn)` | Rounding mod required
✅ | `cvt.rn.relu.f16x2.e2m1x2` | OK
✅ | `cvt.rn.bf16x2.e2m1x2` | OK

### `e2m3` (FP6 E2M3, 12-bit packed pair (16-bit reg, 4 bits unused))

Status | Instruction | Reason
---|---|---
✅ | `cvt.rn.satfinite.e2m3x2.f32` | OK
❌ | `cvt.rn.e2m3x2.f32 (no .satfinite)` | '.satfinite' modifier required for instruction 'cvt' with destination type '.e2m3x2'
✅ | `cvt.rn.satfinite.e2m3x2.f16x2` | OK
✅ | `cvt.rn.satfinite.e2m3x2.bf16x2` | OK
✅ | `cvt.rn.f16x2.e2m3x2` | OK
❌ | `cvt.f16x2.e2m3x2 (no .rn)` | Rounding mod required
✅ | `cvt.rn.relu.f16x2.e2m3x2` | OK
✅ | `cvt.rn.bf16x2.e2m3x2` | OK
❌ | `cvt.f32x2.e2m3x2 (→ pair of f32)` | Unexpected instruction types
❌ | `cvt.rn.f32x2.e2m3x2` | Unexpected instruction types

### `e3m2` (FP6 E3M2, 12-bit packed pair (16-bit reg, 4 bits unused))

Status | Instruction | Reason
---|---|---
✅ | `cvt.rn.satfinite.e3m2x2.f32` | OK
❌ | `cvt.rn.e3m2x2.f32 (no .satfinite)` | '.satfinite' modifier required for instruction 'cvt' with destination type '.e3m2x2'
✅ | `cvt.rn.satfinite.e3m2x2.f16x2` | OK
✅ | `cvt.rn.satfinite.e3m2x2.bf16x2` | OK
✅ | `cvt.rn.f16x2.e3m2x2` | OK
❌ | `cvt.f16x2.e3m2x2 (no .rn)` | Rounding mod required
✅ | `cvt.rn.relu.f16x2.e3m2x2` | OK
✅ | `cvt.rn.bf16x2.e3m2x2` | OK
❌ | `cvt.f32x2.e3m2x2 (→ pair of f32)` | Unexpected instruction types
❌ | `cvt.rn.f32x2.e3m2x2` | Unexpected instruction types

### `e4m3` (FP8 E4M3, 16-bit packed pair)

Status | Instruction | Reason
---|---|---
✅ | `cvt.rn.satfinite.e4m3x2.f32` | OK
❌ | `cvt.rn.e4m3x2.f32 (no .satfinite)` | '.satfinite' modifier required for instruction 'cvt' with destination type '.e4m3x2'
✅ | `cvt.rn.satfinite.e4m3x2.f16x2` | OK
✅ | `cvt.rn.satfinite.e4m3x2.bf16x2` | OK
✅ | `cvt.rn.f16x2.e4m3x2` | OK
❌ | `cvt.f16x2.e4m3x2 (no .rn)` | Rounding mod required
✅ | `cvt.rn.relu.f16x2.e4m3x2` | OK
✅ | `cvt.rn.bf16x2.e4m3x2` | OK
❌ | `cvt.f32x2.e4m3x2 (→ pair of f32)` | Unexpected instruction types
❌ | `cvt.rn.f32x2.e4m3x2` | Unexpected instruction types

### `e5m2` (FP8 E5M2, 16-bit packed pair)

Status | Instruction | Reason
---|---|---
✅ | `cvt.rn.satfinite.e5m2x2.f32` | OK
❌ | `cvt.rn.e5m2x2.f32 (no .satfinite)` | '.satfinite' modifier required for instruction 'cvt' with destination type '.e5m2x2'
✅ | `cvt.rn.satfinite.e5m2x2.f16x2` | OK
✅ | `cvt.rn.satfinite.e5m2x2.bf16x2` | OK
✅ | `cvt.rn.f16x2.e5m2x2` | OK
❌ | `cvt.f16x2.e5m2x2 (no .rn)` | Rounding mod required
✅ | `cvt.rn.relu.f16x2.e5m2x2` | OK
✅ | `cvt.rn.bf16x2.e5m2x2` | OK
❌ | `cvt.f32x2.e5m2x2 (→ pair of f32)` | Unexpected instruction types
❌ | `cvt.rn.f32x2.e5m2x2` | Unexpected instruction types

### Block scale types (UE4M3 / UE8M0 — typically MMA operands only)

Status | Instruction | Reason
---|---|---
❌ | `cvt.rn.satfinite.ue8m0.f32` | Unexpected instruction types
❌ | `cvt.rn.satfinite.ue8m0.f32x2 (encode 2 scales)` | Illegal rounding modifier for instruction 'cvt'
❌ | `cvt.f32.ue8m0 (decode scale → f32)` | Unexpected instruction types

Done. Re-run after CUDA toolkit upgrades to refresh dead-end status.

---

## PTX MMA acceptance survey — sm_120f / imp:builder

### DENSE no-scale (kind::f8f6f4)

Status | Variant | Reason
---|---|---
✅ | `f8f6f4 m16n8k32 e2m1×e2m1 (legacy FP4)` | OK
✅ | `f8f6f4 m16n8k32 e4m3×e4m3 (FP8)` | OK
✅ | `f8f6f4 m16n8k32 e5m2×e5m2 (FP8 alt)` | OK
✅ | `f8f6f4 m16n8k32 e4m3×e2m1 (mixed FP8×FP4)` | OK
✅ | `f8f6f4 m16n8k32 e2m1×e4m3 (mixed FP4×FP8)` | OK
✅ | `f8f6f4 m16n8k32 e2m3×e2m3 (FP6 E2M3)` | OK
✅ | `f8f6f4 m16n8k32 e3m2×e3m2 (FP6 E3M2)` | OK
❌ | `f8f6f4 m16n8k64 e2m1×e2m1 (illegal? K=64 needs sparse)` | Incorrect instruction type specified for mma with shape '.m16n8k64'

### DENSE block-scale (kind::mxf4nvf4 — K=64)

Status | Variant | Reason
---|---|---
✅ | `mxf4nvf4 scale_vec::4X K=64 ue4m3 (Project B)` | OK
✅ | `mxf4nvf4 scale_vec::4X K=64 ue8m0` | OK
❌ | `mxf4nvf4 scale_vec::2X K=64 ue4m3` | Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3'
✅ | `mxf4nvf4 scale_vec::2X K=64 ue8m0 (per-32 scales)` | OK
❌ | `mxf4nvf4 scale_vec::1X K=64 ue4m3` | Illegal modifier '.scale_vec::1X' for instruction 'mma'
❌ | `mxf4nvf4 scale_vec::1X K=64 ue8m0` | Illegal modifier '.scale_vec::1X' for instruction 'mma'
❌ | `mxf4nvf4 scale_vec::8X K=64 ue4m3 (8X exists?)` | Unknown modifier
❌ | `mxf4nvf4 scale_vec::4X K=128 (dense at sparse-K?)` | Illegal matrix shape '.m16n8k128' for instruction 'mma'
❌ | `mxf4nvf4 scale_vec::4X K=32` | Illegal matrix shape '.m16n8k32' for instruction 'mma'

### DENSE block-scale (kind::mxf8f6f4 — K=32)

Status | Variant | Reason
---|---|---
✅ | `mxf8f6f4 1X K=32 e2m1×e2m1 ue8m0 (FP4 with HW scale)` | OK
✅ | `mxf8f6f4 1X K=32 e4m3×e4m3 ue8m0 (FP8 with HW scale)` | OK
✅ | `mxf8f6f4 1X K=32 e5m2×e5m2 ue8m0` | OK
✅ | `mxf8f6f4 1X K=32 e4m3×e2m1 (mixed FP8×FP4)` | OK
✅ | `mxf8f6f4 1X K=32 e2m1×e4m3 (mixed FP4×FP8)` | OK
✅ | `mxf8f6f4 1X K=32 e2m3×e2m3 (FP6 E2M3)` | OK
✅ | `mxf8f6f4 1X K=32 e3m2×e3m2 (FP6 E3M2)` | OK
❌ | `mxf8f6f4 2X K=32 e2m1×e2m1` | Illegal modifier '.scale_vec::2X' for instruction 'mma'
❌ | `mxf8f6f4 1X K=32 e2m1×e2m1 ue4m3 (NVFP4 scale type)` | Incorrect instruction type specified for mma with shape '.m16n8k32'

### SPARSE no-scale (kind::f8f6f4.sp::ordered_metadata, K=64)

Status | Variant | Reason
---|---|---
✅ | `sparse f8f6f4 K=64 e2m1×e2m1 (FP4 2:4 sparse)` | OK
✅ | `sparse f8f6f4 K=64 e4m3×e4m3 (FP8 2:4 sparse)` | OK
✅ | `sparse f8f6f4 K=64 e5m2×e5m2` | OK
✅ | `sparse f8f6f4 K=64 e2m3×e2m3 (FP6)` | OK
✅ | `sparse f8f6f4 K=64 e3m2×e3m2 (FP6)` | OK
✅ | `sparse f8f6f4 K=64 e4m3×e2m1 (mixed)` | OK

### SPARSE block-scale (kind::mxf4nvf4.sp — USER REQUESTED check)

Status | Variant | Reason
---|---|---
✅ | `sparse mxf4nvf4 4X K=128 ue4m3 (the headline-rejected one)` | OK
✅ | `sparse mxf4nvf4 4X K=128 ue8m0` | OK
❌ | `sparse mxf4nvf4 2X K=128 ue4m3` | Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3'
✅ | `sparse mxf4nvf4 2X K=128 ue8m0` | OK
❌ | `sparse mxf4nvf4 1X K=128 ue8m0` | Illegal modifier '.scale_vec::1X' for instruction 'mma'
❌ | `sparse mxf4nvf4 4X K=64 (smaller K)` | Illegal matrix shape '.m16n8k64' for instruction 'Sparse mma'

### SPARSE block-scale (kind::mxf8f6f4.sp — K=64)

Status | Variant | Reason
---|---|---
✅ | `sparse mxf8f6f4 1X K=64 e2m1×e2m1 ue8m0` | OK
✅ | `sparse mxf8f6f4 1X K=64 e4m3×e4m3 ue8m0` | OK
✅ | `sparse mxf8f6f4 1X K=64 e2m3×e2m3 ue8m0 (FP6 sparse blockscale)` | OK

### Sanity baseline (always-supported FP16/BF16 dense MMA)

Status | Variant | Reason
---|---|---
✅ | `FP16 m16n8k16 (sanity)` | OK
✅ | `BF16 m16n8k16 (sanity)` | OK

Done. Re-run after CUDA toolkit upgrades.

---

## PTX async/barrier survey — sm_120f / imp:builder

### cp.async (Ampere-style legacy async copy)

Status | Variant | Reason
---|---|---
✅ | `cp.async.ca.shared.global 4-byte` | OK
✅ | `cp.async.ca.shared.global 8-byte` | OK
✅ | `cp.async.ca.shared.global 16-byte` | OK
✅ | `cp.async.cg.shared.global 16-byte (bypass L1)` | OK
✅ | `cp.async.commit_group` | OK
✅ | `cp.async.wait_group 0` | OK
✅ | `cp.async.wait_all` | OK

### cp.async.bulk (Hopper TMA)

Status | Variant | Reason
---|---|---
❌ | `cp.async.bulk shared::cluster (CTA-pair)` | Illegal modifier
❌ | `cp.async.bulk shared (single-CTA)` | State space incorrect for instruction 'cp.async.bulk'
✅ | `cp.async.bulk.commit_group` | OK
✅ | `cp.async.bulk.wait_group 0` | OK
✅ | `cp.async.bulk.tensor.1d.tile.mbarrier` | OK
✅ | `cp.async.bulk.tensor.2d.tile.mbarrier` | OK
✅ | `cp.async.bulk.tensor.3d.tile.mbarrier` | OK
✅ | `cp.async.bulk.tensor.5d.tile.mbarrier` | OK
❌ | `cp.async.bulk.tensor.2d.im2col.mbarrier (im2col mode)` | Arguments mismatch for instruction 'cp.async.bulk.tensor'
❌ | `cp.async.bulk.tensor.2d.tile + bulk_group (no mbarrier)` | Arguments mismatch for instruction 'cp.async.bulk.tensor'
✅ | `cp.async.bulk.shared::cluster.global (raw bulk no tensor)` | OK

### mbarrier (memory barrier for async ops)

Status | Variant | Reason
---|---|---
✅ | `mbarrier.init.shared::cta.b64` | OK
✅ | `mbarrier.arrive.shared::cta.b64` | OK
✅ | `mbarrier.arrive.expect_tx.shared::cta.b64` | OK
✅ | `mbarrier.try_wait.parity.shared::cta.b64` | OK
✅ | `mbarrier.try_wait.shared::cta.b64` | OK

### st.async (async store)

Status | Variant | Reason
---|---|---
❌ | `st.async.weak.shared::cluster.b32` | Modifier rejected
❌ | `st.async.weak.shared::cluster.b128 (4× b32)` | Modifier rejected
❌ | `st.async.global.b32 (does it exist for global?)` | ptxas fatal   : (C7907) Internal compiler error.

### fence.proxy / async fences

Status | Variant | Reason
---|---|---
✅ | `fence.proxy.async` | OK
✅ | `fence.proxy.async.shared::cta` | OK
✅ | `fence.proxy.async.global` | OK
✅ | `fence.proxy.tensormap::generic` | OK

### Programmatic Dependent Launch (PDL)

Status | Variant | Reason
---|---|---
✅ | `griddepcontrol.wait` | OK
✅ | `griddepcontrol.launch_dependents` | OK

Done. Re-run after CUDA upgrades.

---

## PTX atomics + reductions survey — sm_120f / imp:builder

### atom.global add (numeric types)

Status | Variant | Reason
---|---|---
✅ | `atom.global.add.u32` | OK
✅ | `atom.global.add.u64` | OK
✅ | `atom.global.add.f32` | OK
✅ | `atom.global.add.f64` | OK
✅ | `atom.global.add.noftz.f16 (scalar half)` | OK
✅ | `atom.global.add.noftz.bf16 (scalar bfloat)` | OK
✅ | `atom.global.add.noftz.f16x2 (vector half2)` | OK
✅ | `atom.global.add.noftz.bf16x2 (vector bf16x2)` | OK

### atom.global min/max

Status | Variant | Reason
---|---|---
✅ | `atom.global.min.u32` | OK
✅ | `atom.global.max.u32` | OK
✅ | `atom.global.min.s32` | OK

### atom.global cas / exch / and / or / xor

Status | Variant | Reason
---|---|---
✅ | `atom.global.cas.b32` | OK
✅ | `atom.global.cas.b64` | OK
✅ | `atom.global.exch.b32` | OK
✅ | `atom.global.and.b32` | OK

### red.global (reduce, no return)

Status | Variant | Reason
---|---|---
✅ | `red.global.add.u32` | OK
✅ | `red.global.add.f32` | OK
✅ | `red.global.add.noftz.f16x2 (vector half reduce)` | OK
✅ | `red.global.add.noftz.bf16x2` | OK

### multimem (DSMEM cluster reduction — sm_100+ feature)

Status | Variant | Reason
---|---|---
✅ | `multimem.ld_reduce.add.f32` | OK
✅ | `multimem.st.b32` | OK
✅ | `multimem.red.add.f32` | OK

### redux.sync (warp-level reduction — Volta+)

Status | Variant | Reason
---|---|---
✅ | `redux.sync.add.s32` | OK
✅ | `redux.sync.add.u32` | OK
✅ | `redux.sync.min.u32` | OK
✅ | `redux.sync.max.s32` | OK
✅ | `redux.sync.and.b32` | OK
✅ | `redux.sync.or.b32` | OK
❌ | `redux.sync.add.f32 (FP variant?)` | Instruction 'redux.f32' not supported on .target 'sm_120a'
❌ | `redux.sync.min.f32` | Instruction 'redux.f32' not supported on .target 'sm_120a'

### Cluster sync

Status | Variant | Reason
---|---|---
❌ | `bar.cluster.sync` | Not a name of any known instruction: 'barrier.cluster'
✅ | `bar.cluster.arrive` | OK
✅ | `bar.cluster.wait` | OK

Done.

---

## PTX SFU + math survey — sm_120f / imp:builder

### Special-function-unit (SFU) approximations

Status | Variant | Reason
---|---|---
✅ | `rcp.approx.f32` | OK
✅ | `rcp.approx.ftz.f32` | OK
✅ | `rsqrt.approx.f32` | OK
✅ | `rsqrt.approx.ftz.f32` | OK
✅ | `ex2.approx.f32` | OK
✅ | `ex2.approx.ftz.f32` | OK
✅ | `lg2.approx.f32` | OK
✅ | `sin.approx.f32` | OK
✅ | `cos.approx.f32` | OK
✅ | `tanh.approx.f32` | OK
✅ | `tanh.approx.f16 (scalar half)` | OK
✅ | `tanh.approx.f16x2 (packed half)` | OK
✅ | `tanh.approx.bf16x2 (packed bfloat)` | OK
✅ | `ex2.approx.f16 (does it exist?)` | OK
✅ | `ex2.approx.f16x2` | OK
❌ | `rcp.approx.f16x2` | Unexpected instruction types specified for 'rcp'
❌ | `rsqrt.approx.f16x2` | Unexpected instruction types specified for 'rsqrt'
❌ | `rcp.approx.bf16x2` | Unexpected instruction types specified for 'rcp'

### Division & full-precision approximations

Status | Variant | Reason
---|---|---
✅ | `div.approx.f32` | OK
✅ | `rcp.rn.f32 (full precision)` | OK
✅ | `rsqrt.approx.f64 (double)` | OK
✅ | `sqrt.approx.f32` | OK
✅ | `sqrt.rn.f32` | OK

### Packed FP arithmetic (half2 / bf16x2 native ops)

Status | Variant | Reason
---|---|---
✅ | `add.f16x2` | OK
✅ | `mul.f16x2` | OK
✅ | `fma.rn.f16x2` | OK
✅ | `fma.rn.bf16x2` | OK
✅ | `fma.rn.relu.f16x2 (fused ReLU)` | OK
✅ | `min.f16x2` | OK
✅ | `max.f16x2` | OK

### Bit manipulation / permute

Status | Variant | Reason
---|---|---
✅ | `prmt.b32` | OK
✅ | `prmt.f4e.b32 (forward 4 extract)` | OK
✅ | `bfe.u32 (bit field extract)` | OK
✅ | `bfi.b32 (bit field insert)` | OK
✅ | `fns.b32 (find n-th set bit)` | OK
✅ | `lop3.b32 (lookup-table 3-input boolean)` | OK
✅ | `popc.b32 (population count)` | OK
✅ | `clz.b32 (count leading zeros)` | OK
✅ | `shf.l.wrap.b32 (funnel shift left)` | OK

### dp4a / dp2a (INT8/INT16 dot-product)

Status | Variant | Reason
---|---|---
✅ | `dp4a.s32.s32 (signed INT8 dp4a)` | OK
✅ | `dp4a.u32.u32 (unsigned INT8 dp4a)` | OK
✅ | `dp2a.lo.s32.s32 (INT16 dp2a low)` | OK
✅ | `dp4a.s32.u32 (mixed sign)` | OK

### Misc system / utility

Status | Variant | Reason
---|---|---
✅ | `nanosleep.u32` | OK
✅ | `clock.lo.s64` | OK
✅ | `%smid (SM ID)` | OK
✅ | `%nsmid (number of SMs)` | OK
✅ | `activemask` | OK
✅ | `match.any.sync.b32` | OK
❌ | `match.all.sync.b32` | Predicate output expected for instruction 'match'

Done.

---

## PTX cluster / multimem / TCGEN05 / wgmma survey — sm_120f / imp:builder

Tests sm_100+ "data center Blackwell" features against consumer Blackwell.

### Cluster sync / mapa (cluster shared memory address translation)

Status | Variant | Reason
---|---|---
✅ | `barrier.cluster.arrive` | OK
✅ | `barrier.cluster.wait` | OK
✅ | `barrier.cluster.arrive.relaxed` | OK
✅ | `%cluster_ctaid.x (cluster CTA ID)` | OK
✅ | `%cluster_nctaid.x (cluster size)` | OK
✅ | `%cluster_ctarank` | OK
✅ | `mapa.shared::cluster.u32 (DSMEM addr translate)` | OK
❌ | `mapa.shared::cluster.u64` | Arguments mismatch
✅ | `getctarank.shared::cluster.u32` | OK

### Multimem (DSMEM cluster reduction / store)

Status | Variant | Reason
---|---|---
✅ | `multimem.ld_reduce.add.f32` | OK
❌ | `multimem.ld_reduce.add.f16` | Unexpected instruction types specified for 'multimem.ld_reduce'
❌ | `multimem.ld_reduce.add.f16x2` | Illegal modifier
❌ | `multimem.ld_reduce.add.bf16x2` | Illegal modifier
✅ | `multimem.ld_reduce.add.v4.f32` | OK
❌ | `multimem.ld_reduce.min.f32` | Incorrect type '.f32' for operation '.min' in instruction 'multimem.ld_reduce'
✅ | `multimem.st.f32` | OK
✅ | `multimem.red.add.f32` | OK
❌ | `multimem.red.add.f16x2 (vector half)` | Illegal modifier

### TCGEN05 (Tensor Core Gen 5 — sm_100/sm_103 only?)

Status | Variant | Reason
---|---|---
❌ | `tcgen05.alloc.shared::cta.b32` | Instruction 'tcgen05.alloc' not supported on .target 'sm_120a'
❌ | `tcgen05.dealloc.cta_group::1.b32` | Instruction 'tcgen05.dealloc' not supported on .target 'sm_120a'
❌ | `tcgen05.relinquish_alloc_permit.cta_group::1` | Instruction 'tcgen05.relinquish_alloc_permit' not supported on .target 'sm_120a'
❌ | `tcgen05.commit.cta_group::1.mbarrier` | Instruction 'tcgen05.commit' not supported on .target 'sm_120a'
❌ | `tcgen05.cp.cta_group::1.4x256b (TMEM copy)` | Feature '.4x256b' not supported on .target 'sm_120a'
❌ | `tcgen05.fence::after_thread_sync` | Instruction 'tcgen05.fence' not supported on .target 'sm_120a'
❌ | `tcgen05.shift.cta_group::1.down` | Instruction 'tcgen05.shift' not supported on .target 'sm_120a'
❌ | `tcgen05.wait::ld.sync.aligned` | Instruction 'tcgen05.wait' not supported on .target 'sm_120a'
❌ | `tcgen05.mma.cta_group::1.kind::f16 (TMEM-input MMA)` | Arguments mismatch

### wgmma (Warp-Group MMA, Hopper-style — sm_90 only?)

Status | Variant | Reason
---|---|---
❌ | `wgmma.fence.sync.aligned` | Instruction 'wgmma.fence' not supported on .target 'sm_120a'
❌ | `wgmma.commit_group.sync.aligned` | Instruction 'wgmma.commit_group' not supported on .target 'sm_120a'
❌ | `wgmma.wait_group.sync.aligned 0` | Instruction 'wgmma.wait_group' not supported on .target 'sm_120a'
❌ | `wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16` | Instruction 'wgmma.mma_async with floating point types' not supported on .target 'sm_120a'

### Async stmatrix / ldmatrix (smem ↔ register fragment helpers)

Status | Variant | Reason
---|---|---
✅ | `ldmatrix.sync.aligned.m8n8.x1.shared.b16` | OK
✅ | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | OK
✅ | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 (transposed)` | OK
❌ | `ldmatrix.sync.aligned.m16n16.x1.shared.b8 (8-bit fragments)` | Vector of size 2 is expected for argument 0 of instruction 'ldmatrix'
✅ | `stmatrix.sync.aligned.m8n8.x1.shared.b16` | OK
✅ | `stmatrix.sync.aligned.m8n8.x4.shared.b16` | OK
❌ | `stmatrix.sync.aligned.m16n8.x1.shared.b8` | Modifier '.trans' require for instruction stmatrix with shape '.m16n8'

### Tensormap / TMA descriptor manipulation

Status | Variant | Reason
---|---|---
✅ | `tensormap.replace.tile.global_address.shared::cta.b1024.b64` | OK
✅ | `tensormap.cp_fenceproxy.global.shared::cta` | OK

Done.

---

Full survey complete.
