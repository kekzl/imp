# imp-quantize roadmap: any open-weight model onto one RTX 5090

**Goal:** make `imp-quantize` able to take an arbitrary open-weight release and
produce a checkpoint that runs on a single 32 GB sm_120 card. Today it takes a
BF16/FP16 SafeTensors checkpoint whose experts, if any, are stored one 2-D
tensor per expert. Everything else is refused or excluded.

**Status:** gap list with measurements, not a schedule. Written 2026-08-15
against `a5ba92aa`. Builds on [`roadmap.md`](../roadmap.md) gap 1, which carries
the quantization-quality history; this document is about *coverage*: which
checkpoints can enter the tool at all, and which can come out small enough.

---

## What "any model" can arithmetically mean here

Coverage is bounded before any code is written, and the bound should be stated
so the roadmap is not chasing sizes that cannot fit.

The card is 32 607 MiB. The CUDA primary context takes ~1 680 MiB and the
cuBLAS/CUTLASS reserve a measured ~3 900 MiB (`kMeasuredLibraryReserveBytes`,
`src/memory/plan.h`), both before imp allocates a weight. That leaves
**~27 000 MiB** for weights, KV cache and workspaces, so realistically
**22-24 GiB of weights** once a serving KV pool and workspaces are subtracted.

NVFP4 stores 0.5 bytes per value plus one FP8 micro-scale per 16, i.e. **0.5625
bytes against BF16's 2**, a 3.56x ratio on the tensors that qualify. Whole
checkpoints land near 2.8x because embeddings, norms and biases are copied
through (measured: Qwen3.8-27B, 51.75 GiB to 18.58 GiB, 2.79x).

So the honest ceilings are:

| shape | ceiling on this card |
|---|---|
| dense, fully resident | ~40B parameters (~80 GiB BF16 source) |
| MoE, experts resident | same, counted on *total* parameters |
| MoE, experts host-offloaded | bounded by host RAM (78 GiB here) and PCIe, not VRAM |

"Any model" therefore means: **any model of a size that could fit, in any format
it is published in.** Format coverage is the part that is fixable in this tool.
Size coverage above ~40B dense is not a quantizer problem and is not in scope
here; that is offload and the roadmap's own MoE work.

---

## Where it stands today

Accepted source dtypes are **BF16 and F16 only** (`is_float_dtype`,
`tensor_policy.cpp:14`). Eleven exclusion reasons and four hard refusals are
implemented; the deliberate ones (MLA latent projections, MoE router, vision
tower, norms, embeddings) are each backed by a measurement and should stay.

`--dry-run` now forecasts the output size and the card budget (#1423), so
whether a checkpoint can fit is a seconds-long question rather than a 25-minute
one.

Two things the existing roadmap already settled, which this document must not
re-litigate:

- **Calibrating a model too large to run is a solved problem.** A calibration
  file is keyed by (layer, tensor kind) and the recording hook sits before the
  tier switch, so it can be collected from *any* quantization of the same model
  (roadmap gap 1, point g, 2026-08-01).
- **`--calib` is not automatically an improvement.** On Qwen3-14B the full group
  set costs +2.68 PPL; the damage is the attention pair, and `--calib-groups BD`
  measures **better than round-to-nearest** there (point h, 2026-08-05).

---

## The gaps, ordered by reach per unit of work

Reach was checked against real releases by reading their SafeTensors headers
over HTTP range requests, which costs a few hundred bytes instead of the
download:

| release | dtypes in shard 1 | experts | enters the tool today |
|---|---|---|---|
| Mixtral-8x7B-Instruct | BF16 | 2-D per expert | yes |
| Qwen3-235B-A22B | BF16 | 2-D per expert | yes (too large to fit resident) |
| Qwen3.8-27B | BF16 | dense | yes |
| **DeepSeek-V3** | **F8_E4M3 + F32** | 2-D per expert | read path landed 2026-08-15, write untested |
| **Qwen3.8-2.4T-A95B** | BF16 | **3-D stack `[512, 4096, 8192]`** | **no: stack refused** |

### 1. Read FP8 and F32 sources (largest reach): READ PATH LANDED 2026-08-15

**Status:** the decode and the wiring are in, host-side and covered by the CPU
lane; what is untested is the write, because that needs a GPU and a staged FP8
checkpoint. Both released conventions turned out to be the same layout, which
made this smaller than the section below assumed: an E4M3 weight beside a
`weight_scale_inv` grid of 128x128 tiles, differing only in the scale dtype
(DeepSeek-V3 F32, Qwen3.8-27B-FP8 BF16). The block edge is derived from the two
shapes rather than assumed. Remaining: run it end to end on a real FP8
checkpoint, and answer the two measurement questions below.


The problem it solved: `is_float_dtype` accepted BF16 and F16, so a checkpoint
published only in FP8 was rejected tensor by tensor as "already quantized or
unsupported". FP8 is now a common *primary* release format for large models, so
that was not an edge case but a whole class of releases the tool could not open.

What was built: `fp8_source.{h,cpp}`, a host-only translation unit so the CPU
lane can check the parts that fail silently when wrong. The E4M3 bit layout and
the block stride both produce plausible magnitudes when misread, which is why
they are pinned by test rather than by inspecting a converted checkpoint, and
both are mutation-validated. `main.cpp` pairs each E4M3 weight with its grid
across shards before the write loop, and **consumes** the grid: once the weight
is NVFP4 the old scales describe nothing, and copying them through would leave a
checkpoint whose scales contradict its weights. An E4M3 tensor with no grid is
reported by name instead of falling into the generic dtype exclusion, because
that case is a scalar-scale export needing different handling.

Not done: the write path has only been exercised on a synthetic fixture, since
quantizing needs a GPU. Two things still to settle by measurement, not
assumption: whether FP8 to NVFP4
re-quantization is worth doing at all against just running the FP8 checkpoint
(sm_120 has no FP8 GEMM, so the runtime already expands FP8 to an FP16
companion, `roadmap.md` line 474, which costs VRAM the NVFP4 path would not),
and how much quality a double quantization costs versus the BF16 route when both
are available for the same model.

### 2. Optional embedding and lm_head quantization (largest size lever)

`embed_tokens` is never quantized and `lm_head` only under `--lm-head`. On a
modern large vocabulary that is a quarter of the output: Qwen3.8-27B has
248 320 x 5 120, i.e. **2.37 GiB each in BF16, 4.74 GiB of an 18.60 GiB
checkpoint**. Vocabularies are growing, so this share grows with them.

The exclusion is deliberate and reasoned ("quantizing them costs quality for no
bandwidth win on the decode hot path") and that reasoning is about *speed*. When
the question is whether the model fits at all, the trade is a different one and
should be available as an opt-in with a measured price tag. Needs a PPL
comparison per option before it ships.

**Landed 2026-08-15:** the reporting half, which is what makes the trade
visible. `--dry-run` now breaks down the bytes that did NOT shrink, by reason,
largest first. On Qwen3.8-27B that reads 5.60 GiB, **30 % of the output**: 2 425
MiB embedding, 2 425 MiB lm_head, 875 MiB vision tower. The compression ratio
never says this, and "why is it still this big" has a different fix from "did
quantization work". The opt-in itself still waits on the measurement.

### 3. Confirm `--calib-groups BD` above 14B (smallest work, named open question)

The roadmap states the practical rule ("`BD` on wide-GQA models") and then says
plainly that `n_rep` is 8 on most 70B-class checkpoints and **"whether `BD`
still pays at that size is untested"**. Every large model this roadmap wants to
reach is in that bracket. This is one measurement campaign on an existing code
path, no new code, and it decides whether calibration is a recommendation or a
warning for the sizes that matter most.

### 4. Stacked MoE experts (largest reach among refusals, largest work)

`[n_experts, N, K]` stacks are refused. This blocks Qwen3.8-2.4T-A95B, whose
experts are `[512, 4096, 8192]` with `gate` and `up` fused into one tensor, and
gpt-oss (whose HF release is already MXFP4, so it is not itself a candidate).

The obvious idea, "split the stack into the per-expert 2-D layout the loader
already reads", is worth stating precisely because it is *half* right. The
target layout is real and proven: the local `Gemma-4-26B-A4B-it-NVFP4` is
46 110 per-expert tensors. But `roadmap.md` line 52 already recorded why this is
not a quantizer-only change: gpt-oss-style stacks carry **per-expert biases**
(`experts.down_proj_bias`, `experts.gate_up_proj_bias`, both confirmed in the
header scan) which the loader and the MoE forward do not support, and the fused
`gate_up_proj` needs a per-model layout descriptor to split correctly. So the
cost is loader plus forward plus quantizer, and it was deferred on that basis
rather than on effort alone.

Do not start this without a BF16 source checkpoint with stacks staged locally.
No local model has one, and the axis order and naming cannot be inferred from
the refusal message.

---

## Order, and why

1. **FP8 sources.** Opens a release class that is currently entirely shut, and
   the decode kernel already exists.
2. **`BD` above 14B.** One measurement campaign, no new code, and it settles
   whether the quality lever works where it is most needed.
3. **Embedding/lm_head opt-in.** Biggest size lever, but it trades quality, so
   it needs the measurement discipline that item 2 establishes.
4. **Stacked experts.** Highest reach among the refusals and the only item that
   needs loader and forward work; gate it behind a staged checkpoint.

## Deliberately not proposed

- **Removing the MLA, router, norm or vision-tower exclusions.** Each was found
  by bisection on a checkpoint that loaded and then produced garbage.
- **Chasing dense models beyond ~40B.** No quantizer setting makes them
  resident; that is offload work.
- **Replacing round-to-nearest as the default.** The measured record has
  `--calib` losing at 14B with the full group set, so the default is correct
  until item 2 says otherwise.
