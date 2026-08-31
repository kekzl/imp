<!--
layer: L1
audience: operators
verified: 2026-08-28
commit: be825e4a
-->

# Design decisions

Things imp deliberately does not do: decision, measurement or constraint
behind it, reopen condition. Absences without a reason are gaps and live in
[`LIMITATIONS.md`](LIMITATIONS.md).

## No multi-GPU, no tensor parallelism

**Decision:** single GPU only, and this is not a roadmap item.

**Why:** consumer Blackwell has no NVLink. Tensor parallelism on
bandwidth-bound batch=1 decode moves activations across PCIe every layer:
net-negative on imp's workload. Two 5090s over PCIe are not two datacenter
cards over NVLink.

**Reopens if:** a consumer card ships with a high-bandwidth inter-GPU link, or
the target workload moves to large-batch serving where the collective cost
amortises.

## No CPU inference path, no AVX kernels

**Decision:** the GPU computes. There is no CPU fallback and no AVX-512 expert
path.

**Why (measured, 2026-08-10 on this host):** CPU-computing cold MoE experts
from host RAM budgets at 62.5 GB/s streaming read, 16 threads: 14.0 ms/token
for a 120B-A5B shape, round trips included. Streaming the missing expert into
VRAM and computing on the GPU costs 4.7-8.9 ms for the same token, because
expert selection has strong temporal locality (median reuse distance 2
tokens; an LRU at 40 % residency hits 88.7-96.2 %). The GPU-side design won
and shipped.

**Reopens if:** a model arrives whose active expert set defeats the LRU, or host
memory bandwidth changes class.

## Blackwell only, and specifically consumer Blackwell

**Decision:** `sm_120a` SASS with a `compute_120f` PTX fallback. Nothing else.

**Why:** the delimitation against datacenter Blackwell is a hardware fact,
stated once in
[`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md#target-architecture).
A second architecture means a portability layer in the hot path or a second
set of kernels; both cost more than this project has.

## No FP4 cuBLASLt path

**Decision:** CUTLASS is the primary GEMM path.

**Why:** there are no FP4 cuBLASLt kernels on `sm_120`. FP8 prefill is
unavailable for the same reason. This is not a preference.

## The changelog is not a journal

**Decision:** one to three lines per entry. The investigation lives in `docs/`
and the PR.

**Why:** the CHANGELOG and the release page are the only two artefacts read
by people who never opened the repo; a 15-line entry buries the line the
reader came for.

## No GPU runner in CI

**Decision:** the `test` job stays gated behind `vars.HAS_GPU_RUNNER`, off.

**Why:** a hosted GPU runner costs more than this project spends; a
self-hosted one on the maintainer's only card would contend with the
measurements it exists to protect. Consequence, stated in
[`LIMITATIONS.md`](LIMITATIONS.md): `make verify-fast` before push is the
only thing that ever runs a CUDA kernel against a correctness or performance
check.

## Speculative decoding stays opt-in

**Decision:** drafts are available, none is on by default for every model.

**Why:** the economics are per-model, measured both ways. Prompt-lookup
n-gram is default-on for batch-1 greedy dense because its drafts are free. The
MTP head on Nemotron-3.5 accepts 41 % at depth 1 offline and 39 % on the
serving path, and still costs 51 % of decode: good drafts, but the verify
chunk does not pay (~1.41 tokens emitted per chunk). (Until 2026-08-20 the
serving figure read 0-9 %: an unwritten recurrent snapshot, not the head; the
arithmetic does not change.) NVIDIA's own DSpark drafter measures -42 % in
vLLM on the same card: a verify step costs about four decode steps and
returns 1.7 tokens.

**Reopens if:** the draft path becomes capturable, or a model's verify ratio
drops far enough that the arithmetic changes.

## The two gates that keep speculation off a default server request

**Decision:** speculation stays refused for non-greedy sampling and inside a
budgeted think block. Both were filed as defects (#1538, #1539); both were
measured and both stay.

**Why:** a verify chunk has to earn back what it costs, and on the model
measured it does not. Qwen3-14B-Q6_K, one RTX 5090, 400-token completions,
same prompt, arms alternated:

| arm | tok/s | verify steps |
|---|---|---|
| `temperature: 0.7` (server default) | 157.79, 158.46 | 0 |
| `temperature: 0` (speculation eligible) | 157.61, 157.87 | 6 |

Speculation ON is not faster: 18.6 % acceptance, 5.83 tokens emitted per
verify, 51-68 ms per verify against a 6.3 ms decode step. A verify costs
eight to ten decode steps and returns under six tokens.

Think-block gate, A/B'd the same way (two images, arms alternated), relaxed
to fire only when a chunk could overshoot the budget rather than for the
whole block:

| arm | tok/s | verify steps |
|---|---|---|
| gate as shipped | 162.43, 162.96 | 12 |
| gate relaxed | 157.92, 157.85 | 24 |

Twice the verify steps, 3.0 % less throughput: the gate keeps that part of
the answer from paying for chunks that lose.

Relaxing the think gate alone is also incoherent: the acceptance loop breaks
on `in_think_block` too (`engine_spec_ngram.cpp`), so a relaxed gate without
a matching loop change accepts one token per chunk, strictly worse than
refusing.

**Reopens if:** a model's verify ratio clears its verify cost, which is the same
condition the entry above names. Removing the greedy restriction additionally
needs the verify to accept with min(1, p/q) and resample from the corrected
distribution instead of comparing argmax; that is a feature, and the measurement
above says what it would currently buy.

## The NVFP4 residual add stays a separate kernel

**Decision:** the CUTLASS NVFP4 epilogue keeps beta = 0, and `o_proj` /
`down_proj` on a native-NVFP4 checkpoint keep paying a residual copy plus an
elementwise add. Filed as #1547; built, measured, and not shipped.

**Why:** three measurements, Qwen3-14B-NVFP4 (281 CUTLASS tensors, so the tier
is actually exercised), one RTX 5090, arms alternated, 3 rounds each.

Prefill (the fusion's target): the fused arm won all three paired rounds, but
the within-arm spread of the UNCHANGED arm was 43.8 % (15787, 22707, 20896
tok/s). A delta inside that is not a result; the issue's own ceiling
arithmetic put the prize at ~1.5 %.

  [PROV: commit=4cb36025 date=2026-08-24 hw=RTX5090 model=Qwen3-14B-NVFP4
         quant=NVFP4 cuda=13.3 path=imp-cli --bench
         cmd=`imp-cli --bench --bench-pp 512 --bench-reps 3` and the same with
         `--set speculative.ngram=false`
         n=3 rounds per arm, arms alternated, one process per round]

Decode, speculation off (isolates the kernels): 159.97 against 160.00 tok/s.
Neutral, as expected for a change that only fires at M > 1.

Decode, speculation on (the default): 272.2 against 354.4 tok/s, 23 % down,
spread under 1 %. Mechanism is not the kernel: the fused epilogue rounds once
(FP32 accumulate, one conversion) where copy-plus-add rounds twice, so the
token stream changes; n-gram acceptance fell 99.8 % to 98.8 % and tokens per
verify 8.15 to 6.75 on a deliberately repetitive bench prompt. On a real
request the same evening acceptance is 18.6 %, where a point of acceptance is
noise. The 23 % is a property of the benchmark, not a regression, and not a
reason to take a prefill change that cannot be measured.

The wiring is also not the two-step the issue describes: with the epilogue
taking beta and a test exercising it, `AlphaIsActuallyApplied` left a sticky
`invalid argument` reported by the next GEMM's own guard, three runs of
three, against three clean runs of the same pair without it. Opening this
needs that understood first.

**Reopens if:** the prefill win can be measured above the host's spread, or a
workload is found where the changed rounding does not move the token stream.

## PDL is wired without its device half, and stays that way

**Superseded 2026-08-31.** The device half exists now
(`src/compute/pdl_device.cuh`): registered decode kernels call
`griddepcontrol.wait` before their first global access and
`launch_dependents` after their last input read, and `cuda_graph.cu`
converts an edge only when the CONSUMER is registered (the promise that it
waits; a kernel that does not wait is never registered - the old blanket
list raced greedy determinism once producers triggered). Measured on
Qwen3.8-27B-NVFP4 (final build): M=1 spec-off +1.7% median, 32-stream
aggregate +1.3% median with 3/3 alternating pairs positive in every series,
steady-window GPU idle 13.6% -> 10.8%. `runtime.no_pdl=true` is the control
arm and disables both halves. The entry below is kept as the record of why
the host half alone measured nothing.

**Decision:** the 34 `pdl::launch` sites and the graph-edge rewrite stay, the
device half is not added, and nothing is deleted. #1655 offered both; this is
the third answer.

**Why:** programmatic dependent launch needs a producer calling
`cudaTriggerProgrammaticLaunchCompletion()` and a consumer calling
`cudaGridDependencySynchronize()`. No kernel in `src/` calls either, so a
converted edge releases the consumer exactly where the default edge did. The
mechanism is inert; the header claimed otherwise until this change.

Keeping it costs nothing measurable. Qwen3-8B-Q8_0, one RTX 5090, three
alternating rounds, `runtime.no_pdl` true against false:

| | prefill pp512 | decode tg8192 |
|---|---|---|
| wiring off | 12508, 12630, 12462 tok/s | 385.8, 387.8, 385.5 tok/s |
| wiring on | 12455, 12531, 12091 tok/s | 390.0, 381.8, 382.3 tok/s |

  [PROV: commit=d38ad8a9 date=2026-08-24 hw=RTX5090 model=Qwen3-8B-Q8_0
         quant=Q8_0 cuda=13.3 path=imp-cli --bench
         cmd=`imp-cli --bench --bench-pp 512 --bench-reps 3 --set runtime.no_pdl=<v>`
         n=3 rounds per arm, arms alternated, one process per round]

Off is 0.4 % ahead on prefill and 0.9 % on decode, both inside the arms' own
spread; `is_enabled()` returns before the per-launch set lookup when `no_pdl`
is set, so that lookup is inside the measurement.

Adding the device half means `cudaGridDependencySynchronize()` in every
consumer of every triggering producer, on the decode hot path, for an
unpredicted benefit. Deleting it means touching 34 launch sites on the same
path for a legibility gain, against a measured cost of zero. Neither trade is
worth making blind; what changed is the claim: `pdl.h` states what the
mechanism does today, and the audit's summary rows no longer count PDL as
working sm_120a idiom while its evidence file records `griddepcontrol: 0`.

**Reopens if:** someone completes one producer/consumer pair and measures an
overlap, which is the only thing that would make the rest worth wiring.

