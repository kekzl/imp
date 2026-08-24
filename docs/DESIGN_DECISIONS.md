<!--
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
-->

# Design decisions

Things imp deliberately does not do. Each entry states the decision, the
measurement or constraint behind it, and what would reopen it. An absence with a
reason is a design; an absence without one is a gap, and those live in
[`LIMITATIONS.md`](LIMITATIONS.md).

## No multi-GPU, no tensor parallelism

**Decision:** single GPU only, and this is not a roadmap item.

**Why:** consumer Blackwell has no NVLink. Tensor parallelism on a
bandwidth-bound batch=1 decode has to move activations across PCIe every layer,
which is net-negative on the workload imp targets. Two 5090s over PCIe do not
behave like two datacenter cards over NVLink.

**Reopens if:** a consumer card ships with a high-bandwidth inter-GPU link, or
the target workload moves to large-batch serving where the collective cost
amortises.

## No CPU inference path, no AVX kernels

**Decision:** the GPU computes. There is no CPU fallback and no AVX-512 expert
path.

**Why, and this one was measured rather than assumed.** The obvious version of
the idea, keeping cold MoE experts in host RAM and computing them on the CPU, was
budgeted on this host in 2026-08-10: 62.5 GB/s streaming read, 16 threads. For a
120B-A5B shape that is 14.0 ms/token including round trips. The alternative,
streaming the missing expert into VRAM and letting the GPU compute it, costs
4.7-8.9 ms for the same token, because expert selection has strong temporal
locality: median reuse distance is 2 tokens and an LRU at 40 % residency hits
88.7-96.2 %. The GPU-side design won on measurement, and it is the one that
shipped.

**Reopens if:** a model arrives whose active expert set defeats the LRU, or host
memory bandwidth changes class.

## Blackwell only, and specifically consumer Blackwell

**Decision:** `sm_120a` SASS with a `compute_120f` PTX fallback. Nothing else.

**Why:** the delimitation against datacenter Blackwell is a hardware fact, and it
is stated once, in
[`internals/ARCHITECTURE.md`](internals/ARCHITECTURE.md#target-architecture).
Supporting a second architecture means either a portability layer in the hot path
or a second set of kernels; both cost more than this project has.

## No FP4 cuBLASLt path

**Decision:** CUTLASS is the primary GEMM path.

**Why:** there are no FP4 cuBLASLt kernels on `sm_120`. FP8 prefill is
unavailable for the same reason. This is not a preference.

## The changelog is not a journal

**Decision:** one to three lines per entry. The investigation lives in `docs/`
and the PR.

**Why:** the CHANGELOG and the release page are the only two artefacts read by
people who never opened the repo. A 15-line entry is a research note wearing a
changelog's clothes, and it buries the line the reader came for.

## No GPU runner in CI

**Decision:** the `test` job stays gated behind `vars.HAS_GPU_RUNNER`, off.

**Why:** a hosted GPU runner costs more than this project spends, and a
self-hosted one on the maintainer's only card would contend with the
measurements it exists to protect. The consequence is stated plainly in
[`LIMITATIONS.md`](LIMITATIONS.md) rather than hidden: `make verify-fast` before
push is the only thing that ever runs a CUDA kernel against a correctness or
performance check.

## Speculative decoding stays opt-in

**Decision:** drafts are available, none is on by default for every model.

**Why:** the economics are per-model and were measured both ways. Prompt-lookup
n-gram is default-on for batch-1 greedy dense because its drafts are free. The
MTP head on Nemotron-3.5 accepts 41 % at depth 1 offline and 39 % on the serving
path, and still costs 51 % of decode: the drafts are good, the verify chunk is
what does not pay (~1.41 tokens emitted per chunk). Until 2026-08-20 the serving
figure read 0-9 %, which was an unwritten recurrent snapshot rather than the
head — the arithmetic below did not depend on it, and does not change.
NVIDIA's own DSpark drafter measures -42 % in vLLM on the same card. A verify
step there costs about four decode steps and returns 1.7 tokens.

**Reopens if:** the draft path becomes capturable, or a model's verify ratio
drops far enough that the arithmetic changes.

## The two gates that keep speculation off a default server request

**Decision:** speculation stays refused for non-greedy sampling and inside a
budgeted think block. Both were filed as defects (#1538, #1539); both were
measured and both stay.

**Why:** on this hardware a verify chunk has to earn back what it costs, and on
the model measured it does not. Qwen3-14B-Q6_K, one RTX 5090, 400-token
completions, same prompt, arms alternated:

| arm | tok/s | verify steps |
|---|---|---|
| `temperature: 0.7` (server default) | 157.79, 158.46 | 0 |
| `temperature: 0` (speculation eligible) | 157.61, 157.87 | 6 |

Turning speculation ON is not faster. The log says why: 18.6 % acceptance, 5.83
tokens emitted per verify, 51-68 ms per verify against a 6.3 ms decode step. A
verify costs eight to ten decode steps and returns under six tokens.

The think-block gate was A/B'd the same way, two images, arms alternated, by
relaxing it to fire only when a chunk could overshoot the budget rather than for
the whole block:

| arm | tok/s | verify steps |
|---|---|---|
| gate as shipped | 162.43, 162.96 | 12 |
| gate relaxed | 157.92, 157.85 | 24 |

Twice the verify steps, 3.0 % less throughput. The gate is not an oversight
keeping speculation out of most of a reasoning answer; it is what keeps that
part of the answer from paying for chunks that lose.

Relaxing the think gate alone is also not enough to be coherent: the acceptance
loop breaks on `in_think_block` too (`engine_spec_ngram.cpp`), so a relaxed gate
without a matching loop change accepts one token per chunk, which is strictly
worse than refusing.

**Reopens if:** a model's verify ratio clears its verify cost, which is the same
condition the entry above names. Removing the greedy restriction additionally
needs the verify to accept with min(1, p/q) and resample from the corrected
distribution instead of comparing argmax; that is a feature, and the measurement
above says what it would currently buy.

