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
MTP head on Nemotron-3.5 accepts 43.9 % at depth 1 and still costs 32 % of
decode, because a draft step runs eager while the main decode runs in a graph.
NVIDIA's own DSpark drafter measures -42 % in vLLM on the same card. A verify
step there costs about four decode steps and returns 1.7 tokens.

**Reopens if:** the draft path becomes capturable, or a model's verify ratio
drops far enough that the arithmetic changes.
