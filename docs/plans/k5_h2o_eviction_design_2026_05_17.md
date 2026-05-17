# K5 — H2O token eviction — design scoping memo

**Date**: 2026-05-17
**Status**: design only — no source changes
**Subject**: Should imp implement Heavy-Hitter Oracle (H2O,
[arxiv:2306.14048](https://arxiv.org/abs/2306.14048)) KV-cache token
eviction? Quality-risky on retrieval; only useful in the regime where
NVFP4 KV + K8 CPU offload are both saturated.
**Branch**: `main` (after `79ec86b perf(compute): M5 Slice 2.3`)
**Companion memo**: K8 CPU offload design memo (being scoped in parallel
on 2026-05-17, same docs/plans/ directory).

## Table of contents

1. [Status — is the VRAM pressure even there?](#1-status--is-the-vram-pressure-even-there)
2. [H2O algorithm summary](#2-h2o-algorithm-summary)
3. [Empirical reality check (since 2023)](#3-empirical-reality-check-since-2023)
4. [H2O successors worth tracking](#4-h2o-successors-worth-tracking)
5. [Implementation cost on imp](#5-implementation-cost-on-imp)
6. [Decision matrix](#6-decision-matrix)
7. [Decision recommendation](#7-decision-recommendation)

---

## 1. Status — is the VRAM pressure even there?

H2O exists to solve one problem: "my context does not fit in VRAM under
any quant scheme." That framing has to be checked against where imp
actually sits today, not where it sat when the roadmap entry was
written.

### 1.1 Floor 1 — NVFP4 KV (shipped 2026-05-07/08)

Lever 2 ([lever2_nvfp4_kv_implemented_2026_05_07](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/lever2_nvfp4_kv_implemented_2026_05_07.md))
landed end-to-end NVFP4 KV cache: 4-bit packed K/V + per-group UE4M3
scale, vectorized `cvt.rn.f16x2.e2m1x2` PTX dequant in the inner loop.
Measured compression on Qwen3-8B Q8: 140 KiB/tok → 36 KiB/tok = **3.9×**.
Decode at parity with FP16 baseline (147 vs 147.5 tok/s) after the
follow-on PTX cvt patch.

Max usable context per model class under the current 32 GiB VRAM budget,
extrapolated from the Lever 2 memo's measured Qwen3-8B unlock (16 384 →
40 960 tokens) and the per-model table in
[nvfp4_kv_potential_2026_04_25](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/nvfp4_kv_potential_2026_04_25.md):

| Model class | FP16 KV cap | NVFP4 KV cap | Note |
|---|---|---|---|
| Qwen3-8B Q8 | 16 k | **≈ 40 k** | measured, model-native max |
| Qwen3-32B Q4_K_M | 25.8 k | **≈ 80 k+** | extrapolated, deferred verify |
| Gemma-3-27B Q4 | 22.5 k | **≈ 80 k+** | extrapolated, deferred verify |
| Gemma-4-26B-A4B Q4 | 13.2 k | **≈ 40-50 k** | needs dual head_dim coherence run |
| Qwen3-Coder-30B MoE | KV not the limit | — | expert cache dominates |
| Qwen3.6-35B GDN-MoE | KV not the limit | — | only 10/40 attention layers |

Bottom line: NVFP4 KV alone already moves every dense large model out
of the "doesn't fit" regime at 32 k–80 k contexts. Pure MoE and
GDN-hybrid models are not KV-bound at all.

### 1.2 Floor 2 — K8 CPU offload (design memo today)

The companion K8 design memo (also dated 2026-05-17) covers PCIe-async
prefetch of cold KV blocks to host pinned memory. Once shipped, the
ceiling stops being VRAM capacity and starts being PCIe bandwidth +
prefetch latency. Practical context ceiling under K8 is in the **256 k+
range** (PCIe 5.0 x16 ≈ 64 GB/s, K8 is by construction designed to
hide that behind compute).

K5 and K8 are alternatives, not stack-mates, for the VRAM-pressure
problem:

- K8: hide latency, keep everything, no quality cost. Cost: PCIe BW
  budget + driver complexity.
- K5: drop tokens, save VRAM, **measurable retrieval quality cost**.
  Cost: eviction bookkeeping + retrieval task regressions.

### 1.3 Conclusion

H2O is only relevant in the intersection of three conditions:

1. NVFP4 KV is enabled and the context still doesn't fit (rare; would
   need 100 k+ on a model that can't open with NVFP4 alone).
2. K8 CPU offload is either not shipped or is bandwidth-saturated
   (PCIe budget exhausted by other transfers — e.g. MoE expert
   streaming on Qwen3-Coder-30B-Q6_K, see
   [moe_expert_offload_fix_2026_04_24](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/moe_expert_offload_fix_2026_04_24.md)).
3. The workload is **not retrieval-dominated** (chat / agentic
   scratchpad / summarization) — so the documented quality
   degradation in §3 is acceptable.

None of imp's current production workloads sit in that intersection.
That's the headline.

## 2. H2O algorithm summary

From [arxiv:2306.14048](https://arxiv.org/abs/2306.14048):

- During attention, accumulate per-token attention-score history
  across the layer/head stack. The paper proves a power-law structure:
  a small fraction of tokens carries most of the attention mass.
- Maintain a "Heavy Hitter" set of top-K tokens by cumulative score.
  Evict the rest from the KV cache. Eviction is formulated as dynamic
  submodular maximisation with greedy-quality bounds.
- **Retention 5-20% claimed** at "minor quality loss" on the
  original benchmarks → effective **up to 20× memory reduction**.
- Original eval: Llama-1-7B and OPT-6.7B/30B, on **MMLU,
  CommonsenseQA, XSum**. *None of these are retrieval benchmarks*;
  that's the load-bearing detail §3 picks up.

## 3. Empirical reality check (since 2023)

The H2O paper's "no accuracy drop" claim has been re-tested by every
serious long-context evaluation since:

- **RULER** (NVIDIA, 2024, [arxiv:2404.06654](https://arxiv.org/abs/2404.06654)):
  H2O degrades significantly on long-context retrieval (NIAH at 32 k+).
  NIAH = Needle-In-A-Haystack: drop a fact at random offset into a
  long context, ask the model to retrieve it. H2O evicts the "needle"
  token early because it accrues low attention mass until the query
  arrives — at which point it's already gone.
- **Multi-hop QA** (MuSiQue / HotPotQA evals): ~20 % accuracy loss at
  5 % retention. The supporting documents in multi-hop chains carry
  individually-low attention mass; the eviction policy can't predict
  which "low-mass-now" tokens become "high-mass-later" once a later
  query connects them.
- **Q-Hitter** (MLSys 2024) replicated the retrieval-degradation finding
  and proposed the fix (see §4).

The general principle: **H2O's "minor quality loss" only holds on
benchmarks where the answer is in recent tokens** (chat continuation,
summarisation of the tail). Retrieval, multi-hop reasoning, long-doc
QA, and codebase-spanning tasks are explicitly out of scope of the
original eval. Those happen to be exactly the workloads where users
want long context in the first place.

## 4. H2O successors worth tracking

Four successors have appeared since 2023, each trying to keep the
compression while restoring retrieval quality:

### 4.1 Q-Hitter (MLSys 2024)

Two changes over H2O: (a) **quantization-aware eviction** — score
includes the per-token contribution to attention output magnitude, not
just attention probability, so quantization-amplified tokens stay; (b)
**dynamic threshold** — replaces fixed top-K with an adaptive cutoff
that responds to attention-mass distribution. Claimed -5 % NIAH at 32 k
vs original H2O's -20 %. Engineering cost similar to H2O proper.

### 4.2 SnapKV (2024, [arxiv:2404.14469](https://arxiv.org/abs/2404.14469))

**Observation-window-based selection**: instead of accumulating scores
over the full history, SnapKV looks at attention patterns in a sliding
window of recent tokens to predict which prior tokens will be needed.
Empirically the recent window is a strong predictor for the
near-future, and retention decisions made at observation-window
boundaries are far more accurate than per-step greedy eviction.
Claimed -2 % NIAH at 32 k. Engineering cost: window logic + batched
re-decision points, materially higher than H2O.

### 4.3 PyramidKV (2024, [arxiv:2406.02069](https://arxiv.org/abs/2406.02069))

**Layer-dependent retention**: bottom transformer layers keep most
tokens (they encode low-level lexical features), top layers prune
aggressively (high-level semantic abstraction tolerates eviction).
Retention shape across layers is the "pyramid." Claimed near-lossless
at NIAH 32 k (≤ -1 %). Engineering cost: per-layer policy +
per-layer block-table accounting — the highest-cost option in the
table because every layer becomes a separate eviction tuning problem.

### 4.4 DuoAttention (2024, [arxiv:2410.10819](https://arxiv.org/abs/2410.10819), Han Lab)

Two-branch attention: a **trained retrieval mask** marks "retrieval
heads" that get full attention vs "streaming heads" that get sparse.
Quality is near-lossless on NIAH because the retrieval branch keeps
full KV; cost is the retraining/calibration step required to produce
the mask. Cannot be applied post-hoc without per-model calibration
data.

## 5. Implementation cost on imp

The work breakdown if imp ever takes on any of these (they all share
~80 % of the plumbing):

### 5.1 Score-tracking

Per-token cumulative attention scores are already computed transiently
during softmax in the attention kernels (`attention_paged_*.cu`). The
extra cost is: accumulate the per-token contribution into a
small persistent buffer (one FP16/FP32 score per token per
layer/head, scoped per request). For 8 KV heads × 64 layers × 32 k
tokens × FP16 = 32 MiB per sequence — non-trivial but feasible.
Lower the precision if needed.

### 5.2 Block-level eviction granularity

imp's paged KV uses `kKVBlockSize = 16` (`src/memory/kv_cache.h:12`).
Eviction granularity is therefore 16 tokens, not 1. Eviction policy
must aggregate per-token scores over a block before deciding.
Practical effect: actual retention floor is somewhere between
"5 % of tokens" and "5 % of blocks", which biases the achievable
compression slightly down from the paper's headline number.

### 5.3 Bookkeeping

- Per-request heavy-hitter set (block indices kept).
- Per-request block-table mutation when blocks are evicted (free the
  block back to the pool, remove from the request's logical→physical
  mapping).
- Re-indexing of attention kernel inputs: the gather kernel
  (`src/compute/paged_kv_gather_*.cu`) is already indirected through a
  block table, so the change is mostly in how the block table is
  pruned, not in the kernels themselves. That's a real win — no new
  attention kernel variant required.

### 5.4 Test infrastructure (this is the long pole)

There is currently no NIAH/RULER harness in `tests/` or `scripts/`.
Building a credible eviction-vs-quality eval needs:

- A small NIAH-style probe corpus (~hundreds of prompts × multiple
  context lengths).
- A multi-hop QA probe corpus (HotPotQA or MuSiQue subset).
- Scoring harness that compares H2O-evicted runs to full-KV baseline
  per model.

This is **multi-week effort by itself**, separate from the eviction
code. Without it, the project ships blind.

### 5.5 Total

Multi-week implementation, comparable in scope to K8 CPU offload but
**with quality risk that K8 does not have**.

## 6. Decision matrix

Per-successor comparison at 32 k context (numbers as claimed by each
paper; imp has not independently verified any of them):

| Algorithm | NIAH @ 32 k vs full attn | Engineering cost on imp | Notes |
|---|---|---|---|
| H2O original | -20 % | Medium (1-2 wk + test harness) | Outdated; do not adopt as-is |
| Q-Hitter | -5 % (claimed) | Medium-High | Better; mostly the same plumbing as H2O |
| SnapKV | -2 % (claimed) | High | Observation-window logic adds complexity |
| PyramidKV | -1 % (claimed) | High | Per-layer tuning; highest engineering surface |
| DuoAttention | ~0 % (trained) | Very High | Requires per-model calibration / training step |

All numbers are paper-reported; cross-evaluation in newer surveys is
spotty, so the gaps could be smaller in practice. The qualitative
ordering (original H2O is worst, retrained DuoAttention is best) is
robust across the literature.

## 7. Decision recommendation

**Defer indefinitely.**

The case for H2O on imp does not close: NVFP4 KV is the right
floor for VRAM compression (3.9× at quality parity, no algorithmic
quality risk), the K8 design memo identifies CPU offload as the
right next layer for the 100 k+ regime (no quality risk, only PCIe
budget), and no imp production workload today is in the intersection
where K5 would beat both. The successors (Q-Hitter, SnapKV,
PyramidKV, DuoAttention) cost as much or more to implement than K8
while still trading some retrieval quality.

**Re-eval triggers** (any one of these flips the decision):

- A workload appears where the user needs ≥ 100 k context, the model
  cannot open at that context under NVFP4 KV alone, **and** K8 CPU
  offload is PCIe-saturated by concurrent MoE expert streaming.
- A successor lands with a credibly-trained calibration recipe
  (DuoAttention-style) that closes the retrieval gap to ≤ 1 % on
  RULER 32 k+ without requiring imp-side retraining.
- An imp user reports that retrieval quality on long-context tasks is
  *not* a requirement for their workload — at which point a
  successor like SnapKV (-2 % NIAH at 32 k) becomes an interesting
  predicate-defer pick.

### Why not "investigate SnapKV first"

The third option in the prompt — Phase-1 microbench of SnapKV to see
the real quality curve — was considered and rejected for one reason:
the microbench itself requires building the NIAH/RULER harness (§5.4),
which is the multi-week long pole of the whole project. If that
harness is going to be built, it will pay off whether or not SnapKV is
the chosen algorithm — it's a general long-context-quality measurement
tool. The right sequencing is: **build the long-context quality
harness first, independently of K5**, then re-evaluate K5 once the
harness exists and a workload predicate is met. The harness has its
own writeup as a future docs/plans entry.

## Cross-references

- [`lever2_nvfp4_kv_implemented_2026_05_07`](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/lever2_nvfp4_kv_implemented_2026_05_07.md) — current KV compression floor (3.9×).
- [`kv_research_grade_eval_2026_05_09`](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/kv_research_grade_eval_2026_05_09.md) — earlier H2O evaluation; this memo is the formal scoping.
- [`bitdecoding_long_context_eval_2026_05_14`](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/bitdecoding_long_context_eval_2026_05_14.md) — confirms decode is bandwidth-bound, not attention-math-bound, at consumer-Blackwell scale.
- [`nvfp4_kv_potential_2026_04_25`](../../.claude/projects/-home-kekz-github-com-kekzl-imp/memory/nvfp4_kv_potential_2026_04_25.md) — per-model VRAM headroom under NVFP4 KV.
- K8 CPU offload design memo — being scoped in parallel on 2026-05-17 in `docs/plans/`.
- H2O paper: [arxiv:2306.14048](https://arxiv.org/abs/2306.14048).
- RULER (NIAH at 32 k+ retrieval): [arxiv:2404.06654](https://arxiv.org/abs/2404.06654).
- SnapKV: [arxiv:2404.14469](https://arxiv.org/abs/2404.14469).
- PyramidKV: [arxiv:2406.02069](https://arxiv.org/abs/2406.02069).
- DuoAttention: [arxiv:2410.10819](https://arxiv.org/abs/2410.10819).
