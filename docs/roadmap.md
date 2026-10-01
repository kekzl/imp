# Roadmap

Single-author, single-GPU experiment: "roadmap" means current focus, not
schedule. Shipped work is in [`CHANGELOG.md`](../CHANGELOG.md), competitive
numbers in [`BENCHMARKS.md`](BENCHMARKS.md), limitations in
[`LIMITATIONS.md`](LIMITATIONS.md).

| House rule | |
|---|---|
| Row form | fact + number + decision + ref, one row each |
| Investigation | goes to `docs/plans/`, the PR body or `LIMITATIONS.md`, never into a table cell |
| Lifecycle | entries are closed, corrected or superseded in place, never deleted |
| Citations | `scripts/check_doc_citations.py` requires `path:N anchor`; anchor on line N = ok, anchor moved (once in the file, or the one hit within 25 lines) = `DRIFT` warning, exit 0, `--fix` rewrites N; anchor gone or ambiguous = `DEAD` (#2231: line drift alone turned main red 3x on 2026-09-29; #2185: the existence-only check passed a `weight_map.cpp` cite eleven lines off until 2026-08-31 and two `Makefile` cites 16 and 81 lines off); a bare basename matching two files is resolved by the anchor, else reports `AMBIGUOUS` and passes, so cite the path; a stale `git worktree` checkout makes every basename ambiguous at once |

Detail records: [`plans/2026-09-16-open-rows-closed-detail.md`](plans/2026-09-16-open-rows-closed-detail.md)
(Open rows 1, 2, 4, 7, 13, 14, 15 as they stood when closed),
[`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md)
(serving and kernel rows, 08-25 .. 09-04),
[`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md)
(everything moved out on 2026-08-31).

## Direction

| | |
|---|---|
| Goal | fastest local engine for AI agent workloads on consumer Blackwell |
| Workload | 20k-100k+ tokens per session, context accumulates, streams run in parallel |
| Working regime | aggregate throughput at tens of concurrent streams; batch=1 is settled |
| Foundations (2026-05) | chunked-prefill FMHA + 256 MiB S-matrix, ctx ~4-6k to 32k+ (#453), multi-request decode batching (#454), StreamingLLM auto-enable on full KV (#455) |
| Serving ground (2026-08) | warm weight cache (#956), suspend-to-RAM (#954), request-order independence (#957), gemma-3 IMA fix (#959) |

## Standing position (2026-09-04)

| axis | state |
|---|---|
| GDN hybrid @32 vs vLLM | AHEAD. Qwen3.8-27B 1807.9 vs 1447.8 tok/s (+24.9%, vLLM 0.27.1), 1833.8 vs 1410.7 (+30.0%, 0.28.0), @8 573.0 vs 495.8 (+15.6%), @32 x 1082-token prompts 873.4 vs 497.8 (+75.5%), 3/3 each |
| dense NVFP4 @32 vs vLLM | PARITY OR AHEAD (2026-09-08). Qwen3-14B 38-token prompts 3948.9 vs 3817.6 (+3.4%, 2026-09-03); 982-token prompts 2491.2/2515.5/2482.6 vs 2485.9/2490.9/2497.7 (+0.2/+1.0/-0.6%, 2 of 3; was 1845.0 vs 2478.0 = 0.75x before the grouped FP8 decode attention, #1953) |
| batch=1 | 99.5 tok/s spec-off (2026-09-10, `gdn.m1_fused`) = 89% of the ~112 tok/s roofline (14.5 GB/token at 1628 GB/s resident), was 87.4 = 78% on 2026-08-27; past it only through the MTP verify |
| raw-speed half of [`GOAL.md`](GOAL.md) | MET: batch=1 decode +13-48% vs llama.cpp on every hero (2026-07-12 re-sweep), MoE prefill leads vLLM single-seq, cross-engine PPL parity measured. Everything open below is the agentic half |
| admission at fan-out | `auto` resolves 28 vs a pinned 32 (630 vs 936): admission, not rotation, and 28 sustain full rate under continuous arrival |
| next engine-side post | the dense decode step at 32 streams after the grouped FP8 attention and the small-M launch work (ledger 2026-09-08, second row): the small-M GEMM class re-priced at 83% of resident bandwidth by launch durations (5.49 vs 4.56 ms floor per Qwen3-14B step BEFORE the PDL weight prefetch + q\|k\|v launch, which took ~0.3 ms of that), and the prefill forward at 80% of the FP4 peak (Open 2) |

Both engines measured on one client (`tools/analysis/vllm_conc_ab.sh`, 3
alternating trials, same checkpoint); the "1.58x gap" and "~1.08x pinned" of
2026-08-24/26 compared an imp number from this client against a vLLM number
from a 200-token-gen client. Rows with PROV in [`BENCHMARKS.md`](BENCHMARKS.md)
("re-measured on one client", runs 5-9).

## Open

Ranked by what an agent workload notices first.

| # | item | state | ref |
|---|---|---|---|
| 3 | long context | HALF CLOSED. Quest-class top-k page selection opt-in: Qwen3-8B 32k 160.3 -> 199.5 tok/s (+24.5%), NVFP4-KV 77k 74.3 -> 100.2, concurrent 3x25k +27%, spec verify on the sparse table +28.2%. The retrieval price was the min/max corner bound, not the sparsity: the mean+std page score (`attention.sparse_score_meanstd`, default on since 2026-09-12) reads NIAH 10/10 on Qwen3.8-27B at every budget from 1024 to 8192, against 0/10, 1/10, 2/10 and 7/10 for the bound on the same binary, wall time neutral. NIAH is saturated there, so a small-budget selection refinement has no target on it, and the budget is not a speed lever either (sparse is +9% over dense, 1024 against 8192 inside the spread). Remaining: MLA models, prefill sparsity, StreamingLLM eviction as the only answer under KV-pool pressure (the valve arms on every KV dtype since 2026-09-16; before that F16 only, and an FP8/NVFP4 pool cancelled the request once it ran dry) | #1808, #1818, #1819, [plan](plans/2026-08-28-sparse-decode-attention.md) |
| 10 | one VL tower family | port-sized (InternVL/Pixtral); `vision_tower_supported()` names one layout, and a second model on the SAME tower cost two gates | #1379, #1384 |

## Not gaps


Explicitly NOT gaps: continuous batching, prefix caching, per-request LoRA,
embeddings, the three API dialects, `/metrics`, suspend/resume, sampler surface

## Archive

Moved on 2026-09-28 to [`archive/roadmap_ledger_2026_09_28.md`](archive/roadmap_ledger_2026_09_28.md), text verbatim, relative links re-based with `../`:

| section | archive anchor |
|---|---|
| Open row 12 (closed 2026-09-16) | [Open rows closed before the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-before-the-move) |
| Open rows 5, 6, 8, 9, 11 (closed 2026-10-01) | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) |
| Closed | [Closed](archive/roadmap_ledger_2026_09_28.md#closed) |
| The 2026 bar | [The 2026 bar](archive/roadmap_ledger_2026_09_28.md#the-2026-bar-assessed-2026-08-21) |
| Lever ledger | [Lever ledger](archive/roadmap_ledger_2026_09_28.md#lever-ledger) |
| Batch=1, MTP verify on a GDN hybrid | [Batch=1](archive/roadmap_ledger_2026_09_28.md#batch1) |
| MoE host offload | [MoE host offload](archive/roadmap_ledger_2026_09_28.md#moe-host-offload) |
| First-party NVFP4 quantizer | [NVFP4 quantizer](archive/roadmap_ledger_2026_09_28.md#first-party-nvfp4-quantizer-experimental-calibration-ships) |
| Closed competitive records | [records](archive/roadmap_ledger_2026_09_28.md#closed-competitive-records) |
| Known limitations | [Known limitations](archive/roadmap_ledger_2026_09_28.md#known-limitations) |
| Investigated and shelved | [shelved](archive/roadmap_ledger_2026_09_28.md#investigated-and-shelved) |
