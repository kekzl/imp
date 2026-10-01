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
| Citations | `scripts/check_doc_citations.py` needs `path:N anchor`: anchor on N = ok, moved = `DRIFT` (exit 0, `--fix`), gone or ambiguous = `DEAD`, bare basename on 2 files = `AMBIGUOUS` (passes), so cite the path; [history](archive/roadmap_ledger_2026_09_28.md#house-rule-history) |

## Direction

| | |
|---|---|
| Goal | fastest local engine for AI agent workloads on consumer Blackwell |
| Workload | 20k-100k+ tokens per session, context accumulates, streams run in parallel |
| Working regime | aggregate throughput at tens of concurrent streams; batch=1 is settled |
| Foundations (2026-05) | chunked-prefill FMHA + 256 MiB S-matrix, ctx ~4-6k to 32k+ (#453), multi-request decode batching (#454), StreamingLLM auto-enable on full KV (#455) |
| Serving ground (2026-08) | warm weight cache (#956), suspend-to-RAM (#954), request-order independence (#957), gemma-3 IMA fix (#959) |

## Standing position

| axis | state | measured | ref |
|---|---|---|---|
| GDN hybrid @32 vs vLLM | AHEAD. Qwen3.8-27B 1807.9 vs 1447.8 tok/s (+24.9%, vLLM 0.27.1), 1833.8 vs 1410.7 (+30.0%, 0.28.0), @8 573.0 vs 495.8 (+15.6%), @32 x 1082-token prompts 873.4 vs 497.8 (+75.5%), 3 trials each | 2026-09-02 | [`BENCHMARKS.md`](BENCHMARKS.md#imp-vs-vllm-at-concurrency) runs 1-4 |
| dense NVFP4 @32 vs vLLM | PARITY OR AHEAD. Qwen3-14B 38-token prompts 3948.9 vs 3817.6 (+3.4%, 2026-09-03); 982-token prompts +0.2/+1.0/-0.6% (was 0.75x before #1953) | 2026-09-08 | [`benchmarks_pre_v0.44.md`](archive/benchmarks_pre_v0.44.md) runs 8-10 |
| batch=1 | 99.5 tok/s spec-off (`gdn.m1_fused`) vs the ~112 tok/s roofline (14.5 GB/token at 1628 GB/s resident), was 87.4 = 78% on 2026-08-27; past it only through the MTP verify | 2026-09-10 | `CHANGELOG.md` 0.39.0, [archive](archive/roadmap_ledger_2026_09_28.md#batch1) |
| raw-speed half of [`GOAL.md`](GOAL.md) | MET: batch=1 decode +13-48% vs llama.cpp on every hero, MoE prefill leads vLLM single-seq, cross-engine PPL parity measured | 2026-07-12 | [`PERF.md`](PERF.md), [`GOAL.md`](GOAL.md) |

Method: both engines on one client (`tools/analysis/vllm_conc_ab.sh`), 3 alternating trials, same checkpoint; PROV rows in [`BENCHMARKS.md`](BENCHMARKS.md#imp-vs-vllm-at-concurrency). Superseded rows: [archive](archive/roadmap_ledger_2026_09_28.md#standing-position-rows-superseded-2026-10-01).

## Open

No open rows since 2026-10-01; rows 1-15 are closed (index below), a new row takes number 16, ranked by what an agent workload notices first.

## Not gaps

Continuous batching, prefix caching, per-request LoRA, embeddings, the three API dialects, `/metrics`, suspend/resume, sampler surface.

## Closed rows and archive

| closed row or section | closed or moved | archive anchor | detail record |
|---|---|---|---|
| Open 1, 2, 4, 7, 13, 14, 15 | 2026-09-16 | [Closed](archive/roadmap_ledger_2026_09_28.md#closed) | [`plans/2026-09-16-open-rows-closed-detail.md`](plans/2026-09-16-open-rows-closed-detail.md) |
| Open 12 | 2026-09-16 | [Open rows closed before the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-before-the-move) | `gemm_cublas` pricing in [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md), closure evidence in the row |
| Open 3 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`plans/2026-08-28-sparse-decode-attention.md`](plans/2026-08-28-sparse-decode-attention.md) |
| Open 5 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`plans/2026-10-01-recurrent-state-paging-measured.md`](plans/2026-10-01-recurrent-state-paging-measured.md) |
| Open 6 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`quantization.md`](quantization.md), #2359 |
| Open 8 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | owner decision, [`LIMITATIONS.md`](LIMITATIONS.md#model-specific-blockers), #2359 |
| Open 9 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`API_FEATURES.md`](API_FEATURES.md#video), #2363, #2370 |
| Open 10 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | [`MODELS.md`](MODELS.md#vision), #2375 .. #2389 |
| Open 11 | 2026-10-01 | [Open rows closed after the move](archive/roadmap_ledger_2026_09_28.md#open-rows-closed-after-the-move) | owner decision, AUDIT B84, B36, #2359 |
| The 2026 bar | 2026-09-28 | [The 2026 bar](archive/roadmap_ledger_2026_09_28.md#the-2026-bar-assessed-2026-08-21) | same section |
| Lever ledger | 2026-09-28 | [Lever ledger](archive/roadmap_ledger_2026_09_28.md#lever-ledger) | [`plans/2026-09-04-lever-ledger-detail.md`](plans/2026-09-04-lever-ledger-detail.md) (serving and kernel rows, 08-25 .. 09-04) |
| Batch=1, MTP verify on a GDN hybrid | 2026-09-28 | [Batch=1](archive/roadmap_ledger_2026_09_28.md#batch1) | same section |
| MoE host offload | 2026-09-28 | [MoE host offload](archive/roadmap_ledger_2026_09_28.md#moe-host-offload) | same section |
| First-party NVFP4 quantizer | 2026-09-28 | [NVFP4 quantizer](archive/roadmap_ledger_2026_09_28.md#first-party-nvfp4-quantizer-experimental-calibration-ships) | same section |
| Closed competitive records | 2026-09-28 | [records](archive/roadmap_ledger_2026_09_28.md#closed-competitive-records) | same section |
| Known limitations | 2026-09-28 | [Known limitations](archive/roadmap_ledger_2026_09_28.md#known-limitations) | [`LIMITATIONS.md`](LIMITATIONS.md) |
| Investigated and shelved | 2026-09-28 | [shelved](archive/roadmap_ledger_2026_09_28.md#investigated-and-shelved) | same section |
| Citations rule history (#2231, #2185) | 2026-10-01 | [House rule history](archive/roadmap_ledger_2026_09_28.md#house-rule-history) | same section |
| Standing position (2026-09-04) table | 2026-10-01 | [Standing position rows superseded 2026-10-01](archive/roadmap_ledger_2026_09_28.md#standing-position-rows-superseded-2026-10-01) | same section |
| Everything moved out on 2026-08-31 | 2026-08-31 | - | [`plans/2026-08-31-roadmap-ledger-detail.md`](plans/2026-08-31-roadmap-ledger-detail.md) |

Archive moves keep text verbatim, relative links re-based with `../`.
