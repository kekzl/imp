# REPORT: backlog run 2026-10-18

Items: [`BACKLOG_2026_10_18.md`](BACKLOG_2026_10_18.md). Evidence per item: [`DISPATCH_BACKLOG_2026_10_18.md`](DISPATCH_BACKLOG_2026_10_18.md). Perf entries: [`PERF_LOG.md`](PERF_LOG.md) 2026-10-18.

Commit range: main a6492284 (start) .. 09797f27, PRs #2777, #2778, #2779, #2780, plus this report's PR.

| Status | Count |
|---|---:|
| DONE | 28 |
| WONTFIX | 13 |
| BLOCKED (owner question or missing model) | 10 |
| IN_PROGRESS | 1 (R-119) |
| OPEN (L rows with plans) | 12 |

## Done

| ID | Result |
|---|---|
| B-05, B-06 | `-Wdangling-else` 6 -> 0, `-Wmissing-field-initializers` 14 -> 0 |
| B-08, B-22 | clang-tidy sweep: parentheses 4110, contains 244, min/max 110, 7 modernize checks enabled, 0 findings left for those checks |
| B-23 | duplicated rmsnorm / kv-gather kernels templated; SASS of 1114 of 1123 kernels identical, 9 differ only in the `std::max` select form |
| B-25 | radix-select top-k: 66.77 -> 18.76 us per decode step (Qwen3-4B Q8_0, top-k 40), tokens byte-identical over 588 configs x 12 seeds |
| B-24 | FFN prefill graphs reuse evicted execs: 396 -> 324 instantiates on 40 mixed prompts |
| B-29a, B-29d | batched NVFP4 LM head borrows the prefill SfAtom (Qwen3-14B 46 MiB); nomic-bert frees its Q8_0 sources (pool 128 -> 0 MiB) |
| B-11, B-28, B-09 | shared-workspace fallback through the allocator; release gate runs alloc-interpose (PASS 0/3/0); strict `IMP_FMHA_FA2`, shared cuBLAS helper in `gptq.cu` |
| B-35 | InternVL3.5 video parts (HF `replace_video_token` layout): "A cat and red double decker buses." |
| R-077 | closed: Gemma-4 overlay 256.48 tg vs MMVQ 116.85, no kernel unreachable without a knob |
| R-119.1-119.3 | offload pass flat in width (188.6 ms at 1, 189.71 at 64); n-gram never gives up under offload; token recycling width 4 default there: Qwen3-14B Q6_K `--gpu-layers 0` prose 5.30 -> 6.18, code 6.85 -> 8.85 tok/s, greedy output byte-identical |
| B-01..B-03, B-10, B-12, B-15..B-21, B-30, B-31 | already fixed on main; verified (green test, later ledger entry, commit or re-measurement) |

## Wontfix

| ID | Reason |
|---|---|
| B-04 | `docs/DETERMINISM.md:43` known limit 5 (fresh contexts in one process), workaround documented |
| B-07 | GCC `-Warray-bounds` false positive (`snapshot_boundary.h:88` returns before the loop) |
| B-13, B-14 | C-023 / C-024 measured, not pursued (269.87 vs 269.91 tok/s; sites off the default tiers) |
| B-26, B-27, B-32 | split refuted (SETTLED S-32); flags "not a defect" per the ledger; ledger closed the Gemma-4 delta itself |
| W-01..W-05 | env-gated modes, debug-only aborts, on-demand benchmarks, C-017 measured |
| R-112 | 112.1 NO-GO: selection replaces 110.28 of 256 blocks per layer per step; half the KV resident still costs +10.4 % to +22.4 % per token over PCIe ([record](../plans/2026-10-18-kv-cold-pages-decision.md)) |

## Blocked: questions for the owner

| ID | Question |
|---|---|
| R-054 | Register a self-hosted GPU runner for CI (which machine, which labels)? |
| R-055 | Which binary artifact (static `imp-server` tarball, .deb) and where published? |
| R-080 | Move model-specific block state out of `executor.h` against SETTLED B, yes or no? |
| B-34 | Is FP8 paged KV at head_dim 256 used, so the multitok port is worth it? |
| B-29f | Keep five stacked VRAM reserves or collapse them into one measured margin? |
| B-33, B-29b, B-29c, B-29e | Download a model for each: MoE NVFP4 with K/16 > 512, Nemotron MTP checkpoint, gemma-3 Q8_0 / Q6_K, a dense MXFP4 GGUF? |

## Follow-ups (OPEN, plans in DISPATCH)

| ID | Next step |
|---|---|
| R-119.4-119.6 | width 8 recycling measured worse than 4 (6.09 vs 6.13 prose): lifting caps waits for a better tree drafter (119.5) |
| R-065 | `mmq_imma_q8k128_kernel` at 30.9 / 31.6 % of the INT8 tensor peak: raise it |
| R-049 | forced-width verify harness for widths 8..16, then a block drafter |
| R-101 | rotation (MR-GPTQ) or sequential calibration after plain GPTQ (#2775) |
| R-120 | Gemma-4 audio encoder (needs an audio-capable checkpoint) |
| R-042, R-046, R-047, R-048, R-050, R-051, R-052, R-056 | L rows untouched this run; R-051 / R-052 need model downloads |

## Final sweep

On main 09797f27:

| Gate | Result |
|---|---|
| `make build` | exit 0, 0 warning lines in the log (ccache-backed, not a cold rebuild) |
| `make test-gpu` | 3756 tests ran, 0 failed |
| `make dev-test` | 13/13 |
| `make preflight` | 0 failures |
| `make verify-fast` (#2780 tree) | decode -0.57 %, prefill pp512 +16.17 %, PPL 0.000 %, server c8 +3.50 % |
| inventory re-run | 0 TODO/FIXME/HACK comment markers, 0 open issues, 0 open PRs, CI and CodeQL green on the last 3 main commits |
