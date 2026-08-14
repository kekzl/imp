<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: 81ffa573
-->

# Purge log

Append-only. Every claim removed or corrected during the docs rewrite, with the
reason. A rewrite that deletes without a log is indistinguishable from a rewrite
that lost things.

## 2026-08-13, docs layer rewrite

| file | removed / corrected | reason |
|---|---|---|
| `README.md` | "Qwen3-8B Q8_0 at **~270 tok/s** (CI-gated baseline)" | **two errors in one clause.** The pinned baseline is 287.19, and it is not CI-gated: the `test` job sleeps behind `vars.HAS_GPU_RUNNER`. Replaced by the generated block plus an explicit statement that there is no GPU in CI |
| `README.md` | the ~1000-word Server cell in the feature table | an L0 document listing `imp_memory_reserved_bytes` and green contexts. Moved to `FEATURES.md` and `API.md`; the README carries a six-row extract |
| `README.md` | "**On a single RTX 5090, imp is the fastest engine for single-user and agentic inference**" | an unqualified superlative. Replaced by the per-claim figures in `PERF.md`/`BENCHMARKS.md`, each with its referent |
| `README.md` | inline hero-number bullets | duplicated `BENCHMARKS.md`. SSoT violation: the README now embeds a generated block only |
| `README.md`, `docs/GOAL.md`, `docs/BENCHMARKS.md`, `docs/LIMITATIONS.md` | restatements of "no tcgen05 / TMEM / wgmma / Hopper / sm_100" | the delimitation existed in 8 places, so a reader could not tell which was maintained. Now stated once in `docs/internals/ARCHITECTURE.md#target-architecture`; the rest link. L2 files may still *derive* from it ("no tcgen05, therefore the MMA blocks the issuing warp"), which is rationale rather than restatement |
| dispatch §6.3 | "`/v1/messages` streaming is synthetic (TTFT == total latency)" | **not written into the docs.** Refuted: one shared per-token driver since v0.18.1; `handlers_messages.cpp:146` emits real `content_block_delta` |
| dispatch §6.3 | "no per-request spec-decode toggle" | **not written in.** `handlers_chat_params.cpp:246`, shipped v0.12.3 |
| dispatch §6.3 | "no `cache_control`" | **not written in.** Four call sites, per-breakpoint pin boundary shipped v0.19.2 (#1046) |
| dispatch §2.7 | decode 290-295 tok/s on Qwen3-Coder-30B-A3B, "~1.4x behind vLLM", README headline ~200 tok/s | **not written in.** Dated "May 2026"; the in-tree 2026-07-12 sweep reads that model at 389 tok/s and `roadmap.md` records imp ahead of vLLM on MoE single-sequence prefill. Copying these would have published a regression that did not happen |
| dispatch §4 | "Phi-4" as a model architecture | corrected rather than copied: `model.cpp:316` maps `"phi3"` onto `ModelArch::LLAMA`. Documented as an alias |
| dispatch §2.6 | "`1258 tok/s`, `20x behind vLLM` must be removed" | **already done in June 2026.** Zero live hits; only the archived purge record mentions them. The linter keeps them out |

## Not removed, and why

| kept | reason |
|---|---|
| German prompts in `tools/analysis/degen_corpus.jsonl` | they are multilingual degeneration *fixtures*. Translating them deletes the test. Not a §2.4 violation |
| `docs/roadmap.md`'s 32 unprovenanced figures | it is a dated research record whose numbers *are* the narrative of what was measured or refuted. Excluded from the linter; the reader-facing distillation is `LIMITATIONS.md` / `DESIGN_DECISIONS.md` / `PERF.md`, which are linted |
| every file in `docs/archive/` and `docs/audit/` | records, not documentation |

## 2026-08-14 — the 2.6x prefill variance

| removed | why |
|---|---|
| "prefill varies up to 2.6x across container restarts (cuBLAS autotuning)" in `performance.md` (x3), `internals/BENCHMARKING.md`, `internals/MEMORY.md` (x2) | retracted 2026-08-03 in `docs/audit/AUDIT_ARCH_2026_07_29.md` answer 5: nine process starts of one binary spread **3.50 %**, not 2.6x. The figure was a citation carried forward. Replaced by the measured per-model split in `PERF.md`: **0.6-1.2 %** on the cuBLAS-FP16 model, **37.6 %** on a resident NVFP4 MoE model — so the attribution to cuBLAS was wrong as well as the magnitude. `BENCHMARKS.md` is a record and was left as written, with a dated correction appended below its method note. |

## Found, not fixed here

Code changes, listed so they are not lost:

- `src/core/qtype.h:15` carries a German comment ("Wire-stable 0..31 (kompatibel
  mit GGUF block-quant Wire-Format)"), violating §2.4.
- `src/runtime/engine_init_resolver.cpp:565` says "prefill is never
  graph-captured". `runtime.prefill_graph` defaults to `true` (`config.h:103`),
  so this was filed as a contradiction. **It is not: measured 2026-08-14, no real
  model ever captures a prefill chunk.** Three gates close it independently (the
  512 MiB NVFP4 dequant cap vs the LM head, `kv_append_capturable` requiring F16
  KV, and the last chunk always running eager), and `can_capture` logged nothing
  when it was false. The comment is accidentally right about the outcome and
  wrong about the reason. Diagnostic added in #1417; the default is still `true`.
