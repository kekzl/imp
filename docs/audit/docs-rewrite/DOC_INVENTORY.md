---
layer: L3
audience: agents
verified: 2026-08-13
commit: 1e4fad60
---

# Doc inventory (phase 1, scout)

Read-only pass over every `.md` in the tree. 64 files, 19 015 lines.
Classification is per the dispatch: `KEEP` / `REWRITE` / `MERGE` / `SPLIT` / `DELETE`.

## Scope decision, stated because it shapes everything below

Three groups are **records, not documentation**, and a record is not rewritten:

| group | files | why out of scope |
|---|---|---|
| `docs/archive/**` | 15 | dated point-in-time reports; `docs/archive/README.md` indexes them as history |
| `docs/audit/**` | 12 | append-only ledgers (`PERF_LOG.md`, `SETTLED.md`, `TEST_HARDENING_LOG.md`). `SETTLED.md` is CI-gated by `check-release.sh` |
| `CHANGELOG.md`, `docs/MISSION_JOURNAL.md`, `docs/vram_audit.md`, `AUDIT.md` | 4 | append-only by their own stated convention |

They stay as they are, and the linter must exclude them, otherwise it will
demand a `verified:` refresh on a document whose whole point is that it was true
on one dated afternoon. **31 files are in scope.**

## The dispatch's own premises, checked first

The dispatch dates its facts "Mai 2026" and the tree is at 2026-08-13. Applying
its own rule (§10: code beats docs, measurement beats code) to itself:

| dispatch claim | verdict | evidence |
|---|---|---|
| "161k LOC" | **wrong**, it is 137 776 (src + include) | `wc -l` over `src/ include/` |
| §6.3 "`/v1/messages` streaming is synthetic (TTFT == total latency)" | **stale** | real per-token driver since v0.18.1; `handlers_messages.cpp:146` emits `content_block_delta`, `handlers_chat_stream.cpp:33` is the shared token loop |
| §6.3 "no per-request spec-decode toggle" | **stale** | `handlers_chat_params.cpp:246` reads `body["speculative"]`; shipped v0.12.3 |
| §6.3 "no `cache_control`" | **stale** | 4 call sites incl. `handlers_chat_core.cpp`; per-breakpoint pin boundary shipped v0.19.2 (#1046) |
| §2.6 "`1258 tok/s`, `20x behind vLLM` must be removed" | **already done** | zero hits outside `docs/archive/housekeeping_2026_06_13.md`, which records the June purge |
| §2.6 "TMA is used" | **correct, and the repo contradicts itself** | `gemm_grouped_nvfp4_smallM.cu:65` emits `cp.async.bulk.tensor.2d`, "Emits UTMALDG on SM120". But `CLAUDE.md:87` says "no TMA-WS grouped GEMM" while `docs/sm120.md:31` says `compute_120f` "enables ... TMA warp-specialized grouped GEMM tactics" → `OPEN_QUESTIONS.md` |
| §2.7 perf figures (decode 290-295, "~1.4x behind vLLM", README headline ~200 tok/s) | **stale** | see below |
| §2.6 "Paged KV block=16" | **right as the default, incomplete** | `engine.cpp` `kv_n > 0 ? kv_n : 16`; an NVFP4 run on 2026-08-13 logged `block_size=32` |
| §2.6 forbidden tokens as *features* | **correct and already honoured** | all 40 live hits are negative statements ("no tcgen05/TMEM/wgmma"). The defect is not the claim, it is that it is repeated in 8 places |
| MIT licence | **correct** | `LICENSE` |

The perf row matters most. §2.7 says decode 290-295 tok/s on Qwen3-Coder-30B-A3B
and "~1.4x behind vLLM". The pinned gate is Qwen3-8B-Q8_0 at 287.19 tok/s, the
2026-07-12 hero sweep reads Coder-30B-A3B-NVFP4 at 389, and `docs/roadmap.md`
records imp *ahead* of vLLM on MoE prefill and ahead of llama.cpp by 13-48 % on
batch=1 decode. **Writing §2.7's numbers into the docs would inject exactly the
staleness this dispatch exists to remove**, so they go through phase 2 like every
other claim rather than being copied in.

## Reader-gap analysis

Three questions per audience, against the docs as they stand.

### L0 newcomers (`README.md`)

| question | answered today? |
|---|---|
| Is this for my GPU? | **yes**, `README.md:83` is explicit and honest |
| How do I run it in one command? | **partial** — a quickstart exists but competes with build-from-source for position |
| Where does it lose to llama.cpp/vLLM? | **no** — nothing states the losses in the README; they are scattered in `roadmap.md` "Known limitations" |

### L1 operators (`docs/*.md`)

| question | answered today? |
|---|---|
| Which API fields actually work? | **no** — there is no `API.md`. `usage.md` covers running, not field coverage |
| How do I deploy it behind a proxy with auth? | **no** — no `DEPLOYMENT.md` |
| It returned garbage, what now? | **no** — no `TROUBLESHOOTING.md`. The knowledge exists, in `roadmap.md` prose and `docs/audit/` |

### L2 kernel devs (`docs/internals/*`)

| question | answered today? |
|---|---|
| Where is the decode-attention variant chosen? | **yes**, `attention-dispatch.md` names `executor_attention.cu` |
| What has already been tried and failed? | **yes and it is a strength** — `roadmap.md` "Investigated and shelved" plus `SETTLED.md`. Both are in the wrong place for a kernel dev |
| How do I measure honestly? | **yes**, `BENCHMARKING.md` is the contract |

### L3 agents (`CLAUDE.md`)

| question | answered today? |
|---|---|
| Which target is compiled? | **yes**, `CLAUDE.md` header |
| How do I build and test just this directory? | **no** — only repo-wide targets; there is no per-directory `CLAUDE.md` |
| Where is X chosen? | **partial** — the `code-graph` skill covers it, `CLAUDE.md` routes to it |

## Single-source-of-truth violations

| information | files carrying it | target |
|---|---|---|
| sm_120a vs datacenter-Blackwell delimitation | 8 live (`CLAUDE.md`, `AGENTS.md`, `README.md` x2, `sm120.md`, `sm120_optimal_kernel.md`, `GOAL.md`, `BENCHMARKS.md`, `roadmap.md`) | `docs/internals/ARCHITECTURE.md`, once |
| perf numbers | 11 live files, 32 of them in `roadmap.md` alone | `docs/PERF.md` |
| known gaps | `roadmap.md` "Known limitations", `determinism.md`, `supported-models.md` | `docs/LIMITATIONS.md` |
| non-decisions (multi-GPU, TP, CPU) | `GOAL.md:43`, `roadmap.md:109/377`, `CLAUDE.md` | `docs/DESIGN_DECISIONS.md` |

## Per-file classification (in-scope only)

| file | lines | last | layer | action | why |
|---|---|---|---|---|---|
| `README.md` | 210 | 08-11 | L0 | REWRITE | no "where imp loses", quickstart not first, perf inline instead of generated |
| `CLAUDE.md` | 100 | 08-13 | L3 | KEEP+ | good router; add per-directory children |
| `AGENTS.md` | 103 | 08-13 | L3 | KEEP | subagent roles and guardrails |
| `CONTRIBUTING.md` | 136 | 08-13 | L1 | KEEP | |
| `docs/README.md` | 85 | 08-11 | L1 | REWRITE | index must reflect the new tree |
| `docs/architecture.md` | 229 | 08-01 | L2 | MOVE | → `internals/ARCHITECTURE.md`, absorb the delimitation |
| `docs/sm120.md` | 94 | 07-24 | L2 | MERGE | → `internals/ARCHITECTURE.md` target-architecture section |
| `docs/sm120_optimal_kernel.md` | 225 | 06-17 | L2 | MOVE | → `internals/KERNELS.md` |
| `docs/attention-dispatch.md` | 105 | 08-11 | L2 | MOVE | → `internals/KERNELS.md` dispatch section |
| `docs/MEMORY_ARCHITECTURE.md` | 1444 | 08-03 | L2 | MOVE | → `internals/MEMORY.md`; canonical, do not rewrite content |
| `docs/BENCHMARKING.md` | 106 | 08-13 | L2 | MOVE | → `internals/BENCHMARKING.md` |
| `docs/BENCHMARKS.md` | 439 | 08-13 | L1 | SPLIT | numbers → `PERF.md`; competitive prose stays |
| `docs/performance.md` | 84 | 08-13 | L1 | MERGE | → `PERF.md` (methodology already lives here) |
| `docs/quantization.md` | 603 | 08-10 | L1/L2 | SPLIT | operator-facing "which quant" → `MODELS.md`; AWQ findings → internals |
| `docs/quant-pipeline.md` | 66 | 07-30 | L2 | MOVE | → `internals/` |
| `docs/supported-models.md` | 153 | 08-12 | L1 | REWRITE | becomes `MODELS.md` with the status legend |
| `docs/usage.md` | 446 | 08-10 | L1 | SPLIT | → `QUICKSTART.md` + `DEPLOYMENT.md` + `API.md` |
| `docs/determinism.md` | 221 | 08-10 | L1 | KEEP | contract doc, already precise |
| `docs/GOAL.md` | 189 | 08-11 | L1 | SPLIT | non-goals → `DESIGN_DECISIONS.md` |
| `docs/roadmap.md` | 401 | 08-13 | L1 | SPLIT | gaps → `LIMITATIONS.md`, numbers → `PERF.md`, shelved → `DESIGN_DECISIONS.md` |
| `docs/nsys_profiling.md` | 195 | 07-30 | L2 | MOVE | → `internals/BENCHMARKING.md` companion |
| `docs/vision_gemma4v_spec.md` | 59 | 05-31 | L2 | KEEP | porting reference, still accurate |
| `docs/plans/*` (3) | 514 | | L2 | KEEP | dated design records |
| `tests/README.md` | 158 | 07-30 | L3 | KEEP | becomes `tests/CLAUDE.md` companion |
| `tests/refs/README.md` | 97 | 07-10 | L2 | KEEP | |
| `tools/roofline/README.md` | 90 | 08-06 | L2 | KEEP | |
| `tools/eval/niah/README.md` | 27 | 07-08 | L2 | KEEP | |

No file is classified `DELETE`. That is a finding rather than an omission: the
tree has no abandoned documentation, it has **duplication and missing audiences**.
The purge in phase 3 is therefore about removing repeated claims from files that
survive, not about removing files.

## What phase 2 must verify

Everything in `FEATURES.md`'s future scope: the model list, the quant list, the
API surface and the kernel list from dispatch §4, plus the six dispatch premises
marked stale above, so the correction is on the record rather than in a reply.
