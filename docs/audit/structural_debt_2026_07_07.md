# Structural audit — 2026-07-07

Fifth structural pass (priors: `structural_consistency_2026_06_06.md`,
`structural_debt_2026_06_08/09`, `housekeeping_2026_06_13.md`,
`vram_audit_2026_07_07.md`). Since the last full pass: 152 commits (PRs
~#650–#887), so this sweep focused on the surfaces added since mid-June —
speculative decoding (suffix drafter, MTP verify, spec graph-capture), the
server endpoints (`/v1/messages`, `/v1/responses`, `/v1/embeddings`), and the
config/env retirement (#879) fallout. Method per the codebase-audit skill:
fan-out sweep (duplication, dead code, stale comments, server consistency,
test health), then every candidate re-verified against the code before
filing. File-size gate: 0 violations, 31 warnings (all below hard-review).

## Confirmed findings → issues

| # | Finding | Issue |
|---|---------|-------|
| 1 | Rate limit + max-concurrent admission gates only `/v1/chat/completions`, `/v1/completions`, `/v1/responses` — `/v1/messages` and `/v1/embeddings` bypass both (`main.cpp:131,146`; the Anthropic shim calls `handle_chat_completions()` directly, no pre-routing re-entry) | [#888](https://github.com/kekzl/imp/issues/888) |
| 2 | `handle_embeddings` holds `state.mtx` across the whole multi-input computation while `/health`, `/metrics`, `/v1/models` block on it with unbounded `lock_guard` — liveness-probe hazard, undercuts the #874 `/health` design | [#889](https://github.com/kekzl/imp/issues/889) |
| 3 | `/v1/embeddings` parameter gaps: no `ensure_model_loaded` (bogus model → 200), `--max-input-tokens` not enforced, `encoding_format`/`dimensions` silently ignored | [#890](https://github.com/kekzl/imp/issues/890) |
| 4 | Contract divergences: `/v1/completions` validates `n` then ignores it; `logprobs` typed bool (Completions spec: int → spec-valid request 400s); `/v1/messages` unhandled exceptions escape as OpenAI-shaped envelopes | [#891](https://github.com/kekzl/imp/issues/891) |
| 5 | Three near-identical SSE token loops (chat_stream / messages / responses, ~200–250 LOC shared body each) with confirmed drift: `parallel_tool_calls=false` suppression exists only in the OpenAI copy; structural-stop block copied a 4th time into the non-stream path | [#892](https://github.com/kekzl/imp/issues/892) |
| 6 | Post-#879 leftovers: ~14 comment/log sites in `src/exec/` still reference retired `IMP_*` env vars, incl. a self-contradicting user-facing ERROR log (`pre_dequant_phase3_cutlass.cu:490` recommends `IMP_MXFP4_FP16_FALLBACK=force` while hardcoding `allow_force=false`); niah README documents retired configs; stale TODO in `gemm_grouped.cu:147` | [#893](https://github.com/kekzl/imp/issues/893) |
| 7 | Dead-code nits: `MtpDraftWorkspace::d_router_logits` (alloc+free, never read/written), `launch_cluster_1d`, `pdl::disable_kernel`, `LayerOffload::is_offloaded`, `MemAccount::device_peak_used` — all definition-only across `src/ tests/ tools/` | [#894](https://github.com/kekzl/imp/issues/894) |
| 8 | `test_quant_integration.cu:1707` self-skips on Q5_K all-NaN caused by stale cuBLAS state from a prior test — a masked product-code bug, and future real Q5_K regressions skip silently | [#895](https://github.com/kekzl/imp/issues/895) |
| 9 | Zero test coverage: MTP econ guard, `engine_spec_capture` path, `/health` 503 surface, nomic-bert encoder (always-skips in CI), `--vram-budget` flag parse, `/v1/responses` e2e | [#896](https://github.com/kekzl/imp/issues/896) |
| 10 | `mtp_mrope_kernel` computes plain RoPE inline — no YaRN/rope-scaling; on a rope-scaled MTP model the drafter silently loses acceptance (same drift class as the #880 MLA mscale bug) | [#897](https://github.com/kekzl/imp/issues/897) |

Live env-var ground truth (for #893): the only `getenv("IMP_*")` read sites are
`IMP_DETERMINISTIC`, `IMP_FMHA_FA2`, `IMP_CONFIG`, and the diagnostics
`IMP_SPEC_TRACE`, `IMP_JUMP_TRACE`, `IMP_PPL_DUMP` (plus test-only helpers).

## Checked and NOT flagged — do not re-chase

- **`flush_text` clamp "missing" in messages/responses streams** — benign:
  `substr(0, up_to)`/`erase(0, up_to)` clamp the count themselves. Cited in
  #892 only as drift evidence, not as a bug.
- **MoE prefill quant-variant family** (`executor_forward_moe_batch.cu`,
  5 × gate→up→activation→down skeleton) — consistent, no drift found.
  Abstracting it would be speculative; leave until a real divergence bug.
- **n-gram vs suffix drafter backward-match loop** (~10 LOC shared primitive)
  — too small to be worth coupling the two drafters.
- **MTP head as a hand-copied decoder-layer forward** — intrinsic to the
  design (bespoke single-token workspace path, graph-friendly). The one
  concrete drift found is the rope-scaling gap (#897); do not attempt to
  "unify with the executor layer path" wholesale.
- **`qkt_mxf4nvf4_validate`** — test-only harness, part of the FP4-attention
  research program (deliberately kept, settled prior).
- **Request parsing across OpenAI/Anthropic/responses** — genuinely shared
  (transform-to-common-body + `parse_chat_request_params`), not duplicated.
- **Hardware-capability comment claims** in code added since mid-June — clean;
  no wgmma/tcgen05/FP8-prefill false claims (the FP8 *FMHA* mentions are
  accurate: that path exists on sm_120).
- **Main docs** (`README.md`, `docs/architecture.md`, `docs/sm120.md`,
  `BENCHMARKING.md`) — no references to retired flags or env vars. The only
  doc drift is the niah README (#893).
- **`/v1/messages` missing-`max_tokens` defaulting** (real Anthropic API
  400s) — documented as intentional at `handlers_chat.cpp:722`.
- **3 `DISABLED_` tests + green_ctx smoke tests** — inventoried in #896;
  justified or low-value, no separate action.
- **CMake orphan check** — clean; `executor_attention_{prefill,qkv,decode}.cu`
  are unity-included from `executor_attention.cu`, not orphans.
