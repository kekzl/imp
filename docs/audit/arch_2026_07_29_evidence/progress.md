# Audit progress

Report assembled at the end from `audit_scratch/part_*.md` in section order.

## Done
- Phase 0 orient: LOC (`largest_files.txt`), thresholds, AGENTS.md, ARCHMAP.md, attention-dispatch.md
- Phase 1 censuses (all in `audit_scratch/`):
  - `enums.txt`, `qtype.txt`, `arch_dispatch_sites.txt`, `quant_dispatch_sites.txt`, `qtype_switches.txt`
  - `legacy_hunt.txt`, `clone_pairs.txt` (own token clone detector `clones.py`)
  - `alloc_census.txt`, `check_alloc_sites.txt`, `sync_census.txt`, `launch_check_census.txt`
  - `layering.txt`, `layering_tally.txt`, `raii.txt`, `process_diag_install.txt`
  - `family_sizes.txt`, `server.txt`, `tests.txt`, `fallback_hunt.txt`

## Key facts already nailed (file:line)
- 16 archs in `src/model/model_arch.h:7`; table-driven registry `src/model/model.cpp:171-347`.
- `select_attn_prefill_path` (`src/compute/attention_dispatch_decision.h`) is a MIRROR — only
  `tests/test_routing_decision.cpp:7` includes it; `attention_dispatch.cu:33` only *mentions* it in a
  comment. Same for `select_moe_prefill_path` (`src/exec/moe_prefill_decision.h`).
- `ProcessDiag` = process-global static (`src/runtime/process_diag.cpp:57`), 28 kernel-affecting flags
  mirrored in `process_diag_install()` (:64-105), installed ONLY by tool mains
  (`tools/imp-cli/main.cpp:134`, `tools/imp-server/main.cpp:64`). `engine.cpp:783-790` promotes
  exactly ONE (deterministic_gemm) for C-API embeddings.
- `attention.fa2_hd256` is dual-sourced: `runtime_config()` in exec (`executor_attention_prefill.cu:51`)
  vs `process_diag_fa2_hd256()` in the kernel (`attention_fmha_sm120.cu:1900`).
- I1 "only src/memory talks to the driver" is a RATCHET: `tools/alloc_allowlist.txt` = 74 files/492 sites.
- Post-launch checks: 440 launch lines / 436 checks. Only gap:
  `src/vision/qwen3vl_encoder_kernels.cu` 9 launches / 0 checks.
- 6 module-static cuBLAS handles + 11-entry hand-maintained `cuda_static_reset.cpp` registry.
- CI runs `ctest -L unit` only; labels `gpu` and `perf` never run in CI (`CMakeLists.txt:857-882`).
- Backward edges: exec→runtime 27 files (22× config.h), compute→runtime 21, compute→model 9.

## Remaining
Tracks C..H detail, then assemble.

## Final state (audit complete)
- Report assembled: AUDIT_ARCH_2026_07_29.md (19 sections, ~2100 lines).
- Late correction applied: the GPU CI job (ci.yml:379-435) is fully written and dormant behind vars.HAS_GPU_RUNNER, not missing. F-5 / R-20 / scorecard-G / §4.4 / §12.1 / §12.7 updated accordingly.
- No GPU job was run at any point (card 100% busy, 29207/32607 MiB, mmm-comfy).
- No file under src/ tools/ tests/ docs/ was modified.

## Findings shipped (2026-08-02)
- PR #1205 (c35fad75): F-1 CRITICAL, F-2 HIGH; F-3 partially (see the note in the report).
- PR #1206 (b79a5f67): F-4, F-7, F-19, F-21, F-22 + new blocking CI job 'Launch guards'.
- Report updated in place: F-4's count was WRONG (3 files -> 1; 440/436 -> 407/407 in-scope);
  F-22 was scored LOW and turned up a real production bug (discarded ChatTemplate::init).
- Neither PR ran verify-fast: GPU busy at both push times (mmm-comfy, then mmm-imp-vision).
