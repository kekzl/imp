# Phase 1 Warnings Triage

Build: `imp:bringup`, RelWithDebInfo, IMP_BUILD_TESTS=ON, IMP_BUILD_BENCH=ON.
Total warnings: **15** (well under the 500 threshold for a dedicated subagent).
Total errors: **0**.

## Categories

### 1. `_COUNT` enum sentinel not handled in switch — 1 warning
- `src/model/tensor_kind_name.cpp:6:12` — `enumeration value '_COUNT' not handled in switch [-Wswitch]`
- **Action:** suppress at source. `_COUNT` is the size sentinel and is intentionally not a real value.
- **Triage:** non-blocking, cosmetic. **Defer** — fix only if a wider warnings sweep happens.

### 2. `?:` with omitted middle operand (GNU extension) — 6 warnings, all in `tests/test_llm_compressor_loader.cpp` (lines 315, 317, 330, 349)
- `[-Wpedantic]` only — GCC accepts the form; clang would too.
- **Action:** non-blocking. Test-only file.
- **Triage:** **Defer**.

### 3. `system()` return value ignored — 8 warnings, all in `tests/test_llm_compressor_loader.cpp`
- `[-Wunused-result]`. The test calls `system("python3 ...")` to scaffold dummy SafeTensors weights; ignoring the return is fine for the test pattern.
- **Action:** could wrap in `(void)system(...)` but that's churn for a test.
- **Triage:** **Defer**.

## Decision

Build is **green**. None of the warning classes mask correctness or performance issues. **Phase 1 success criterion met (0 errors, warnings triaged).** No fixes applied — none of the categories are dead-path candidates (FA4 stubs / half-rate matmul fallbacks); they are isolated test-scaffolding lints.
