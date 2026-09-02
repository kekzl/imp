---
name: find-stubs
description: Use when asking whether something in imp is actually finished - hunting stubs, placeholders, request fields that are parsed and then ignored, code paths or kernels that never run, gate-based features that silently no-op, tests that assert nothing. Triggers on "is this implemented", "unfinished", "stub", "placeholder", "dead path", "does this flag do anything", "accepted but ignored", "kernel never launched", "test asserts nothing", "neutral A/B on a gated feature". Do NOT use for structural debt / god files / duplication (codebase-audit) or for who-calls-what (code-graph).
---

# Finding unfinished code - imp

Cheap and loud first, expensive and precise last: run 1 -> 5 -> 6; 3 and 4 are one-off sweeps. Every rung carries its baseline on imp (2026-09-02, `c3d9689e`; earlier 2026-08-19, `ad820790`); without one you read normal as a finding. Scope `src include tools`, `--glob '!build*'`.

CI gates already covering rungs (`scripts/ci_static_gates.sh`, blocking in `Build`, hooks): `tools/check_dead_inline_accessors.py` (rung 3/4 for header inlines), `tools/check_test_lanes.py --report` (tests in NO CI lane, pinned; stronger than rung 6), `tools/check_launch_guards.py`, `tools/check_log_fatal.py` (FATAL logs that do not stop). Run them before hand-sweeping their territory.

## 1. Textual markers (5 minutes)

```bash
rg -n -i --glob '!build*' -e '\b(STUB|NOT[_ ]?IMPLEMENTED|UNIMPLEMENTED)\b' src include tools   # 7 (12 at ad820790), all productive
rg -n -i --glob '!build*' -e '\b(TODO|FIXME|XXX|HACK|WIP|PLACEHOLDER)\b' src include tools        # 27, mostly image placeholders (domain term)
rg -n -i --glob '!build*' -e 'for now|temporar|simplified|fallback for|approximation|assume[sd]? that' src include tools   # 67
rg -n --glob '!build*' -e 'assert\(false\)|assert\(0\)|std::abort|__trap\(\)|#if\s+0\b' src include tools   # 13
```

The sharp pattern found `gemm_grouped_dispatch` with four `tier not implemented` branches and zero callers (removed #1479), `ssm_graph_ban` (only caller was its own test, #1513), the unwired grouped mxf4nvf4 small-M kernel (#1753). `should be` (~92), `revisit`, `naive impl` are excluded on purpose.

## 2. Compiler as detector

`-Wswitch-enum` names the enum values a dispatch does not handle (`QType`, arch, sampler). Scratch configure only; device code via `-Xcompiler`.

## 3. AST matchers (clang-query, one matcher per run, mount at `/src` because `build-dev/compile_commands.json` hardcodes it)

```bash
printf 'set output diag\nmatch %s\nquit\n' "$MATCHER" > /tmp/q.txt
docker run --rm -v $(pwd):/src -v /tmp/q.txt:/q.txt -w /src imp:toolchain bash -c \
  'apt-get update -qq && apt-get install -y -qq clang-tools >/dev/null 2>&1;
   clang-query -p build-dev -f /q.txt $(find src tools -name "*.cpp" | sort)' \
  | grep -E "^[0-9]+ matches|/src/(src|tools)/.*binds here"
```

| Matcher | On imp |
|---|---|
| `functionDecl(isDefinition(), unless(isImplicit()), isExpansionInMainFile(), hasBody(compoundStmt(statementCountIs(0))))` | 24 raw, mostly out-of-line trivial destructors |
| same with `statementCountIs(1)` + literal `returnStmt` | 1 (a deliberate no-op callback); a second hit is worth reading |
| `cxxCatchStmt(isExpansionInMainFile(), has(compoundStmt(statementCountIs(0))))` | 9 raw before dedupe |

`isExpansionInMainFile()` is mandatory (6086 matches from nlohmann/CUTLASS headers without it). Dedupe by `file:line` (a TU in two targets reports twice). `.cu` is out of scope (rung 4). `clang-tidy` adds `bugprone-empty-catch`; `.clang-tidy` disables `bugprone-branch-clone` deliberately; `make tidy` runs in `imp:builder`.

## 4. Reachability, not syntax (one-off)

- Host: `-ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--print-gc-sections`; coverage `-fprofile-instr-generate -fcoverage-mapping` + `llvm-cov show --region-coverage-lt=1` after a REAL server run (`make coverage`, `scripts/coverage_server.sh`).
- Device census by diff (`cuobjdump --dump-resource-usage` reports 0 functions here):

```bash
nm build-dev/imp-server | grep -oP '__device_stub__\S+' | sed 's/.*__device_stub__/_/' \
  | c++filt | sed 's/(.*//; s/<.*//' | grep -v '^_' | sort -u > kernels_present.txt
nsys stats --report cuda_gpu_kern_sum --format csv run.nsys-rep | cut -d, -f8 | sort -u > kernels_launched.txt
comm -23 kernels_present.txt kernels_launched.txt
```

The difference = dead kernels AND live kernels behind a condition that is never true (`ssm_graph_ban` cost 3x decode). Under CUDA graphs a host log line never proves a kernel ran: `nsys --cuda-graph-trace=node` presence plus a time differential does. `rg -c '__global__' src/` counts declarations too (482 mentions); `ccg kernels` lists definitions (code-graph, enrich currently broken).

## 4b. Gate-based features that silently no-op

A feature behind a dtype, capacity or shape gate reads NEUTRAL when the gate closes. Proof of activity = launch counts or a log line present in ONE arm only. Cases: sparse attention on NVFP4 (dtype gate opened, consumer branches still read the dense table, #1818); spec verify chunks on the sparse table (scratch rows = 8 vs a 33-row chunk, #1807); stream-K workspace sized 0 B refused every launch (1960 -> 1400 CUTLASS launches, 22k -> 4.2k tok/s); `kv_cache.growable` no-op on planned loads (ceiling == commit, #1794). Rule: when a feature was gated per dtype, the consumer path for that dtype was probably never wired.

## 5. Accepted but ignored (the expensive class)

```bash
FIELDS=$(sed -n '/^struct ChatRequestParams/,/^};/p' tools/imp-server/handlers_internal.h \
  | grep -vE '^\s*//' \
  | rg -o '^\s*(?:float|int|bool|size_t|std::string|std::vector<[^;]*>)\s+([^;]+);' -r '$1' \
  | tr ',' '\n' | rg -o '^\s*([a-z_][a-z_0-9]*)' -r '$1' | sort -u)
for f in $FIELDS; do
  n=$(rg -c --glob '!build*' -e "[.>]$f\b" tools/imp-server src | grep -v handlers_chat_params.cpp \
      | awk -F: '{s+=$2} END {print s+0}')
  [ "$n" -le 2 ] && echo "SUSPECT $f ($n uses outside the parser)"
done
```

`tr ',' '\n'` matters (comma-separated declarators hide `include_usage` and the `*_explicit` flags). Baseline: 55 fields, 10 candidates, all benign (one write, one read each): `cache_prompt`, `enable_thinking_requested`, `image_error`, `include_usage`, `json_schema_str`, `max_stop_len`, `rep_pen_explicit`, `requested_model`, `top_k_explicit`, `top_p_explicit`. The finding is a field with 0, or a new name. Read both sites; the count never decides. Same class in config: a field parsed by the binder and never read (codebase-audit recipe "config field never read").

## 6. Tests that check nothing

```bash
rg -n 'DISABLED_' tests                                                            # 9, read all
awk -f .claude/skills/find-stubs/tests_without_assertions.awk $(find tests -name '*.cpp' -o -name '*.cu')   # 53
python3 tools/check_test_lanes.py --report                                         # 1054 in no CI lane (pinned), 1624 in ctest -L unit
rg -c 'GTEST_SKIP' tests | awk -F: '{s+=$2} END {print s}'                          # 208, mostly legitimate (no GPU, no model)
```

A one-line regex over test bodies stops at the first line-initial `}` and flagged 184 of ~2600; the awk script splits on `^TEST`/`^}` and drops harness-delegating bodies (read its header). A filter like `DetEvalE2ETest.*` that matches 0 tests prints PASSED. A test whose inputs cannot reach the defect passes its own mutant: uniform-random rows never produced the FP8 `out / row_scale` overflow (fixed by `SsmPrefillFp8TinyRowsStayFinite`); `PagedOracle` at HD128 never ran the HD256 word-load path (#1817). Mutation baseline 90.4% (`docs/audit/MUTATION_BASELINE.md`); the dominant escape is "input does not reach the failure".

## Pitfalls

- A finding is a candidate (codebase-audit).
- Refresh the counts above when the tree moves; if a rung becomes a gate, give it a baseline file and fail on growth, like `tools/check_filesize.py`.
- Absence of a marker means nothing: the worst recorded memory defects carried a CORRECT log line naming a number the code did not use (#1746/#1705, defect classes in `docs/internals/MEMORY.md`).
- A control arm that distinguishes nothing is not a control (`imp-cli --version` prints usage in every arm).
