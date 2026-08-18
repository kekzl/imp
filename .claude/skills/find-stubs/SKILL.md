---
name: find-stubs
description: Use when asking whether something in imp is actually finished — hunting stubs, placeholders, request fields that are parsed and then ignored, code paths or kernels that never run, tests that assert nothing. Triggers on "is this implemented", "unfinished", "stub", "placeholder", "dead path", "does this flag do anything", "accepted but ignored", "kernel never launched", "test asserts nothing". Do NOT use for structural debt / god files / duplication (codebase-audit) or for who-calls-what (code-graph).
---

# Finding unfinished code — imp

Cheap and loud first, expensive and precise last. Run **1 → 5 → 6**; 3 and 4 are
one-off sweeps worth a baseline file afterwards.

**Every rung carries its measured yield on imp** (2026-08-19, `ad820790`). That
is the whole point: without a baseline you read normal as a finding, and these
sweeps over-flag by construction. Scope is `src include tools`, `--glob '!build*'`.

## 1. Textual markers — 5 minutes

```bash
rg -n -i --glob '!build*' -e '\b(TODO|FIXME|XXX|HACK|WIP|STUB|PLACEHOLDER|NOT[_ ]?IMPLEMENTED|UNIMPLEMENTED)\b' src include tools   # 39
rg -n -i --glob '!build*' -e 'for now|temporar|simplified|fallback for|approximation|assume[sd]? that' src include tools            # 68
rg -n --glob '!build*' -e 'assert\(false\)|assert\(0\)|std::abort|__trap\(\)|#if\s+0\b' src include tools                           # 10
```

This repo does not use TODO/FIXME (one hit each). The productive tokens are
`placeholder` (25), `stub` (6), `not implemented` (6) — read all 39. `should be`
(90 hits alone), `revisit` and `naive impl` (0) are left out of the second
pattern deliberately.

## 2. Compiler as detector

`-Wswitch-enum` is the one that pays: it names the enum values a dispatch claims
to support and does not handle (`QType`, arch, sampler). Scratch configure only,
never the tree; device code needs it through `-Xcompiler`.

## 3. AST matchers — what grep cannot see

`clang-query` is in no imp image. Install on the fly, mount at **`/src`**
(`build-dev/compile_commands.json` hardcodes it), **one matcher per run** —
three in one script silently produced two result blocks.

```bash
printf 'set output diag\nmatch %s\nquit\n' "$MATCHER" > /tmp/q.txt
docker run --rm -v $(pwd):/src -v /tmp/q.txt:/q.txt -w /src imp:toolchain bash -c \
  'apt-get update -qq && apt-get install -y -qq clang-tools >/dev/null 2>&1;
   clang-query -p build-dev -f /q.txt $(find src tools -name "*.cpp" | sort)' \
  | grep -E "^[0-9]+ matches|/src/(src|tools)/.*binds here"
```

**`isExpansionInMainFile()` in every matcher is mandatory, not hygiene:** one TU
without it returned **6086 matches**, all from `nlohmann/json` and CUTLASS
headers. With it, 0. **Dedupe by `file:line`** — a TU that is compiled into more
than one target is processed once per entry, so its hits are reported twice
(`tool_call.cpp` alone accounts for 4 of the 9 empty catches, as two sites).

| matcher | on imp |
|---|---|
| `functionDecl(isDefinition(), unless(isImplicit()), isExpansionInMainFile(), hasBody(compoundStmt(statementCountIs(0))))` | 24 raw — mostly out-of-line trivial destructors |
| same, `compoundStmt(statementCountIs(1), hasAnySubstatement(returnStmt(hasReturnValue(ignoringImplicit(anyOf(cxxBoolLiteral(), integerLiteral(), cxxNullPtrLiteralExpr(), initListExpr(), floatLiteral()))))))` | **1** — a deliberate no-op lambda callback, so a second hit is worth reading |
| `cxxCatchStmt(isExpansionInMainFile(), has(compoundStmt(statementCountIs(0))))` | 9 raw, before dedupe |

`.cu` files are out of scope here (they need full nvcc flags) — that is what
rung 4 is for. `clang-tidy` adds `bugprone-empty-catch`, `misc-unused-parameters`,
`bugprone-branch-clone` (identical if/else = usually an unfinished special case);
`make tidy` already wires it.

## 4. Reachability, not syntax — one-off

Host: `-ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--print-gc-sections`;
anything collected is reachable from no entry point. Coverage:
`-fprofile-instr-generate -fcoverage-mapping`, then `llvm-cov show --region-coverage-lt=1`
after a **real server run**, not just the suite.

Device (no gcov for CUDA) — census by diff. `cuobjdump --dump-resource-usage`
reports **0 functions** on this build; use the host-side stubs:

```bash
nm build-dev/imp-server | grep -oP '__device_stub__\S+' | sed 's/.*__device_stub__/_/' \
  | c++filt | sed 's/(.*//; s/<.*//' | grep -v '^_' | sort -u > kernels_present.txt   # 308
nsys stats --report cuda_gpu_kern_sum --format csv run.nsys-rep | cut -d, -f8 | sort -u > kernels_launched.txt
comm -23 kernels_present.txt kernels_launched.txt
```

The difference is dead kernels **and** live kernels behind a condition that is
never true — `ssm_graph_ban` was the second kind and cost 3x decode.

## 5. Accepted but ignored — the expensive class

A field parsed out of JSON, stored in a struct, never read. Syntactically
perfect, no warning, silently wrong; the highest-value rung for `imp-server`.

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

The `tr ',' '\n'` matters: half this struct is comma-separated declarators, and
without it `include_usage` and the three `*_explicit` flags are invisible.
**Baseline: 53 fields, 10 candidates, all benign** — one write, one read each.
**The finding is a field with 0**, or a new name in that list. Read both sites of
every candidate; the count never decides.

## 6. Tests that check nothing

```bash
rg -n 'DISABLED_' tests                                                        # 8 — read all
awk -f .claude/skills/find-stubs/tests_without_assertions.awk $(find tests -name '*.cpp' -o -name '*.cu')   # 44
```

A one-line regex over test bodies does **not** work here — it stops at the first
line-initial `}`, which any helper lambda provides, and flagged 184 of 2410. The
awk script splits on `^TEST` / `^}` and drops harness-delegating bodies; read its
header before trusting the 44. `GTEST_SKIP` is 178 hits and mostly legitimate
(no GPU, no model) — filter to skips whose condition is neither, or you drown.

A green test that asserts nothing is the most dangerous form of unfinished.

## Pitfalls

- **A finding is a candidate.** Verify at the code before reporting — same rule,
  same reason as skill `codebase-audit`.
- **Refresh the baselines above when the tree moves**, or the next reader scores
  normal as a regression. If any rung becomes a CI gate, give it a baseline file
  and fail on growth, like `tools/check_filesize.py`.
- **Absence of a marker means nothing.** None of the defects in
  [`MEMORY.md`](../../../docs/internals/MEMORY.md) B8 carried one, and the worst
  of them was a *correct* log line describing the wrong quantity.
