<!--
layer: L3
audience: agents
verified: 2026-08-13
commit: b61fbf5f
-->

# Agent evaluation

The dispatch's acceptance criterion: a subagent given **only** `CLAUDE.md` (root
plus one directory) must answer three questions correctly. Run 2026-08-13
against `CLAUDE.md` + `src/compute/CLAUDE.md`, with the agent forbidden from
reading, grepping or listing anything else.

The point of the exercise is that it can fail. It did, twice, and both failures
were real defects in the files rather than in the questions.

## Round 1 — two failures

| question | verdict |
|---|---|
| (a) where is the decode-attention variant chosen? | **medium confidence, and the answer it gave was wrong** |
| (b) build and test only this directory | **NOT DERIVABLE** |
| (c) which target is compiled | pass, high confidence |

**(a)** The agent answered `src/compute/attention_dispatch.cu`, reasoning from
the line "picks the FMHA chain per (phase × dtype × layer)". It flagged its own
doubt: *"The text never uses the word 'decode' together with the dispatch file."*
That doubt was correct. The decode variant is chosen in
`src/exec/executor_attention_decode.cu`, at the `dispatch_record::set_attn_decode`
calls; `attention_dispatch.cu` handles the **prefill** FMHA chain. The file had
written "phase × dtype × layer" in a way that implied it covered both.

**(b)** *"Both files only give whole-repo build and test targets, not a way to
scope a build or test run to `src/compute/`."* Also correct, and the omission was
mine: the repository **does** split its test binaries per module
(`test-compute`, `test-attention`, `test-quant`, `test-kv`, `test-moe-gdn`,
built by `imp_add_test_module` in `CMakeLists.txt`). The file simply did not
mention them.

## Fixes

- `src/compute/CLAUDE.md` now says which file decides the decode variant and
  which decides prefill, and labels `attention_dispatch.cu` as prefill-only.
- It documents the per-module binaries, the `docker run` invocation, and that
  `ctest` registers only label aggregates so selection happens with
  `--gtest_filter` rather than `ctest -R`.

Both edits pushed the file over its 800-token budget, and `docs_lint.py` refused
it until it was compressed back under. That is the budget working: the file got
**more useful and no longer**.

## Round 2 — one gap left, then closed

| question | verdict |
|---|---|
| (a) | **pass, high confidence**, correct file named |
| (b) | test half derivable and correct; build half still incomplete |
| (c) | **pass, high confidence** |

The remaining criticism was exact: *"das Verzeichnis-CLAUDE.md springt vom
Make-Target direkt zu einem `docker run` mit `imp:test` und
`./build/test-attention`, ohne je zu sagen, welcher Befehl Image und Binary
erzeugt."* True. Fixed by naming the artefacts `make build` produces, in the
comment beside it.

## Standing caveats, so this is not read as more than it is

- One agent, one directory, one run per round. A different model, or
  `src/runtime/`, may find other gaps.
- The agent could still Read the two allowed paths, so this measures whether the
  files *contain* the answers, not whether an agent finds them under time
  pressure.
- Both rounds correctly refused to answer for a CMake target name, which neither
  file provides and which the questions did not really ask for. Read as a
  precision signal, not a gap.

## Verdict

Passes after two rounds. The test earned its place: it found one **wrong**
statement and one missing capability that a human reviewer of the same file had
already read past twice.
