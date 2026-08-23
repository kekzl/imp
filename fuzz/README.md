<!--
layer: L3
audience: agents
verified: 2026-08-22
commit: 6a1d363c
-->

# fuzz — targets for the parsers that take untrusted bytes

`docs/audit/SETTLED.md` S-28 claimed these surfaces were "fuzzed, in CI"
(#1620). No fuzz target existed anywhere in the tree; the four files it named
are two hand-written fault-injection batteries and two seeded property tests
whose generator output is asserted **valid** before use.

## Two ways to run the same targets

| | build | driver | where |
|---|---|---|---|
| corpus + mutator | `make dev` | `tests/test_fuzz_corpus.cpp` | CPU lane, every PR, ~0.7 s |
| libFuzzer | `-DIMP_FUZZERS=ON`, clang | `fuzz_<target>` binaries | on demand |

```bash
docker run --rm -v $PWD:/src -w /src silkeh/clang:18 bash -c '
  cmake -B build-fuzz -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
        -DIMP_FUZZERS=ON -DIMP_BUILD_TESTS=OFF -DIMP_BUILD_SERVER=OFF &&
  cmake --build build-fuzz -j$(nproc) --target fuzz_json_schema &&
  ./build-fuzz/fuzz_json_schema -max_total_time=600'
```

## Targets

| target | entry | shipped defects |
|---|---|---|
| `fuzz_json_schema` | `parse_json_schema` | #1564 desync, #1609 depth |
| `fuzz_regex` | `RegexNfa::compile` + `step` | #1608 unbounded `{n,m}`, #1609 depth |
| `fuzz_gbnf` | `parse_gbnf` | #1609 depth |
| `fuzz_tool_stream` | `StreamToolCallFilter::feed` | #1554 mid-codepoint delta (twice) |
| `fuzz_safetensors` | `load_safetensors` on a real file | #1603-#1605 |
| `fuzz_tokenizer_json` | `Tokenizer::load` on a real file | #1606 |

## What these do not cover

- **`fuzz_safetensors` reaches the header scan, not the weight upload.** With no
  `config.json` beside the file no `Model` is built, so the wild pointer #1603
  produces is never dereferenced. Measured: against the reverted fixes the
  target stays green under ASan while the other four go red. The upload path
  needs a GPU and is covered by `tests/test_safetensors_loader.cpp`.
- **A wrong parse is invisible here.** #1564 (a truncated schema) and #1567 (a
  dropped keyword) produce a valid-looking tree, not a crash. A fuzzer with no
  oracle cannot see that; the unit tests can.
- **No coverage feedback in the CPU lane.** The mutator is four byte operations
  over a fixed corpus with a fixed seed, a few thousand executions. It catches a
  re-introduction of a known defect class, not a new one.

## Adding a corpus entry

It earns its place by having broken something, or by reaching a branch nothing
else reaches. Measure it: against the reverted fix the entry must produce the
failure. A 200-level nesting entry passed where 20000 crashes, and an entry
built from `[` never reached a parser that recurses on `{`.
