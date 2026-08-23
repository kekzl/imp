#pragma once

// Fuzz targets for the parsers that take untrusted bytes (#1620).
//
// Two consumers, one body each:
//
//   1. libFuzzer, one binary per target, built with
//      `cmake -DIMP_FUZZERS=ON` and a clang toolchain. Each .cpp defines the
//      standard `LLVMFuzzerTestOneInput` entry point, so the targets are also
//      OSS-Fuzz shaped if that ever happens.
//   2. `tests/test_fuzz_corpus.cpp`, which drives the SAME functions over a
//      committed corpus plus a deterministic mutator, with g++, in the CPU
//      lane CI actually runs. That is what makes "fuzzed in CI" true rather
//      than aspirational - `docs/audit/SETTLED.md` S-28 claimed it for two
//      seeded property tests and two hand-written fault-injection batteries,
//      none of which mutate anything.
//
// The corpus build compiles every target with IMP_FUZZ_NO_ENTRY so the
// LLVMFuzzerTestOneInput definitions do not collide in one binary.
//
// What belongs here: a parser reachable from a file or a request body that a
// user does not control. What does not: anything needing a GPU, a model, or
// more than a few milliseconds per input.

#include <cstddef>
#include <cstdint>

// Return value: 0 for "input processed". A non-zero return means the target
// detected a violated invariant it can express without crashing - only
// imp_fuzz_tool_stream does that today. Under libFuzzer the same condition
// aborts instead, because that is the only thing libFuzzer saves an input for.
extern "C" {

// JSON Schema -> SchemaNode tree (src/compute/json_schema.cpp).
// Reached from `response_format.json_schema` and from tool `parameters`.
int imp_fuzz_json_schema(const uint8_t* data, size_t size);

// Regex -> Thompson NFA (RegexNfa::compile). Reached from
// `response_format.regex`, `guided_regex`, and a schema's `pattern`.
int imp_fuzz_regex(const uint8_t* data, size_t size);

// GBNF grammar -> rule table (src/compute/gbnf_parser.cpp). Reached from
// `response_format.grammar` / `guided_grammar`.
int imp_fuzz_gbnf(const uint8_t* data, size_t size);

// The tool-call stream filter, fed the input in chunks. Reached from every
// streaming response; a mid-codepoint cut here shipped twice (#1554).
#ifdef IMP_FUZZ_HAVE_TOOL_STREAM
int imp_fuzz_tool_stream(const uint8_t* data, size_t size);
#endif

// SafeTensors shard loader, against a real file. This is the surface that
// carried four out-of-bounds accesses (#1603-#1606), and it had no
// fault-injection battery of any kind.
int imp_fuzz_safetensors(const uint8_t* data, size_t size);

// tokenizer.json loader, against a real file (#1606: a negative token id was
// an out-of-bounds vector write during load).
int imp_fuzz_tokenizer_json(const uint8_t* data, size_t size);

}  // extern "C"
