#pragma once

// Fuzz targets for parsers taking untrusted bytes (#1620): libFuzzer binaries (clang,
// -DIMP_FUZZERS=ON) and tests/test_fuzz_corpus.cpp (same functions, g++, corpus+mutator, what CI runs).
// In scope: parsers reachable from a file/request body a user doesn't control; not GPU/model work.

#include <cstddef>
#include <cstdint>

// Return 0 for "input processed"; non-zero means the target detected a violated invariant
// without crashing (only imp_fuzz_tool_stream does). Under libFuzzer the same condition aborts instead.
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

// GGUF loader, against a real file (AUDIT_arch_2026 F1-1: `n_dims > 4` was a
// stack write in the loader; F1-5/F1-7/F1-11 the same class in metadata).
int imp_fuzz_gguf(const uint8_t* data, size_t size);

// Vision (mmproj) GGUF loader, dry pass (F1-2: no bounds check at all, and a
// hand-copied fork of the GGUF parse loop that never got the shared checks).
int imp_fuzz_mmproj(const uint8_t* data, size_t size);

}  // extern "C"
