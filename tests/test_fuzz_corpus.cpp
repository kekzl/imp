// The fuzz targets, driven in the CPU lane (#1620).
//
// `docs/audit/SETTLED.md` S-28 recorded that the parser surfaces are "fuzzed,
// in CI". They were not: two of the four files it named are hand-written
// fault-injection batteries with no randomness at all, and the other two are
// seeded property tests whose generator's output is asserted VALID before use.
// Nothing mutated anything, and no fuzz target existed anywhere in the tree.
//
// This file closes the gap from the cheap end. It drives the same
// `imp_fuzz_*` entry points libFuzzer drives (fuzz/), over:
//
//   1. a committed corpus of the inputs that actually broke something, so a
//      re-introduction is a named test failure rather than a fuzzing session
//      someone has to remember to run;
//   2. a deterministic mutator over that corpus, fixed seed, so a failure
//      reproduces from the test name alone.
//
// It is NOT a substitute for a real fuzzing run: no coverage feedback, no
// corpus growth, a few thousand executions instead of billions. It is the part
// that runs on every pull request. `cmake -DIMP_FUZZERS=ON` with clang builds
// the libFuzzer binaries for the long runs.
//
// Budget: this has to stay inside the CPU lane's seconds. Keep the per-target
// iteration counts small enough that the whole file is well under a second,
// and put depth/size bombs in the corpus rather than hoping the mutator finds
// them.

#include <gtest/gtest.h>

#include "fuzz_targets.h"

#include <cstdint>
#include <string>
#include <vector>

namespace {

using Target = int (*)(const uint8_t*, size_t);

struct TargetSpec {
    const char* name;
    Target fn;
    std::vector<std::string> corpus;
};

// xorshift64*, so the sequence is identical on every platform and compiler.
// std::mt19937 would also be reproducible, but this keeps the mutator's own
// behaviour readable in one line.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    size_t below(size_t n) { return n ? static_cast<size_t>(next() % n) : 0; }
};

// Four mutations, chosen because they are the ones that produced the shipped
// defects: a flipped byte (a dtype string one character off), a truncation (a
// header cut mid-value), an insertion (a nesting level too many), and a splice
// of two corpus entries (a valid prefix with a hostile tail).
std::string mutate(const std::string& in, const std::string& other, Rng& rng) {
    std::string s = in;
    switch (rng.below(5)) {
        case 0:
            if (!s.empty())
                s[rng.below(s.size())] = static_cast<char>(rng.next() & 0xFF);
            break;
        case 1:
            if (!s.empty())
                s.resize(rng.below(s.size()));
            break;
        case 2:
            s.insert(rng.below(s.size() + 1), 1 + rng.below(8), static_cast<char>(rng.next() & 0xFF));
            break;
        case 3:
            if (!s.empty())
                s.erase(rng.below(s.size()), 1 + rng.below(4));
            break;
        default:
            if (!other.empty())
                s = s.substr(0, rng.below(s.size() + 1)) + other.substr(rng.below(other.size()));
            break;
    }
    return s;
}

void run_target(const TargetSpec& spec, int iterations, uint64_t seed) {
    ASSERT_FALSE(spec.corpus.empty()) << spec.name << " has no corpus";

    // 1. The corpus verbatim. Every entry is an input that broke something.
    for (size_t i = 0; i < spec.corpus.size(); i++) {
        const auto& c = spec.corpus[i];
        EXPECT_EQ(spec.fn(reinterpret_cast<const uint8_t*>(c.data()), c.size()), 0)
            << spec.name << ": corpus entry " << i << " reported a violated invariant";
    }

    // 2. Mutations of it. A crash, a hang or a sanitizer report is the finding;
    // the return value only carries invariants a target can check itself.
    Rng rng(seed);
    for (int i = 0; i < iterations; i++) {
        const std::string& base = spec.corpus[rng.below(spec.corpus.size())];
        const std::string& other = spec.corpus[rng.below(spec.corpus.size())];
        const std::string in = mutate(base, other, rng);
        EXPECT_EQ(spec.fn(reinterpret_cast<const uint8_t*>(in.data()), in.size()), 0)
            << spec.name << ": mutation " << i << " (seed " << seed << ") reported a violation";
    }
}

// ---- corpora ----
//
// Rule for adding to these: an entry earns its place by having broken
// something, or by reaching a branch nothing else reaches. A generic "valid
// input" belongs in a unit test, not here.

std::vector<std::string> schema_corpus() {
    return {
        R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"]})",
        // #1564: consumed nothing, truncated the rest of the schema.
        R"({"type":"object","additionalProperties":{"type":"number"},"properties":{"a":{"type":"string"}}})",
        // #1564: constrained the model to the empty string.
        R"({"type":"integer","enum":[1,2,3]})",
        R"({"enum":[true,false]})",
        R"({"const":42})",
        // #1567: accepted and silently dropped.
        R"({"type":"integer","minimum":1,"maximum":5})",
        R"({"allOf":[{"type":"string"}],"not":{"type":"number"}})",
        // #1609: one stack frame per level. Two things had to be measured
        // against the reverted fix to get this entry right: the parser
        // recurses on '{' (a run of '[' is rejected at the first character and
        // reaches nothing), and 200 levels is not enough to overflow a worker
        // stack. This shape is the one from the issue, at a depth that
        // actually takes the process down.
        [] {
            std::string d;
            for (int i = 0; i < 20000; i++)
                d += R"({"items":)";
            d += "{}";
            for (int i = 0; i < 20000; i++)
                d += "}";
            return d;
        }(),
        R"({"items":{"items":{"items":{"items":{"type":"string"}}}}})",
        // Structure the parser has to survive without a value.
        R"({"$ref":"#/$defs/x","$defs":{"x":{"type":"string"}}})",
        R"({"type":"string","pattern":"^a{3,5}$"})",
        "{",
        "",
    };
}

std::vector<std::string> regex_corpus() {
    return {
        "^[a-z0-9_]{1,32}$",
        R"(\d{4}-\d{2}-\d{2})",
        "(foo|bar)+baz",
        // #1608: a clone count with no bound, and a digit run that overflowed.
        "a{2000000000}",
        "a{99999999999999999999999}",
        "(((a{100}){100}){100}){100}",
        // #1609: one frame per '('. Same reasoning as the schema corpus.
        std::string(20000, '(') + "a" + std::string(20000, ')'),
        "[",
        "a{",
        "a{,}",
        "",
    };
}

std::vector<std::string> gbnf_corpus() {
    return {
        "root ::= \"a\" | \"b\"",
        "root ::= obj\nobj ::= \"{\" pair (\",\" pair)* \"}\"\npair ::= \"x\"",
        "root ::= \"a\"{1,1024}",
        "root ::= \"a\"{0,100000}",
        "root ::= " + std::string(20000, '(') + "\"a\"" + std::string(20000, ')'),
        "root ::= root",
        "root ::=",
        "",
    };
}

#ifdef IMP_FUZZ_HAVE_TOOL_STREAM
std::vector<std::string> tool_stream_corpus() {
    // First byte is the knob (family + chunk size); the rest is the body.
    auto with_knob = [](uint8_t k, const std::string& body) {
        return std::string(1, static_cast<char>(k)) + body;
    };
    return {
        with_knob(0x01, R"(<tool_call>{"name": "f", "arguments": {"a": 1}}</tool_call>)"),
        // #1554: the emit boundary cut these in half.
        with_knob(0x01,
                  "<tool_call>{\"name\": \"note\", \"arguments\": {\"t\": "
                  "\"ÄÖÜäöüß ÄÖÜäöüß\"}}</tool_call>"),
        with_knob(0x09, "<tool_call>{\"name\": \"note\", \"arguments\": {\"t\": \"中文中文\"}}</tool_call>"),
        with_knob(0x11, "<tool_call>{\"name\": \"note\", \"arguments\": {\"t\": \"a😀b😀c\"}}</tool_call>"),
        with_knob(0x02, "<function=f>{\"a\": \"ü\"}</function>"),
        with_knob(0x03, "plain text with no call at all"),
        with_knob(0x04, "<tool_call>{\"name\":"),
    };
}
#endif

std::vector<std::string> safetensors_corpus() {
    // 8-byte little-endian header length, then the JSON header, then data.
    auto blob = [](const std::string& header, size_t payload) {
        std::string s(8, '\0');
        const uint64_t n = header.size();
        for (int i = 0; i < 8; i++)
            s[i] = static_cast<char>((n >> (8 * i)) & 0xFF);
        s += header;
        s.append(payload, '\0');
        return s;
    };
    return {
        blob(R"({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}})", 16),
        // #1604: validated at 2 bytes/elem, read at 4.
        blob(R"({"w": {"dtype": "I16", "shape": [4], "data_offsets": [0, 8]}})", 8),
        blob(R"({"w": {"dtype": "U16", "shape": [4], "data_offsets": [0, 8]}})", 8),
        // #1603: offset_start never validated on this branch.
        blob(R"({"w": {"dtype": "F8_E8M0", "shape": [4,4], "data_offsets": [9223372036854775807, 0]}})", 64),
        // #1605: no overflow and no sign guard on the shape product.
        blob(R"({"w": {"dtype": "F32", "shape": [-4], "data_offsets": [0, 16]}})", 16),
        blob(R"({"w": {"dtype": "F32", "shape": [4294967296, 4294967296], "data_offsets": [0, 0]}})", 16),
        blob(R"({"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 9223372036854775807]}})", 16),
        blob("{}", 0),
        blob("not json at all", 0),
        std::string(8, '\xff'),  // header length past any file
        "",
    };
}

std::vector<std::string> tokenizer_corpus() {
    return {
        R"({"model":{"type":"BPE","vocab":{"a":0,"b":1},"merges":[]}})",
        // #1606: vocab_[-1] = token, during load.
        R"({"model":{"type":"BPE","vocab":{"a":-1,"b":0},"merges":[]}})",
        R"({"model":{"type":"BPE","vocab":{"a":0,"huge":2147483647},"merges":[]}})",
        R"({"model":{"type":"BPE","vocab":{"a":0},"merges":[]},"added_tokens":[{"id":-5,"content":"<x>","special":1}]})",
        R"({"model":{"type":"Unigram","vocab":[["a",0.0],["b",-1.0]]}})",
        R"({"model":{"type":"BPE","vocab":{"ü":0,"😀":1},"merges":["ü 😀"]}})",
        R"({"model":{}})",
        "{",
        "",
    };
}

// Minimal GGUF writer for the two GGUF-shaped corpora: header, KV pairs, tensor
// infos, aligned data. Type ids per the GGUF spec (UINT32=4, STRING=8, ARRAY=9;
// ggml F32=0, F16=1). The hostile entries are the AUDIT_arch_2026 F1 cases.
struct GgufWriter {
    std::string s;
    void u32(uint32_t v) { raw(&v, 4); }
    void u64(uint64_t v) { raw(&v, 8); }
    void f32(float v) { raw(&v, 4); }
    void str(const std::string& v) {
        u64(v.size());
        s += v;
    }
    void raw(const void* p, size_t n) { s.append(static_cast<const char*>(p), n); }
    void align(size_t a) {
        while (s.size() % a)
            s.push_back('\0');
    }
    void header(uint64_t n_tensors, uint64_t n_kv) {
        u32(0x46554747u);
        u32(3);
        u64(n_tensors);
        u64(n_kv);
    }
    void kv_u32(const std::string& k, uint32_t v) {
        str(k);
        u32(4);
        u32(v);
    }
    void kv_str(const std::string& k, const std::string& v) {
        str(k);
        u32(8);
        str(v);
    }
    void kv_u32_array(const std::string& k, const std::vector<uint32_t>& a) {
        str(k);
        u32(9);
        u32(4);
        u64(a.size());
        for (auto v : a)
            u32(v);
    }
    void kv_str_array(const std::string& k, const std::vector<std::string>& a) {
        str(k);
        u32(9);
        u32(8);
        u64(a.size());
        for (const auto& v : a)
            str(v);
    }
    // `n_dims` is passed separately so it can disagree with `dims` (F1-1).
    void tensor(const std::string& name, uint32_t n_dims, const std::vector<uint64_t>& dims, uint32_t type,
                uint64_t offset) {
        str(name);
        u32(n_dims);
        for (auto d : dims)
            u64(d);
        u32(type);
        u64(offset);
    }
    void f32_data(int n) {
        align(32);
        for (int i = 0; i < n; i++)
            f32(0.01f * static_cast<float>(i));
    }
};

std::vector<std::string> gguf_corpus() {
    auto llama = [](uint32_t n_dims, uint64_t ne0, uint64_t offset, uint32_t block_count) {
        GgufWriter w;
        w.header(1, 2);
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", block_count);
        w.tensor("token_embd.weight", n_dims, {ne0, 4}, 0, offset);
        w.f32_data(16);
        return w.s;
    };
    auto tokenizer = [](uint32_t bos) {
        GgufWriter w;
        w.header(1, 4);
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 0);
        w.kv_str_array("tokenizer.ggml.tokens", {"a", "b", "c"});
        w.kv_u32("tokenizer.ggml.bos_token_id", bos);
        w.tensor("token_embd.weight", 2, {4, 4}, 0, 0);
        w.f32_data(16);
        return w.s;
    };
    auto gemma4 = [](uint32_t block_count, const std::vector<uint32_t>& pattern) {
        GgufWriter w;
        w.header(1, 5);
        w.kv_str("general.architecture", "gemma4");
        w.kv_u32("gemma4.block_count", block_count);
        w.kv_u32_array("gemma4.attention.sliding_window_pattern", pattern);
        w.kv_u32("gemma4.attention.key_length", 256);
        w.kv_u32("gemma4.attention.key_length_swa", 512);
        w.tensor("token_embd.weight", 2, {4, 4}, 0, 0);
        w.f32_data(16);
        return w.s;
    };
    GgufWriter huge;
    huge.header(uint64_t{1} << 60, 0);
    GgufWriter five;  // F1-1 well-formed: n_dims = 5 with five dim words
    five.header(1, 2);
    five.kv_str("general.architecture", "llama");
    five.kv_u32("llama.block_count", 0);
    five.tensor("token_embd.weight", 5, {4, 4, 1, 1, 1}, 0, 0);
    five.f32_data(16);
    return {
        llama(2, 4, 0, 0),
        llama(5, 4, 0, 0),  // F1-1: n_dims > 4, words missing
        five.s,
        llama(2, 4, uint64_t{1} << 63, 0),  // offset past the file
        llama(2, uint64_t{1} << 40, 0, 0),  // dim product overflow
        llama(2, 4, 0, 2147483647u),        // F1-5: block_count past the cap
        tokenizer(0x40000000u),             // F1-7: bos outside the vocab
        tokenizer(1),
        gemma4(8, {1}),  // F1-5: pattern shorter than block_count
        huge.s,          // tensor_count = 2^60
        "GGUF",
        "",
    };
}

std::vector<std::string> mmproj_corpus() {
    auto clip = [](uint32_t n_dims, uint64_t offset, uint32_t block_count, uint32_t patch) {
        GgufWriter w;
        w.header(1, 6);
        w.kv_str("general.architecture", "clip");
        w.kv_u32("clip.vision.block_count", block_count);
        w.kv_u32("clip.vision.embedding_length", 8);
        w.kv_u32("clip.vision.attention.head_count", 2);
        w.kv_u32("clip.vision.image_size", 28);
        w.kv_u32("clip.vision.patch_size", patch);
        w.tensor("v.patch_embd.weight", n_dims, {8, 8}, 1, offset);
        w.align(32);
        w.s.append(128, '\0');
        return w.s;
    };
    GgufWriter huge;
    huge.header(uint64_t{1} << 60, 0);
    return {
        clip(2, 0, 2, 14),
        clip(5, 0, 2, 14),                  // F1-2: n_dims > 4, same fork
        clip(2, uint64_t{1} << 63, 2, 14),  // F1-2: offset past the file, no bounds check
        clip(2, 0, 2147483647u, 14),        // block_count sized layers before any cap
        clip(2, 0, 2, 0),                   // patch_size = 0 divided image_size
        huge.s,
        "GGUF",
        "",
    };
}

// Iteration counts: enough that the mutator reaches past the corpus, small
// enough that the file stays inside the CPU lane's budget. The file-backed
// targets get fewer because each execution writes and removes a temp file.
TEST(FuzzCorpus, JsonSchema) {
    run_target({"json_schema", imp_fuzz_json_schema, schema_corpus()}, 1500, 0xA11CE);
}

TEST(FuzzCorpus, Regex) { run_target({"regex", imp_fuzz_regex, regex_corpus()}, 1500, 0xB0B); }

TEST(FuzzCorpus, Gbnf) { run_target({"gbnf", imp_fuzz_gbnf, gbnf_corpus()}, 1500, 0xC0FFEE); }

// tool_call.h needs nlohmann, which only the server build fetches, so this one
// target is absent from the sanitizer build. The other five carry that lane.
#ifdef IMP_FUZZ_HAVE_TOOL_STREAM
TEST(FuzzCorpus, ToolStreamFilter) {
    run_target({"tool_stream", imp_fuzz_tool_stream, tool_stream_corpus()}, 1500, 0xD00D);
}
#endif

TEST(FuzzCorpus, SafeTensorsLoader) {
    run_target({"safetensors", imp_fuzz_safetensors, safetensors_corpus()}, 250, 0xE1F);
}

TEST(FuzzCorpus, TokenizerJson) {
    run_target({"tokenizer_json", imp_fuzz_tokenizer_json, tokenizer_corpus()}, 250, 0xF00D);
}

TEST(FuzzCorpus, GgufLoader) { run_target({"gguf", imp_fuzz_gguf, gguf_corpus()}, 250, 0xA5A5); }

TEST(FuzzCorpus, MmprojLoader) { run_target({"mmproj", imp_fuzz_mmproj, mmproj_corpus()}, 250, 0x5A5A); }

}  // namespace
