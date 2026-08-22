// Built only when the server sources are in (nlohmann via tool_call.h).
#define IMP_FUZZ_HAVE_TOOL_STREAM 1
// Streaming tool-call filter (#1620). This is where #1554 lived, twice: the
// emit boundary is byte arithmetic on a UTF-8 buffer, and the first fix went
// to the other chunker. The fuzzer feeds the input in varying chunk sizes,
// because the defect only appears at particular boundaries.

#include "fuzz_targets.h"
#include "fuzz_common.h"

#include "tool_stream_filter.h"
#include "stream_pipeline.h"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>

extern "C" int imp_fuzz_tool_stream(const uint8_t* data, size_t size) {
    imp_fuzz::quiet_logs();
    if (size < 2 || size > 65536)
        return 0;
    // First byte picks the template family and the chunk size, so one input
    // covers "same bytes, different arrival pattern" - the axis the defect
    // lives on.
    const uint8_t knob = data[0];
    const auto fam = static_cast<imp::ChatTemplateFamily>(knob % 6);
    const size_t chunk = 1 + (knob >> 3) % 32;
    const std::string body(reinterpret_cast<const char*>(data + 1), size - 1);

    // The invariant below only holds for well-formed input: the filter passes
    // bytes through, so ill-formed UTF-8 in gives ill-formed UTF-8 out and that
    // is correct behaviour, not a defect. The mutator produces such input
    // constantly (one flipped byte inside a 2-byte character is enough), and
    // the first version of this target reported those as findings.
    const bool body_is_utf8 = imp_fuzz::is_valid_utf8(body);

    try {
        imp::server::StreamToolCallFilter filter(fam);
        using Kind = imp::server::StreamToolCallFilter::Segment::Kind;
        for (size_t off = 0; off < body.size(); off += chunk) {
            for (auto& seg : filter.feed(body.substr(off, chunk))) {
                if (seg.kind == Kind::CALL_ARGS_DELTA && body_is_utf8) {
                    // The invariant #1554 broke: every delta is JSON-encoded on
                    // its own, so every delta has to be whole UTF-8. Abort
                    // rather than return, so the fuzzer records it as a crash
                    // with the reproducing input.
                    if (imp::stream::utf8_complete_len(seg.text) != seg.text.size()) {
                        fprintf(stderr, "fuzz: CALL_ARGS_DELTA ends mid-codepoint\n");
#ifdef IMP_FUZZ_NO_ENTRY
                        // Corpus runner: report, so the test names the case.
                        return 1;
#else
                        // libFuzzer only records crashes, and only a crash
                        // saves the reproducing input.
                        abort();
#endif
                    }
                }
            }
        }
        (void)filter.finish();
    } catch (const std::bad_alloc&) {}
    return 0;
}

#ifndef IMP_FUZZ_NO_ENTRY
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    return imp_fuzz_tool_stream(data, size);
}
#endif
