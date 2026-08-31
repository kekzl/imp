#include "runtime/spec_trace.h"

#include <algorithm>

#include "exec/executor.h"
#include "core/logging.h"

#include <cstdio>

namespace imp {

std::string spec_trace_top2_gaps(const float* logits, int n_rows, size_t vocab) {
    std::string s;
    for (int j = 0; j < n_rows; ++j) {
        const float* row = logits + static_cast<size_t>(j) * vocab;
        // Two passes would read 2 MB twice; one pass keeping the best two is
        // enough and this runs per verify step when the flag is on.
        float b1 = -3e38f, b2 = -3e38f;
        int i1 = -1, i2 = -1;
        for (size_t t = 0; t < vocab; ++t) {
            const float x = row[t];
            if (x > b1) {
                b2 = b1;
                i2 = i1;
                b1 = x;
                i1 = static_cast<int>(t);
            } else if (x > b2) {
                b2 = x;
                i2 = static_cast<int>(t);
            }
        }
        char buf[96];
        snprintf(buf, sizeof(buf), "%s%d>%d:%.4f", j ? "," : "", i1, i2, b1 - b2);
        s += buf;
    }
    return s;
}

void spec_trace_emit_verify(int p0, int t0, const std::vector<int32_t>* draft, int mc_cands,
                            const int32_t* argmax, int chunk_len, GraphExecutor* exec, float* d_logits,
                            std::vector<float>& h_logits, int vocab, cudaStream_t stream) {
    std::string s = "[verify] p0=" + std::to_string(p0) + " t0=" + std::to_string(t0) +
                    (mc_cands ? " mc_cands=" + std::to_string(mc_cands) : "") + " draft=[";
    // mc: `draft` is the first of mc_cands contiguous candidate vectors; print
    // every candidate ('|'-separated) - the argmax rows below are grouped the
    // same way, (1 + depth) rows per candidate.
    const int n_cands = std::max(1, mc_cands);
    for (int c = 0; draft && c < n_cands; ++c) {
        const auto& d = draft[c];
        if (c) s += "|";
        for (size_t j = 0; j < d.size(); ++j)
            s += std::to_string(d[j]) + (j + 1 < d.size() ? "," : "");
    }
    s += "] argmax=[";
    for (int j = 0; j < chunk_len; ++j)
        s += std::to_string(argmax[j]) + (j + 1 < chunk_len ? "," : "");
    s += "]";

    // The top-2 gap is the whole reason this file exists. The bonus token off
    // the last chunk row decides whether generation stops, and
    // docs/LIMITATIONS.md records it coming out as <|im_end|> where
    // single-token decode keeps writing - without ever saying by how much.
    if (d_logits != nullptr && !h_logits.empty() && exec != nullptr) {
        exec->project_logits_all(chunk_len, d_logits, stream);
        const size_t v = static_cast<size_t>(vocab);
        if (cudaMemcpyAsync(h_logits.data(), d_logits, static_cast<size_t>(chunk_len) * v * sizeof(float),
                            cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
            cudaStreamSynchronize(stream) == cudaSuccess) {
            s += " gap=[" + spec_trace_top2_gaps(h_logits.data(), chunk_len, v) + "]";
        }
    }
    IMP_LOG_INFO("%s", s.c_str());
}

}  // namespace imp
