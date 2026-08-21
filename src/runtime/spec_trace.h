#pragma once

// diagnostics.spec_trace helpers, split out of engine_spec_ngram.cpp so the
// speculation loop is not carrying its diagnostics: the top-2 formatting below
// is the only part of that file that reads full logits, and it pushed the TU
// over the size gate when it landed.

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace imp {

// "id1>id2:gap,..." for each of n_rows rows of [n_rows, vocab] float logits.
//
// The gap is (top1 - top2) in logit units. It exists to answer one question:
// whether a chunk row's verdict is a confident call or a coin flip. The bonus
// token off the last row of a verify chunk decides whether generation stops,
// and docs/LIMITATIONS.md records it coming out as <|im_end|> where
// single-token decode keeps writing - without ever saying by how much.
std::string spec_trace_top2_gaps(const float* logits, int n_rows, size_t vocab);

class GraphExecutor;

// Build and log the "[verify] ..." line. Takes the pieces rather than the
// Engine so the diagnostics do not need engine.h.
void spec_trace_emit_verify(int p0, int t0, const std::vector<int32_t>* draft, int mc_cands,
                            const int32_t* argmax, int chunk_len, GraphExecutor* exec, float* d_logits,
                            std::vector<float>& h_logits, int vocab, cudaStream_t stream);

}  // namespace imp
