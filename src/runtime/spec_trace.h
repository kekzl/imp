#pragma once

// diagnostics.spec_trace helpers, split out of engine_spec_ngram.cpp so the
// speculation loop is not carrying its diagnostics: the top-2 formatting
// below is the only part of that file that reads full logits.

#include <cstddef>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace imp {

// "id1>id2:gap,..." for each of n_rows rows of [n_rows, vocab] float logits.
//
// gap = top1 - top2 in logit units: whether a row's verdict is a confident
// call or a coin flip. The bonus token off the last verify-chunk row decides
// whether generation stops (docs/LIMITATIONS.md: it can read <|im_end|>
// without saying by how much).
std::string spec_trace_top2_gaps(const float* logits, int n_rows, size_t vocab);

class GraphExecutor;

// Build and log the "[verify] ..." line. Takes the pieces rather than the
// Engine so the diagnostics do not need engine.h.
void spec_trace_emit_verify(int p0, int t0, const std::vector<int32_t>* draft, int mc_cands,
                            const int32_t* argmax, int chunk_len, GraphExecutor* exec, float* d_logits,
                            std::vector<float>& h_logits, int vocab, cudaStream_t stream);

// Bound batched verify, one line per step when `on`:
// "spec-batch-trace n=N T=T fwd=MS head=MS | id:nd:m:e ..." (request id, drafts verified, drafts matched,
// tokens emitted). Read by tools/analysis/mtp_trace_hist.py.
void spec_trace_emit_batch_step(bool on, int rows_per_req, double fwd_ms, double head_ms,
                                const std::vector<int>& ids, const std::vector<int>& n_draft,
                                const std::vector<int>& matched, const std::vector<int>& emitted);

}  // namespace imp
