#pragma once

// Sparse decode attention: token -> block geometry, as a pure function.
//
// It lives outside the .cu because the arithmetic is where the defect was
// (#1819): every conversion used the compile-time kKVBlockSize (16) while a
// model with n_kv_heads <= 4 runs a 32-token block, so `sparse_topk_tokens=N`
// bought 2N tokens of budget, `sparse_min_ctx` engaged at twice its stated
// length, and sink/recent covered twice their configured windows. Host-only
// and dependency-free, so a CPU test can pin it at both block sizes.

#include <algorithm>

namespace imp {

struct SparseGeometry {
    int budget_blocks = 0;
    int sink_blocks = 0;
    int recent_blocks = 0;
    int engage_blocks = 0;
    int max_ctx_blocks = 0;
    bool budget_raised = false;  // budget was below sink+recent and got lifted
};

// max_ctx_tokens: the engine's effective max_seq_len. The +16 slack mirrors the
// spec verify row tables (engine_spec_capture.cpp table_cap "+ 16"); the
// dispatch gate compares the incoming table stride against this capacity.
inline SparseGeometry sparse_geometry(int topk_tokens, int sink_tokens, int recent_tokens, int min_ctx,
                                      int max_ctx_tokens, int block_size) {
    const int bs = block_size > 0 ? block_size : 1;
    const auto up = [bs](int tokens) { return (std::max(tokens, 0) + bs - 1) / bs; };

    SparseGeometry g;
    g.max_ctx_blocks = up(max_ctx_tokens) + 16;
    g.sink_blocks = up(sink_tokens);
    // The recent window always covers at least the partial tail block.
    g.recent_blocks = std::max(1, up(recent_tokens));
    g.budget_blocks = up(topk_tokens);
    if (g.budget_blocks <= g.sink_blocks + g.recent_blocks) {
        g.budget_blocks = g.sink_blocks + g.recent_blocks + 1;
        g.budget_raised = true;
    }
    // Identity below sparse_min_ctx (the selection's win only outgrows its
    // overhead past ~12k measured); the table rows must hold an identity copy
    // up to that length.
    g.engage_blocks = std::min(g.max_ctx_blocks, std::max(g.budget_blocks, up(min_ctx)));
    return g;
}

}  // namespace imp
