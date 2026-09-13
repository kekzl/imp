#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace imp {

// Instrumentation-only probe for contextual FFN sparsity (Vector 1, "Break the Memory Wall").
// Counts, per hard-coded threshold {0.005,0.01,0.02,0.05,0.1}, intermediate-dim rows i with
// |silu(gate[i])*up[i]| < t - i.e. columns of w_down a per-token-aware kernel could skip.
// Per-layer counters accumulate across every dense-FFN decode step; flush_ffn_sparsity_probe_log()
// drains them to stderr and resets. Off unless ffn.sparsity_probe=true; public functions
// short-circuit before any device work when off.
void probe_ffn_silu_sparsity(int layer, const __half* gate, const __half* up, int K,
                             cudaStream_t stream);

void flush_ffn_sparsity_probe_log();

}  // namespace imp
