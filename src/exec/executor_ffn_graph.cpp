// Piecewise FFN prefill graphs (#2435): see exec/ffn_graph_cache.h.

#include "core/dispatch_policy.h"
#include "exec/executor.h"
#include "exec/executor_debug.h"
#include "exec/pre_dequant_internal.h"

namespace imp {

void GraphExecutor::set_ffn_graphs_enabled(bool on, bool cuda_graphs) {
    // Host-resident experts stage through per-call copies (build_staged_device_args_).
    ffn_graphs_enabled_ = on && cuda_graphs && model_ != nullptr &&
                          !pre_dequant_internal::has_host_resident_experts(*model_);
    if (!ffn_graphs_enabled_)
        ffn_graphs_.clear();
}

int GraphExecutor::ffn_graph_begin_(const InferenceState& state, int n, cudaStream_t stream) {
    if (!ffn_graphs_enabled_ || ffn_graphs_.disabled() || !state.is_prefill || state.spec_verify_chunk ||
        state.force_fp16_gemm)
        return 0;
    const int rows = FfnGraphCache::bucket_rows(n);
    if (rows <= 0 || rows > max_tokens_)
        return 0;
    // Host state the FFN reads or writes per call, which a replay would skip.
    if (calib_ || lora_ != nullptr || offload_mgr_ != nullptr || fp32_accum_buf_ != nullptr ||
        model_->profile().gated_residual)
        return 0;
    const auto& diag = dispatch_policy().diagnostics;
    if (diag.profile || !diag.moe_expert_hist.empty() || debug_forward_enabled() ||
        dump_hidden_dir() != nullptr)
        return 0;
    if (moe_prefill_uncapturable() || nvfp4_dequant_uncapturable())
        return 0;
    // Inside the serial prefill graph's capture: that graph already holds the whole forward.
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone)
        return 0;
    // Phase views are carved from the row count: carve at `rows` so a bucket keeps its addresses.
    if (!ws_.resize_workspace(rows, stream))
        return 0;
    // Pad rows feed norms, routing and GEMMs but no output; zeroed so they stay finite.
    if (rows > n) {
        for (Tensor* t : {&hidden_, &residual_, &norm_out_}) {
            const size_t row_bytes = t->nbytes() / static_cast<size_t>(t->shape[0]);
            IMP_CUDA_CHECK_LOG(
                cudaMemsetAsync(static_cast<char*>(t->data) + static_cast<size_t>(n) * row_bytes, 0,
                                static_cast<size_t>(rows - n) * row_bytes, stream));
        }
    }
    return rows;
}

void GraphExecutor::run_ffn_phase_(int layer, cudaStream_t stream) {
    if (dispatch_policy().moe.skip)  // debug: skip all FFN/MoE to isolate attention bugs
        return;
    const bool moe = layer_has_moe(layer);
    if (!moe && !layer_has_dense_ffn(layer))
        return;
    auto fn = [&] {
        if (moe)
            run_moe_ffn(layer, stream);
        else
            run_ffn(layer, stream);
    };
    if (ffn_graph_rows_ <= 0) {
        fn();
        return;
    }
    struct RowsScope {
        int& cur;
        int saved;
        ~RowsScope() { cur = saved; }
    } scope{cur_n_tokens_, cur_n_tokens_};
    cur_n_tokens_ = ffn_graph_rows_;
    ffn_graphs_.run(layer, ffn_graph_rows_, ws_.generation(), stream, fn);
}

}  // namespace imp
