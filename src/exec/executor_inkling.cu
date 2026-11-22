// GraphExecutor's Inkling2-Loop branch: the standard path embeds, norms and projects; the 48 layer
// applications run in exec/inkling_forward.h.

#include "exec/executor.h"

#include "core/logging.h"
#include "exec/inkling_forward.h"
#include "model/inkling_weights.h"

#include <stdexcept>

namespace imp {

std::unique_ptr<InklingRuntime> make_inkling_runtime(const Model& model, VRAMAllocator* va, int max_tokens,
                                                     int max_ctx) {
    const std::string& dir = model.inkling_dir;
    if (dir.empty())
        return nullptr;
    auto rt = std::make_unique<InklingRuntime>();
    const bool ok = va && model.token_embedding().data && model.output_proj().data &&
                    model.output_norm().data && load_inkling_weights(dir, rt->w, *va, /*with_vocab=*/false) &&
                    rt->f.init(rt->w, *va, max_tokens, max_ctx);
    if (!ok)
        throw std::runtime_error("inkling2_loop: " + dir + " could not be loaded (see the error above)");
    return rt;
}

int GraphExecutor::run_inkling_(const InferenceState& state, Tensor& h, int n, cudaStream_t stream) {
    if (!inkling_)
        return model_->config().n_layers;
    // Host read of row 0's position: this path is eager and serves one sequence.
    int pos0 = 0;
    inkling_->ids.resize(static_cast<size_t>(n));
    IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
    if (cudaMemcpy(&pos0, state.positions, sizeof pos0, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(inkling_->ids.data(), state.token_ids, static_cast<size_t>(n) * sizeof(int32_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        h.qtype != QType::F16 ||
        !inkling_->f.run(static_cast<half*>(h.data), n, pos0, stream, inkling_->ids.data()))
        throw std::runtime_error("inkling2_loop: forward failed at position " + std::to_string(pos0));
    return 0;
}

// Prefix reuse: the position the next prefill resumes at (0 for every other arch).
int GraphExecutor::inkling_resume(std::span<const int32_t> tokens, cudaStream_t stream) {
    return inkling_ ? inkling_->f.rewind(tokens, stream) : 0;
}

}  // namespace imp
