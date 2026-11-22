// GraphExecutor's Inkling2-Loop branch: the standard path embeds, norms and projects; the 48 layer
// applications run in exec/inkling_forward.h.

#include "exec/executor.h"

#include "core/logging.h"
#include "exec/inkling_forward.h"
#include "model/inkling_weights.h"

namespace imp {

bool GraphExecutor::init_inkling(const std::string& dir, int max_ctx) {
    inkling_weights_ = std::make_unique<InklingWeights>();
    if (!load_inkling_weights(dir, *inkling_weights_, /*with_vocab=*/false))
        return false;
    inkling_ = std::make_unique<InklingForward>();
    return inkling_->init(*inkling_weights_, max_tokens_, max_ctx);
}

void GraphExecutor::run_inkling_(const InferenceState& state, Tensor& h, int n, cudaStream_t stream) {
    // Host read of row 0's position: this path is eager and serves one sequence.
    int pos0 = 0;
    cudaStreamSynchronize(stream);
    if (cudaMemcpy(&pos0, state.positions, sizeof pos0, cudaMemcpyDeviceToHost) != cudaSuccess ||
        h.qtype != QType::F16 || !inkling_->run(static_cast<half*>(h.data), n, pos0, stream))
        IMP_LOG_ERROR("inkling2_loop: forward failed (%d rows at position %d)", n, pos0);
}

}  // namespace imp
