#pragma once

// Exact, pre-upload demand for the engine-persistent (T2) tenants whose size
// is a pure function of the model shape and the engine config
// (docs/MEMORY_ARCHITECTURE.md B5 point 1, A7 step 4b/6).
//
// Why this exists: the T2 arena acquires its Region when it is OPENED, so
// sizing it correctly is what reserves those bytes against everything that
// allocates later — including the pre-dequant cache build, which expands into
// whatever free VRAM it finds and, on gpt-oss-20b at server defaults, drove
// the card to 0.0 MiB free and left a 31.64 MiB workspace to be rejected
// (AUDIT B23). Getting the capacity right is therefore not tuning; it is the
// mechanism.
//
// Pure and CUDA-free so it can be unit-tested on the CPU lane. It deliberately
// covers only the two tenants whose formulas are exact today; the arena keeps
// a floor for the rest until the remaining tenants are migrated.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace imp {

class Model;
struct EngineConfig;

// The shape facts the two functions below actually consume. Extracting this
// makes them testable in the CPU lane — the header has always claimed they were
// "pure and CUDA-free so it can be unit-tested", and until this existed they
// took a Model and nothing tested them at all. The arena is sized from these
// numbers, so a wrong one is the #1103 failure class (under-reservation), which
// is not something to leave to review.
struct ExecShape {
    int max_seq_len_cfg = 0;
    bool is_ssm = false;
    bool is_moe = false;
    int n_experts = 0;
    int expert_d_ff = 0;
    int d_ff = 0;
    int d_model = 0;
    // max_logit_tokens as executor_workspace.cu computes it: max(max_batch, 8).
    // NOT the context length — I mistook it for that once and wrongly ruled the
    // sampling scratch out as ~115 MiB when it is ~1 MiB (AUDIT B52/B53).
    int max_batch_size = 1;
    // (N, logical K) of every weight the MMVQ / dequant paths can see. Logical
    // K means NVFP4's packed byte dim already doubled.
    std::vector<std::pair<int64_t, int64_t>> weights;
};

struct ExecT2Demand {
    // MMVQ (Q8_1-input GEMV) scratch: max_tokens * ceil(maxK/32) * 36 * 2.
    size_t mmvq_scratch = 0;
    // Sampling result scratch: 2 (parity) * max_logit_tokens * SAMPLE_SCRATCH.
    size_t sample_scratch = 0;
    // gemm_nvfp4 dequant workspace: the largest single NVFP4 dequant target,
    // capped at 512 MiB (targets above the cap are served by the uncapturable
    // path, exactly as allocate_nvfp4_dequant_workspace decides).
    size_t nvfp4_dequant = 0;

    size_t total() const { return mmvq_scratch + nvfp4_dequant + sample_scratch; }
};

// max_tokens as GraphExecutor will compute it: min(max_seq_len, 4096), then
// capped to 2048 for SSM+MoE hybrids.
//
// NOTE: this replicates the AS-BUILT condition, which reads has_gdn_ before it
// is assigned and therefore fires only for SSM+MoE, never for pure GDN
// (AUDIT B18). Replicating the *intended* condition would size the arena at
// T=2048 while the executor allocates at T=4096 — a 2x under-reservation.
int exec_max_tokens(const Model& model, int max_seq_len);

// Largest logical K across the weight tensors the MMVQ / dequant paths see.
int exec_max_weight_k(const Model& model);

// The batch is an engine decision, not a model fact, and the sampling scratch is
// sized from it — so the caller passes it rather than assembling an ExecShape.
ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len, int max_batch_size);

// Batch defaults to 1 (i.e. the max_logit_tokens floor of 8).
ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len);

// ── Pure core ────────────────────────────────────────────────────────
// The Model overloads above are thin wrappers over these. Everything the
// arithmetic depends on is in ExecShape, so these run in the CPU lane.
ExecShape exec_shape_of(const Model& model);

int exec_max_tokens(const ExecShape& shape, int max_seq_len);
int exec_max_weight_k(const ExecShape& shape);
ExecT2Demand exec_t2_demand(const ExecShape& shape, int max_seq_len);

}  // namespace imp
