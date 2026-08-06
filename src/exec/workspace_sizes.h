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
#include <string>
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
    // Shape inputs for the FP8 activation reduction scratch. Mirrors the
    // max_dim ladder in executor_workspace_buffers.cu — replicated rather than
    // approximated, for the same reason exec_max_tokens replicates the as-built
    // GDN cap: a demand function that merely resembles the site is a plan that
    // fits by luck.
    bool use_fp8_prefill = false;
    int n_heads = 0;
    int head_dim = 0;
    int ssm_inner_size = 0;
    int ssm_conv_channels = 0;
    int ssm_dt_rank = 0;
    // Columns the Qwen3.5-style attention output gate writes. The gate half of a
    // fused q_proj is split out into a buffer BORROWED from the SSM z buffer
    // (executor_attention.cu), so the two sizings are coupled. 0 when the model
    // has no gate, i.e. when no layer's q_proj is wider than n_heads * head_dim.
    int attn_gate_cols = 0;
    // (N, logical K) of every weight the MMVQ / dequant paths can see. Logical
    // K means NVFP4's packed byte dim already doubled.
    std::vector<std::pair<int64_t, int64_t>> weights;
    // Raw facts for the dp4a input-staging cluster (q8_1 + d8, the FFN block
    // mask, the Q4_K/Q5_K prefill pair). Deliberately NOT derived from
    // `weights`: the site scans a narrower tensor list and reads shape[1] /
    // shape[2] RAW — no NVFP4 doubling, no output_proj, no gdn_gate — so
    // reusing exec_max_weight_k() here would size the scratch from a K the
    // kernels never see. Replicated rather than approximated, same rule as
    // exec_max_tokens.
    // IMMA (INT8 tensor-core) prefill activation scratch, mmq_q8_imma.cu. TWO
    // maxima, kept apart on purpose: the dense route runs at max_tokens rows
    // against a dense K, the MoE route at max_tokens * top_k rows against an
    // expert K, and the two never co-occur. Multiplying rows_max by K_max would
    // charge ~5x what either route can actually reach (measured 2026-07-31:
    // dense 8B peaks at M=2048 K=12288, the 30B MoE at M=16384 K=2048).
    int imma_dense_max_k = 0;        // max shape[1] over Q8_0 dense weights
    int imma_expert_max_k = 0;       // max shape[2] over IMMA-routed expert stacks
    int mmvq_max_k = 0;              // max dense shape[1] / expert-packed shape[2]
    int mmvq_max_expert_down_k = 0;  // max expert_down_packed shape[2], 0 when dense
    int n_experts_active = 0;        // top_k; the MoE down projection quantizes that many
    bool has_sub5bit_dense = false;  // any dense Q4_K/Q5_K weight — gates the prefill pair
    // MLA (DeepSeek) QKV scratch, and the opt-in absorbed-decode latent cache.
    // n_layers is only read by the absorb term, which is per-layer.
    int n_layers = 0;
    int kv_lora_rank = 0;  // > 0 IS the is_mla() predicate (model_config.h)
    int qk_rope_head_dim = 0;
    int qk_nope_head_dim = 0;
    int v_head_dim = 0;
    bool mla_absorb = false;  // runtime_config().attention.mla_absorb
    // Chunk-capture K/V scratch (executor_workspace_buffers.cu). The site takes
    // the MAX over the per-layer arrays rather than the config scalars —
    // hybrids have layers with different kv-head counts — so carry both maxima
    // rather than n_kv_heads/head_dim, which would fit by luck on uniform
    // models and under-reserve on the others.
    int kv_heads_max = 0;
    int head_dim_max = 0;
    // min(speculative.capture_ctx_cap, max_seq_len) as engine_spec_capture.cpp
    // computes it. 0 = the captured-verify path is off, charge nothing.
    int capture_ctx_cap = 0;
};

// Replicated constants, pinned against their definitions by static_asserts in
// executor_workspace_buffers.cu. They live here so this stays CUDA-free.
constexpr size_t kExecBlockQ81Bytes = 48;  // sizeof(block_q8_1), compute/gemm.h
constexpr int kExecKVBlockSize = 16;       // kKVBlockSize, memory/kv_cache.h
// kGemmCublasWorkspaceBytes + kGemmBenchScratchBytes, compute/gemm.h.
constexpr size_t kExecCublasWorkspaceBytes = 64ull << 20;
constexpr size_t kExecBenchScratchBytes = 32ull << 20;
// kGrouped3xStagingBytes + kGrouped3xWorkspaceBytes,
// compute/gemm_cutlass_grouped_3x.h.
constexpr size_t kExecGrouped3xStagingBytes = 1ull << 20;
constexpr size_t kExecGrouped3xWorkspaceBytes = 1ull << 20;
// mmq_q8_imma.cu's split-K partial slices. A BOUND, not a measurement: the path
// needs M * N * used floats, runs only when M <= 32, and its tile guard
// (`n_tiles * S < 256`, checked before the last doubling, kBN = 128) caps
// N * used at 512 * 128 — so the buffer cannot exceed 32 * 65536 * 4 B.
// (Measured peak on Qwen3-8B-Q8_0: 6.0 MiB.)
constexpr size_t kExecImmaSplitkBytes = 8ull << 20;

struct ExecT2Demand {
    // MMVQ (Q8_1-input GEMV) scratch: max_tokens * ceil(maxK/32) * 36 * 2.
    size_t mmvq_scratch = 0;
    // Sampling result scratch: 2 (parity) * max_logit_tokens * SAMPLE_SCRATCH.
    size_t sample_scratch = 0;
    // FP8 activation reduction scratch: the per-block absmax array plus two
    // scalars. Charged only when FP8 prefill is on.
    size_t fp8_reduction = 0;
    // Batched-MoE pointer/scale arrays, all sized from n_experts and all "< 4 KB"
    // by their own comments. Small, but the point of charging them is that the
    // arena is SIZED for its tenants rather than absorbing them into slack.
    size_t moe_arrays = 0;
    // gemm_nvfp4 dequant workspace: the largest single NVFP4 dequant target,
    // capped at 512 MiB (targets above the cap are served by the uncapturable
    // path, exactly as allocate_nvfp4_dequant_workspace decides).
    size_t nvfp4_dequant = 0;
    // dp4a input staging: the q8_1/d8 decode pair, the FFN sparsity mask and
    // the Q4_K/Q5_K dense-prefill pair. One sizing family (the max-K scan), one
    // degradation contract ("dp4a path disabled"), so they migrate as a unit.
    size_t quant_scratch = 0;
    // Split-K paged-attention partials: max_batch x n_heads x splits x (2+hd).
    size_t splitk_scratch = 0;
    // DRY-penalty staging (compute/sampling_penalties.cu): max_seq_len token
    // ids + their float penalties, taken once at engine init.
    size_t dry_penalty = 0;
    // MLA QKV scratch (kv_a + latent + k_rope + kv_b), plus the absorbed-decode
    // latent cache and its score scratch when attention.mla_absorb is on. Zero on
    // every non-MLA model, which is every model except DeepSeek's.
    size_t mla_scratch = 0;
    // The cuBLASLt workspace and the algo-selection bench scratch that
    // gemm_init() takes (compute/gemm.cu). A fixed 96 MiB, independent of the
    // model: cuBLASLt's workspace is a ceiling offered to its heuristic, not a
    // shape-derived size. Charged unconditionally because gemm_init() runs for
    // every model, encoder-only included.
    size_t cublas_workspace = 0;
    // CUTLASS 3.x grouped-GEMM staging + workspace (compute/gemm_cutlass_grouped_3x.cu),
    // taken by gemm_grouped_3x_nvfp4_prewarm(). MoE models only — engine.cpp
    // gates the prewarm on profile().is_moe, and a model without experts cannot
    // reach the grouped path at all.
    size_t grouped3x = 0;
    // IMMA prefill activation scratch + its split-K partials (mmq_q8_imma.cu).
    // Charged whenever the model carries weights the IMMA routes can take;
    // zero on native-NVFP4 models, which never enter them.
    size_t imma_scratch = 0;
    // Chunk-capture K/V pair for the captured speculative verify. Sized from
    // the capture context cap, charged only when that path is eligible.
    size_t chunk_capture = 0;

    size_t total() const {
        return mmvq_scratch + nvfp4_dequant + sample_scratch + moe_arrays + fp8_reduction + quant_scratch +
               splitk_scratch + mla_scratch + dry_penalty + cublas_workspace + grouped3x + imma_scratch;
    }

    // "mmvq 21.1 + nvfp4 192.0 + sample 1.0 + moe 0.00 MiB". Lives here rather
    // than at the log site so adding a tenant does not grow engine.cpp, which
    // sits on its file-size hard limit.
    std::string describe() const;
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

// The batch, the FP8 prefill decision and attention.mla_absorb are engine and
// runtime-config facts, not model facts, and three tenants are sized from them —
// so the caller passes them rather than this header reaching for a global (there
// is no process-global RuntimeConfig; it is per-Engine by design).
ExecT2Demand exec_t2_demand(const Model& model, int max_seq_len, int max_batch_size,
                            bool use_fp8_prefill, bool mla_absorb = false,
                            int capture_ctx_cap = 0);
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

// Columns the SSM z buffer must hold. It serves two tenants: the recurrent z /
// gate projection (ssm_inner_size) and the attention output gate, which borrows
// the same allocation. On every hybrid staged today the two are exactly equal
// (4096 == 4096 on Qwen3.6-35B-A3B and Ornith-35B, 6144 == 6144 on the 27B), so
// the borrow currently fits by arithmetic coincidence rather than by
// construction — a checkpoint whose recurrent inner size is narrower than
// n_heads * head_dim would overrun it, silently and only in prefill.
int exec_ssm_z_cols(const ExecShape& shape);
int exec_ssm_z_cols(const Model& model);

// Width of the attention output gate, 0 when the model has no gate. Both SSM z
// sizings call this so the buffer and its charge cannot drift apart.
int exec_attn_gate_cols(const Model& model);

// The (rows, K) pair the `imma_scratch` charge was computed from — the larger
// of the dense and MoE routes. engine.cpp hands it to
// mmq_q8_imma_preallocate() so the scratch is taken ONCE at the charged size.
// Growing it incrementally instead would strand every intermediate slice in the
// bump arena, and the sum of a growth staircase is not what was charged.
struct ImmaScratchShape {
    int rows = 0;
    int k = 0;
    bool valid() const { return rows > 0 && k > 0; }
};
ImmaScratchShape exec_imma_scratch_shape(const ExecShape& shape, int max_seq_len);
ImmaScratchShape exec_imma_scratch_shape(const Model& model, int max_seq_len);

}  // namespace imp
