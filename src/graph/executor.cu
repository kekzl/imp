#include "graph/executor.h"
#include "compute/embedding.h"
#include "compute/layernorm.h"
#include "compute/rope.h"
#include "compute/gemm.h"
#include "compute/gemm_grouped.h"
#include "compute/activation.h"
#include "compute/attention.h"
#include "compute/attention_paged.h"
#include "compute/moe_routing.h"
#include "compute/sampling.h"
#include "compute/ssm.h"
#include "quant/quant_gemm.h"
#include "quant/dequant_gpu.h"
#include "core/logging.h"
#include "memory/kv_cache.h"
#include "runtime/pdl.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#ifdef __CUDA_FP8_TYPES_EXIST__
#include <cuda_fp8.h>
#endif
#include <cstring>
#include <cmath>
#include <algorithm>

namespace imp {

// ---------------------------------------------------------------------------
// Small CUDA kernels used by the executor
// ---------------------------------------------------------------------------

// Element-wise addition: a[i] += b[i], for FP16 data
__global__ void elementwise_add_fp16_kernel(half* a, const half* b, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    // Process two elements at a time using half2 for efficiency
    int64_t n2 = n / 2;
    if (idx < n2) {
        half2* a2 = reinterpret_cast<half2*>(a);
        const half2* b2 = reinterpret_cast<const half2*>(b);
        half2 va = a2[idx];
        half2 vb = b2[idx];
        a2[idx] = __hadd2(va, vb);
    }
    // Handle the last odd element
    if (idx == 0 && (n & 1)) {
        a[n - 1] = __hadd(a[n - 1], b[n - 1]);
    }
}

// Element-wise addition: a[i] += b[i], for FP32 data
__global__ void elementwise_add_fp32_kernel(float* a, const float* b, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t i = idx; i < n; i += static_cast<int64_t>(blockDim.x) * gridDim.x) {
        a[i] += b[i];
    }
}

// Copy K/V for a set of tokens into paged KV cache blocks.
// Each token's K (or V) slice is copied to the correct slot in the right block.
//
// data_in:          [n_tokens, n_kv_heads * head_dim] contiguous
// positions:        [n_tokens] position of each token in the sequence
// block_tables:     [n_sequences, max_blocks_per_seq] or [max_blocks] block IDs
// cache_base:       base pointer of the KV pool for this layer (block 0)
// block_stride:     elements per block = kKVBlockSize * n_kv_heads * head_dim
// row_elems:        n_kv_heads * head_dim (elements per token)
// max_blocks_per_seq: stride for 2D block table (0 = legacy flat)
// n_sequences:      number of sequences in the batch
__global__ void write_kv_cache_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    half* cache_base,
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        // Batched: for decode, token i = sequence i
        int seq_idx = token_idx;  // 1 token per sequence in decode
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        // Single-sequence or legacy path
        block_id = block_tables[block_idx];
    }

    half* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                           + static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// FP16 -> FP8 E4M3 quantization + write to paged KV cache
#ifdef __CUDA_FP8_TYPES_EXIST__
__global__ void write_kv_cache_fp8_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    __nv_fp8_e4m3* cache_base,  // FP8 cache
    float inv_scale,            // 1.0 / kv_scale
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

    __nv_fp8_e4m3* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                                     + static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        float val = __half2float(src[i]) * inv_scale;
        dst[i] = __nv_fp8_e4m3(val);
    }
}
#else
// Software fallback: clamp FP16 to FP8 E4M3 range and pack to uint8_t
__global__ void write_kv_cache_fp8_kernel(
    const half* data_in,
    const int* positions,
    const int* block_tables,
    uint8_t* cache_base,        // FP8 cache (as raw bytes)
    float inv_scale,            // 1.0 / kv_scale
    int block_stride,
    int row_elems,
    int block_size,
    int n_tokens,
    int max_blocks_per_seq,
    int n_sequences
) {
    int token_idx = blockIdx.x;
    if (token_idx >= n_tokens) return;

    int pos = positions[token_idx];
    int block_idx = pos / block_size;
    int slot_in_block = pos % block_size;

    int block_id;
    if (max_blocks_per_seq > 0 && n_sequences > 1) {
        int seq_idx = token_idx;
        block_id = block_tables[seq_idx * max_blocks_per_seq + block_idx];
    } else {
        block_id = block_tables[block_idx];
    }

    uint8_t* dst = cache_base + static_cast<int64_t>(block_id) * block_stride
                               + static_cast<int64_t>(slot_in_block) * row_elems;
    const half* src = data_in + static_cast<int64_t>(token_idx) * row_elems;

    // FP8 E4M3 range: [-448, 448]
    const float fp8_max = 448.0f;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        float val = __half2float(src[i]) * inv_scale;
        val = fminf(fmaxf(val, -fp8_max), fp8_max);
        // Simple rounding: convert float to FP8 E4M3 bit pattern
        // Sign(1) | Exponent(4) | Mantissa(3)
        uint32_t bits = __float_as_uint(val);
        uint8_t sign = (bits >> 24) & 0x80;
        int exponent = ((bits >> 23) & 0xFF) - 127 + 7; // rebias to E4M3
        uint8_t mantissa = (bits >> 20) & 0x07;
        if (exponent <= 0) {
            exponent = 0;
            mantissa = 0;
        } else if (exponent >= 15) {
            exponent = 15;
            mantissa = 0x06; // max finite for E4M3 (no inf/nan encoding)
        }
        dst[i] = sign | (static_cast<uint8_t>(exponent) << 3) | mantissa;
    }
}
#endif

// FP16 -> FP32 conversion kernel (for gate logits before softmax)
__global__ void fp16_to_fp32_kernel(const half* __restrict__ in,
                                    float* __restrict__ out,
                                    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __half2float(in[idx]);
    }
}

// Add FP16 bias to each row of FP32 matrix: out[i,j] += bias[j]
// Grid: n_tokens, Block: 256, each thread handles multiple expert indices.
__global__ void add_fp16_bias_to_fp32_kernel(float* __restrict__ data,
                                              const half* __restrict__ bias,
                                              int n_tokens, int n_cols) {
    int token = blockIdx.x;
    if (token >= n_tokens) return;
    float* row = data + static_cast<int64_t>(token) * n_cols;
    for (int j = threadIdx.x; j < n_cols; j += blockDim.x) {
        row[j] += __half2float(bias[j]);
    }
}

// Scale FP32 expert weights in-place: weights[i] *= scale
__global__ void scale_fp32_kernel(float* __restrict__ data, float scale, int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= scale;
    }
}

// FP32 -> FP16 conversion kernel (for scatter output back to compute_dtype)
__global__ void fp32_to_fp16_kernel(const float* __restrict__ in,
                                    half* __restrict__ out,
                                    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(in[idx]);
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

static void elementwise_add(Tensor& a, const Tensor& b, cudaStream_t stream) {
    int64_t n = a.numel();
    if (a.dtype == DType::FP16) {
        int64_t n2 = (n + 1) / 2;
        int threads = 256;
        int blocks = static_cast<int>((n2 + threads - 1) / threads);
        elementwise_add_fp16_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<half*>(a.data),
            static_cast<const half*>(b.data),
            n);
    } else {
        int threads = 256;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        elementwise_add_fp32_kernel<<<blocks, threads, 0, stream>>>(
            static_cast<float*>(a.data),
            static_cast<const float*>(b.data),
            n);
    }
}

// Create a view of the first n_tokens rows from a [max_tokens, cols] buffer.
// Never modifies the source tensor.
static Tensor slice_rows(const Tensor& buf, int n_tokens) {
    if (n_tokens == static_cast<int>(buf.shape[0])) return buf;
    // buf.slice(0, n) returns a view with shape[0] = n, same data pointer.
    return buf.slice(0, n_tokens);
}

// Dispatch GEMM based on weight quantization type.
// For Q4_0/Q4_1: uses fused quant_gemm_int4 with packed nibbles + scales.
// For Q8_0/Q6_K (with dequant_scratch): dequant into scratch, then cuBLAS gemm.
// For NONE/F16/BF16: uses standard cuBLAS gemm.
static void gemm_dispatch(const Tensor& input, const Tensor& weight,
                           const Tensor& scales, GGMLQuantType qtype,
                           Tensor& output, void* dequant_scratch,
                           cudaStream_t stream) {
    if (qtype == GGMLQuantType::Q4_0 || qtype == GGMLQuantType::Q4_1) {
        // weight is [N, K/2] packed nibbles, scales is [N, num_groups]
        quant_gemm_int4(input, weight, scales, output, stream);
    } else if (dequant_scratch != nullptr && dequant_gpu_supported(qtype)) {
        // Raw quantized bytes on GPU — dequant into scratch, then GEMM
        int rows = static_cast<int>(weight.shape[0]);
        int cols = static_cast<int>(weight.shape[1]);

        // Check for pre-existing CUDA errors (from async ops like per-expert dequant)
        {
            cudaError_t pre_err = cudaStreamSynchronize(stream);
            if (pre_err != cudaSuccess) {
                static int pre_count = 0;
                if (++pre_count <= 5) {
                    fprintf(stderr, "imp::gemm_dispatch: PRE-EXISTING error before dequant: %s "
                            "(qtype=%u, %dx%d)\n",
                            cudaGetErrorString(pre_err), (unsigned)qtype, rows, cols);
                }
                cudaGetLastError(); // Clear the error
            }
        }

        dequant_gpu(weight.data, dequant_scratch, qtype, rows, cols, stream);
        // Check for dequant kernel errors before cuBLASLt (sticky errors cause status 14)
        cudaError_t dq_err = cudaStreamSynchronize(stream);
        if (dq_err != cudaSuccess) {
            fprintf(stderr, "imp::gemm_dispatch: dequant sync error: %s (qtype=%u, %dx%d)\n",
                    cudaGetErrorString(dq_err), (unsigned)qtype, rows, cols);
        }
        dq_err = cudaGetLastError();
        if (dq_err != cudaSuccess) {
            fprintf(stderr, "imp::gemm_dispatch: dequant kernel error: %s (qtype=%u, %dx%d)\n",
                    cudaGetErrorString(dq_err), (unsigned)qtype, rows, cols);
            return;  // Don't call cuBLASLt with poisoned CUDA state
        }
        Tensor w_fp16(dequant_scratch, DType::FP16, weight.ndim, weight.shape, true);
        gemm(input, w_fp16, output, 1.0f, 0.0f, stream);
    } else {
        // Standard FP16/BF16 GEMM
        gemm(input, weight, output, 1.0f, 0.0f, stream);
    }
}

// ---------------------------------------------------------------------------
// GraphExecutor lifetime
// ---------------------------------------------------------------------------

GraphExecutor::~GraphExecutor() {
    free_buffers();
}

bool GraphExecutor::init(const Model& model, DType compute_dtype, bool use_pdl,
                         int max_batch_size, int max_seq_len) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    use_pdl_ = use_pdl;

    const auto& cfg = model.config();

    // Detect model features for workspace sizing
    has_moe_ = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    has_ssm_ = (cfg.ssm_inner_size > 0);
    has_dense_ffn_ = (cfg.d_ff > 0);

    // Compute max expert FFN hidden dim from actual packed tensor shapes.
    // cfg.expert_d_ff may not match the actual tensor dimensions (e.g. Nemotron-H).
    max_expert_eff_ = cfg.expert_d_ff;
    if (has_moe_) {
        for (int li = 0; li < cfg.n_layers; li++) {
            const auto& L = model.layer(li);
            // gate/up packed: shape [n_experts, expert_d_ff, d_model]
            for (const auto* p : {&L.expert_gate_packed, &L.expert_up_packed}) {
                if (p->data && p->ndim >= 3)
                    max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(p->shape[1]));
            }
            // down packed: shape [n_experts, d_model, expert_d_ff]
            if (L.expert_down_packed.data && L.expert_down_packed.ndim >= 3)
                max_expert_eff_ = std::max(max_expert_eff_, static_cast<int>(L.expert_down_packed.shape[2]));
        }
        if (max_expert_eff_ != cfg.expert_d_ff) {
            IMP_LOG_WARN("expert_d_ff mismatch: config=%d, actual packed tensors=%d — using %d",
                         cfg.expert_d_ff, max_expert_eff_, max_expert_eff_);
        }
    }

    // Use engine-provided max_seq_len if given, otherwise fall back to model config.
    int effective_seq_len = (max_seq_len > 0) ? max_seq_len : cfg.max_seq_len;
    max_tokens_ = std::min(effective_seq_len, 4096);
    if (max_tokens_ <= 0) {
        max_tokens_ = 4096;
    }

    // Logits buffer only needs to hold tokens that require LM head projection:
    // - Prefill: 1 (last token only)
    // - Decode:  n_sequences (one per batch slot)
    max_logit_tokens_ = std::max(max_batch_size, 1);

    // Allocate persistent workspace (hidden, residual, norm_out, logits)
    allocate_persistent_workspace(max_tokens_);

    // Compute shared workspace sizes and allocate unified pool
    compute_shared_sizes(max_tokens_);
    allocate_shared_workspace(max_tokens_);

    // Allocate auxiliary buffers (dequant scratch, MoE staging, routing)
    allocate_auxiliary_buffers();

    // Build SSM layer index mapping
    if (has_ssm_) {
        ssm_layer_map_.resize(cfg.n_layers, -1);
        int ssm_idx = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            if (model_->layer(i).ssm_in.data != nullptr) {
                ssm_layer_map_[i] = ssm_idx++;
            }
        }
        IMP_LOG_INFO("SSM layers: %d out of %d total", ssm_idx, cfg.n_layers);
    }

    // Enable Programmatic Dependent Launch on custom kernels if requested.
    if (use_pdl_ && pdl::is_available()) {
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp16_kernel));
        pdl::enable(reinterpret_cast<const void*>(&elementwise_add_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&write_kv_cache_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp16_to_fp32_kernel));
        pdl::enable(reinterpret_cast<const void*>(&fp32_to_fp16_kernel));
        IMP_LOG_INFO("PDL enabled on executor kernels");
    } else if (use_pdl_) {
        IMP_LOG_WARN("PDL requested but not available on this device/CUDA version");
        use_pdl_ = false;
    }

    initialized_ = true;

    IMP_LOG_INFO("GraphExecutor initialized: max_tokens=%d, d_model=%d, "
                 "n_layers=%d, dtype=%s, pdl=%s",
                 max_tokens_, cfg.d_model, cfg.n_layers,
                 dtype_name(compute_dtype_),
                 use_pdl_ ? "on" : "off");
    return true;
}

// ---------------------------------------------------------------------------
// Unified workspace allocation
// ---------------------------------------------------------------------------

void GraphExecutor::compute_shared_sizes(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int ff  = cfg.d_ff;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    // Attention phase: q, k, v, attn_out, proj_out
    attn_shared_size_ = align256(static_cast<size_t>(max_tokens) * nh * hd * es)    // q
                       + align256(static_cast<size_t>(max_tokens) * nkv * hd * es)  // k
                       + align256(static_cast<size_t>(max_tokens) * nkv * hd * es)  // v
                       + align256(static_cast<size_t>(max_tokens) * nh * hd * es)   // attn_out
                       + align256(static_cast<size_t>(max_tokens) * d * es);        // proj_out

    // Dense FFN phase: gate, up, swiglu, ffn_out
    if (has_dense_ffn_ && ff > 0) {
        ffn_shared_size_ = align256(static_cast<size_t>(max_tokens) * ff * es)   // gate_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // up_out
                          + align256(static_cast<size_t>(max_tokens) * ff * es)  // swiglu_out
                          + align256(static_cast<size_t>(max_tokens) * d * es);  // ffn_out
    }

    // MoE phase
    if (has_moe_) {
        int ne    = cfg.n_experts;
        int top_k = cfg.n_experts_active;
        int eff   = max_expert_eff_;
        int expanded = max_tokens * top_k;

        moe_shared_size_ = align256(static_cast<size_t>(max_tokens) * ne * sizeof(float))  // gate_logits
                          + align256(static_cast<size_t>(expanded) * d * es)                // gathered
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_gate
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_up
                          + align256(static_cast<size_t>(expanded) * eff * es)              // expert_swiglu
                          + align256(static_cast<size_t>(expanded) * d * es)                // expert_down
                          + align256(static_cast<size_t>(max_tokens) * d * sizeof(float));  // scatter_out
    }

    // SSM phase
    if (has_ssm_) {
        int inner = cfg.ssm_inner_size;
        int n_groups = cfg.ssm_group_count;
        int state_size = cfg.ssm_state_size;
        int n_heads = cfg.ssm_dt_rank;
        int conv_channels = inner + 2 * n_groups * state_size;
        int ssm_in_dim = inner + conv_channels + n_heads;

        ssm_shared_size_ = align256(static_cast<size_t>(max_tokens) * ssm_in_dim * es)       // proj
                          + align256(static_cast<size_t>(max_tokens) * conv_channels * es)   // xBC
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // y
                          + align256(static_cast<size_t>(max_tokens) * inner * es)           // z
                          + align256(static_cast<size_t>(max_tokens) * d * es)               // out
                          + align256(static_cast<size_t>(max_tokens) * n_heads * es);        // dt
    }
}

void GraphExecutor::allocate_persistent_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int v = cfg.vocab_size;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t hidden_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t logits_sz   = align256(static_cast<size_t>(max_logit_tokens_) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz + logits_sz;

    cudaError_t err = cudaMalloc(&persistent_workspace_, total);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate persistent workspace (%zu bytes): %s",
                      total, cudaGetErrorString(err));
        return;
    }
    persistent_workspace_size_ = total;

    char* ptr = static_cast<char*>(persistent_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    hidden_   = make(d, hidden_sz);
    residual_ = make(d, residual_sz);
    norm_out_ = make(d, norm_out_sz);

    {
        int64_t shape[2] = {static_cast<int64_t>(max_logit_tokens_), static_cast<int64_t>(v)};
        logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += logits_sz;
    }

    IMP_LOG_INFO("Persistent workspace: %.2f MiB (hidden+residual+norm+logits)",
                 total / (1024.0 * 1024.0));
}

void GraphExecutor::allocate_shared_workspace(int max_tokens) {
    size_t max_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (max_shared == 0) return;

    cudaError_t err = cudaMalloc(&shared_workspace_, max_shared);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate shared workspace (%zu bytes): %s",
                      max_shared, cudaGetErrorString(err));
        return;
    }
    shared_workspace_size_ = max_shared;
    shared_workspace_max_tokens_ = max_tokens;

    IMP_LOG_INFO("Shared workspace: %.2f MiB = max(attn=%.1f, ffn=%.1f, moe=%.1f, ssm=%.1f MiB) "
                 "— saved %.2f MiB vs separate allocation",
                 max_shared / (1024.0 * 1024.0),
                 attn_shared_size_ / (1024.0 * 1024.0),
                 ffn_shared_size_ / (1024.0 * 1024.0),
                 moe_shared_size_ / (1024.0 * 1024.0),
                 ssm_shared_size_ / (1024.0 * 1024.0),
                 (attn_shared_size_ + ffn_shared_size_ + moe_shared_size_ + ssm_shared_size_
                  - max_shared) / (1024.0 * 1024.0));

    // Pre-allocate MoE routing buffers (separate from shared workspace)
    if (has_moe_) {
        const auto& cfg = model_->config();
        moe_routing_buffers_.allocate(max_tokens, cfg.n_experts, cfg.n_experts_active);
    }
}

void GraphExecutor::allocate_auxiliary_buffers() {
    const auto& cfg = model_->config();

    // Dequant scratch buffer for on-the-fly weight dequantization.
    {
        size_t max_weight_elems = 0;
        for (int i = 0; i < cfg.n_layers; i++) {
            const auto& L = model_->layer(i);
            for (const auto* w : {&L.wq, &L.wk, &L.wv, &L.wo,
                                   &L.w_gate, &L.w_up, &L.w_down,
                                   &L.w_gate_shared, &L.w_up_shared, &L.w_down_shared,
                                   &L.ssm_in, &L.ssm_out}) {
                if (w->data) max_weight_elems = std::max(max_weight_elems,
                                                          static_cast<size_t>(w->numel()));
            }
        }
        if (max_weight_elems > 0) {
            dequant_scratch_size_ = max_weight_elems * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&dequant_scratch_, dequant_scratch_size_);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate dequant scratch (%zu bytes): %s",
                              dequant_scratch_size_, cudaGetErrorString(err));
                dequant_scratch_ = nullptr;
                dequant_scratch_size_ = 0;
            } else {
                IMP_LOG_INFO("Dequant scratch buffer: %.2f MiB",
                             dequant_scratch_size_ / (1024.0 * 1024.0));
            }
        }
    }

    // MoE dequant and staging buffers
    if (has_moe_) {
        int d   = cfg.d_model;
        int eff = max_expert_eff_;

        // Dequant buffer: 1 expert slot
        {
            size_t expert_fp16_elems = static_cast<size_t>(eff) * d;
            size_t dequant_sz = expert_fp16_elems * sizeof(uint16_t);
            cudaError_t err = cudaMalloc(&moe_dequant_buf_, dequant_sz);
            if (err != cudaSuccess) {
                IMP_LOG_ERROR("Failed to allocate MoE dequant buffer (%zu bytes): %s",
                              dequant_sz, cudaGetErrorString(err));
                moe_dequant_buf_ = nullptr;
                moe_dequant_buf_size_ = 0;
            } else {
                moe_dequant_buf_size_ = dequant_sz;
                IMP_LOG_INFO("MoE dequant buffer: %.2f MiB (1 expert slot)",
                             dequant_sz / (1024.0 * 1024.0));
            }
        }

        // Staging buffer for host→device expert weight transfer
        {
            size_t max_expert_raw = 0;
            for (int li = 0; li < model_->n_layers(); li++) {
                const auto& L = model_->layer(li);
                auto check = [&](const Tensor& p, GGMLQuantType qt) {
                    if (!p.data || p.ndim < 3) return;
                    size_t rb = ggml_quant_row_bytes(qt, p.shape[2]);
                    size_t expert_raw = static_cast<size_t>(p.shape[1]) * rb;
                    max_expert_raw = std::max(max_expert_raw, expert_raw);
                };
                check(L.expert_up_packed, L.expert_up_qtype);
                check(L.expert_down_packed, L.expert_down_qtype);
                check(L.expert_gate_packed, L.expert_gate_qtype);
            }
            if (max_expert_raw > 0) {
                cudaError_t err = cudaMalloc(&moe_raw_staging_buf_, max_expert_raw);
                if (err != cudaSuccess) {
                    IMP_LOG_ERROR("Failed to allocate MoE staging buffer (%zu bytes): %s",
                                  max_expert_raw, cudaGetErrorString(err));
                    moe_raw_staging_buf_ = nullptr;
                    moe_raw_staging_size_ = 0;
                } else {
                    moe_raw_staging_size_ = max_expert_raw;
                    IMP_LOG_INFO("MoE staging buffer: %.2f MiB (1 expert raw)",
                                 max_expert_raw / (1024.0 * 1024.0));
                }
            }
        }
    }
}

void GraphExecutor::free_buffers() {
    moe_routing_buffers_.free();
    if (moe_dequant_buf_) {
        cudaFree(moe_dequant_buf_);
        moe_dequant_buf_ = nullptr;
        moe_dequant_buf_size_ = 0;
    }
    if (moe_raw_staging_buf_) {
        cudaFree(moe_raw_staging_buf_);
        moe_raw_staging_buf_ = nullptr;
        moe_raw_staging_size_ = 0;
    }
    if (dequant_scratch_) {
        cudaFree(dequant_scratch_);
        dequant_scratch_ = nullptr;
        dequant_scratch_size_ = 0;
    }
    if (shared_workspace_) {
        cudaFree(shared_workspace_);
        shared_workspace_ = nullptr;
        shared_workspace_size_ = 0;
    }
    if (persistent_workspace_) {
        cudaFree(persistent_workspace_);
        persistent_workspace_ = nullptr;
        persistent_workspace_size_ = 0;
    }
    ssm_layer_map_.clear();
    initialized_ = false;
}

// ---------------------------------------------------------------------------
// Shared workspace configuration (pure pointer arithmetic, no allocation)
// ---------------------------------------------------------------------------

void GraphExecutor::configure_attn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    q_        = make(nh * hd,  align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    k_        = make(nkv * hd, align256(static_cast<size_t>(max_tokens) * nkv * hd * es));
    v_        = make(nkv * hd, align256(static_cast<size_t>(max_tokens) * nkv * hd * es));
    attn_out_ = make(nh * hd,  align256(static_cast<size_t>(max_tokens) * nh * hd * es));
    proj_out_ = make(d,        align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_ffn_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d  = cfg.d_model;
    int ff = cfg.d_ff;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    gate_out_   = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    up_out_     = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    swiglu_out_ = make(ff, align256(static_cast<size_t>(max_tokens) * ff * es));
    ffn_out_    = make(d,  align256(static_cast<size_t>(max_tokens) * d * es));
}

void GraphExecutor::configure_moe_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d     = cfg.d_model;
    int ne    = cfg.n_experts;
    int top_k = cfg.n_experts_active;
    int eff   = max_expert_eff_;
    size_t es = dtype_size(compute_dtype_);
    int expanded = max_tokens * top_k;

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    // gate_logits: FP32
    {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(ne)};
        moe_gate_logits_ = Tensor(ptr, DType::FP32, 2, shape, true);
        ptr += align256(static_cast<size_t>(max_tokens) * ne * sizeof(float));
    }

    auto make_moe = [&](Tensor& t, int64_t rows, int64_t cols, size_t aligned_sz, DType dt) {
        int64_t shape[2] = {rows, cols};
        t = Tensor(ptr, dt, 2, shape, true);
        ptr += aligned_sz;
    };

    make_moe(moe_gathered_,      expanded, d,   align256(static_cast<size_t>(expanded) * d * es),   compute_dtype_);
    make_moe(moe_expert_gate_,   expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_up_,     expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_swiglu_, expanded, eff, align256(static_cast<size_t>(expanded) * eff * es), compute_dtype_);
    make_moe(moe_expert_down_,   expanded, d,   align256(static_cast<size_t>(expanded) * d * es),   compute_dtype_);
    make_moe(moe_scatter_out_,   max_tokens, d, align256(static_cast<size_t>(max_tokens) * d * sizeof(float)), DType::FP32);
}

void GraphExecutor::configure_ssm_workspace(int max_tokens) {
    const auto& cfg = model_->config();
    int d = cfg.d_model;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int state_size = cfg.ssm_state_size;
    int n_heads = cfg.ssm_dt_rank;
    int conv_channels = inner + 2 * n_groups * state_size;
    int ssm_in_dim = inner + conv_channels + n_heads;
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };
    char* ptr = static_cast<char*>(shared_workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, true);
        ptr += aligned_sz;
        return t;
    };

    ssm_proj_buf_ = make(ssm_in_dim,    align256(static_cast<size_t>(max_tokens) * ssm_in_dim * es));
    ssm_xBC_buf_  = make(conv_channels,  align256(static_cast<size_t>(max_tokens) * conv_channels * es));
    ssm_y_buf_    = make(inner,          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_z_buf_    = make(inner,          align256(static_cast<size_t>(max_tokens) * inner * es));
    ssm_out_buf_  = make(d,              align256(static_cast<size_t>(max_tokens) * d * es));
    ssm_dt_buf_   = make(n_heads,        align256(static_cast<size_t>(max_tokens) * n_heads * es));
}

bool GraphExecutor::resize_workspace(int new_max_tokens, cudaStream_t stream) {
    if (new_max_tokens == shared_workspace_max_tokens_ || new_max_tokens <= 0) return true;
    if (new_max_tokens > max_tokens_) new_max_tokens = max_tokens_;  // never exceed init-time max

    // Recompute shared sizes for the new token count
    int saved_max = max_tokens_;
    max_tokens_ = new_max_tokens;
    compute_shared_sizes(new_max_tokens);
    max_tokens_ = saved_max;

    size_t new_shared = std::max({attn_shared_size_, ffn_shared_size_,
                                  moe_shared_size_, ssm_shared_size_});
    if (new_shared == 0) return true;

    if (new_shared != shared_workspace_size_) {
        if (shared_workspace_) {
            cudaFreeAsync(shared_workspace_, stream);
        }
        cudaError_t err = cudaMallocAsync(&shared_workspace_, new_shared, stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("Failed to resize shared workspace to %zu bytes: %s",
                          new_shared, cudaGetErrorString(err));
            shared_workspace_ = nullptr;
            shared_workspace_size_ = 0;
            return false;
        }
        shared_workspace_size_ = new_shared;
    }
    shared_workspace_max_tokens_ = new_max_tokens;
    return true;
}

bool GraphExecutor::layer_has_attention(int layer) const {
    return model_->layer(layer).wq.data != nullptr;
}

bool GraphExecutor::layer_has_ssm(int layer) const {
    return model_->layer(layer).ssm_in.data != nullptr;
}

bool GraphExecutor::layer_has_moe(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.moe_gate.data != nullptr;
}

bool GraphExecutor::layer_has_dense_ffn(int layer) const {
    const auto& ly = model_->layer(layer);
    return ly.w_up.data != nullptr && ly.moe_gate.data == nullptr;
}

Tensor GraphExecutor::view_tokens(const Tensor& buf, int n_tokens) const {
    // buf is always [max_tokens_, cols] from allocate_buffers.
    // Return a [n_tokens, cols] view.
    return slice_rows(buf, n_tokens);
}

// ---------------------------------------------------------------------------
// KV cache write
// ---------------------------------------------------------------------------

void GraphExecutor::write_kv_cache(int layer, const InferenceState& state,
                                   cudaStream_t stream) {
    if (!state.kv_cache || !state.block_tables) return;

    // Map global layer index to KV cache layer index
    int kv_layer = layer;
    if (!kv_layer_map_.empty()) {
        kv_layer = kv_layer_map_[layer];
        if (kv_layer < 0) return;  // not an attention layer
    }

    KVCache* cache = state.kv_cache;
    int n        = state.n_tokens;
    int nkv      = cache->n_kv_heads();
    int hd       = cache->head_dim();
    int row_elems    = nkv * hd;
    int block_stride = kKVBlockSize * row_elems;

    int threads = std::min(row_elems, 256);
    int nblocks = n;   // one CUDA block per token

    bool use_fp8 = (cache->dtype() == DType::FP8_E4M3);

    if (use_fp8) {
        // FP8 E4M3 quantized KV cache write path
        float inv_scale = 1.0f;  // default; in production, scale comes from QuantConfig

        // K view: [n_tokens, nkv * hd]
        Tensor kv = view_tokens(k_, n);
#ifdef __CUDA_FP8_TYPES_EXIST__
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->k_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->k_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#endif

        // V view
        Tensor vv = view_tokens(v_, n);
#ifdef __CUDA_FP8_TYPES_EXIST__
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->v_ptr(kv_layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#endif
    } else {
        // Standard FP16 KV cache write path
        // K view: [n_tokens, nkv * hd]
        Tensor kv = view_tokens(k_, n);
        write_kv_cache_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<half*>(cache->k_ptr(kv_layer, 0)),
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);

        // V view
        Tensor vv = view_tokens(v_, n);
        write_kv_cache_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<half*>(cache->v_ptr(kv_layer, 0)),
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
    }
}

// ---------------------------------------------------------------------------
// Attention sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_attention(int layer, const InferenceState& state,
                                  cudaStream_t stream) {
    // Configure shared workspace for attention phase
    configure_attn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n   = state.n_tokens;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (cfg.d_model / nh);
    float eps = cfg.rms_norm_eps;

    // Sized views for this call (never mutates member tensors).
    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);
    Tensor qv = view_tokens(q_,        n);
    Tensor kk = view_tokens(k_,        n);
    Tensor vv = view_tokens(v_,        n);
    Tensor ao = view_tokens(attn_out_, n);
    Tensor po = view_tokens(proj_out_, n);

    // 1+2. Fused residual save + RMSNorm: residual = hidden, then
    //       norm_out = rmsnorm(hidden) -- hidden is copied to residual in-place
    //       by rmsnorm_residual (which computes norm(x) with x being the input,
    //       and writes x back to residual before normalizing).
    //       We use residual as the "save" destination: copy hidden -> residual,
    //       then compute rmsnorm(residual) into norm_out.
    //       Actually rmsnorm_residual does: x += residual, then norm(x).
    //       For attention, we want: residual = hidden, norm_out = norm(hidden).
    //       So we just copy + norm (residual is read later for add-back).
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ly.attn_norm, no, eps, stream);

    // 3. QKV projections:  [n, d] @ W^T -> [n, proj_dim]
    //    Uses quantized GEMM when weights are INT4-packed, standard cuBLAS otherwise.
    //    For Q8_0/Q6_K: dequant into scratch buffer, then cuBLAS GEMM.
    gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, qv, dequant_scratch_, stream);
    gemm_dispatch(no, ly.wk, ly.wk_scales, ly.wk_qtype, kk, dequant_scratch_, stream);
    gemm_dispatch(no, ly.wv, ly.wv_scales, ly.wv_qtype, vv, dequant_scratch_, stream);

    // 4. QK-norm (Qwen3): RMSNorm each head's Q/K vector independently
    if (ly.attn_q_norm.data != nullptr) {
        // Reshape Q from [n, nh*hd] to [n*nh, hd] for per-head RMSNorm
        int64_t q_flat[2] = {static_cast<int64_t>(n) * nh, static_cast<int64_t>(hd)};
        Tensor q_flat_view = qv.reshape(2, q_flat);
        rmsnorm(q_flat_view, ly.attn_q_norm, q_flat_view, eps, stream);
    }
    if (ly.attn_k_norm.data != nullptr) {
        int64_t k_flat[2] = {static_cast<int64_t>(n) * nkv, static_cast<int64_t>(hd)};
        Tensor k_flat_view = kk.reshape(2, k_flat);
        rmsnorm(k_flat_view, ly.attn_k_norm, k_flat_view, eps, stream);
    }

    // 5. Reshape Q, K for RoPE: [1, n, heads, hd] (rope_forward expects 4D)
    int64_t q4r[4] = {1, n, nh,  hd};
    int64_t k4r[4] = {1, n, nkv, hd};
    Tensor q4r_t = qv.reshape(4, q4r);
    Tensor k4r_t = kk.reshape(4, k4r);

    // 6. RoPE (in-place on Q and K) — supports partial RoPE via rope_dim
    rope_forward(q4r_t, k4r_t, state.positions, hd, cfg.rope_theta, 1.0f,
                 cfg.rope_dim, stream);

    // 7. Attention
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));

    if (state.is_prefill) {
        // Flash attention: [batch=1, seq, heads, hd]
        int64_t q4s[4]  = {1, n, nh,  hd};
        int64_t kv4s[4] = {1, n, nkv, hd};
        int64_t o4s[4]  = {1, n, nh,  hd};

        Tensor q4  = qv.reshape(4, q4s);
        Tensor k4  = kk.reshape(4, kv4s);
        Tensor v4  = vv.reshape(4, kv4s);
        Tensor o4  = ao.reshape(4, o4s);

        // Use runtime dispatch: TCGEN05 (sm_120) > WMMA (sm_90) > scalar
        attention_prefill_dispatch(q4, k4, v4, o4, scale, /*causal=*/true, stream);

        // Persist K, V into cache for later decode steps
        write_kv_cache(layer, state, stream);
    } else {
        // Decode: write new token's K/V to cache first
        write_kv_cache(layer, state, stream);

        // Paged attention: Q shape depends on batch size
        int n_seq = state.n_sequences;
        // For decode, n_tokens == n_sequences (one token per seq)
        int64_t qd[4] = {n_seq, 1, nh, hd};
        int64_t od[4] = {n_seq, 1, nh, hd};
        Tensor q4 = qv.reshape(4, qd);
        Tensor o4 = ao.reshape(4, od);

        KVCache* cache = state.kv_cache;
        int total_blk  = cache->total_blocks();
        DType cache_dtype = cache->dtype();
        int64_t cs[4]  = {static_cast<int64_t>(total_blk),
                          static_cast<int64_t>(kKVBlockSize),
                          static_cast<int64_t>(nkv),
                          static_cast<int64_t>(hd)};
        // Use mapped KV layer index for hybrid models (attention layers only)
        int kv_layer = layer;
        if (!kv_layer_map_.empty()) {
            kv_layer = kv_layer_map_[layer];
        }
        Tensor k_c(cache->k_ptr(kv_layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(kv_layer, 0), cache_dtype, 4, cs, true);

        if (cache_dtype == DType::FP8_E4M3) {
            // FP8 paged attention with on-the-fly dequant
            float kv_scale = 1.0f;  // default; in production, from QuantConfig
            paged_attention_decode_fp8(q4, k_c, v_c, o4,
                                        state.block_tables, state.context_lens,
                                        kKVBlockSize, scale, kv_scale,
                                        state.max_context_len, stream);
        } else {
            paged_attention_decode(q4, k_c, v_c, o4,
                                    state.block_tables, state.context_lens,
                                    kKVBlockSize, scale, state.max_context_len,
                                    stream);
        }
    }

    // 8. O projection: [n, nh*hd] @ wo^T -> [n, d]
    gemm_dispatch(ao, ly.wo, ly.wo_scales, ly.wo_qtype, po, dequant_scratch_, stream);

    // 9. Residual connection: hidden = proj_out + residual
    //    With PDL enabled, elementwise_add's tail can overlap with the next
    //    kernel on the stream (e.g., the RMSNorm at the start of FFN).
    elementwise_add(po, r, stream);
    cudaMemcpyAsync(h.data, po.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
}

// ---------------------------------------------------------------------------
// FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for dense FFN phase
    configure_ffn_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    // cur_n_tokens_ is set by forward_logits before the layer loop.
    int n   = cur_n_tokens_;
    float eps = cfg.rms_norm_eps;

    Tensor h  = view_tokens(hidden_,     n);
    Tensor r  = view_tokens(residual_,   n);
    Tensor no = view_tokens(norm_out_,   n);
    Tensor go = view_tokens(gate_out_,   n);
    Tensor uo = view_tokens(up_out_,     n);
    Tensor so = view_tokens(swiglu_out_, n);
    Tensor fo = view_tokens(ffn_out_,    n);

    // 1+2. Fused: save residual + RMSNorm
    //       residual = hidden, norm_out = rmsnorm(hidden)
    //       Nemotron-H uses attn_norm for ALL layer types (no separate ffn_norm).
    const Tensor& ffn_norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ffn_norm_w, no, eps, stream);

    // 3. Gate and Up projections
    gemm_dispatch(no, ly.w_gate, ly.w_gate_scales, ly.w_gate_qtype, go, dequant_scratch_, stream);
    gemm_dispatch(no, ly.w_up,   ly.w_up_scales,   ly.w_up_qtype,   uo, dequant_scratch_, stream);

    // 4. SwiGLU: out = silu(gate) * up
    swiglu(go, uo, so, stream);

    // 5. Down projection
    gemm_dispatch(so, ly.w_down, ly.w_down_scales, ly.w_down_qtype, fo, dequant_scratch_, stream);

    // 6. Fused residual add: hidden = ffn_out + residual
    //    Use rmsnorm_residual for the NEXT layer's norm if this isn't the last layer.
    //    For now, use elementwise_add + copy (the fused version is used where
    //    we can chain into the next norm).
    elementwise_add(fo, r, stream);
    cudaMemcpyAsync(h.data, fo.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
}

// ---------------------------------------------------------------------------
// MoE FFN sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_moe_ffn(int layer, cudaStream_t stream) {
    // Configure shared workspace for MoE phase
    configure_moe_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    int n       = cur_n_tokens_;
    int d       = cfg.d_model;
    int ne      = cfg.n_experts;
    int top_k   = cfg.n_experts_active;
    int eff     = max_expert_eff_;
    float eps   = cfg.rms_norm_eps;
    size_t es   = dtype_size(compute_dtype_);
    int expanded = n * top_k;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual: residual = hidden
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);

    // 2. RMSNorm on hidden -> norm_out
    //    Nemotron-H uses attn_norm for ALL layer types (no separate ffn_norm).
    const Tensor& norm_w = (ly.ffn_norm.data != nullptr) ? ly.ffn_norm : ly.attn_norm;
    rmsnorm(h, norm_w, no, eps, stream);

    // 3. Gate logits: norm_out [n, d_model] @ moe_gate [n_experts, d_model]^T
    //    -> gate_logits [n, n_experts]
    //
    //    moe_topk_gating requires FP32 gate logits. Our GEMM computes in FP16
    //    (matching the weight dtype). We compute into a temporary FP16 region
    //    (reuse the beginning of moe_gathered_ which is large enough for
    //    n * n_experts elements in FP16) and then cast to FP32.
    {
        // Temporary FP16 buffer for gate logits: [n, n_experts]
        // moe_gathered_ is [max_tokens * top_k, d_model] in compute_dtype,
        // which is >= n * n_experts elements for any practical model.
        int64_t gl_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(ne)};
        Tensor gate_logits_tmp(moe_gathered_.data, compute_dtype_, 2, gl_shape, true);

        gemm(no, ly.moe_gate, gate_logits_tmp, 1.0f, 0.0f, stream);

        // Cast to FP32 for the gating softmax
        Tensor gate_logits_f32 = slice_rows(moe_gate_logits_, n);
        int64_t numel = static_cast<int64_t>(n) * ne;
        int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);
        if (compute_dtype_ == DType::FP16) {
            fp16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const half*>(gate_logits_tmp.data),
                static_cast<float*>(gate_logits_f32.data),
                numel);
        } else {
            // BF16 or FP32 -- for FP32 inputs this is a plain copy,
            // for BF16 we'd need a bf16->fp32 kernel. For now only FP16 is
            // supported as compute_dtype.
            cudaMemcpyAsync(gate_logits_f32.data, gate_logits_tmp.data,
                            static_cast<size_t>(numel) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // 3b. Router bias (Nemotron/DeepSeek V3): added to sigmoid OUTPUTS for
    //     expert SELECTION only.  The raw logits are passed unbiased to sigmoid
    //     so that the resulting probabilities used as expert WEIGHTS are unbiased.
    //     See llama.cpp build_moe_ffn(): "leave probs unbiased as it's later
    //     used to get expert weights".
    const void* router_bias_ptr = ly.moe_router_bias.data;

    // 4. Top-k gating: softmax/sigmoid + top-k selection + sort
    //    Use pre-allocated routing buffers to avoid per-call cudaMalloc.
    Tensor gate_logits_f32 = slice_rows(moe_gate_logits_, n);
    MoeRoutingResult routing;
    bool use_sigmoid = cfg.moe_sigmoid_gating;
    bool norm_weights = cfg.expert_weights_norm;
    if (moe_routing_buffers_.pool) {
        moe_topk_gating(gate_logits_f32, top_k, moe_routing_buffers_, routing, stream, use_sigmoid, norm_weights, router_bias_ptr);
    } else {
        moe_topk_gating(gate_logits_f32, top_k, routing, stream, use_sigmoid, norm_weights, router_bias_ptr);
    }

    // 4b. Expert weight scaling (Nemotron: scale = 2.5)
    if (cfg.expert_weights_scale != 1.0f) {
        int64_t n_weights = static_cast<int64_t>(n) * top_k;
        int threads_s = 256;
        int blocks_s = static_cast<int>((n_weights + threads_s - 1) / threads_s);
        scale_fp32_kernel<<<blocks_s, threads_s, 0, stream>>>(
            static_cast<float*>(routing.expert_weights.data),
            cfg.expert_weights_scale, n_weights);
    }

    // 5. Gather: reorder tokens by expert assignment
    //    norm_out [n, d_model] -> gathered [expanded, d_model]
    {
        int64_t gath_shape[2] = {static_cast<int64_t>(expanded),
                                  static_cast<int64_t>(d)};
        Tensor gathered(moe_gathered_.data, compute_dtype_, 2, gath_shape, true);
        moe_gather(no, routing, gathered, stream);
    }

    // 6. Per-expert FFN via grouped GEMM
    //
    //    Read expert_offsets from device to host to determine per-expert token
    //    counts. This is a small transfer (n_experts+1 ints).
    std::vector<int32_t> h_offsets(ne + 1);
    cudaMemcpyAsync(h_offsets.data(), routing.expert_offsets.data,
                    static_cast<size_t>(ne + 1) * sizeof(int32_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Build per-expert tensor views for grouped GEMM.
    // Two paths:
    // - Pre-dequanted: expert_w_gate[e] etc. are FP16 on GPU (legacy / unquantized packed)
    // - On-the-fly dequant: expert_*_packed is raw Q6_K/Q8_0/Q4_0 on GPU, dequant per GEMM
    bool use_packed_dequant = (ly.expert_up_packed.data != nullptr &&
                               moe_dequant_buf_ != nullptr);

    // Non-gated expert FFN detection: no gate weights (Nemotron uses SiLU(up(x)) instead of SwiGLU)
    // Note: can't use expert_w_gate.empty() because loader pre-allocates the vector for all layers.
    // Instead check if gate data is actually present (packed or first unpacked entry).
    bool non_gated_experts = (ly.expert_gate_packed.data == nullptr &&
                              (ly.expert_w_gate.empty() || ly.expert_w_gate[0].data == nullptr));

    // Validate expert_d_ff matches packed tensor shapes (critical for buffer offsets)
    if (use_packed_dequant) {
        int64_t ref_eff = non_gated_experts
            ? ly.expert_up_packed.shape[1]
            : ly.expert_gate_packed.shape[1];
        int64_t down_eff = ly.expert_down_packed.shape[2];
        if (ref_eff != eff || down_eff != eff) {
            IMP_LOG_ERROR("CRITICAL: expert_d_ff mismatch! config=%d, packed.shape=%ld, "
                         "down_packed.shape[2]=%ld. Using packed tensor shapes instead.",
                         eff, (long)ref_eff, (long)down_eff);
            eff = static_cast<int>(ref_eff);
        }
    }

    // Helper: dequant one expert's weight from packed tensor into dequant scratch slot 0.
    // Returns a Tensor view into the scratch buffer with shape [rows, cols], FP16.
    // Uses slot 0 always -- safe because all ops are on the same stream, so the previous
    // GEMM reading from slot 0 completes before the next dequant writes to it.
    auto dequant_expert = [&](const Tensor& packed, GGMLQuantType qtype,
                              int expert_idx) -> Tensor {
        int64_t rows = packed.shape[1];
        int64_t cols = packed.shape[2];
        size_t row_bytes = ggml_quant_row_bytes(qtype, cols);
        size_t expert_raw = static_cast<size_t>(rows) * row_bytes;
        size_t total_raw = static_cast<size_t>(packed.shape[0]) * expert_raw;
        size_t offset = static_cast<size_t>(expert_idx) * expert_raw;

        // Bounds check: verify offset + expert_raw <= total allocated
        if (offset + expert_raw > total_raw) {
            fprintf(stderr, "imp::dequant_expert: OOB! expert %d offset=%zu + raw=%zu > total=%zu "
                    "(packed shape [%ld,%ld,%ld] qtype=%u)\n",
                    expert_idx, offset, expert_raw, total_raw,
                    (long)packed.shape[0], (long)packed.shape[1], (long)packed.shape[2],
                    (unsigned)qtype);
        }

        // Check dequant buffer is large enough
        size_t dequant_needed = static_cast<size_t>(rows) * cols * sizeof(uint16_t);
        if (dequant_needed > moe_dequant_buf_size_) {
            fprintf(stderr, "imp::dequant_expert: dequant buffer too small! "
                    "need=%zu have=%zu (rows=%ld cols=%ld)\n",
                    dequant_needed, moe_dequant_buf_size_, (long)rows, (long)cols);
        }

        const char* src;
        if (!packed.on_device) {
            // Expert weights offloaded to host — copy this expert's raw bytes to GPU staging buffer
            const char* host_ptr = static_cast<const char*>(packed.data) + offset;
            cudaMemcpyAsync(moe_raw_staging_buf_, host_ptr, expert_raw,
                            cudaMemcpyHostToDevice, stream);
            src = static_cast<const char*>(moe_raw_staging_buf_);
        } else {
            src = static_cast<const char*>(packed.data) + offset;
        }

        char* dst = static_cast<char*>(moe_dequant_buf_);  // always slot 0

        dequant_gpu(src, dst, qtype, static_cast<int>(rows), static_cast<int>(cols), stream);

        int64_t shape[2] = {rows, cols};
        return Tensor(dst, DType::FP16, 2, shape, true);
    };

    {
        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        // Per-expert gate + up projections: dequant and GEMM one expert at a time.
        // This reuses a single dequant buffer slot, avoiding the need to allocate
        // O(n_unique_active_experts) worth of dequant memory.
        for (int e = 0; e < ne; ++e) {
            int start = h_offsets[e];
            int count = h_offsets[e + 1] - start;
            if (count == 0) continue;

            int64_t count64 = static_cast<int64_t>(count);

            // Input: gathered tokens for this expert
            int64_t a_shape[2] = {count64, static_cast<int64_t>(d)};
            Tensor a_view(gathered_base + static_cast<size_t>(start) * d * es,
                          compute_dtype_, 2, a_shape, true);

            // Gate projection (only for gated experts like SwiGLU)
            if (!non_gated_experts) {
                int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                Tensor c_view(expert_gate_base + static_cast<size_t>(start) * eff * es,
                              compute_dtype_, 2, c_shape, true);
                Tensor b = use_packed_dequant
                    ? dequant_expert(ly.expert_gate_packed, ly.expert_gate_qtype, e)
                    : ly.expert_w_gate[e];
                gemm(a_view, b, c_view, 1.0f, 0.0f, stream);
            }

            // Up projection: A @ W_up^T -> C_up
            {
                int64_t c_shape[2] = {count64, static_cast<int64_t>(eff)};
                Tensor c_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                              compute_dtype_, 2, c_shape, true);
                Tensor b = use_packed_dequant
                    ? dequant_expert(ly.expert_up_packed, ly.expert_up_qtype, e)
                    : ly.expert_w_up[e];
                gemm(a_view, b, c_view, 1.0f, 0.0f, stream);
            }

        }

        // Activation: SwiGLU (gated) or relu^2 (non-gated, Nemotron-H)
        {
            int64_t act_shape[2] = {static_cast<int64_t>(expanded),
                                     static_cast<int64_t>(eff)};
            if (non_gated_experts) {
                // Non-gated: out = relu^2(up)  [Nemotron-H uses squared ReLU]
                // Write result into swiglu buffer for down projection
                Tensor up_buf(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                Tensor swiglu_buf(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                cudaMemcpyAsync(swiglu_buf.data, up_buf.data,
                                static_cast<size_t>(expanded) * eff * es,
                                cudaMemcpyDeviceToDevice, stream);
                relu_sqr_inplace(swiglu_buf, stream);
            } else {
                // Gated: out = SiLU(gate) * up
                Tensor gate_buf(moe_expert_gate_.data, compute_dtype_, 2, act_shape, true);
                Tensor up_buf(moe_expert_up_.data, compute_dtype_, 2, act_shape, true);
                Tensor swiglu_buf(moe_expert_swiglu_.data, compute_dtype_, 2, act_shape, true);
                swiglu(gate_buf, up_buf, swiglu_buf, stream);
            }
        }

        // Per-expert down projections: dequant and GEMM one expert at a time.
        for (int e = 0; e < ne; ++e) {
            int start = h_offsets[e];
            int count = h_offsets[e + 1] - start;
            if (count == 0) continue;

            int64_t count64 = static_cast<int64_t>(count);

            int64_t a_shape[2] = {count64, static_cast<int64_t>(eff)};
            Tensor a_view(expert_swiglu_base + static_cast<size_t>(start) * eff * es,
                          compute_dtype_, 2, a_shape, true);
            int64_t c_shape[2] = {count64, static_cast<int64_t>(d)};
            Tensor c_view(expert_down_base + static_cast<size_t>(start) * d * es,
                          compute_dtype_, 2, c_shape, true);
            Tensor b = use_packed_dequant
                ? dequant_expert(ly.expert_down_packed, ly.expert_down_qtype, e)
                : ly.expert_w_down[e];
            gemm(a_view, b, c_view, 1.0f, 0.0f, stream);
        }
    }

    // 7. Scatter: weighted scatter-add expert outputs back to token positions.
    //    Output is FP32 (atomicAdd on floats). Must zero-init before atomic adds.
    {
        int64_t expert_out_shape[2] = {static_cast<int64_t>(expanded),
                                        static_cast<int64_t>(d)};
        Tensor expert_down_view(moe_expert_down_.data, compute_dtype_,
                                2, expert_out_shape, true);
        Tensor scatter_out = slice_rows(moe_scatter_out_, n);
        cudaMemsetAsync(scatter_out.data, 0,
                        static_cast<size_t>(n) * d * sizeof(float), stream);
        moe_scatter(expert_down_view, routing, scatter_out, stream);
    }

    // 8. Convert scatter output FP32 -> compute_dtype into hidden
    {
        int64_t numel = static_cast<int64_t>(n) * d;
        int threads = 256;
        int blocks = static_cast<int>((numel + threads - 1) / threads);
        if (compute_dtype_ == DType::FP16) {
            fp32_to_fp16_kernel<<<blocks, threads, 0, stream>>>(
                static_cast<const float*>(moe_scatter_out_.data),
                static_cast<half*>(h.data),
                numel);
        } else {
            // FP32 compute_dtype: just copy
            cudaMemcpyAsync(h.data, moe_scatter_out_.data,
                            static_cast<size_t>(numel) * sizeof(float),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // 8b. Shared expert FFN: all tokens pass through an additional
    //     dense FFN whose output is added to the routed expert output.
    //     Reuses MoE workspace buffers (routed computation is complete).
    //     Supports both gated (Qwen3: gate+up+SwiGLU) and non-gated (Nemotron: up+SiLU).
    if (ly.w_up_shared.data != nullptr) {
        int eff_shared = static_cast<int>(ly.w_up_shared.shape[0]);
        bool shared_gated = (ly.w_gate_shared.data != nullptr);

        // Reuse moe_expert_gate_, moe_expert_up_, moe_expert_swiglu_ as scratch.
        int64_t sh_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(eff_shared)};
        Tensor sh_up(moe_expert_up_.data, compute_dtype_, 2, sh_shape, true);
        Tensor sh_swiglu(moe_expert_swiglu_.data, compute_dtype_, 2, sh_shape, true);

        // Down projection output: [n, d_model]. Reuse moe_expert_down_.
        int64_t sh_down_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(d)};
        Tensor sh_down(moe_expert_down_.data, compute_dtype_, 2, sh_down_shape, true);

        // Up projection
        gemm_dispatch(no, ly.w_up_shared, Tensor(), ly.w_up_shared_qtype,
                      sh_up, dequant_scratch_, stream);

        if (shared_gated) {
            // Gated: gate + SwiGLU
            Tensor sh_gate(moe_expert_gate_.data, compute_dtype_, 2, sh_shape, true);
            gemm_dispatch(no, ly.w_gate_shared, Tensor(), ly.w_gate_shared_qtype,
                          sh_gate, dequant_scratch_, stream);
            swiglu(sh_gate, sh_up, sh_swiglu, stream);
        } else {
            // Non-gated: relu^2(up)  [Nemotron-H uses squared ReLU]
            cudaMemcpyAsync(sh_swiglu.data, sh_up.data,
                            static_cast<size_t>(n) * eff_shared * dtype_size(compute_dtype_),
                            cudaMemcpyDeviceToDevice, stream);
            relu_sqr_inplace(sh_swiglu, stream);
        }

        // Down projection
        gemm_dispatch(sh_swiglu, ly.w_down_shared, Tensor(), ly.w_down_shared_qtype,
                      sh_down, dequant_scratch_, stream);

        // Add shared expert output to hidden (which already has routed expert output)
        elementwise_add(h, sh_down, stream);
    }

    // 9. Residual connection: hidden += residual
    elementwise_add(h, r, stream);

    // 10. Free routing result tensors only if allocated by moe_topk_gating.
    //     When using pre-allocated buffers, memory belongs to moe_routing_buffers_.
    if (routing.owns_memory) {
        cudaFree(routing.expert_indices.data);
        cudaFree(routing.expert_weights.data);
        cudaFree(routing.sorted_token_ids.data);
        cudaFree(routing.expert_offsets.data);
    }
}

// ---------------------------------------------------------------------------
// SSM (Mamba2) sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_ssm(int layer, const InferenceState& state,
                            cudaStream_t stream) {
    // Configure shared workspace for SSM phase
    configure_ssm_workspace(shared_workspace_max_tokens_);

    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);
    int n = state.n_tokens;
    float eps = cfg.rms_norm_eps;
    int inner = cfg.ssm_inner_size;
    int n_groups = cfg.ssm_group_count;
    int ssize = cfg.ssm_state_size;
    int conv_kernel = cfg.ssm_conv_kernel;
    int conv_channels = inner + 2 * n_groups * ssize;
    int n_heads = cfg.ssm_dt_rank;
    int head_dim_ssm = inner / n_heads;

    Tensor h  = view_tokens(hidden_,   n);
    Tensor r  = view_tokens(residual_, n);
    Tensor no = view_tokens(norm_out_, n);

    // 1. Save residual + RMSNorm
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ly.attn_norm, no, eps, stream);

    // 2. ssm_in projection: [n, d_model] @ ssm_in^T -> [n, ssm_in_dim]
    //    ssm_in_dim = inner(z) + conv_channels(xBC) + n_heads(dt)
    Tensor proj = view_tokens(ssm_proj_buf_, n);
    gemm_dispatch(no, ly.ssm_in, Tensor(), ly.ssm_in_qtype, proj, dequant_scratch_, stream);

    // 3. Split projection output [n, total_dim] into z, xBC, dt by column slices.
    //    proj layout: each row has [z(inner) | xBC(conv_channels) | dt(n_heads)].
    //    Use cudaMemcpy2DAsync to extract strided column slices into contiguous buffers.
    size_t es = dtype_size(compute_dtype_);
    int total_dim = inner + conv_channels + n_heads;
    size_t src_pitch = static_cast<size_t>(total_dim) * es;

    Tensor z_buf = view_tokens(ssm_z_buf_, n);
    cudaMemcpy2DAsync(z_buf.data, static_cast<size_t>(inner) * es,
                      proj.data, src_pitch,
                      static_cast<size_t>(inner) * es, n,
                      cudaMemcpyDeviceToDevice, stream);

    Tensor xBC_in = view_tokens(ssm_xBC_buf_, n);
    {
        char* xBC_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner) * es;
        cudaMemcpy2DAsync(xBC_in.data, static_cast<size_t>(conv_channels) * es,
                          xBC_src, src_pitch,
                          static_cast<size_t>(conv_channels) * es, n,
                          cudaMemcpyDeviceToDevice, stream);
    }

    Tensor dt_buf = view_tokens(ssm_dt_buf_, n);
    {
        char* dt_src = static_cast<char*>(proj.data) + static_cast<size_t>(inner + conv_channels) * es;
        cudaMemcpy2DAsync(dt_buf.data, static_cast<size_t>(n_heads) * es,
                          dt_src, src_pitch,
                          static_cast<size_t>(n_heads) * es, n,
                          cudaMemcpyDeviceToDevice, stream);
    }

    // 4. Conv1d on xBC
    //    Output reuses ssm_proj_buf_ (proj is done, safe to reuse).
    int64_t conv_out_shape[2] = {static_cast<int64_t>(n), static_cast<int64_t>(conv_channels)};
    Tensor xBC_out(ssm_proj_buf_.data, compute_dtype_, 2, conv_out_shape, true);

    int ssm_idx = ssm_layer_map_[layer];
    void* conv_st = (state.ssm_state && ssm_idx >= 0)
                    ? state.ssm_state->conv_state(state.ssm_seq_id, ssm_idx)
                    : nullptr;

    if (conv_st) {
        if (state.is_prefill) {
            ssm_conv1d_prefill(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                               xBC_out, conv_kernel, stream);
        } else {
            ssm_conv1d_decode(conv_st, xBC_in, ly.ssm_conv1d_w, ly.ssm_conv1d_b,
                              xBC_out, conv_kernel, stream);
        }
    }

    // 5. SiLU on full conv output (x, B, and C together).
    //    Mamba2 applies SiLU to the ENTIRE conv1d output, not just x.
    //    This matches causal_conv1d_fn(..., activation="silu").
    silu_inplace(xBC_out, stream);

    // 6-7. Split conv output into x/B/C per token, run SSM scan.
    int BC_size = n_groups * ssize;
    Tensor y_buf = view_tokens(ssm_y_buf_, n);

    void* h_st = (state.ssm_state && ssm_idx >= 0)
                 ? state.ssm_state->h_state(state.ssm_seq_id, ssm_idx)
                 : nullptr;

    if (h_st) {
        for (int t = 0; t < n; t++) {
            char* row = static_cast<char*>(xBC_out.data)
                        + static_cast<size_t>(t) * conv_channels * es;

            int64_t x_shape[1] = {static_cast<int64_t>(inner)};
            Tensor x_t(row, compute_dtype_, 1, x_shape, true);

            int64_t bc_shape[1] = {static_cast<int64_t>(BC_size)};
            Tensor B_t(row + static_cast<size_t>(inner) * es,
                       compute_dtype_, 1, bc_shape, true);
            Tensor C_t(row + static_cast<size_t>(inner + BC_size) * es,
                       compute_dtype_, 1, bc_shape, true);

            int64_t dt_shape[1] = {static_cast<int64_t>(n_heads)};
            Tensor dt_t(static_cast<char*>(dt_buf.data)
                        + static_cast<size_t>(t) * n_heads * es,
                        compute_dtype_, 1, dt_shape, true);

            int64_t y_shape[1] = {static_cast<int64_t>(inner)};
            Tensor y_t(static_cast<char*>(y_buf.data)
                       + static_cast<size_t>(t) * inner * es,
                       compute_dtype_, 1, y_shape, true);

            // Pass h_dtype from SSMState for FP16 h_state support
            DType h_dtype = (state.ssm_state) ? state.ssm_state->h_dtype() : DType::FP32;
            ssm_scan_decode(x_t, B_t, C_t, dt_t,
                            ly.ssm_a, ly.ssm_d, ly.ssm_dt_b, h_st,
                            y_t, n_heads, head_dim_ssm, ssize, n_groups, h_dtype, stream);
        }
    }

    // 8. Gating: y = y * SiLU(z)  [BEFORE GroupRMSNorm, per llama.cpp reference]
    silu_inplace(z_buf, stream);
    elementwise_mul(y_buf, z_buf, y_buf, stream);

    // 9. Group RMSNorm on y  [AFTER gating, per llama.cpp reference]
    group_rmsnorm(y_buf, ly.ssm_norm_w, y_buf, n_groups, eps, stream);

    // 10. ssm_out projection: [n, inner] @ ssm_out^T -> [n, d_model]
    Tensor out_buf = view_tokens(ssm_out_buf_, n);
    gemm_dispatch(y_buf, ly.ssm_out, Tensor(), ly.ssm_out_qtype, out_buf, dequant_scratch_, stream);

    // 11. Residual add: hidden = output + residual
    elementwise_add(out_buf, r, stream);
    cudaMemcpyAsync(h.data, out_buf.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);

}

// ---------------------------------------------------------------------------
// Full forward pass
// ---------------------------------------------------------------------------

void GraphExecutor::forward_logits(const InferenceState& state,
                                   Tensor& logits_out,
                                   cudaStream_t stream) {
    if (!initialized_) {
        IMP_LOG_ERROR("GraphExecutor::forward_logits called before init()");
        return;
    }

    const auto& cfg = model_->config();
    int n = state.n_tokens;
    if (n <= 0) {
        IMP_LOG_ERROR("n_tokens must be positive, got %d", n);
        return;
    }
    if (n > max_tokens_) {
        IMP_LOG_ERROR("n_tokens (%d) exceeds max_tokens (%d)", n, max_tokens_);
        return;
    }

    // Store for use by run_ffn (which doesn't receive the InferenceState).
    cur_n_tokens_ = n;

    // Clear any stale CUDA error state before starting the forward pass.
    cudaGetLastError();

    // All member tensors are [max_tokens_, cols]. view_tokens creates [n, cols]
    // views on the fly without modifying the members.

    // ---- Step 1: Embedding lookup ----
    //    For Q8_0/Q6_K embeddings, dequantizes only the needed rows on the fly.
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup(model_->token_embedding(), state.token_ids, n, h,
                     model_->tok_emb_qtype_, stream);

    // ---- Step 2: Transformer/Hybrid layers ----
    for (int i = 0; i < cfg.n_layers; ++i) {
        // Layer offloading: ensure weights are on GPU, prefetch next layer
        if (offload_mgr_) {
            offload_mgr_->ensure_layer(i, stream);
            if (i + 1 < cfg.n_layers) {
                offload_mgr_->prefetch_layer(i + 1);
            }
        }

        // Attention or SSM (mutually exclusive per layer)
        if (layer_has_attention(i)) {
            run_attention(i, state, stream);
        } else if (layer_has_ssm(i)) {
            run_ssm(i, state, stream);
        }

        // FFN: MoE, dense, or none (attention-only layers may have no FFN)
        if (layer_has_moe(i)) {
            run_moe_ffn(i, stream);
        } else if (layer_has_dense_ffn(i)) {
            run_ffn(i, stream);
        }

        // Release offloaded layer (restore host pointers)
        if (offload_mgr_) {
            offload_mgr_->release_layer(i);
        }

    }

    // ---- Step 3+4: Final RMSNorm + LM head projection ----
    // Only project the tokens that actually need sampling:
    //   Prefill: last token only (all others just populate KV cache)
    //   Decode:  all tokens (one per sequence)
    if (state.is_prefill) {
        Tensor h_last  = view_tokens(hidden_,   n).slice(n - 1, n);
        Tensor no_last = view_tokens(norm_out_,  1);
        rmsnorm(h_last, model_->output_norm(), no_last, cfg.rms_norm_eps, stream);

        Tensor lg = view_tokens(logits_, 1);
        gemm(no_last, model_->output_proj(), lg, 1.0f, 0.0f, stream);
        logits_out = lg;
    } else {
        Tensor h_final  = view_tokens(hidden_,   n);
        Tensor no_final = view_tokens(norm_out_, n);
        rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream);

        Tensor lg = view_tokens(logits_, n);
        gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);
        logits_out = lg;
    }
}

int32_t GraphExecutor::forward(const InferenceState& state, cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);

    // Check for CUDA errors after the forward pass
    {
        cudaError_t err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("CUDA error after forward: %s", cudaGetErrorString(err));
        }
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            IMP_LOG_ERROR("CUDA last error: %s", cudaGetErrorString(err));
        }
    }

    // Sample from the last token's logits.
    // forward_logits returns [1, V] for prefill, [n, V] for decode.
    // For single-token forward, always use row 0.
    Tensor last_logits = logits.slice(0, 1);
    int64_t vocab_shape[1] = {last_logits.shape[1]};
    last_logits = last_logits.reshape(1, vocab_shape);

    int32_t token;
    if (state.temperature <= 0.0f || state.top_k == 1) {
        token = sample_greedy(last_logits, stream);
    } else {
        int top_k  = state.top_k > 0  ? state.top_k  : 50;
        float top_p = state.top_p > 0.0f ? state.top_p : 1.0f;
        unsigned int seed = state.seed >= 0
                                ? static_cast<unsigned int>(state.seed)
                                : 42u;
        token = sample_topk_topp(last_logits, top_k, top_p,
                                 state.temperature, seed, stream);
    }

    return token;
}

std::vector<int32_t> GraphExecutor::forward_batch(const InferenceState& state,
                                                  cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);

    int n_seq = state.n_sequences;
    std::vector<int32_t> tokens(n_seq);

    // Helper: flatten [1, V] to [V] for sampling
    auto flatten_logits = [](Tensor t) -> Tensor {
        int64_t vocab_shape[1] = {t.shape[t.ndim - 1]};
        return t.reshape(1, vocab_shape);
    };

    if (state.is_prefill || n_seq <= 1) {
        // Single sequence or prefill: logits is [1, V] (forward_logits already sliced)
        Tensor last_logits = flatten_logits(logits.slice(0, 1));
        tokens[0] = (state.temperature <= 0.0f || state.top_k == 1)
            ? sample_greedy(last_logits, stream)
            : sample_topk_topp(last_logits,
                               state.top_k > 0 ? state.top_k : 50,
                               state.top_p > 0.0f ? state.top_p : 1.0f,
                               state.temperature,
                               state.seed >= 0 ? static_cast<unsigned int>(state.seed) : 42u,
                               stream);
    } else {
        // Batched decode: n_tokens == n_sequences, each row is one sequence's logits
        for (int i = 0; i < n_seq; i++) {
            Tensor seq_logits = flatten_logits(logits.slice(i, i + 1));
            tokens[i] = (state.temperature <= 0.0f || state.top_k == 1)
                ? sample_greedy(seq_logits, stream)
                : sample_topk_topp(seq_logits,
                                   state.top_k > 0 ? state.top_k : 50,
                                   state.top_p > 0.0f ? state.top_p : 1.0f,
                                   state.temperature,
                                   state.seed >= 0 ? static_cast<unsigned int>(state.seed + i) : (42u + i),
                                   stream);
        }
    }

    return tokens;
}

} // namespace imp
