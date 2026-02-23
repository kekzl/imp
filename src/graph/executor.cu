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
#include "quant/quant_gemm.h"
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
// For NONE/F16/BF16: uses standard cuBLAS gemm.
static void gemm_dispatch(const Tensor& input, const Tensor& weight,
                           const Tensor& scales, GGMLQuantType qtype,
                           Tensor& output, cudaStream_t stream) {
    if (qtype == GGMLQuantType::Q4_0 || qtype == GGMLQuantType::Q4_1) {
        // weight is [N, K/2] packed nibbles, scales is [N, num_groups]
        quant_gemm_int4(input, weight, scales, output, stream);
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

bool GraphExecutor::init(const Model& model, DType compute_dtype, bool use_pdl) {
    if (initialized_) {
        free_buffers();
    }

    model_ = &model;
    compute_dtype_ = compute_dtype;
    use_pdl_ = use_pdl;

    const auto& cfg = model.config();

    // Cap max_tokens at a reasonable limit.
    max_tokens_ = std::min(cfg.max_seq_len, 4096);
    if (max_tokens_ <= 0) {
        max_tokens_ = 4096;
    }

    allocate_buffers(max_tokens_);

    // Enable Programmatic Dependent Launch on custom kernels if requested.
    // The PDL attribute is sticky: once set, it applies to all future launches
    // of that kernel. cuBLAS/cuBLASLt kernels already have PDL support built-in
    // when using CUDA 13.1; our custom kernels need explicit annotation.
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

void GraphExecutor::allocate_buffers(int max_tokens) {
    const auto& cfg = model_->config();
    int d   = cfg.d_model;
    int ff  = cfg.d_ff;
    int v   = cfg.vocab_size;
    int nh  = cfg.n_heads;
    int nkv = cfg.n_kv_heads;
    int hd  = cfg.head_dim > 0 ? cfg.head_dim : (d / nh);
    size_t es = dtype_size(compute_dtype_);

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    size_t hidden_sz     = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t residual_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t norm_out_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t q_sz          = align256(static_cast<size_t>(max_tokens) * nh * hd * es);
    size_t k_sz          = align256(static_cast<size_t>(max_tokens) * nkv * hd * es);
    size_t v_sz          = align256(static_cast<size_t>(max_tokens) * nkv * hd * es);
    size_t attn_out_sz   = align256(static_cast<size_t>(max_tokens) * nh * hd * es);
    size_t proj_out_sz   = align256(static_cast<size_t>(max_tokens) * d * es);
    size_t gate_out_sz   = align256(static_cast<size_t>(max_tokens) * ff * es);
    size_t up_out_sz     = align256(static_cast<size_t>(max_tokens) * ff * es);
    size_t swiglu_out_sz = align256(static_cast<size_t>(max_tokens) * ff * es);
    size_t ffn_out_sz    = align256(static_cast<size_t>(max_tokens) * d * es);
    // Logits are always FP32 for accurate sampling (softmax, argmax)
    size_t logits_sz     = align256(static_cast<size_t>(max_tokens) * v * sizeof(float));

    size_t total = hidden_sz + residual_sz + norm_out_sz +
                   q_sz + k_sz + v_sz + attn_out_sz + proj_out_sz +
                   gate_out_sz + up_out_sz + swiglu_out_sz + ffn_out_sz +
                   logits_sz;

    cudaError_t err = cudaMalloc(&workspace_, total);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate executor workspace (%zu bytes): %s",
                      total, cudaGetErrorString(err));
        return;
    }
    workspace_size_ = total;

    IMP_LOG_INFO("Executor workspace: %.2f MiB", total / (1024.0 * 1024.0));

    // Bump-allocate tensor views into the workspace.
    // All tensors are 2-D: [max_tokens, cols].
    char* ptr = static_cast<char*>(workspace_);

    auto make = [&](int64_t cols, size_t aligned_sz) -> Tensor {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), cols};
        Tensor t(ptr, compute_dtype_, 2, shape, /*on_device=*/true);
        ptr += aligned_sz;
        return t;
    };

    hidden_     = make(d,        hidden_sz);
    residual_   = make(d,        residual_sz);
    norm_out_   = make(d,        norm_out_sz);
    q_          = make(nh * hd,  q_sz);
    k_          = make(nkv * hd, k_sz);
    v_          = make(nkv * hd, v_sz);
    attn_out_   = make(nh * hd,  attn_out_sz);
    proj_out_   = make(d,        proj_out_sz);
    gate_out_   = make(ff,       gate_out_sz);
    up_out_     = make(ff,       up_out_sz);
    swiglu_out_ = make(ff,       swiglu_out_sz);
    ffn_out_    = make(d,        ffn_out_sz);
    // Logits: FP32 for accurate sampling
    {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(v)};
        logits_ = Tensor(ptr, DType::FP32, 2, shape, /*on_device=*/true);
        ptr += logits_sz;
    }

    // Allocate MoE buffers if the model uses Mixture-of-Experts layers.
    if (cfg.n_experts > 0 && cfg.n_experts_active > 0) {
        allocate_moe_buffers(max_tokens);
    }
}

void GraphExecutor::free_buffers() {
    free_moe_buffers();
    if (workspace_) {
        cudaFree(workspace_);
        workspace_ = nullptr;
        workspace_size_ = 0;
    }
    initialized_ = false;
}

void GraphExecutor::allocate_moe_buffers(int max_tokens) {
    const auto& cfg = model_->config();
    int d       = cfg.d_model;
    int eff     = cfg.expert_d_ff;
    int ne      = cfg.n_experts;
    int top_k   = cfg.n_experts_active;
    size_t es   = dtype_size(compute_dtype_);
    int expanded = max_tokens * top_k;  // each token is routed to top_k experts

    auto align256 = [](size_t x) -> size_t { return (x + 255) & ~size_t(255); };

    // Gate logits: [max_tokens, n_experts] in FP32 (required by moe_topk_gating)
    size_t gate_logits_sz   = align256(static_cast<size_t>(max_tokens) * ne * sizeof(float));
    // Gathered tokens: [expanded, d_model] in compute_dtype
    size_t gathered_sz      = align256(static_cast<size_t>(expanded) * d * es);
    // Expert gate projection: [expanded, expert_d_ff] in compute_dtype
    size_t expert_gate_sz   = align256(static_cast<size_t>(expanded) * eff * es);
    // Expert up projection: [expanded, expert_d_ff] in compute_dtype
    size_t expert_up_sz     = align256(static_cast<size_t>(expanded) * eff * es);
    // Expert SwiGLU output: [expanded, expert_d_ff] in compute_dtype
    size_t expert_swiglu_sz = align256(static_cast<size_t>(expanded) * eff * es);
    // Expert down projection: [expanded, d_model] in compute_dtype
    size_t expert_down_sz   = align256(static_cast<size_t>(expanded) * d * es);
    // Scatter output: [max_tokens, d_model] in FP32 (atomicAdd requires float)
    size_t scatter_out_sz   = align256(static_cast<size_t>(max_tokens) * d * sizeof(float));

    size_t total = gate_logits_sz + gathered_sz + expert_gate_sz +
                   expert_up_sz + expert_swiglu_sz + expert_down_sz +
                   scatter_out_sz;

    cudaError_t err = cudaMalloc(&moe_workspace_, total);
    if (err != cudaSuccess) {
        IMP_LOG_ERROR("Failed to allocate MoE workspace (%zu bytes): %s",
                      total, cudaGetErrorString(err));
        return;
    }
    moe_workspace_size_ = total;

    IMP_LOG_INFO("MoE workspace: %.2f MiB (top_k=%d, n_experts=%d, expert_d_ff=%d)",
                 total / (1024.0 * 1024.0), top_k, ne, eff);

    // Bump-allocate tensor views into the MoE workspace.
    char* ptr = static_cast<char*>(moe_workspace_);

    // gate_logits: FP32
    {
        int64_t shape[2] = {static_cast<int64_t>(max_tokens), static_cast<int64_t>(ne)};
        moe_gate_logits_ = Tensor(ptr, DType::FP32, 2, shape, /*on_device=*/true);
        ptr += gate_logits_sz;
    }

    // The rest use compute_dtype unless stated otherwise.
    auto make_moe = [&](Tensor& t, int64_t rows, int64_t cols, size_t aligned_sz, DType dt) {
        int64_t shape[2] = {rows, cols};
        t = Tensor(ptr, dt, 2, shape, /*on_device=*/true);
        ptr += aligned_sz;
    };

    make_moe(moe_gathered_,       expanded, d,   gathered_sz,      compute_dtype_);
    make_moe(moe_expert_gate_,    expanded, eff, expert_gate_sz,   compute_dtype_);
    make_moe(moe_expert_up_,      expanded, eff, expert_up_sz,     compute_dtype_);
    make_moe(moe_expert_swiglu_,  expanded, eff, expert_swiglu_sz, compute_dtype_);
    make_moe(moe_expert_down_,    expanded, d,   expert_down_sz,   compute_dtype_);
    make_moe(moe_scatter_out_,    max_tokens, d, scatter_out_sz,   DType::FP32);

    // Pre-allocate MoE routing buffers to avoid per-call cudaMalloc
    moe_routing_buffers_.allocate(max_tokens, ne, top_k);
}

void GraphExecutor::free_moe_buffers() {
    moe_routing_buffers_.free();
    if (moe_workspace_) {
        cudaFree(moe_workspace_);
        moe_workspace_ = nullptr;
        moe_workspace_size_ = 0;
    }
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
            static_cast<__nv_fp8_e4m3*>(cache->k_ptr(layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(kv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->k_ptr(layer, 0)),
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
            static_cast<__nv_fp8_e4m3*>(cache->v_ptr(layer, 0)),
            inv_scale,
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
#else
        write_kv_cache_fp8_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<uint8_t*>(cache->v_ptr(layer, 0)),
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
            static_cast<half*>(cache->k_ptr(layer, 0)),
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);

        // V view
        Tensor vv = view_tokens(v_, n);
        write_kv_cache_kernel<<<nblocks, threads, 0, stream>>>(
            static_cast<const half*>(vv.data),
            state.positions,
            state.block_tables,
            static_cast<half*>(cache->v_ptr(layer, 0)),
            block_stride, row_elems, kKVBlockSize, n,
            state.max_blocks_per_seq, state.n_sequences);
    }
}

// ---------------------------------------------------------------------------
// Attention sub-pass for one layer
// ---------------------------------------------------------------------------

void GraphExecutor::run_attention(int layer, const InferenceState& state,
                                  cudaStream_t stream) {
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
    gemm_dispatch(no, ly.wq, ly.wq_scales, ly.wq_qtype, qv, stream);
    gemm_dispatch(no, ly.wk, ly.wk_scales, ly.wk_qtype, kk, stream);
    gemm_dispatch(no, ly.wv, ly.wv_scales, ly.wv_qtype, vv, stream);

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

    // 6. RoPE (in-place on Q and K)
    rope_forward(q4r_t, k4r_t, state.positions, hd, cfg.rope_theta, 1.0f, stream);

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
        Tensor k_c(cache->k_ptr(layer, 0), cache_dtype, 4, cs, true);
        Tensor v_c(cache->v_ptr(layer, 0), cache_dtype, 4, cs, true);

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
    gemm_dispatch(ao, ly.wo, ly.wo_scales, ly.wo_qtype, po, stream);

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
    cudaMemcpyAsync(r.data, h.data, h.nbytes(),
                    cudaMemcpyDeviceToDevice, stream);
    rmsnorm(h, ly.ffn_norm, no, eps, stream);

    // 3. Gate and Up projections
    gemm_dispatch(no, ly.w_gate, ly.w_gate_scales, ly.w_gate_qtype, go, stream);
    gemm_dispatch(no, ly.w_up,   ly.w_up_scales,   ly.w_up_qtype,   uo, stream);

    // 4. SwiGLU: out = silu(gate) * up
    swiglu(go, uo, so, stream);

    // 5. Down projection
    gemm_dispatch(so, ly.w_down, ly.w_down_scales, ly.w_down_qtype, fo, stream);

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
    const auto& cfg = model_->config();
    const auto& ly  = model_->layer(layer);

    int n       = cur_n_tokens_;
    int d       = cfg.d_model;
    int ne      = cfg.n_experts;
    int top_k   = cfg.n_experts_active;
    int eff     = cfg.expert_d_ff;
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
    rmsnorm(h, ly.ffn_norm, no, eps, stream);

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

    // 4. Top-k gating: softmax + top-k selection + sort
    //    Use pre-allocated routing buffers to avoid per-call cudaMalloc.
    Tensor gate_logits_f32 = slice_rows(moe_gate_logits_, n);
    MoeRoutingResult routing;
    if (moe_routing_buffers_.pool) {
        moe_topk_gating(gate_logits_f32, top_k, moe_routing_buffers_, routing, stream);
    } else {
        moe_topk_gating(gate_logits_f32, top_k, routing, stream);
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
    // A_gate/A_up: views into gathered [count, d_model]
    // B_gate: ly.expert_w_gate[e]  [expert_d_ff, d_model]
    // B_up:   ly.expert_w_up[e]    [expert_d_ff, d_model]
    // C_gate: views into moe_expert_gate_ [count, expert_d_ff]
    // C_up:   views into moe_expert_up_   [count, expert_d_ff]
    // C_down: views into moe_expert_down_ [count, d_model]
    {
        std::vector<Tensor> A_gate, B_gate, C_gate;
        std::vector<Tensor> A_up,   B_up,   C_up;
        std::vector<Tensor> A_down, B_down, C_down;

        char* gathered_base     = static_cast<char*>(moe_gathered_.data);
        char* expert_gate_base  = static_cast<char*>(moe_expert_gate_.data);
        char* expert_up_base    = static_cast<char*>(moe_expert_up_.data);
        char* expert_swiglu_base= static_cast<char*>(moe_expert_swiglu_.data);
        char* expert_down_base  = static_cast<char*>(moe_expert_down_.data);

        for (int e = 0; e < ne; ++e) {
            int start = h_offsets[e];
            int count = h_offsets[e + 1] - start;
            if (count == 0) continue;

            int64_t count64 = static_cast<int64_t>(count);

            // A = gathered[start:start+count, :] -- [count, d_model]
            int64_t a_shape[2] = {count64, static_cast<int64_t>(d)};
            Tensor a_view(gathered_base + static_cast<size_t>(start) * d * es,
                          compute_dtype_, 2, a_shape, true);

            // Gate projection
            int64_t c_gate_shape[2] = {count64, static_cast<int64_t>(eff)};
            Tensor c_gate_view(expert_gate_base + static_cast<size_t>(start) * eff * es,
                               compute_dtype_, 2, c_gate_shape, true);
            A_gate.push_back(a_view);
            B_gate.push_back(ly.expert_w_gate[e]);  // [expert_d_ff, d_model]
            C_gate.push_back(c_gate_view);

            // Up projection
            int64_t c_up_shape[2] = {count64, static_cast<int64_t>(eff)};
            Tensor c_up_view(expert_up_base + static_cast<size_t>(start) * eff * es,
                             compute_dtype_, 2, c_up_shape, true);
            A_up.push_back(a_view);
            B_up.push_back(ly.expert_w_up[e]);  // [expert_d_ff, d_model]
            C_up.push_back(c_up_view);

            // Down projection (A = swiglu output, not the gathered input)
            int64_t a_down_shape[2] = {count64, static_cast<int64_t>(eff)};
            Tensor a_down_view(expert_swiglu_base + static_cast<size_t>(start) * eff * es,
                               compute_dtype_, 2, a_down_shape, true);
            int64_t c_down_shape[2] = {count64, static_cast<int64_t>(d)};
            Tensor c_down_view(expert_down_base + static_cast<size_t>(start) * d * es,
                               compute_dtype_, 2, c_down_shape, true);
            A_down.push_back(a_down_view);
            B_down.push_back(ly.expert_w_down[e]);  // [d_model, expert_d_ff]
            C_down.push_back(c_down_view);
        }

        // Gate projections: gathered @ W_gate^T -> expert_gate
        gemm_grouped(A_gate, B_gate, C_gate, stream);

        // Up projections: gathered @ W_up^T -> expert_up
        gemm_grouped(A_up, B_up, C_up, stream);

        // SwiGLU on the full contiguous expert buffers.
        // gate and up outputs are in expert-sorted order (same layout as gathered),
        // so we can apply SwiGLU as one contiguous operation over all expanded tokens.
        {
            int64_t swiglu_shape[2] = {static_cast<int64_t>(expanded),
                                        static_cast<int64_t>(eff)};
            Tensor gate_buf(moe_expert_gate_.data, compute_dtype_, 2, swiglu_shape, true);
            Tensor up_buf(moe_expert_up_.data, compute_dtype_, 2, swiglu_shape, true);
            Tensor swiglu_buf(moe_expert_swiglu_.data, compute_dtype_, 2, swiglu_shape, true);
            swiglu(gate_buf, up_buf, swiglu_buf, stream);
        }

        // Down projections: swiglu_out @ W_down^T -> expert_down
        gemm_grouped(A_down, B_down, C_down, stream);
    }

    // 7. Scatter: weighted scatter-add expert outputs back to token positions.
    //    Output is FP32 (atomicAdd on floats).
    {
        int64_t expert_out_shape[2] = {static_cast<int64_t>(expanded),
                                        static_cast<int64_t>(d)};
        Tensor expert_down_view(moe_expert_down_.data, compute_dtype_,
                                2, expert_out_shape, true);
        Tensor scatter_out = slice_rows(moe_scatter_out_, n);
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

    // All member tensors are [max_tokens_, cols]. view_tokens creates [n, cols]
    // views on the fly without modifying the members.

    // ---- Step 1: Embedding lookup ----
    Tensor h = view_tokens(hidden_, n);
    embedding_lookup(model_->token_embedding(), state.token_ids, n, h, stream);

    // ---- Step 2: Transformer layers ----
    bool has_moe = (cfg.n_experts > 0 && cfg.n_experts_active > 0);
    for (int i = 0; i < cfg.n_layers; ++i) {
        run_attention(i, state, stream);
        if (has_moe && !model_->layer(i).expert_w_gate.empty()) {
            run_moe_ffn(i, stream);
        } else {
            run_ffn(i, stream);
        }
    }

    // ---- Step 3: Final RMSNorm ----
    Tensor h_final = view_tokens(hidden_,   n);
    Tensor no_final = view_tokens(norm_out_, n);
    rmsnorm(h_final, model_->output_norm(), no_final, cfg.rms_norm_eps, stream);

    // ---- Step 4: LM head projection ----
    Tensor lg = view_tokens(logits_, n);
    gemm(no_final, model_->output_proj(), lg, 1.0f, 0.0f, stream);

    logits_out = lg;
}

int32_t GraphExecutor::forward(const InferenceState& state, cudaStream_t stream) {
    Tensor logits;
    forward_logits(state, logits, stream);

    // Sample from the last token's logits.
    // logits: [n_tokens, vocab_size] FP32. We want the last row as 1-D [vocab_size].
    Tensor last_logits = logits.slice(state.n_tokens - 1, state.n_tokens);
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
    int n_tok = state.n_tokens;
    std::vector<int32_t> tokens(n_seq);

    // Helper: flatten [1, V] to [V] for sampling
    auto flatten_logits = [](Tensor t) -> Tensor {
        int64_t vocab_shape[1] = {t.shape[t.ndim - 1]};
        return t.reshape(1, vocab_shape);
    };

    if (state.is_prefill || n_seq <= 1) {
        // Single sequence or prefill: sample from last token
        Tensor last_logits = flatten_logits(logits.slice(n_tok - 1, n_tok));
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
