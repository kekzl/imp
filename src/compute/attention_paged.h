#pragma once

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstdint>

namespace imp {

// Q8_1 epilogue of the split-K reduce (batch 1, head_dim % 32 == 0): the reduce also writes O as
// Q8_1 blocks (q8, d8), bit-identical to quantize_fp16_to_q8_1(O), and sets `written`. A launcher
// argument per call (audit D-9): no armed state outlives the call that was handed it.
struct PagedQ8Epilogue {
    void* q8 = nullptr;  // block_q8_1 [head_dim * n_heads / 32]
    float* d8 = nullptr;
    bool written = false;
};

// The reduce takes it at most once per call, at batch 1 and head_dim % 32 == 0; marks it written.
[[nodiscard]] inline bool paged_q8_epilogue_take(PagedQ8Epilogue* e, int batch_size, int head_dim) {
    if (e == nullptr || e->q8 == nullptr || e->written || batch_size != 1 || head_dim % 32 != 0)
        return false;
    e->written = true;
    return true;
}

// Arguments of the raw-pointer paged decode launchers below (audit D-11): built with designated
// initializers, so a swapped n_heads / n_kv_heads / head_dim / block_size reads wrong at the call
// site instead of compiling silently. A launcher ignores the fields its kernel has no use for.
struct PagedDecodeArgs {
    const half* Q = nullptr;        // [batch, 1, n_heads, head_dim]
    const void* K_cache = nullptr;  // half (F16 KV) or uint8_t (FP8, NVFP4) blocks
    const void* V_cache = nullptr;
    const uint8_t* K_scales = nullptr;  // NVFP4 E4M3 group scales
    const uint8_t* V_scales = nullptr;
    half* O = nullptr;                  // unsplit output
    float* partial = nullptr;           // split-K (m, l, o) partials for paged_attention_launch_reduce
    const int* block_tables = nullptr;  // [batch, max_num_blocks]
    const int* context_lens = nullptr;  // [batch]
    int batch_size = 0;
    int n_heads = 0;
    int n_kv_heads = 0;
    int head_dim = 0;  // the FP8 tile / four-token launchers are HD=128 only and do not read it
    int block_size = 0;
    float scale = 0.0f;
    float kv_scale = 1.0f;  // FP8 per-tensor KV scale
    int max_num_blocks = 0;
    int num_splits = 1;
    int sliding_window = 0;
    float softcap = 0.0f;
    const half* attn_sinks = nullptr;
    int heads_per_cta = 0;  // 0 = auto
    cudaStream_t stream = nullptr;
};

// Split-K partial scratch of one paged decode call (audit D-9), a per-call field: null = no split-K.
// A split holds batch * n_heads * (2 + head_dim) floats; launchers cap num_splits to the size.
struct PagedSplitKScratch {
    void* ptr = nullptr;
    size_t size = 0;
};

// Bytes of num_splits (m, l, o[head_dim]) partials per (batch, head).
constexpr size_t paged_splitk_bytes(int batch_size, int n_heads, int head_dim, int num_splits) {
    return static_cast<size_t>(batch_size) * n_heads * num_splits * (2 + head_dim) * sizeof(float);
}

// F16 decode rule: num_splits when it fits, else the most splits that fit, at least 1; null = 1.
[[nodiscard]] inline int paged_splitk_cap(const PagedSplitKScratch& s, int batch_size, int n_heads,
                                          int head_dim, int num_splits) {
    if (s.ptr == nullptr)
        return 1;
    if (paged_splitk_bytes(batch_size, n_heads, head_dim, num_splits) <= s.size)
        return num_splits;
    const int max_splits = static_cast<int>(s.size / paged_splitk_bytes(batch_size, n_heads, head_dim, 1));
    return std::max(std::min(num_splits, std::max(max_splits, 1)), 1);
}

// Arguments of the 6 Tensor-level paged decode entry points below (audit D-11), built with
// designated initializers. An entry point ignores the fields its KV dtype has no use for.
// block_tables: [batch,max_blocks] int32. context_lens: [batch] int32. sliding_window: 0=off,
// >0=last N KV. n_sinks (StreamingLLM): >0 also attends [0,n_sinks); needs sliding_window>0 and
// ctx_len>n_sinks+sliding_window, else plain sliding-window/full attention. softcap: 0=off,
// >0 applies tanh(score/cap)*cap. attn_sinks (gpt-oss #547): per-head learned logits [n_heads]
// FP16, a virtual softmax column (unrelated to n_sinks). max_blocks_per_seq: 0 = from max_context_len.
struct PagedDecodeTensorArgs {
    const Tensor* Q = nullptr;  // [batch, 1, n_heads, head_dim] FP16
    const Tensor* K_cache = nullptr;
    const Tensor* V_cache = nullptr;
    Tensor* O = nullptr;
    const void* K_scales = nullptr;  // INT8/INT4: half per head; NVFP4 / MXFP4-KV: E4M3 / UE8M0 bytes
    const void* V_scales = nullptr;
    const int* block_tables = nullptr;
    const int* context_lens = nullptr;
    int block_size = 0;
    float scale = 0.0f;
    float kv_scale = 1.0f;  // FP8 per-tensor KV scale
    int max_context_len = 0;
    int sliding_window = 0;
    float softcap = 0.0f;
    cudaStream_t stream = nullptr;
    int max_blocks_per_seq = 0;
    int n_sinks = 0;
    const void* attn_sinks = nullptr;
    int v_head_dim = 0;  // F16 only; see paged_attention_decode
    PagedQ8Epilogue* q8_epi = nullptr;
    PagedSplitKScratch splitk{};
};

// Paged attention decode (single query/sequence), F16 KV. v_head_dim: 0=same as head_dim. MLA
// asymmetric QK/V: pass v_head_dim to read only that many V elements per slot (V allocated at
// head_dim stride) and write O at v_head_dim width.
void paged_attention_decode(const PagedDecodeTensorArgs& a);

// Token-tiled FP8 split-K kernel (attention_paged_fp8_tile.cu). Dispatched from
// paged_attention_decode_fp8 for head_dim=128 / block_size % 16 == 0; writes the same
// partial_out layout as the pipeline kernel (reduce kernel shared).
[[nodiscard]] bool paged_attention_splitk_fp8_tile_supported(int head_dim, int block_size);
void paged_attention_splitk_fp8_tile_launch(const PagedDecodeArgs& a);

// GQA-batched tile variant: one block per KV head computes all G Q heads from a
// shared smem tile (L2 KV traffic /G). Grid.y = n_kv_heads; the launcher's split
// count should be raised accordingly (see paged_attention_decode_fp8).
[[nodiscard]] bool paged_attention_splitk_fp8_tile_gqa_supported(int head_dim, int block_size, int n_heads,
                                                   int n_kv_heads);
int paged_attention_splitk_fp8_tile_gqa_splits(int batch_size, int n_heads, int n_kv_heads, int head_dim,
                                               int block_size, int max_context_len,
                                               const PagedSplitKScratch& splitk);
void paged_attention_splitk_fp8_tile_gqa_launch(const PagedDecodeArgs& a);

// FP8 E4M3 paged attention decode, on-the-fly dequant. Q:[batch,1,n_heads,hd] FP16.
// K/V_cache:[num_blocks,n_kv_heads,block_size,hd] FP8_E4M3. kv_scale: per-tensor FP32
// (val = fp8_val * kv_scale). NVFP4 4-token variant (process_diag_paged_nvfp4_multitok), HD=128/256,
// E4M3 scales; num_splits>1 fills `partial` for the reduce. Returns false for unsupported shapes.
[[nodiscard]] bool paged_attention_decode_nvfp4_multitok_launch(const PagedDecodeArgs& a);

// NVFP4 multitok with Q-head grouping: a CTA converts each K/V row once for heads_per_cta Q
// heads of one KV head. HD=128/256. heads_per_cta 0=auto (largest of 4/3/2 dividing GQA ratio),
// 1=not served. Returns false when the shape is not served.
[[nodiscard]] bool paged_attention_nvfp4_multitok_gqa_launch(const PagedDecodeArgs& a);
int paged_attention_nvfp4_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested);

// NVFP4 split-K decode on m16n8k16 tensor cores, one CTA per (batch, KV head, split), all Q heads
// of the KV head (GQA ratio 2..8). HD=128/256, block_size 16, num_splits >= 2 into `partial`.
[[nodiscard]] bool paged_attention_nvfp4_mma_launch(const PagedDecodeArgs& a);

// HD=128 FP8 decode with four tokens per warp iteration
// (process_diag_paged_fp8_multitok, attention_paged_fp8_multitok.cu). Called by
// paged_attention_decode_fp8 when the knob is on and split-K is off.
void paged_attention_decode_fp8_multitok_hd128(const PagedDecodeArgs& a);

// HD=128 FP8 decode, 16 lanes/KV row, heads_per_cta Q heads/CTA sharing KV loads. GQA: called by
// paged_attention_decode_fp8 at one split; MHA: before the four-token kernel when split-K is off.
// heads_per_cta 0=auto (largest of 5/4/3/2/1 dividing the ratio). False for HD!=128, ratio>16.
[[nodiscard]] bool paged_attention_fp8_multitok_gqa_launch(const PagedDecodeArgs& a);
// Split-K instance of the FP8 grouped kernel: grid (batch, n_kv_heads x groups, num_splits), one
// (m, l, o) partial per head in the shared reduce layout; num_splits from the caller.
[[nodiscard]] bool paged_attention_splitk_fp8_multitok_gqa_launch(const PagedDecodeArgs& a);
int paged_attention_fp8_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested);

// F16 decode, 4 tokens/warp iteration, heads_per_cta Q heads/CTA sharing KV loads (HD=128/256,
// GQA ratio 1..8). Called when the knob is on, split-K is off and n_sinks==0; heads_per_cta
// 0=auto (largest of 4/2/1 dividing the ratio, HD=256 caps at 2). Returns false otherwise.
[[nodiscard]] bool paged_attention_decode_f16_multitok_launch(const PagedDecodeArgs& a);

// Split-K instance of the F16 multitok kernel: grid (batch, n_kv_heads x
// groups, num_splits), one (m, l, o) partial per head in the shared reduce
// layout. Same shape rules as the plain launch; num_splits from the caller.
[[nodiscard]] bool paged_attention_splitk_f16_multitok_launch(const PagedDecodeArgs& a);

// Q heads per CTA the F16 multitok kernels will use for this shape (0 =
// shape not served): the largest of 4 / 2 / 1 dividing the GQA ratio, HD=256/512
// capped at 2; `requested` (1/2/4) wins when it divides.
int paged_attention_f16_multitok_heads_per_cta(int head_dim, int n_q_per_kv, int requested);

// Resident CTAs per SM of the F16 split-K multitok instance for (head_dim, hpc); 0 = not served.
int paged_attention_f16_multitok_splitk_ctas_per_sm(int head_dim, int hpc);

// Split count of the F16 multitok decode, one wave (n_ctas * s <= ctas_per_sm * num_sms). A budget
// <= 2 runs unsplit (Qwen3-8B 16 x 1131: unsplit 48.7 us, 2 splits 49.4, 3 = 1.13 waves 58.2).
// Else the fewest s in [3, budget], >= 4 KV blocks per split, with the fewest per-warp block rounds.
inline int paged_attention_f16_multitok_split_count(int n_ctas, int ctas_per_sm, int num_sms,
                                                    int num_ctx_blocks, int warps_per_cta = 8) {
    if (n_ctas <= 0 || ctas_per_sm <= 0 || num_sms <= 0 || num_ctx_blocks <= 0)
        return 1;
    const int budget = std::min((ctas_per_sm * num_sms) / n_ctas, 64);
    if (budget <= 2)
        return 1;
    const int s_max = std::min(budget, std::max(num_ctx_blocks / 4, 1));
    auto rounds = [&](int s) { return (((num_ctx_blocks + s - 1) / s) + warps_per_cta - 1) / warps_per_cta; };
    int best_s = s_max;
    for (int s = s_max - 1; s >= std::min(3, s_max); s--)
        if (rounds(s) <= rounds(best_s))
            best_s = s;
    return best_s;
}

// FP8 E4M3 paged attention decode, on-the-fly dequant. K/V_cache:[num_blocks,n_kv_heads,block_size,hd]
// FP8_E4M3, kv_scale per tensor (val = fp8_val * kv_scale).
void paged_attention_decode_fp8(const PagedDecodeTensorArgs& a);

// INT8 dp4a paged attention decode, per-head scales. Q:[batch,1,n_heads,hd] FP16.
// K/V_cache:[num_blocks,block_size,n_kv_heads,hd] INT8. K/V_scales:[...,n_kv_heads] FP16.
void paged_attention_decode_int8(const PagedDecodeTensorArgs& a);

// INT4 paged attention decode, packed 2/byte, per-head scales. Q:[batch,1,n_heads,hd] FP16.
// K/V_cache:[num_blocks,block_size,n_kv_heads,hd/2] packed uint8. K/V_scales FP16.
void paged_attention_decode_int4(const PagedDecodeTensorArgs& a);

// NVFP4 paged attention decode: packed FP4 (E2M1), per-token-head-group_of_16 UE4M3 scales.
// Q:[batch,1,n_heads,hd] FP16. K/V_cache:[num_blocks,block_size,n_kv_heads,hd/2] packed uint8.
// K/V_scales:[...,hd/16] UE4M3 bytes.
void paged_attention_decode_nvfp4(const PagedDecodeTensorArgs& a);

// MXFP4-KV paged attention decode: same layout as NVFP4 but scales are UE8M0 instead of E4M3
// (design memo 3.1.2). Q:[batch,1,n_heads,hd] FP16. K/V_cache packed uint8 [.../hd/2].
// K/V_scales:[...,hd/16] UE8M0 bytes.
void paged_attention_decode_mxfp4_kv(const PagedDecodeTensorArgs& a);

// Single source of truth for whether paged DECODE applies attn sinks (gpt-oss #547) for a KV
// dtype - a property of the kernels, not the model. Wire a new dtype here too, or the fallback
// silently misses it (#1339, #1345). FP16 sinks since #547, FP8 since #1346.
[[nodiscard]] bool paged_attention_applies_sinks(QType kv_dtype);

// Which head_dims a KV dtype's paged DECODE launchers template for (#1674); same wire-a-dtype
// contract as paged_attention_applies_sinks. Unknown dtype returns true (never refuse unchecked);
// the resolver falls back to FP16 KV so an unserved head_dim (throws, SETTLED S-22) never reaches
// a launcher.
[[nodiscard]] bool paged_attention_serves_head_dim(QType kv_dtype, int head_dim);

// True only if FP8 KV gets the FAST paged decode kernels at this head_dim; otherwise
// LAUNCH_FP8_FALLBACK runs silently. Fast kernels (fp8_multitok[_gqa].cu) are head_dim=128
// only: 4 FP8 bytes/lane is what makes one uint32 load/lane work.
constexpr bool paged_fp8_decode_has_fast_kernel(int head_dim) { return head_dim == 128; }

// Terminal for a launcher that has no template for `head_dim`. One function so
// the 17 sites are one line each and the message is written once.
[[noreturn]] void paged_attention_unsupported_head_dim(const char* fn, int head_dim);

// Launch the split-K reduce kernel (shared across FP16/FP8/INT8). q8_epi: PagedQ8Epilogue, taken
// when eligible.
void paged_attention_launch_reduce(float* partial, half* O, int batch_size, int n_heads, int head_dim,
                                   int num_splits, cudaStream_t stream, const half* attn_sinks = nullptr,
                                   PagedQ8Epilogue* q8_epi = nullptr);

}  // namespace imp
