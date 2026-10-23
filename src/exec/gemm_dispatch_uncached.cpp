// GEMM routes for a weight without a cache entry: the GGUF blocks read in place (dp4a, IMMA) and the
// dequant -> cuBLAS safety net. Split out of executor_gemm_dispatch.cpp (file-size gate).
#include "exec/gemm_dispatch_fallback.h"

#include <utility>

#include "compute/gemm.h"
#include "compute/gemm_q4k.h"
#include "compute/gemm_q8_0_dense.h"
#include "compute/mmq_q8_imma.h"
#include "core/logging.h"
#include "core/process_diag.h"
#include "core/tensor_kind.h"
#include "exec/executor_gemv_helpers.h"
#include "exec/executor_kernels.h"
#include "exec/gemm_kernel_registry.h"
#include "exec/weight_dequant.h"
#include "exec/quant_scratch.h"
#include "exec/weight_caches.h"
#include "quant/dequant_gpu.h"

namespace imp {

// The dequant scratch is sized on the largest per-layer weight; the LM head can be far
// larger (Gemma-4: 262144 x 2816 = 1408 MiB FP16 vs 44 MiB, #2046). Whole-weight dequant
// into it is only legal when it fits.
static bool dequant_scratch_fits(const QuantScratch* qs, const Tensor& weight) {
    if (qs == nullptr || qs->dequant == nullptr)
        return false;
    const size_t need = static_cast<size_t>(weight.shape[0]) * static_cast<size_t>(weight.shape[1]) *
                        sizeof(uint16_t);
    if (need <= qs->dequant_size)
        return true;
    static bool warned = false;
    if (!warned) {
        warned = true;
        IMP_LOG_WARN(
            "gemm_dispatch_uncached_fallback: %ld x %ld %s weight needs %.1f MiB, dequant scratch is "
            "%.1f MiB; whole-weight dequant declined",
            (long)weight.shape[0], (long)weight.shape[1], qtype_name(weight.qtype), need / (1024.0 * 1024.0),
            qs->dequant_size / (1024.0 * 1024.0));
    }
    return false;
}

// Q8_0 / Q4_K decode rows (M <= 64): dp4a on the GGUF blocks in place, weight read once per
// 16/8/4-row activation tile (#2688). Q8_0's plane IMMA declines under graph capture (weight
// copy) and Q4_K's IMMA tile is 128 rows, so both cost a whole-weight pass per few rows before.
static bool dp4a_decode_rows(const Tensor& input, const Tensor& weight, Tensor& output,
                             const GemmContext& ctx) {
    const auto* qs = ctx.qscratch;
    const int M = static_cast<int>(input.shape[0]);
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    const size_t blocks = static_cast<size_t>(M) * (K / 32);
    if (M > 64 || !qs || !qs->q8_1_prefill_buf || !qs->d8_prefill_buf ||
        blocks * sizeof(block_q8_1) > qs->q8_1_prefill_bytes || blocks * sizeof(float) > qs->d8_prefill_bytes)
        return false;
    const auto* x = reinterpret_cast<const __half*>(input.data);
    auto* y = reinterpret_cast<__half*>(output.data);
    const bool ok = weight.qtype == QType::Q8_0
                        ? gemm_q8_0_dp4a_dense(weight.data, x, y, qs->q8_1_prefill_buf, qs->d8_prefill_buf, M,
                                               N, K, ctx.beta, ctx.stream)
                        : gemm_q4k_dp4a_dense_rows(weight.data, x, y, qs->q8_1_prefill_buf,
                                                   qs->d8_prefill_buf, M, N, K, ctx.beta, ctx.stream);
    static bool s_logged[2] = {};  // activity proof per outcome, once each
    if (!s_logged[ok]) {
        s_logged[ok] = true;
        IMP_LOG_INFO("uncached %s decode rows M=%d N=%d K=%d beta=%.0f: %s", qtype_name(weight.qtype), M, N,
                     K, ctx.beta, ok ? "dp4a in place" : "declined");
    }
    return ok;
}

// Rows via IMMA on the GGUF blocks: Q8_0 4..32 (gemm.q8_0_imma_smallm; dp4a reads the weight once per
// 16 rows, at 2..3 once too with one launch less), IQ4_NL / IQ4_XS / Q5_K from 2 (gemm.gguf_imma_smallm;
// the alternative is a dequant of the whole weight per call).
static bool gguf_imma_rows(const Tensor& input, const Tensor& weight, Tensor& output,
                           const GemmContext& ctx) {
    const int M = static_cast<int>(input.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    const bool q8 = weight.qtype == QType::Q8_0;
    const bool on = (q8 && M <= 32) ? ctx.q8_0_imma_smallm : ctx.gguf_imma_smallm;
    if (M < (q8 ? 4 : 2) || !on ||
        !mmq_gguf_imma_smallm_gemm(weight.data, weight.qtype, reinterpret_cast<const __half*>(input.data),
                                   reinterpret_cast<__half*>(output.data), M,
                                   static_cast<int>(weight.shape[0]), K, ctx.stream, ctx.beta,
                                   ctx.act_quant_hint_matches(input.data, M, K)))
        return false;
    static bool s_logged[128] = {};  // activity proof, once per qtype
    if (!s_logged[std::to_underlying(weight.qtype) & 127]) {
        s_logged[std::to_underlying(weight.qtype) & 127] = true;
        IMP_LOG_INFO("uncached %s rows M=%d N=%ld K=%d: raw IMMA in place", qtype_name(weight.qtype), M,
                     (long)weight.shape[0], K);
    }
    return true;
}

// F16 rows on unpadded strides with the weight bytes present: what every in-place kernel reads.
static bool dense_f16_rows(const Tensor& input, const Tensor& weight, const Tensor& output) {
    return !weight.dropped_source && input.shape[0] >= 1 && input.qtype == QType::F16 &&
           output.qtype == QType::F16 && input.stride[0] == weight.shape[1] &&
           output.stride[0] == weight.shape[0];
}

// Q8_0: raw IMMA at 4..32, else dp4a rows. Q4_K: dp4a for decode / M=1, IMMA for prompt rows (tier
// Undefined on gemma-3: every dense Q4_K weight).
static bool q8_q4k_rows(const Tensor& input, const Tensor& weight, Tensor& output, const GemmContext& ctx,
                        bool decode_rows, bool short_prompt) {
    const int M = static_cast<int>(input.shape[0]);
    const bool q8 = weight.qtype == QType::Q8_0;
    // Above 32 rows only wide Q8_0 weights: N=48 (GDN alpha/beta) is 4 BM=128 CTAs, cuBLAS beats it.
    const bool wide_prompt = M > 32 && weight.shape[0] >= 1024 && !process_diag_deterministic();
    if (q8 && (decode_rows || short_prompt || wide_prompt) && gguf_imma_rows(input, weight, output, ctx))
        return true;
    // M=1 is left to dp4a_token_gemv: the row-tile kernel reads a Q8_0 GDN weight at ~33 % of DRAM peak.
    if (M > 1 && (decode_rows || (q8 && short_prompt)) && dp4a_decode_rows(input, weight, output, ctx))
        return true;
    return !q8 && M >= 2 && ctx.q4k_imma_prefill &&
           mmq_q4k_imma_gemm(weight.data, reinterpret_cast<const __half*>(input.data),
                             reinterpret_cast<__half*>(output.data), M, static_cast<int>(weight.shape[0]),
                             static_cast<int>(weight.shape[1]), ctx.stream, ctx.beta,
                             q4k_split_ok(decode_rows));
}

// Q6_K (UD ssm_out): the k256 kernel at 2..32 rows, as imma_source_prefill.
static bool q6k_rows(const Tensor& input, const Tensor& weight, Tensor& output, const GemmContext& ctx,
                     bool rows_ok) {
    const int M = static_cast<int>(input.shape[0]);
    return M >= 2 && M <= 32 && ctx.q6k_imma_smallm && rows_ok &&
           mmq_q6k_imma_gemm(weight.data, reinterpret_cast<const __half*>(input.data),
                             reinterpret_cast<__half*>(output.data), M, static_cast<int>(weight.shape[0]),
                             static_cast<int>(weight.shape[1]), ctx.stream, ctx.beta);
}

bool gemm_uncached_in_place(const Tensor& input, const Tensor& weight, Tensor& output, const GemmContext& ctx,
                            bool decode_rows) {
    if (!dense_f16_rows(input, weight, output))
        return false;
    // Prompt rows only off runtime.deterministic, and for Q8_0 / Q6_K only a whole prompt under 33
    // tokens: the kernel changes at 33 rows (#2152).
    const bool det = process_diag_deterministic();
    const bool short_prompt = input.shape[0] <= 32 && !det;
    switch (weight.qtype) {
        case QType::Q8_0:
        case QType::Q4_K:
            return q8_q4k_rows(input, weight, output, ctx, decode_rows, short_prompt);
        case QType::IQ4_NL:
        case QType::IQ4_XS:
        case QType::Q5_K:
            return (decode_rows || !det) && gguf_imma_rows(input, weight, output, ctx);
        case QType::Q6_K:
            return q6k_rows(input, weight, output, ctx, decode_rows || short_prompt);
        default:
            return false;
    }
}

// M=1 token GEMV on the other dp4a GGUF types (IQ4_XS, Q5_K, Q6_K, ...), beta 0 or the beta=1 residual
// form in place (y = W x + y). The dequant below rebuilt the whole weight per token (UD-Q4_K_M down).
static bool dp4a_token_gemv(const Tensor& input, const Tensor& weight, Tensor& output,
                            const GemmContext& ctx) {
    const auto* qs = ctx.qscratch;
    const int N = static_cast<int>(weight.shape[0]);
    const int K = static_cast<int>(weight.shape[1]);
    if (input.shape[0] != 1 || !is_dp4a_qtype(weight.qtype) || weight.dropped_source ||
        ctx.gemm_no_dp4a_gemv || (ctx.beta != 0.0f && ctx.beta != 1.0f) || input.qtype != QType::F16 ||
        output.qtype != QType::F16 || qs == nullptr || qs->q8_1_buf == nullptr || qs->d8_buf == nullptr ||
        K % 32 != 0 || K / 32 > qs->q8_1_max_blocks)
        return false;
    auto* q8 = static_cast<block_q8_1*>(qs->q8_1_buf);
    auto* y = static_cast<half*>(output.data);
    quantize_fp16_to_q8_1(static_cast<const half*>(input.data), q8, qs->d8_buf, K, ctx.stream);
    if (ctx.beta == 1.0f)
        dispatch_gemv_residual(weight.qtype, weight.data, q8, qs->d8_buf, y, y, N, K, ctx.stream);
    else
        dispatch_dp4a_gemv(weight.qtype, weight.data, q8, qs->d8_buf, y, N, K, ctx.stream);
    static bool s_logged[2] = {};  // activity proof per beta, once each
    if (!s_logged[ctx.beta == 1.0f]) {
        s_logged[ctx.beta == 1.0f] = true;
        IMP_LOG_INFO("uncached %s token GEMV N=%d K=%d beta=%.0f: dp4a in place", qtype_name(weight.qtype), N,
                     K, ctx.beta);
    }
    return true;
}

// Uncached fallback: safety net for weights without a WeightHandle
// (kInvalidTensorID, budget-exhausted) and for M=1 beta!=0 residual-add.
void gemm_dispatch_uncached_fallback(const Tensor& input, const Tensor& weight, Tensor& output,
                                     const GemmContext& ctx, bool decode_rows) {
    const auto* wc = ctx.wcache;
    const auto* qs = ctx.qscratch;
    if (!wc || !qs)
        return;

    const QType qtype = weight.qtype;
    const int M = static_cast<int>(input.shape[0]);
    if (gemm_uncached_in_place(input, weight, output, ctx, decode_rows) ||
        dp4a_token_gemv(input, weight, output, ctx))
        return;

    if (ctx.beta != 0.0f) {
        auto it = wc->fp16.find(weight.data);
        if (it != wc->fp16.end()) {
            gemm(input, it->second, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        if (weight_dequant_supported(qtype) && !weight.dropped_source && dequant_scratch_fits(qs, weight)) {
            int rows = static_cast<int>(weight.shape[0]);
            int cols = static_cast<int>(weight.shape[1]);
            // The one degradation landing on the per-token path was silent
            // (AUDIT_ARCH_2026 A2-8): every decode step here dequantizes and copies
            // the whole weight before the GEMM. Warned once per process, with the size, so a trace can catch
            // it.
            static bool s_warned_full_dequant = false;
            if (!s_warned_full_dequant) {
                s_warned_full_dequant = true;
                IMP_LOG_WARN(
                    "gemm_dispatch_uncached_fallback: beta=%.3f on a %s weight without a cached FP16 "
                    "copy: the whole %d x %d weight (%.1f MiB dequantised) is materialised per "
                    "call from here on, on the decode path",
                    ctx.beta, qtype_name(qtype), rows, cols,
                    static_cast<double>(rows) * cols * 2.0 / (1024.0 * 1024.0));
            }
            dequant_weight_fp16(weight, qs->dequant, ctx.stream);
            Tensor w_fp16(qs->dequant, QType::F16, weight.ndim, weight.shape, true);
            gemm(input, w_fp16, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        if (qtype == QType::F16 || qtype == QType::BF16) {
            gemm(input, weight, output, 1.0f, ctx.beta, ctx.stream);
            return;
        }
        IMP_LOG_ERROR("gemm_dispatch_uncached_fallback: beta=%.3f but no FP16 path for qtype=%d", ctx.beta,
                      std::to_underlying(qtype));
        return;
    }

    // Generic dequant catch-all (M>1 prefill for uncached weights)
    if (M > 1 && input.qtype == QType::F16 && !weight.dropped_source) {
        static bool s_logged[128] = {};  // which in-place route declined, once per qtype
        if (!s_logged[std::to_underlying(qtype) & 127]) {
            s_logged[std::to_underlying(qtype) & 127] = true;
            IMP_LOG_INFO("uncached %s M=%d N=%ld K=%ld: per-call dequant (in stride %ld, out stride %ld)",
                         qtype_name(qtype), M, (long)weight.shape[0], (long)weight.shape[1],
                         (long)input.stride[0], (long)output.stride[0]);
        }
        GemmKernelArgs args{};
        args.input = &input;
        args.output = &output;
        args.stream = ctx.stream;
        args.beta = ctx.beta;
        args.weight_payload = &weight;
        args.dequant_scratch = qs->dequant;
        args.dequant_scratch_size = qs->dequant_size;
        GemmStrategy strat{StorageTier::FP16, QType::NONE, /*m_is_one=*/false};
        if (GemmKernelRegistry::instance().dispatch(strat, args) == GemmDispatchResult::Ok)
            return;
    }

    if (weight.dropped_source) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            IMP_LOG_WARN(
                "gemm_dispatch_uncached_fallback: dropped weight at final fallback! "
                "M=%d qtype=%d kind=%s — overlay coverage gap.",
                M, std::to_underlying(qtype), tensor_kind_name(weight.kind));
        }
        return;
    }

    // Block-quant types (Q4_K, Q5_K, Q8_0, etc.) and native FP8: dequant to FP16 then cuBLAS.
    if (weight_dequant_supported(qtype)) {
        if (!dequant_scratch_fits(qs, weight)) {
            // gemm() has no block-quant arm: it would read the Q6_K/Q8_0 bytes as FP16 and run
            // past the allocation. Output stays untouched; the WARN above names the weight.
            IMP_LOG_ERROR("gemm_dispatch_uncached_fallback: no route for a %ld x %ld %s weight at M=%d",
                          (long)weight.shape[0], (long)weight.shape[1], qtype_name(qtype), M);
            return;
        }
        dequant_weight_fp16(weight, qs->dequant, ctx.stream);
        Tensor w_fp16(qs->dequant, QType::F16, weight.ndim, weight.shape, true);
        gemm(input, w_fp16, output, 1.0f, 0.0f, ctx.stream);
        return;
    }

    gemm(input, weight, output, 1.0f, 0.0f, ctx.stream);
}

void gguf_source_fallback(const WeightHandle& h, const Tensor& input, Tensor& output, const GemmContext& ctx,
                          bool decode_rows) {
    if (ctx.wcache == nullptr || !ctx.wcache->q8_planes_only.contains(h.source_data)) {
        Tensor weight(const_cast<void*>(h.source_data), h.source_qtype, 2, h.shape, true);
        gemm_dispatch_uncached_fallback(input, weight, output, ctx, decode_rows);
        return;
    }
    static bool logged = false;
    if (!logged) {
        logged = true;
        IMP_LOG_INFO(
            "gemm: IMMA declined a Q8_0 weight without source (M=%lld beta=%.0f), FP16 from its planes",
            static_cast<long long>(input.shape[0]), ctx.beta);
    }
    const auto* qs = ctx.qscratch;
    const size_t need = static_cast<size_t>(h.shape[0]) * static_cast<size_t>(h.shape[1]) * sizeof(half);
    IMP_CHECK(qs && qs->dequant && qs->dequant_size >= need,
              "gguf_source_fallback: no dequant scratch for a %lld x %lld Q8_0 weight without source",
              static_cast<long long>(h.shape[0]), static_cast<long long>(h.shape[1]));
    IMP_CHECK(mmq_q8_imma_planes_to_f16(h.source_data, static_cast<half*>(qs->dequant), ctx.stream),
              "gguf_source_fallback: a Q8_0 weight without source has no IMMA planes");
    Tensor w16(qs->dequant, QType::F16, 2, h.shape, true);
    gemm(input, w16, output, 1.0f, ctx.beta, ctx.stream);
}

}  // namespace imp
