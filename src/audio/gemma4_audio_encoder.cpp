#include "audio/gemma4_audio_encoder.h"

#include "audio/gemma4_audio_kernels.h"
#include "compute/gemm_lt_ex.h"
#include "core/cublas_check.h"
#include "core/logging.h"
#include "memory/engine_arena.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace imp {

namespace {

// C[M, N] = A[M, K] @ B[N, K]^T, row-major FP16, FP32 accumulation (the 4096-wide FFW sum).
void gemm_nt(const half* A, const half* B, half* C, int M, int N, int K, cudaStream_t stream) {
    const float one = 1.0f;
    const float zero = 0.0f;
    IMP_CUBLAS_CHECK_LOG(lt_gemm_ex(CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &one, B, CUDA_R_16F, K, A, CUDA_R_16F,
                                    K, &zero, C, CUDA_R_16F, N, CUBLAS_COMPUTE_32F, stream));
}

int half_len(int n) { return ((n - 1) / 2) + 1; }  // Conv k3 s2 p1 output length

struct Geometry {
    int F, F1, T, M1, M2, sub_width;
};

Geometry geometry(const Gemma4AudioConfig& c, int frames) {
    Geometry g{};
    g.F = frames;
    g.F1 = half_len(frames);
    g.T = half_len(g.F1);
    g.M1 = half_len(c.n_mels);
    g.M2 = half_len(g.M1);
    g.sub_width = g.M2 * c.sub_channels1;
    return g;
}

size_t aligned(size_t b) { return (b + 255) & ~size_t{255}; }

bool finite_bounds(float lo, float hi) { return std::isfinite(lo) || std::isfinite(hi); }

std::vector<float> to_host(const float* d, size_t n) {
    std::vector<float> h(n);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    return h;
}

std::vector<float> to_host(const half* d, size_t n) {
    std::vector<half> tmp(n);
    IMP_CUDA_CHECK_LOG(cudaMemcpy(tmp.data(), d, n * sizeof(half), cudaMemcpyDeviceToHost));
    std::vector<float> h(n);
    for (size_t i = 0; i < n; ++i)
        h[i] = __half2float(tmp[i]);
    return h;
}

}  // namespace

int Gemma4AudioEncoder::max_frames_for(const Gemma4AudioFrontend& fe) {
    const int64_t padded = ((static_cast<int64_t>(fe.max_samples) + fe.pad_to_multiple - 1) /
                            fe.pad_to_multiple) *
                           fe.pad_to_multiple;
    return static_cast<int>(((padded + (fe.frame_length / 2) - (fe.frame_length + 1)) / fe.hop_length) + 1);
}

size_t Gemma4AudioEncoder::demand_bytes(const Gemma4AudioConfig& c, int max_frames) {
    const Geometry g = geometry(c, max_frames);
    const size_t T = static_cast<size_t>(g.T);
    const size_t H = static_cast<size_t>(c.hidden_size);
    size_t b = 0;
    b += aligned(static_cast<size_t>(g.F) * c.n_mels * 4);
    b += aligned(static_cast<size_t>(c.sub_channels0) * g.F1 * g.M1 * 4);
    b += aligned(static_cast<size_t>(c.sub_channels1) * T * g.M2 * 4);
    b += aligned(static_cast<size_t>(g.F1)) + aligned(T);
    b += aligned(T * H * 4);
    b += aligned(T * std::max(H, static_cast<size_t>(g.sub_width)) * 2);
    b += 2 * aligned(T * 4 * H * 2);  // tmp16, wide16
    b += 5 * aligned(T * H * 2);      // q, k, v, o, h16
    b += 2 * aligned(T * static_cast<size_t>(c.output_proj_dims) * 2);
    return b;
}

bool Gemma4AudioEncoder::init(const Gemma4AudioModel& m, int max_frames) {
    m_ = &m;
    max_frames_ = max_frames;
    const Gemma4AudioConfig& c = m.config;
    const Geometry g = geometry(c, max_frames);
    const size_t T = static_cast<size_t>(g.T);
    const size_t H = static_cast<size_t>(c.hidden_size);
    bool ok = true;
    auto take = [&](size_t bytes) -> void* {
        auto slab = engine_arena().take_bytes(aligned(bytes));
        if (slab.empty()) {
            ok = false;
            return nullptr;
        }
        taken_ += aligned(bytes);
        return slab.data();
    };
    d_feat_ = static_cast<float*>(take(static_cast<size_t>(g.F) * c.n_mels * 4));
    d_c0_ = static_cast<float*>(take(static_cast<size_t>(c.sub_channels0) * g.F1 * g.M1 * 4));
    d_c1_ = static_cast<float*>(take(static_cast<size_t>(c.sub_channels1) * T * g.M2 * 4));
    d_mask1_ = static_cast<uint8_t*>(take(static_cast<size_t>(g.F1)));
    d_valid_ = static_cast<uint8_t*>(take(T));
    d_x_ = static_cast<float*>(take(T * H * 4));
    d_a16_ = static_cast<half*>(take(T * std::max(H, static_cast<size_t>(g.sub_width)) * 2));
    d_tmp16_ = static_cast<half*>(take(T * 4 * H * 2));
    d_wide16_ = static_cast<half*>(take(T * 4 * H * 2));
    d_q_ = static_cast<half*>(take(T * H * 2));
    d_k_ = static_cast<half*>(take(T * H * 2));
    d_v_ = static_cast<half*>(take(T * H * 2));
    d_o_ = static_cast<half*>(take(T * H * 2));
    d_h16_ = static_cast<half*>(take(T * H * 2));
    d_proj16_ = static_cast<half*>(take(T * static_cast<size_t>(c.output_proj_dims) * 2));
    d_emb_in16_ = static_cast<half*>(take(T * static_cast<size_t>(c.output_proj_dims) * 2));
    if (!ok)
        IMP_LOG_ERROR("Audio encoder: engine arena exhausted (%zu bytes taken, %zu demanded)", taken_,
                      demand_bytes(c, max_frames));
    return ok;
}

void Gemma4AudioEncoder::linear_(const half* in, const Gemma4AudioLinear& l, half* out, int rows, int n_out,
                                 int n_in, cudaStream_t stream) {
    const half* src = in;
    if (finite_bounds(l.in_min, l.in_max)) {
        launch_g4a_clamp16(in, d_tmp16_, static_cast<int64_t>(rows) * n_in, l.in_min, l.in_max, stream);
        src = d_tmp16_;
    }
    gemm_nt(src, static_cast<const half*>(l.w.data), out, rows, n_out, n_in, stream);
    if (finite_bounds(l.out_min, l.out_max))
        launch_g4a_clamp16(out, out, static_cast<int64_t>(rows) * n_out, l.out_min, l.out_max, stream);
}

void Gemma4AudioEncoder::ffw_(const Gemma4AudioFfw& w, int rows, cudaStream_t stream) {
    const Gemma4AudioConfig& c = m_->config;
    const int H = c.hidden_size;
    G4aNormArgs pre;
    pre.in32 = d_x_;
    pre.w = static_cast<const float*>(w.pre_norm.data);
    pre.out16 = d_a16_;
    pre.clip = c.gradient_clipping;
    pre.eps = c.rms_norm_eps;
    launch_g4a_rmsnorm(pre, rows, H, stream);
    linear_(d_a16_, w.up, d_wide16_, rows, 4 * H, H, stream);
    launch_g4a_silu16(d_wide16_, static_cast<int64_t>(rows) * 4 * H, stream);
    linear_(d_wide16_, w.down, d_h16_, rows, H, 4 * H, stream);
    G4aNormArgs post;
    post.in16 = d_h16_;
    post.w = static_cast<const float*>(w.post_norm.data);
    post.residual = d_x_;
    post.out32 = d_x_;
    post.clip = c.gradient_clipping;
    post.eps = c.rms_norm_eps;
    post.scale = c.residual_weight;
    launch_g4a_rmsnorm(post, rows, H, stream);
}

void Gemma4AudioEncoder::block_(const Gemma4AudioLayer& L, int rows, cudaStream_t stream) {
    const Gemma4AudioConfig& c = m_->config;
    const int H = c.hidden_size;
    ffw_(L.ffw1, rows, stream);

    G4aNormArgs pre;
    pre.in32 = d_x_;
    pre.w = static_cast<const float*>(L.norm_pre_attn.data);
    pre.out16 = d_a16_;
    pre.clip = c.gradient_clipping;
    pre.eps = c.rms_norm_eps;
    launch_g4a_rmsnorm(pre, rows, H, stream);
    linear_(d_a16_, L.q, d_q_, rows, H, H, stream);
    linear_(d_a16_, L.k, d_k_, rows, H, H, stream);
    linear_(d_a16_, L.v, d_v_, rows, H, H, stream);
    const float k_scale = static_cast<float>(std::log1p(std::numbers::e) / std::numbers::ln2);
    launch_g4a_local_attention(d_q_, d_k_, d_v_, static_cast<const float*>(L.rel_k.data),
                               static_cast<const float*>(L.q_scale.data), d_valid_, d_o_, rows, c.num_heads,
                               c.head_dim(), c.context_left - 1, c.rel_positions(), k_scale, c.logit_cap,
                               stream);
    linear_(d_o_, L.post, d_h16_, rows, H, H, stream);
    G4aNormArgs post;
    post.in16 = d_h16_;
    post.w = static_cast<const float*>(L.norm_post_attn.data);
    post.residual = d_x_;
    post.out32 = d_x_;
    post.clip = c.gradient_clipping;
    post.eps = c.rms_norm_eps;
    launch_g4a_rmsnorm(post, rows, H, stream);

    // Light conv: norm -> linear_start -> GLU -> causal depthwise conv -> clip, norm, SiLU -> linear_end.
    G4aNormArgs lc;
    lc.in32 = d_x_;
    lc.w = static_cast<const float*>(L.lconv_pre_norm.data);
    lc.out16 = d_a16_;
    lc.eps = c.rms_norm_eps;
    launch_g4a_rmsnorm(lc, rows, H, stream);
    linear_(d_a16_, L.lin_start, d_wide16_, rows, 2 * H, H, stream);
    launch_g4a_glu16(d_wide16_, d_o_, rows, H, stream);
    launch_g4a_dwconv_norm_silu(d_o_, static_cast<const float*>(L.dw_conv.data),
                                static_cast<const float*>(L.lconv_conv_norm.data), d_a16_, rows, H,
                                c.conv_kernel, c.gradient_clipping, c.rms_norm_eps, stream);
    linear_(d_a16_, L.lin_end, d_h16_, rows, H, H, stream);
    launch_g4a_add16_to32(d_x_, d_h16_, static_cast<int64_t>(rows) * H, stream);

    ffw_(L.ffw2, rows, stream);
    G4aNormArgs out;
    out.in32 = d_x_;
    out.w = static_cast<const float*>(L.norm_out.data);
    out.out32 = d_x_;
    out.clip = c.gradient_clipping;
    out.eps = c.rms_norm_eps;
    launch_g4a_rmsnorm(out, rows, H, stream);
}

bool Gemma4AudioEncoder::encode(const AudioFeatures& f, half* d_out, cudaStream_t stream,
                                Gemma4AudioTaps* taps) {
    if (m_ == nullptr || f.n_frames <= 0 || f.n_frames > max_frames_ || f.n_mels != m_->config.n_mels ||
        f.mel.size() != static_cast<size_t>(f.n_frames) * f.n_mels ||
        f.mask.size() != static_cast<size_t>(f.n_frames))
        return false;
    const Gemma4AudioConfig& c = m_->config;
    const Geometry g = geometry(c, f.n_frames);
    const int T = g.T;
    const int H = c.hidden_size;
    if (f.n_tokens <= 0 || f.n_tokens > T)
        return false;
    std::vector<uint8_t> mask1(static_cast<size_t>(g.F1));
    std::vector<uint8_t> valid(static_cast<size_t>(T));
    for (int i = 0; i < g.F1; ++i)
        mask1[static_cast<size_t>(i)] = f.mask[2 * static_cast<size_t>(i)];
    for (int i = 0; i < T; ++i)
        valid[static_cast<size_t>(i)] = mask1[2 * static_cast<size_t>(i)];
    IMP_CUDA_CHECK_LOG(
        cudaMemcpyAsync(d_feat_, f.mel.data(), f.mel.size() * sizeof(float), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_mask1_, mask1.data(), mask1.size(), cudaMemcpyHostToDevice, stream));
    IMP_CUDA_CHECK_LOG(cudaMemcpyAsync(d_valid_, valid.data(), valid.size(), cudaMemcpyHostToDevice, stream));

    // Subsampling: two Conv2d k3 s2 + channel LayerNorm + ReLU; layer1 input masked by mask[::2].
    launch_g4a_conv2d_s2(d_feat_, static_cast<const float*>(m_->sub0_conv.data), d_c0_, 1, c.sub_channels0,
                         g.F, c.n_mels, stream);
    launch_g4a_chan_ln_relu(d_c0_, static_cast<const float*>(m_->sub0_norm.data), c.sub_channels0, g.F1, g.M1,
                            c.rms_norm_eps, d_mask1_, d_c0_, nullptr, stream);
    launch_g4a_conv2d_s2(d_c0_, static_cast<const float*>(m_->sub1_conv.data), d_c1_, c.sub_channels0,
                         c.sub_channels1, g.F1, g.M1, stream);
    launch_g4a_chan_ln_relu(d_c1_, static_cast<const float*>(m_->sub1_norm.data), c.sub_channels1, T, g.M2,
                            c.rms_norm_eps, nullptr, nullptr, d_a16_, stream);
    gemm_nt(d_a16_, static_cast<const half*>(m_->input_proj.data), d_h16_, T, H, g.sub_width, stream);
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(d_x_, 0, static_cast<size_t>(T) * H * sizeof(float), stream));
    launch_g4a_add16_to32(d_x_, d_h16_, static_cast<int64_t>(T) * H, stream);
    if (taps != nullptr) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        taps->subsample = to_host(d_x_, static_cast<size_t>(T) * H);
    }

    for (size_t i = 0; i < m_->layers.size(); ++i) {
        block_(m_->layers[i], T, stream);
        if (taps != nullptr && i == 0) {
            IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
            taps->layer0 = to_host(d_x_, static_cast<size_t>(T) * H);
        }
    }

    launch_g4a_f32_to_16(d_x_, d_a16_, static_cast<int64_t>(T) * H, stream);
    gemm_nt(d_a16_, static_cast<const half*>(m_->out_proj_w.data), d_proj16_, T, c.output_proj_dims, H,
            stream);
    launch_g4a_add_bias16(d_proj16_, static_cast<const float*>(m_->out_proj_b.data), T, c.output_proj_dims,
                          stream);
    if (taps != nullptr) {
        IMP_CUDA_CHECK_LOG(cudaStreamSynchronize(stream));
        taps->tower = to_host(d_proj16_, static_cast<size_t>(T) * c.output_proj_dims);
    }

    // Embedder: RMSNorm without scale, then the projection, valid rows only (they lead: the
    // padding sits at the end of the clip).
    G4aNormArgs en;
    en.in16 = d_proj16_;
    en.out16 = d_emb_in16_;
    en.eps = c.rms_norm_eps;
    launch_g4a_rmsnorm(en, f.n_tokens, c.output_proj_dims, stream);
    gemm_nt(d_emb_in16_, static_cast<const half*>(m_->embed_proj.data), d_out, f.n_tokens, c.lm_d_model,
            c.output_proj_dims, stream);
    return cudaStreamSynchronize(stream) == cudaSuccess;
}

}  // namespace imp
