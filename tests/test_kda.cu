// KDA scans (src/compute/kda_recurrent.cu, kda_scan_chunk.cu) against the host reference
// tests/kda_reference.h, which test_kda_reference.cpp pins to fla's torch code.
// Inputs mirror the executor: conv_f32 rows Q|K|V (FP32), f and beta FP16, A as -exp(A_log).
// Tolerance: 3e-3 abs on o and state (FP16 f/beta rounding is folded into the reference inputs,
// the rest is FP32 vs FP64 accumulation over 128-term dots across up to 300 steps).

#include "compute/kda.h"
#include "kda_reference.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace imp {
namespace {

constexpr int kD = kKdaHeadDim;

template <typename T>
T* to_device(const std::vector<T>& h) {
    T* d = nullptr;
    EXPECT_EQ(cudaMalloc(&d, std::max<size_t>(h.size(), 1) * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return d;
}

template <typename T>
std::vector<T> to_host(const T* d, size_t n) {
    std::vector<T> h(n);
    EXPECT_EQ(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return h;
}

// One sequence's case, inputs rounded the way the device sees them.
struct Case {
    kda_ref::Inputs in;
    std::vector<float> s0;
};

Case make_case(uint64_t seed, int T, int H) {
    Case c;
    c.in = kda_ref::make_inputs(seed, T, H, kD, kD, -5.0f, &c.s0);
    for (auto* v : {&c.in.f, &c.in.b})
        for (auto& x : *v)
            x = __half2float(__float2half(x));
    return c;
}

// Device buffers for n_seq sequences of rows[i] each, concatenated, plus a slot pool.
struct DevBatch {
    float* conv = nullptr;
    half* f = nullptr;
    half* beta = nullptr;
    float* neg_a = nullptr;
    float* dtb = nullptr;
    half* y = nullptr;
    int rows = 0;
    int H = 0;
    void free_all() {
        cudaFree(conv), cudaFree(f), cudaFree(beta), cudaFree(neg_a), cudaFree(dtb), cudaFree(y);
    }
};

// A_log / dt_bias are per layer: every sequence shares cases[0]'s.
DevBatch upload(const std::vector<Case>& cases) {
    DevBatch b;
    b.H = cases[0].in.H;
    const int H = b.H, HK = H * kD;
    std::vector<float> conv;
    std::vector<half> f, beta;
    for (const auto& c : cases) {
        for (int t = 0; t < c.in.T; ++t) {
            const size_t o = static_cast<size_t>(t) * HK;
            conv.insert(conv.end(), c.in.q.begin() + o, c.in.q.begin() + o + HK);
            conv.insert(conv.end(), c.in.k.begin() + o, c.in.k.begin() + o + HK);
            conv.insert(conv.end(), c.in.v.begin() + o, c.in.v.begin() + o + HK);
            for (int i = 0; i < HK; ++i)
                f.push_back(__float2half(c.in.f[o + i]));
            for (int i = 0; i < H; ++i)
                beta.push_back(__float2half(c.in.b[(t * H) + i]));
        }
        b.rows += c.in.T;
    }
    std::vector<float> neg_a(H);
    for (int i = 0; i < H; ++i)
        neg_a[i] = -std::exp(cases[0].in.A_log[i]);
    b.conv = to_device(conv);
    b.f = to_device(f);
    b.beta = to_device(beta);
    b.neg_a = to_device(neg_a);
    b.dtb = to_device(cases[0].in.dt_bias);
    EXPECT_EQ(cudaMalloc(&b.y, static_cast<size_t>(b.rows) * HK * sizeof(half)), cudaSuccess);
    return b;
}

KdaScanArgs args_of(const DevBatch& b, void* state, int n_seq, int n_tokens) {
    KdaScanArgs a;
    a.conv_f32 = b.conv;
    a.conv_channels = 3 * b.H * kD;
    a.f = b.f;
    a.beta = b.beta;
    a.neg_a = b.neg_a;
    a.dt_bias = b.dtb;
    a.h_state = state;
    a.y = b.y;
    a.n_seq = n_seq;
    a.n_tokens = n_tokens;
    a.n_heads = b.H;
    return a;
}

// Reference for one case with the shared layer parameters of `layer`.
void reference(const Case& c, const Case& layer, std::vector<double>& state, std::vector<double>& out) {
    kda_ref::Inputs in = c.in;
    in.A_log = layer.in.A_log;
    in.dt_bias = layer.in.dt_bias;
    state.assign(c.s0.begin(), c.s0.end());
    kda_ref::forward(in, state, out);
}

double max_abs_diff(const std::vector<double>& ref, const half* got, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i)
        m = std::max(m, std::abs(ref[i] - static_cast<double>(__half2float(got[i]))));
    return m;
}

template <typename S>
double state_diff(const std::vector<double>& ref, const std::vector<S>& got) {
    double m = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double g = 0.0;
        if constexpr (std::is_same_v<S, __nv_bfloat16>)
            g = __bfloat162float(got[i]);
        else
            g = got[i];
        m = std::max(m, std::abs(ref[i] - g));
    }
    return m;
}

void run_single(int T, int H, bool chunk) {
    const Case c = make_case(1000 + T, T, H);
    DevBatch b = upload({c});
    float* st = to_device(c.s0);
    const KdaScanArgs a = args_of(b, st, 1, T);
    if (chunk)
        kda_scan_chunk(a, nullptr);
    else
        kda_scan_recurrent(a, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<double> rs, ro;
    reference(c, c, rs, ro);
    const auto y = to_host(b.y, ro.size());
    const auto s = to_host(st, rs.size());
    EXPECT_LT(max_abs_diff(ro, y.data(), ro.size()), 3e-3) << "T=" << T << " chunk=" << chunk;
    EXPECT_LT(state_diff(rs, s), 3e-3) << "T=" << T << " chunk=" << chunk;
    cudaFree(st);
    b.free_all();
}

TEST(KdaScan, RecurrentMatchesReference) {
    for (int T : {1, 2, 7, 33})
        run_single(T, 4, false);
}

TEST(KdaScan, ChunkMatchesReference) {
    for (int T : {16, 17, 64, 100, 300})
        run_single(T, 4, true);
}

// Ragged prefill: sequences of different lengths, each on its own slot of a pool, one launch.
TEST(KdaScan, ChunkRaggedSlotsMatchReference) {
    const int H = 2;
    const std::vector<int> lens = {70, 5, 129};
    std::vector<Case> cases;
    for (size_t i = 0; i < lens.size(); ++i)
        cases.push_back(make_case(77 + i, lens[i], H));
    DevBatch b = upload(cases);
    const size_t slab = static_cast<size_t>(H) * kD * kD;
    const std::vector<int> slots = {2, 0, 1};
    std::vector<float> pool(slab * 3);
    for (size_t i = 0; i < cases.size(); ++i)
        std::copy(cases[i].s0.begin(), cases[i].s0.end(), pool.begin() + (slots[i] * slab));
    float* d_pool = to_device(pool);
    std::vector<int> offs = {0};
    for (int l : lens)
        offs.push_back(offs.back() + l);
    int* d_offs = to_device(offs);
    int* d_slots = to_device(slots);
    KdaScanArgs a = args_of(b, d_pool, static_cast<int>(lens.size()), 0);
    a.seq_row_offsets = d_offs;
    a.seq_slots = d_slots;
    a.seq_stride = static_cast<int64_t>(slab);
    kda_scan_chunk(a, nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto y = to_host(b.y, static_cast<size_t>(b.rows) * H * kD);
    const auto p = to_host(d_pool, pool.size());
    for (size_t i = 0; i < cases.size(); ++i) {
        std::vector<double> rs, ro;
        reference(cases[i], cases[0], rs, ro);
        EXPECT_LT(max_abs_diff(ro, y.data() + (static_cast<size_t>(offs[i]) * H * kD), ro.size()), 3e-3)
            << "seq " << i;
        const std::vector<float> got(p.begin() + (slots[i] * slab), p.begin() + ((slots[i] + 1) * slab));
        EXPECT_LT(state_diff(rs, got), 3e-3) << "seq " << i;
    }
    cudaFree(d_pool), cudaFree(d_offs), cudaFree(d_slots);
    b.free_all();
}

// Batched decode on a BF16 slot pool: one token per sequence, three steps.
TEST(KdaScan, RecurrentBatchedBf16Decode) {
    const int H = 2, n_seq = 3, steps = 3;
    std::vector<Case> full;
    for (int i = 0; i < n_seq; ++i)
        full.push_back(make_case(500 + i, steps, H));
    const size_t slab = static_cast<size_t>(H) * kD * kD;
    std::vector<__nv_bfloat16> pool(slab * n_seq);
    for (int i = 0; i < n_seq; ++i)
        for (size_t e = 0; e < slab; ++e)
            pool[(i * slab) + e] = __float2bfloat16(full[i].s0[e]);
    // Reference starts from the BF16-rounded state, like the device.
    for (int i = 0; i < n_seq; ++i)
        for (size_t e = 0; e < slab; ++e)
            full[i].s0[e] = __bfloat162float(pool[(i * slab) + e]);
    __nv_bfloat16* d_pool = to_device(pool);
    const std::vector<int> slots = {1, 2, 0};
    int* d_slots = to_device(slots);
    std::vector<std::vector<double>> last_y(n_seq);
    for (int t = 0; t < steps; ++t) {
        std::vector<Case> step(n_seq);
        for (int i = 0; i < n_seq; ++i) {
            const auto& src = full[slots[i]];
            Case& c = step[i];
            c.in = src.in;
            c.in.T = 1;
            const size_t hk = static_cast<size_t>(H) * kD;
            auto slice = [&](const std::vector<float>& v, size_t w) {
                return std::vector<float>(v.begin() + (t * w), v.begin() + ((t + 1) * w));
            };
            c.in.q = slice(src.in.q, hk), c.in.k = slice(src.in.k, hk), c.in.v = slice(src.in.v, hk);
            c.in.f = slice(src.in.f, hk), c.in.b = slice(src.in.b, H);
        }
        step[0].in.A_log = full[0].in.A_log;
        step[0].in.dt_bias = full[0].in.dt_bias;
        DevBatch b = upload(step);
        KdaScanArgs a = args_of(b, d_pool, n_seq, 1);
        a.seq_slots = d_slots;
        a.seq_stride = static_cast<int64_t>(slab);
        a.state_bf16 = true;
        kda_scan_recurrent(a, nullptr);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        if (t + 1 == steps) {
            const auto y = to_host(b.y, static_cast<size_t>(n_seq) * H * kD);
            for (int i = 0; i < n_seq; ++i) {
                Case whole = full[slots[i]];
                std::vector<double> rs, ro;
                reference(whole, full[0], rs, ro);
                const size_t per = static_cast<size_t>(H) * kD;
                std::vector<double> last(ro.end() - per, ro.end());
                // BF16 state round trip per step: 2^-8 relative on |S| <= ~2.
                EXPECT_LT(max_abs_diff(last, y.data() + (i * per), per), 3e-2) << "seq " << i;
            }
        }
        b.free_all();
    }
    cudaFree(d_pool), cudaFree(d_slots);
}

// The chunk and recurrent scans agree on the same prefill (both from the same FP32 state).
TEST(KdaScan, ChunkEqualsRecurrent) {
    const int T = 96, H = 4;
    const Case c = make_case(4242, T, H);
    DevBatch b = upload({c});
    float* s1 = to_device(c.s0);
    float* s2 = to_device(c.s0);
    kda_scan_chunk(args_of(b, s1, 1, T), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto y1 = to_host(b.y, static_cast<size_t>(T) * H * kD);
    kda_scan_recurrent(args_of(b, s2, 1, T), nullptr);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const auto y2 = to_host(b.y, y1.size());
    double m = 0.0;
    for (size_t i = 0; i < y1.size(); ++i)
        m = std::max(m, static_cast<double>(std::abs(__half2float(y1[i]) - __half2float(y2[i]))));
    EXPECT_LT(m, 3e-3);
    cudaFree(s1), cudaFree(s2);
    b.free_all();
}

}  // namespace
}  // namespace imp
