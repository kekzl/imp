#pragma once

// Host reference for KDA: a C++ port of fla's naive_recurrent_kda plus the Ling-3.0 front end
// (l2norm with +1e-6, lower-bound sigmoid gate, sigmoid beta), FP64 accumulation.
// Pinned against fla's torch code by tests/test_kda_reference.cpp (tests/refs/kda_golden.h).

#include <cmath>
#include <cstdint>
#include <vector>

namespace imp::kda_ref {

// 31-bit LCG of tests/refs/gen_kda_golden.py: value in [-1, 1) times amp, rounded to float.
struct Lcg {
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed % (1ull << 31)) {}
    double next() {
        s = ((s * 1103515245ull) + 12345ull) % (1ull << 31);
        return (static_cast<double>(s) / static_cast<double>(1ull << 30)) - 1.0;
    }
    std::vector<float> fill(size_t n, double amp) {
        std::vector<float> v(n);
        for (auto& x : v)
            x = static_cast<float>(next() * amp);
        return v;
    }
};

// Raw per-token inputs, row-major: q, k, f [T, H, K]; v [T, H, V]; b [T, H]; A_log [H];
// dt_bias [H*K]. state [H, K, V] is read as the initial state and overwritten with the final one.
struct Inputs {
    int T = 0, H = 0, K = 0, V = 0;
    float lower_bound = -5.0f;
    std::vector<float> q, k, v, f, b, A_log, dt_bias;
};

inline void forward(const Inputs& in, std::vector<double>& state, std::vector<double>& out) {
    const int T = in.T, H = in.H, K = in.K, V = in.V;
    out.assign(static_cast<size_t>(T) * H * V, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(K));
    std::vector<double> q(K), k(K), decay(K), kv(V);
    for (int t = 0; t < T; ++t) {
        for (int h = 0; h < H; ++h) {
            const size_t qk_off = ((static_cast<size_t>(t) * H) + h) * K;
            double qq = 0.0, kk = 0.0;
            for (int c = 0; c < K; ++c) {
                qq += static_cast<double>(in.q[qk_off + c]) * in.q[qk_off + c];
                kk += static_cast<double>(in.k[qk_off + c]) * in.k[qk_off + c];
            }
            const double qi = 1.0 / std::sqrt(qq + 1e-6), ki = 1.0 / std::sqrt(kk + 1e-6);
            const double a_exp = std::exp(static_cast<double>(in.A_log[h]));
            for (int c = 0; c < K; ++c) {
                q[c] = in.q[qk_off + c] * qi * scale;
                k[c] = in.k[qk_off + c] * ki;
                const double z = a_exp * (static_cast<double>(in.f[qk_off + c]) + in.dt_bias[(h * K) + c]);
                decay[c] = std::exp(in.lower_bound / (1.0 + std::exp(-z)));
            }
            const double beta = 1.0 / (1.0 + std::exp(-static_cast<double>(in.b[(t * H) + h])));
            double* S = state.data() + (static_cast<size_t>(h) * K * V);
            for (int c = 0; c < K; ++c)
                for (int e = 0; e < V; ++e)
                    S[(c * V) + e] *= decay[c];
            for (int e = 0; e < V; ++e) {
                double acc = 0.0;
                for (int c = 0; c < K; ++c)
                    acc += k[c] * S[(c * V) + e];
                kv[e] = acc;
            }
            const size_t v_off = ((static_cast<size_t>(t) * H) + h) * V;
            for (int c = 0; c < K; ++c)
                for (int e = 0; e < V; ++e)
                    S[(c * V) + e] += beta * k[c] * (in.v[v_off + e] - kv[e]);
            for (int e = 0; e < V; ++e) {
                double acc = 0.0;
                for (int c = 0; c < K; ++c)
                    acc += q[c] * S[(c * V) + e];
                out[v_off + e] = acc;
            }
        }
    }
}

// Inputs in the golden's fill order: q, k, v, f, b, A_log, dt_bias, then the initial state.
inline Inputs make_inputs(uint64_t seed, int T, int H, int K, int V, float lower_bound,
                          std::vector<float>* state0) {
    Lcg rng(seed);
    Inputs in;
    in.T = T, in.H = H, in.K = K, in.V = V, in.lower_bound = lower_bound;
    const size_t tk = static_cast<size_t>(T) * H * K;
    in.q = rng.fill(tk, 1.0);
    in.k = rng.fill(tk, 1.0);
    in.v = rng.fill(static_cast<size_t>(T) * H * V, 1.0);
    in.f = rng.fill(tk, 3.0);
    in.b = rng.fill(static_cast<size_t>(T) * H, 3.0);
    in.A_log = rng.fill(H, 1.5);
    in.dt_bias = rng.fill(static_cast<size_t>(H) * K, 1.0);
    if (state0)
        *state0 = rng.fill(static_cast<size_t>(H) * K * V, 0.5);
    return in;
}

}  // namespace imp::kda_ref
