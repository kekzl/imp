#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace imp {

// GraphExecutor::greedy_argmax_all arguments in the old parameter order (audit D-11, B-133);
// n_rows, d_out and stream required, the rest default to the old trailing defaults: no penalty
// history, no top-m harvest, no ban lists, FP16-activation LM head, no per-row histories.
struct GreedyArgmaxArgs {
    int n_rows = 0;
    int32_t* d_out = nullptr;
    cudaStream_t stream = nullptr;
    const int32_t* d_hist = nullptr;
    int n_hist = 0;
    const int32_t* d_draft = nullptr;
    float rep_pen = 1.0f;
    float freq_pen = 0.0f;
    float pres_pen = 0.0f;
    int32_t* d_topm = nullptr;
    int topm = 0;
    const int32_t* d_banned = nullptr;
    int n_banned = 0;
    bool allow_w4a4_head = false;
    const int32_t* d_banned_alt = nullptr;
    int n_banned_alt = 0;
    const uint8_t* h_row_alt = nullptr;
    const int32_t* const* h_row_hist = nullptr;
    const int* h_row_hist_n = nullptr;
    const float* h_row_pens = nullptr;
};

}  // namespace imp
