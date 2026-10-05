#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace imp {

// LFM2 gated short conv over in_proj rows bcx [rows, 3H] = [B | C | x]:
//   v_t = B_t * x_t;  y_t = C_t * sum_k w[c, k] * v_{t - (L - 1) + k}
// v before the chunk comes from the conv window (float [H, L], oldest first, the SSM pool layout).
// Single sequence: n_seq = 1, n_tok rows, state = that sequence's window, seq_slots = nullptr.
// Batched decode: n_seq rows of one token each, windows at pool + seq_slots[s] * slot_stride_bytes.
// Every window is advanced to its last L values of v.
void shortconv_forward(void* state_or_pool, const int* seq_slots, int64_t slot_stride_bytes, const half* bcx,
                       const half* weight, half* y, int n_seq, int n_tok, int hidden, int kernel_size,
                       cudaStream_t stream);

}  // namespace imp
