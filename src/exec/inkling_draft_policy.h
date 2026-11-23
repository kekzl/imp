#pragma once
// Inkling2 DSpark draft: how many of a block's draft tokens join the verify call (exec/inkling_draft.h).

#include <algorithm>
#include <cstdint>

namespace imp {

// A draft row costs 0.89 ms in the verify call and a confirmed one saves about 4 ms: rows below a chance
// of 0.22 lose time. The head is overconfident (a stated 0.5-0.7 holds 37 % of the time), so the chance is
// the measured one: kDraftHit[row, 3 = later][kDraftConfEdge bucket], 27960 positions, temperature 0 and 0.8.
inline constexpr float kDraftKeep = 0.22f;
inline constexpr int kDraftConfBuckets = 11;
inline constexpr float kDraftConfEdge[kDraftConfBuckets - 1] = {0.3f,  0.5f,  0.7f,  0.8f,   0.9f,
                                                                0.95f, 0.98f, 0.99f, 0.995f, 0.999f};
inline constexpr float kDraftHit[4][kDraftConfBuckets] = {
    {0.09f, 0.21f, 0.37f, 0.48f, 0.57f, 0.66f, 0.74f, 0.79f, 0.87f, 0.89f, 0.97f},
    {0.08f, 0.14f, 0.28f, 0.37f, 0.43f, 0.53f, 0.63f, 0.71f, 0.80f, 0.86f, 0.96f},
    {0.06f, 0.15f, 0.23f, 0.32f, 0.39f, 0.51f, 0.60f, 0.73f, 0.79f, 0.86f, 0.97f},
    {0.06f, 0.12f, 0.22f, 0.28f, 0.40f, 0.54f, 0.60f, 0.72f, 0.79f, 0.87f, 0.96f}};

// The measured chance that draft row `row` holds when the rows before it did and the head states conf.
constexpr float inkling_draft_hit(int row, float conf) {
    int b = 0;
    while (b < kDraftConfBuckets - 1 && conf >= kDraftConfEdge[b])
        ++b;
    return kDraftHit[std::min(row, 3)][b];
}

// Rows of the draft tok[0..n) (stated confidences conf) worth verifying: the leading ones while the
// chance that all of them hold stays at kDraftKeep; a token outside [0, vocab) ends the draft too.
constexpr int inkling_draft_rows(const int32_t* tok, const float* conf, int n, int vocab) {
    int rows = 0;
    float chance = 1.0f;
    while (rows < n && tok[rows] >= 0 && tok[rows] < vocab &&
           (chance *= inkling_draft_hit(rows, conf[rows])) >= kDraftKeep)
        ++rows;
    return rows;
}

}  // namespace imp
