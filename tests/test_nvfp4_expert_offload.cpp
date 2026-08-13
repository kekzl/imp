// The slot layout that lets host-resident NVFP4 experts reach the fused decode
// kernels unchanged (2026-08-13).
//
// The GGUF host path (#1370) works because one expert is one contiguous byte
// range, so the LRU cache's fixed-stride pool IS the array the kernels index.
// An NVFP4 expert is TWO ranges — packed FP4 weights and FP8 micro-scales — and
// the trick still works only because the kernels take separate bases and
// separate strides for them.
//
// That makes the arithmetic load-bearing in a way a comment cannot pin: the
// cache WRITES a slot at one address and the kernel READS it at another,
// computed independently. If those two ever disagree the kernel reads a
// neighbouring expert's scales, which is coherent-looking and wrong — the
// failure mode #1403 measured ("the capital of France is the city of the same
// name"), not a crash. So the central test here is that the two agree.
//
// CPU-only: the layout is pure arithmetic and takes its inputs as data.

#include <gtest/gtest.h>

#include "exec/nvfp4_expert_offload.h"

#include <vector>

using namespace imp;

namespace {

// Qwen3-30B-A3B-NVFP4: d_model 2048, moe_intermediate 768.
constexpr int64_t kGateN = 768, kGateK = 2048;   // gate/up: [768, 2048]
constexpr int64_t kDownN = 2048, kDownK = 768;   // down:    [2048, 768]

}  // namespace

TEST(NvFP4SlotLayout, HoldsBothRangesOfOneExpert) {
    const auto l = nvfp4_slot_layout(kGateN, kGateK);
    EXPECT_EQ(l.packed_bytes, static_cast<size_t>(kGateN) * (kGateK / 2));   // 768 KiB
    EXPECT_EQ(l.ms_bytes, static_cast<size_t>(kGateN) * (kGateK / 16));      //  96 KiB
    EXPECT_GE(l.slot_bytes(), l.packed_bytes + l.ms_bytes);
}

// The kernels load packed weights through uint2 (gemv_nvfp4_row), so every
// slot's packed block must start 8-byte aligned. Both the stride and the
// micro-scale offset feed that address, so both have to hold.
//
// The 8 is written out rather than taken from kNvFP4SlotAlign on purpose: it
// is the KERNEL's requirement, and a test that reads the layout's own constant
// passes unchanged if that constant is set to 1.
TEST(NvFP4SlotLayout, EverySlotStartsWhereAUint2LoadCanReadIt) {
    constexpr size_t kUint2Align = 8;  // sizeof(uint2), from gemv_nvfp4_row
    for (const auto [N, K] : std::vector<std::pair<int64_t, int64_t>>{
             {kGateN, kGateK}, {kDownN, kDownK}, {1, 16}, {17, 48}, {3, 80}, {129, 176}}) {
        const auto l = nvfp4_slot_layout(N, K);
        ASSERT_GT(l.slot_bytes(), 0u) << "N=" << N << " K=" << K;
        EXPECT_EQ(l.slot_bytes() % kUint2Align, 0u) << "stride, N=" << N << " K=" << K;
        EXPECT_EQ(l.packed_off() % kUint2Align, 0u) << "scale offset, N=" << N << " K=" << K;
        EXPECT_GE(l.packed_off(), l.packed_bytes);
        EXPECT_LE(l.packed_off() + l.ms_bytes, l.slot_bytes());
    }
}

// THE property the whole path rests on: the kernel reads a slot's micro-scales
// from `micro_scales + idx * expert_stride_ms`, and that has to land INSIDE
// slot idx — the same slot whose packed weights it is decoding.
//
// This is not automatic. The natural stride for a scale array is its own size
// (`ms_bytes`), which is what a contiguous per-expert scale buffer uses and
// what the resident path passes. Using it here would walk the scale base
// forward far slower than the slot stride, so slot 1's weights would be decoded
// with scales from inside slot 0 — coherent-looking, wrong output. Passing the
// SLOT stride for both is what makes the two halves stay together.
//
// The assertion below fails for `expert_stride_ms = ms_bytes` and for any other
// stride that is not the slot stride.
TEST(NvFP4SlotLayout, ScaleAddressStaysInsideItsOwnSlot) {
    for (const auto [N, K] : std::vector<std::pair<int64_t, int64_t>>{{kGateN, kGateK},
                                                                     {kDownN, kDownK}}) {
        const auto l = nvfp4_slot_layout(N, K);
        constexpr size_t kPoolBase = 0x10000000;  // stand-in for the layer pool base
        const size_t stride = l.slot_bytes();

        // The two bases the dispatch hands the kernel.
        const size_t packed_base = kPoolBase;
        const size_t ms_base = kPoolBase + l.packed_off();

        for (int idx = 0; idx < 24; ++idx) {  // 3 projections x top_k 8
            const size_t slot_begin = kPoolBase + static_cast<size_t>(idx) * stride;
            const size_t slot_end = slot_begin + stride;

            const size_t w = packed_base + static_cast<size_t>(idx) * stride;
            const size_t ms = ms_base + static_cast<size_t>(idx) * stride;

            EXPECT_EQ(w, slot_begin) << "weights, slot " << idx;
            EXPECT_GE(ms, slot_begin + l.packed_bytes) << "scales overlap weights, slot " << idx;
            EXPECT_LE(ms + l.ms_bytes, slot_end) << "scales spill into slot " << (idx + 1);
        }
    }
}

// Neighbouring slots must not overlap — an expert's scales spilling into the
// next slot's packed block is the silent-wrong-answer case.
TEST(NvFP4SlotLayout, SlotsDoNotOverlap) {
    const auto l = nvfp4_slot_layout(kDownN, kDownK);
    EXPECT_LE(l.packed_off() + l.ms_bytes, l.slot_bytes());
    EXPECT_LE(l.packed_bytes, l.packed_off());
}

// K must be a multiple of the 16-value micro-block the kernels hard-code. A
// checkpoint that violates it has no valid scale mapping, so the layout reports
// "path does not apply" rather than rounding it into something plausible.
TEST(NvFP4SlotLayout, RejectsShapesTheKernelsCannotAddress) {
    EXPECT_EQ(nvfp4_slot_layout(768, 2044).slot_bytes(), 0u);  // K % 16 != 0
    EXPECT_EQ(nvfp4_slot_layout(0, 2048).slot_bytes(), 0u);
    EXPECT_EQ(nvfp4_slot_layout(768, 0).slot_bytes(), 0u);
    EXPECT_EQ(nvfp4_slot_layout(-1, 2048).slot_bytes(), 0u);
}

// Phase 0 promotes host-resident weights only for MoE experts: dense weights
// have no host path, so labelling one NVFP4 where it sits would produce bytes
// nothing can serve.
TEST(NvFP4ExpertKey, MatchesPerExpertMoeWeightsOnly) {
    EXPECT_TRUE(is_expert_key("L0.expert_w_gate.0"));
    EXPECT_TRUE(is_expert_key("L47.expert_w_down.127"));
    EXPECT_TRUE(is_expert_key("L5.expert_w_up.7"));

    EXPECT_FALSE(is_expert_key("L5.wq"));
    EXPECT_FALSE(is_expert_key("L5.w_gate"));  // the DENSE gate, not an expert
    EXPECT_FALSE(is_expert_key("out_proj"));
    EXPECT_FALSE(is_expert_key(""));
    EXPECT_FALSE(is_expert_key("L5"));
    EXPECT_FALSE(is_expert_key("expert_w_gate.0"));  // no layer prefix
}

namespace {

// One projection's experts as the loader leaves them: host-resident, NVFP4,
// 2-D [N, K/2], each carrying its own micro-scale block.
std::vector<Tensor> make_host_experts(int n, int64_t N, int64_t K_packed) {
    static std::vector<char> backing(1 << 20);
    std::vector<Tensor> v(n);
    for (int i = 0; i < n; ++i) {
        v[i].data = backing.data() + i * 16;
        v[i].scales = backing.data() + 4096 + i * 16;
        v[i].qtype = QType::NVFP4;
        v[i].on_device = false;
        v[i].ndim = 2;
        v[i].shape[0] = N;
        v[i].shape[1] = K_packed;
        v[i].tensor_scale = 1.0f;
    }
    return v;
}

}  // namespace

TEST(NvFP4HostExpertsServable, AcceptsAPromotedHostResidentProjection) {
    EXPECT_TRUE(nvfp4_host_experts_servable(make_host_experts(8, kGateN, kGateK / 2)));
}

// Each rejection below is a way the dispatch would otherwise hand the staging
// loop something it cannot use.
TEST(NvFP4HostExpertsServable, RejectsWhatTheStagingLoopCannotUse) {
    EXPECT_FALSE(nvfp4_host_experts_servable({}));  // non-gated model: no gate projection

    auto device_resident = make_host_experts(8, kGateN, kGateK / 2);
    for (auto& e : device_resident)
        e.on_device = true;
    EXPECT_FALSE(nvfp4_host_experts_servable(device_resident))
        << "device-resident experts belong to the resident path, not the cache";

    // Phase 0 did not promote — the #1403 state. Serving it is what produced a
    // fluent wrong answer, so it must not look servable.
    auto unpromoted = make_host_experts(8, kGateN, kGateK / 2);
    for (auto& e : unpromoted)
        e.qtype = QType::INT8;
    EXPECT_FALSE(nvfp4_host_experts_servable(unpromoted));

    auto no_scales = make_host_experts(8, kGateN, kGateK / 2);
    for (auto& e : no_scales)
        e.scales = nullptr;
    EXPECT_FALSE(nvfp4_host_experts_servable(no_scales));

    // A ragged projection would give one launch two different strides.
    auto ragged = make_host_experts(8, kGateN, kGateK / 2);
    ragged[5].shape[0] = kGateN / 2;
    EXPECT_FALSE(nvfp4_host_experts_servable(ragged));

    // A defect in a LATER expert must be caught too — checking experts[0] only
    // is the cheap mistake here, and every routed expert reaches a kernel.
    auto late_host_miss = make_host_experts(8, kGateN, kGateK / 2);
    late_host_miss[7].scales = nullptr;
    EXPECT_FALSE(nvfp4_host_experts_servable(late_host_miss));
}
