// FP8 E4M3 KV scale per layer: kv_cache.fp8_scale = legacy | checkpoint | calibrate.
// Scale selection is pure (core/kv_fp8_scale.h); this TU wires it to the model and the warmup.

#include "runtime/engine.h"
#include "runtime/engine_internal.h"
#include "runtime/kv_fp8_scale_init.h"
#include "runtime/config.h"
#include "runtime/request.h"
#include "exec/executor.h"
#include "memory/kv_cache_manager.h"
#include "model/tokenizer.h"
#include "core/kv_fp8_scale.h"
#include "core/logging.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>

namespace imp {

namespace {

// Fixed calibration text: prose, dialogue, code, numbers. Static per model load, never request data
// (a scale from live traffic makes request B depend on request A, docs/DETERMINISM.md).
constexpr const char* kFp8KvCalibrationText =
    "The harbour was quiet when the ferry came in, its lights smeared across the water. Mara counted "
    "the passengers twice, because the manifest said forty-one and she saw only thirty-eight.\n"
    "\"Did anyone stay below?\" she asked. The captain shrugged. \"Three always do. They sleep through "
    "everything, even the horn.\"\n\n"
    "In 1887 the town had 2,413 inhabitants; by 1952 it had shrunk to 906, and today the census lists "
    "1,174 people, 37% of them over sixty-five. The average rainfall is 1,260 mm per year.\n\n"
    "def moving_average(xs, k):\n"
    "    if k <= 0:\n"
    "        raise ValueError(\"k must be positive\")\n"
    "    out, acc = [], 0.0\n"
    "    for i, x in enumerate(xs):\n"
    "        acc += x\n"
    "        if i >= k:\n"
    "            acc -= xs[i - k]\n"
    "        out.append(acc / min(i + 1, k))\n"
    "    return out\n\n"
    "Question: A train leaves at 09:40 and travels 312 km at 78 km/h. When does it arrive?\n"
    "Answer: 312 / 78 = 4 hours, so it arrives at 13:40.\n\n"
    "- Preheat the oven to 200 C.\n- Mix 250 g flour, 2 eggs and 300 ml milk.\n- Bake for 25 minutes.\n\n"
    "Die Sitzung wurde um 14 Uhr eroeffnet. Le comite a approuve le budget. El informe final llega "
    "manana. Summary: the committee approved the budget, the final report arrives tomorrow, and the "
    "next meeting is on the third Thursday of the month.";

// E4M3 edge counts of the calibration K/V under three scale choices, summed over layers: legacy (1.0
// floor), shared (max of K, V) and split, at the configured headroom. Measures what a scale moves.
void log_fp8_kv_edge_counts(const std::vector<uint32_t>& hist, const std::vector<float>& amax,
                            float headroom) {
    const size_t n_kv = amax.size() / 2;
    if (hist.size() < 2 * n_kv * kFp8KvHistBins)
        return;
    const char* arms[3] = {"legacy", "shared", "split"};
    for (int arm = 0; arm < 3; ++arm) {
        Fp8KvEdgeCounts sum[2];
        for (size_t l = 0; l < n_kv; ++l) {
            const float ka = amax[2 * l];
            const float va = amax[(2 * l) + 1];
            Fp8KvScalePair s;
            if (arm == 0) {
                const float legacy = fp8_kv_pow2_ceil(std::max(1.0f, std::max(ka, va) / kFp8KvE4M3Max));
                s = {legacy, legacy};
            } else {
                s = fp8_kv_scales_from_amax(ka, va, headroom, arm == 2);
            }
            for (int t = 0; t < 2; ++t) {
                const Fp8KvEdgeCounts c = fp8_kv_edge_counts(&hist[((2 * l) + t) * kFp8KvHistBins],
                                                             t == 0 ? s.k : s.v);
                sum[t].nonzero += c.nonzero;
                sum[t].flushed += c.flushed;
                sum[t].subnormal += c.subnormal;
                sum[t].saturated += c.saturated;
            }
        }
        IMP_LOG_INFO(
            "kv_cache.fp8_scale=calibrate edges %s: K flushed %llu subnormal %llu saturated %llu of %llu, "
            "V flushed %llu subnormal %llu saturated %llu of %llu",
            arms[arm], static_cast<unsigned long long>(sum[0].flushed),
            static_cast<unsigned long long>(sum[0].subnormal),
            static_cast<unsigned long long>(sum[0].saturated),
            static_cast<unsigned long long>(sum[0].nonzero), static_cast<unsigned long long>(sum[1].flushed),
            static_cast<unsigned long long>(sum[1].subnormal),
            static_cast<unsigned long long>(sum[1].saturated),
            static_cast<unsigned long long>(sum[1].nonzero));
    }
}

}  // namespace

Fp8KvScaleMode resolve_fp8_kv_scale_mode(const RuntimeConfig& rc) {
    const auto mode = parse_fp8_kv_scale_mode(rc.kv_cache.fp8_scale);
    if (!mode)
        throw std::runtime_error("kv_cache.fp8_scale=\"" + rc.kv_cache.fp8_scale +
                                 "\" is not one of legacy|checkpoint|calibrate");
    return *mode;
}

void apply_checkpoint_fp8_kv_scales(GraphExecutor& ex, const Model& model,
                                    const std::vector<int>& kv_layer_map, QType kv_dtype,
                                    const RuntimeConfig& rc) {
    const Fp8KvScaleMode mode = resolve_fp8_kv_scale_mode(rc);  // refuses a typo on every dtype
    if (kv_dtype != QType::FP8_E4M3 || mode != Fp8KvScaleMode::Checkpoint)
        return;
    const auto& mc = model.config();
    const int n_kv = static_cast<int>(ex.kv_scales().size());
    const bool split = rc.kv_cache.fp8_scale_split;
    std::vector<Fp8KvScalePair> scales(n_kv, Fp8KvScalePair{0.0f, 0.0f});
    int present = 0;
    int below_one = 0;
    for (int l = 0; l < static_cast<int>(kv_layer_map.size()) && l < mc.n_layers; ++l) {
        const int kv = kv_layer_map[l];
        if (kv < 0 || kv >= n_kv || mc.layer_kv_shared(l))
            continue;
        const auto& layer = model.layer(l);
        if (layer.kv_ckpt_k_scale > 0.0f || layer.kv_ckpt_v_scale > 0.0f)
            ++present;
        const Fp8KvScalePair s = fp8_kv_scales_from_checkpoint(layer.kv_ckpt_k_scale, layer.kv_ckpt_v_scale,
                                                               split);
        scales[kv] = {std::max(scales[kv].k, s.k), std::max(scales[kv].v, s.v)};
    }
    for (Fp8KvScalePair& s : scales) {
        s.k = s.k > 0.0f ? s.k : 1.0f;
        s.v = s.v > 0.0f ? s.v : 1.0f;
        below_one += (s.k < 1.0f || s.v < 1.0f) ? 1 : 0;
    }
    ex.set_static_kv_scales(scales);
    if (present == 0)
        IMP_LOG_WARN(
            "kv_cache.fp8_scale=checkpoint: no k_scale / v_scale tensor in the checkpoint, "
            "static 1.0 on all %d KV layers (the vLLM default)",
            n_kv);
    else
        IMP_LOG_INFO(
            "kv_cache.fp8_scale=checkpoint: %d of %d KV layers carry a checkpoint scale, %d "
            "below 1.0 (static, power of two)",
            present, n_kv, below_one);
}

uint64_t prefix_cache_fingerprint(uint64_t model_fingerprint, QType kv_dtype, const RuntimeConfig& rc) {
    if (kv_dtype != QType::FP8_E4M3)
        return model_fingerprint;
    const Fp8KvScaleMode mode = resolve_fp8_kv_scale_mode(rc);
    if (mode == Fp8KvScaleMode::Legacy)
        return model_fingerprint;
    // One FNV-1a step over the mode byte (+16 split): a cache written under another mode never matches.
    const uint64_t tag = (static_cast<uint64_t>(mode) + 1u) + (rc.kv_cache.fp8_scale_split ? 16u : 0u);
    return (model_fingerprint ^ tag) * 0x100000001b3ULL;
}

bool prefix_cache_restore_allowed(QType kv_dtype, const RuntimeConfig& rc) {
    return kv_dtype != QType::FP8_E4M3 || resolve_fp8_kv_scale_mode(rc) != Fp8KvScaleMode::Calibrate;
}

void calibrate_fp8_kv_scales(Engine& engine, QType kv_dtype, int prefill_chunk_size, bool run_forward) {
    const RuntimeConfig& rc = engine.runtime_config();
    GraphExecutor* ex = engine.executor();
    if (ex == nullptr || kv_dtype != QType::FP8_E4M3 ||
        resolve_fp8_kv_scale_mode(rc) != Fp8KvScaleMode::Calibrate)
        return;
    if (!run_forward) {
        if (!ex->kv_scales_calibrated())
            IMP_LOG_WARN(
                "kv_cache.fp8_scale=calibrate: no calibration ran (warmup skipped its forward, or a "
                "WARN above refused it), the legacy FP8 KV scale applies");
        return;
    }
    Tokenizer* tok = engine.model() ? engine.model()->tokenizer() : nullptr;
    if (tok == nullptr) {
        IMP_LOG_WARN("kv_cache.fp8_scale=calibrate: no tokenizer, keeping the legacy FP8 KV scale");
        return;
    }
    std::vector<int32_t> ids = tok->encode(kFp8KvCalibrationText);
    if (tok->add_bos() && tok->bos_id() >= 0)
        ids.insert(ids.begin(), tok->bos_id());
    const int chunk = prefill_chunk_size;
    if (ids.empty() || (chunk > 0 && static_cast<int>(ids.size()) > chunk)) {
        // One chunk only: fp8_kv_inv_scale_ records the amax of the first write per layer.
        IMP_LOG_WARN(
            "kv_cache.fp8_scale=calibrate: %zu calibration tokens do not fit one prefill chunk "
            "(%d), keeping the legacy FP8 KV scale",
            ids.size(), chunk);
        return;
    }

    ex->begin_kv_scale_calibration();
    auto req = std::make_shared<Request>();
    req->input_tokens = ids;
    req->max_tokens = 1;
    req->temperature = 0.0f;
    req->ignore_eos = true;
    engine.add_request(req);  // public path, assigns the id
    for (int i = 0; i < 8 && req->status != RequestStatus::FINISHED; i++)
        (void)engine.step();
    engine.kv_manager()->free_sequence(req->id);
    engine.reset_ssm_state(req->id);
    while (engine.kv_manager()->evict_cached_block()) {
    }  // no block written under the calibration scale survives
    req->status = RequestStatus::CANCELLED;

    const std::vector<float>& amax = ex->kv_calibration_amax();  // [2l] K, [2l+1] V
    const float headroom = rc.kv_cache.fp8_scale_headroom;
    const bool split = rc.kv_cache.fp8_scale_split;
    const size_t n_kv = amax.size() / 2;
    std::vector<Fp8KvScalePair> scales(n_kv);
    std::string per_layer;
    int below_one = 0;
    int missing = 0;
    for (size_t l = 0; l < n_kv; ++l) {
        const float ka = amax[2 * l];
        const float va = amax[(2 * l) + 1];
        missing += (ka <= 0.0f && va <= 0.0f) ? 1 : 0;  // not written by the calibration prefill: 1.0
        scales[l] = fp8_kv_scales_from_amax(ka, va, headroom, split);
        below_one += (scales[l].k < 1.0f || scales[l].v < 1.0f) ? 1 : 0;
        char buf[80];
        std::snprintf(buf, sizeof(buf), "%s%zu:%.3g/%.3g/2^%d/2^%d", l ? " " : "", l, ka, va,
                      std::ilogb(scales[l].k), std::ilogb(scales[l].v));
        per_layer += buf;
    }
    ex->set_static_kv_scales(scales);
    IMP_LOG_INFO(
        "kv_cache.fp8_scale=calibrate: %zu tokens, headroom %.2f, split %d, %d of %zu KV layers below "
        "1.0, %d not written",
        ids.size(), headroom, split ? 1 : 0, below_one, n_kv, missing);
    IMP_LOG_INFO("kv_cache.fp8_scale=calibrate: layer:K amax/V amax/K scale/V scale %s", per_layer.c_str());
    log_fp8_kv_edge_counts(ex->kv_calibration_hist(), amax, headroom);
}

}  // namespace imp
