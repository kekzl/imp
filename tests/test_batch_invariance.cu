// =============================================================================
// AUDIT_arch_2026 D-2: what joining a batch costs on the NATIVE NVFP4 path.
//
// WHY THIS EXISTS
//   M=1 decode is served by `gemv_nvfp4_kpar` with FP16 activations (W4A16)
//   for every weight that is in the NVFP4 decode cache; M <= 32 — note the
//   bound, it is not 2..32 — is served by `gemm_nvfp4_smallm_v2_a4`, which
//   quantizes the ACTIVATION rows to NVFP4 first (W4A4). Any weight the M=1
//   branch declines therefore takes the W4A4 kernel even solo, which a 2 %
//   error planted in that kernel proved on the checkpoint measured here
//   (src/exec/executor_gemm_dispatch.cu:183-237 and :356-415).
//   Every quality number imp had for that step came from either
//   `ForwardPassTest.DecodeLogitsInvariantToBatchComposition` — a 2-layer FP16
//   synthetic model that never runs an NVFP4 kernel — or the server-level
//   degeneration battery, which is coherence, not a numerical distance.
//   PR #1766 said so itself: "Teacher-forced PPL runs prefill (M=2048,
//   CUTLASS) and cannot reach the batched-decode-only path".
//
//   This test reaches it: the SAME token sequence is teacher-forced through
//   the SAME engine at M=1 and at M=32, one real checkpoint, and the two runs
//   are compared as distributions (NLL over the true continuation, top-1
//   agreement, per-token logprob delta) rather than as text.
//
// WHAT IT IS NOT
//   Not a determinism test: `IMP_DETERMINISTIC=1` removes run-to-run noise so
//   the delta measured is the batch effect. The M=1-vs-M=1 control arm is in
//   here for exactly that reason — it must read zero, or the M=32 number means
//   nothing (SETTLED: "a test whose input cannot reach the defect is worthless").
//   Not a batch-invariance guarantee either: imp deliberately has none
//   (docs/determinism.md). It is the missing NUMBER behind that scope decision.
// =============================================================================

#include <gtest/gtest.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "compute/gemm.h"
#include "exec/executor.h"
#include "memory/kv_cache.h"
#include "model/model.h"
#include "model/safetensors_loader.h"
#include "model/tokenizer.h"
#include "runtime/config.h"
#include "runtime/engine.h"
#include "test_models.h"

namespace imp {
namespace {

namespace fs = std::filesystem;

constexpr int kBatchRows = 32;    // the top of the smallM window (M <= 32)
constexpr int kPromptLen = 24;    // tokens prefilled before the measured window
constexpr int kDecodeSteps = 64;  // teacher-forced decode steps per arm
constexpr size_t kNeededMiB = 14000;

// ~1.5 kB of ordinary English prose: long enough to tokenize past
// kPromptLen + kDecodeSteps + kBatchRows offsets on any BPE vocabulary, and
// ordinary enough that the model is not in a degenerate low-entropy regime
// (a repetitive prompt would drive every logprob to ~0 and hide the effect).
constexpr const char* kCorpus =
    "The bandwidth of a memory system is not the same thing as its latency, and confusing the two "
    "leads to designs that look fast on paper and stall in practice. A decode step reads every "
    "weight of the model exactly once, so its floor is set by how many bytes the weights occupy "
    "divided by how fast the device can stream them. Arithmetic intensity is low, caches do not "
    "help much, and the only real levers are making the weights smaller or making the reads wider. "
    "Prefill is the opposite case: many tokens share the same weights, the matrices become tall, "
    "and the machine becomes limited by how many multiply-accumulate operations it can retire per "
    "cycle. Engineers who measure only one of the two phases end up optimizing the wrong half of "
    "the problem, and the resulting speedups do not survive contact with a real serving workload. "
    "Batching changes the shape of every matrix multiplication in the network. When a single "
    "request is served alone, the activation side of each layer is one row wide, and the kernel "
    "that runs is a matrix-vector product. When thirty-two requests are served together, the same "
    "layer becomes a small matrix-matrix product, a different kernel, and sometimes a different "
    "numerical format for the activations themselves. The weights do not change, the arithmetic "
    "does. Whether that difference is visible in the output distribution is an empirical question "
    "and not one that can be settled by inspecting the kernel source, which is why this file "
    "measures it against a real checkpoint instead of arguing about it.";

// -----------------------------------------------------------------------------
// One decode step, driven straight at the executor so the fed token is OURS.
// The engine's own loop samples; teacher forcing needs the opposite. Buffers
// are allocated once and reused — 3 arms x 64 steps of cudaMalloc/cudaFree
// would dominate the runtime and add allocator noise to the measurement.
// -----------------------------------------------------------------------------
class TeacherForcedDriver {
public:
    TeacherForcedDriver(GraphExecutor& ex, KVCache& kv, int max_rows, int max_blocks_per_seq, int vocab)
        : ex_(ex), kv_(kv), max_blocks_per_seq_(max_blocks_per_seq), vocab_(vocab) {
        cudaMalloc(&d_tokens_, max_rows * sizeof(int32_t));
        cudaMalloc(&d_pos_, max_rows * sizeof(int));
        cudaMalloc(&d_ctx_, max_rows * sizeof(int));
        cudaMalloc(&d_bt_, (size_t)max_rows * max_blocks_per_seq * sizeof(int));
        host_logits_.resize((size_t)max_rows * vocab);
    }
    ~TeacherForcedDriver() {
        cudaFree(d_tokens_);
        cudaFree(d_pos_);
        cudaFree(d_ctx_);
        cudaFree(d_bt_);
    }
    TeacherForcedDriver(const TeacherForcedDriver&) = delete;
    TeacherForcedDriver& operator=(const TeacherForcedDriver&) = delete;

    // Prefill one sequence (n_sequences = 1, exactly as the engine prefills)
    // into an explicit set of physical blocks.
    void prefill(const std::vector<int32_t>& tokens, const std::vector<int>& blocks) {
        const int n = static_cast<int>(tokens.size());
        // Workspace discipline, copied from Engine::step_prefill: slot 1 is the
        // single-row decode workspace and its row buffers are one token tall,
        // so a prefill left on it dies in slice_rows (shape[0]=1).
        if (ex_.has_decode_workspace())
            ex_.use_workspace(0);
        (void)ex_.resize_workspace(n, nullptr);
        std::vector<int> positions(n);
        for (int i = 0; i < n; i++)
            positions[i] = i;
        cudaMemcpy(d_tokens_, tokens.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_, positions.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bt_, blocks.data(), blocks.size() * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ctx_, &n, sizeof(int), cudaMemcpyHostToDevice);

        InferenceState state;
        state.token_ids = d_tokens_;
        state.positions = d_pos_;
        state.n_tokens = n;
        state.is_prefill = true;
        state.n_sequences = 1;
        state.kv_cache = &kv_;
        state.block_tables = d_bt_;
        state.max_blocks_per_seq = static_cast<int>(blocks.size());
        state.max_context_len = n;
        state.context_lens = d_ctx_;

        Tensor logits;
        ex_.forward_logits(state, logits, nullptr);
        cudaDeviceSynchronize();
    }

    // One decode step for `n` rows. block_tables is [n, max_blocks_per_seq]
    // row-major. Returns the host-side [n, vocab] logits.
    const std::vector<float>& step(const std::vector<int32_t>& tokens, const std::vector<int>& positions,
                                   const std::vector<int>& ctx_lens, const std::vector<int>& block_tables) {
        const int n = static_cast<int>(tokens.size());
        // Same slot rule the engine applies per step (engine_scheduler.cpp):
        // one row on the dedicated decode workspace, more rows on slot 0 sized
        // to the batch. Getting this wrong would silently change which kernel
        // the arm runs, which is the very thing being compared.
        if (n == 1 && ex_.has_decode_workspace()) {
            ex_.use_workspace(1);
        } else {
            if (ex_.active_workspace() == 1)
                ex_.use_workspace(0);
            (void)ex_.resize_workspace(n, nullptr);
        }
        cudaMemcpy(d_tokens_, tokens.data(), n * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pos_, positions.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_ctx_, ctx_lens.data(), n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bt_, block_tables.data(), block_tables.size() * sizeof(int), cudaMemcpyHostToDevice);

        InferenceState state;
        state.token_ids = d_tokens_;
        state.positions = d_pos_;
        state.n_tokens = n;
        state.is_prefill = false;
        state.n_sequences = n;
        state.kv_cache = &kv_;
        state.block_tables = d_bt_;
        state.max_blocks_per_seq = max_blocks_per_seq_;
        state.max_context_len = *std::max_element(ctx_lens.begin(), ctx_lens.end());
        state.context_lens = d_ctx_;

        Tensor logits;
        ex_.forward_logits(state, logits, nullptr);
        cudaDeviceSynchronize();
        cudaMemcpy(host_logits_.data(), logits.data, (size_t)n * vocab_ * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return host_logits_;
    }

private:
    GraphExecutor& ex_;
    KVCache& kv_;
    int max_blocks_per_seq_;
    int vocab_;
    int32_t* d_tokens_ = nullptr;
    int* d_pos_ = nullptr;
    int* d_ctx_ = nullptr;
    int* d_bt_ = nullptr;
    std::vector<float> host_logits_;
};

// Per-step record for the row under test.
struct StepRecord {
    double target_logprob = 0.0;  // log p(true next token)
    int argmax = -1;
    double max_abs_logit = 0.0;  // largest |logit| in the row, for scale
};

// Numerically stable log-softmax of one row, evaluated at `target`.
StepRecord score_row(const float* logits, int vocab, int32_t target) {
    StepRecord r;
    double m = -1e300;
    for (int i = 0; i < vocab; i++)
        if (logits[i] > m) {
            m = logits[i];
            r.argmax = i;
        }
    double sum = 0.0;
    for (int i = 0; i < vocab; i++)
        sum += std::exp(static_cast<double>(logits[i]) - m);
    r.target_logprob = static_cast<double>(logits[target]) - m - std::log(sum);
    for (int i = 0; i < vocab; i++)
        r.max_abs_logit = std::max(r.max_abs_logit, std::fabs(static_cast<double>(logits[i])));
    return r;
}

struct ArmResult {
    std::vector<StepRecord> steps;
    double mean_nll() const {
        double s = 0.0;
        for (const StepRecord& r : steps)
            s += -r.target_logprob;
        return steps.empty() ? 0.0 : s / static_cast<double>(steps.size());
    }
};

// Compare two arms on the row under test. `label` names the comparison.
struct Comparison {
    double nll_delta = 0.0;    // mean NLL(b) - mean NLL(a)
    double mean_abs_lp = 0.0;  // mean |logprob(b) - logprob(a)| of the true token
    double max_abs_lp = 0.0;
    double top1_agreement = 0.0;  // fraction of steps with the same argmax
    int top1_flips = 0;
};

Comparison compare(const ArmResult& a, const ArmResult& b) {
    Comparison c;
    const size_t n = std::min(a.steps.size(), b.steps.size());
    double sum_abs = 0.0;
    int same = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = std::fabs(b.steps[i].target_logprob - a.steps[i].target_logprob);
        sum_abs += d;
        c.max_abs_lp = std::max(c.max_abs_lp, d);
        if (a.steps[i].argmax == b.steps[i].argmax)
            same++;
    }
    c.nll_delta = b.mean_nll() - a.mean_nll();
    c.mean_abs_lp = n ? sum_abs / static_cast<double>(n) : 0.0;
    c.top1_agreement = n ? static_cast<double>(same) / static_cast<double>(n) : 0.0;
    c.top1_flips = static_cast<int>(n) - same;
    return c;
}

void print_comparison(const char* label, const ArmResult& a, const ArmResult& b, const Comparison& c) {
    printf(
        "[batch-invariance %-18s] nll %.6f -> %.6f (delta %+.6f) | mean|dlogp| %.3e max %.3e | "
        "top-1 %.4f (%d flips of %zu)\n",
        label, a.mean_nll(), b.mean_nll(), c.nll_delta, c.mean_abs_lp, c.max_abs_lp, c.top1_agreement,
        c.top1_flips, b.steps.size());
}

constexpr const char* kDefaultModel = "/models/Qwen3-14B-NVFP4";

std::string model_path() {
    if (const char* p = getenv(imp_test::kEnvModelNvfp4))
        return p;
    return kDefaultModel;
}

size_t device_free_mib() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess)
        return 0;
    return free_b >> 20;
}

EngineConfig batch_invariance_config() {
    EngineConfig cfg;
    cfg.max_batch_size = kBatchRows;
    cfg.max_seq_len = 512;
    cfg.use_cuda_graphs = false;  // the driver calls forward_logits directly
    cfg.use_pdl = false;
    cfg.use_fp8_prefill = false;
    cfg.use_nvfp4_decode = 2;
    cfg.kv_cache_dtype = QType::F16;  // pin it: an auto upgrade would confound the arms
    cfg.compute_dtype = QType::F16;
    cfg.kv_block_size = 16;
    cfg.use_green_contexts = false;
    cfg.gpu_layers = -1;
    cfg.use_prefix_caching = false;
    cfg.use_mxfp4_prefill = false;
    cfg.dual_path_quant = false;
    return cfg;
}

// Run the measured row through `n_rows` decode steps. Row 0 is always the
// sequence under test; rows 1..n-1 carry DIFFERENT content at DIFFERENT
// lengths, so a bug that keys off a per-row context length cannot hide.
ArmResult run_arm(TeacherForcedDriver& drv, const std::vector<int32_t>& toks, int n_rows, int blocks_per_seq,
                  int vocab) {
    // Row 0 is the sequence under test: corpus from token 0, full prompt.
    // Rows 1+ start at a different offset and are shortened by up to 4 tokens,
    // so neither the content nor the per-row context length matches row 0.
    std::vector<int> starts(n_rows), lens(n_rows);
    for (int r = 0; r < n_rows; r++) {
        starts[r] = (r == 0) ? 0 : (kPromptLen / 2 + r);
        lens[r] = kPromptLen - (r == 0 ? 0 : (r % 5));
    }

    for (int r = 0; r < n_rows; r++) {
        std::vector<int> blocks(blocks_per_seq);
        for (int b = 0; b < blocks_per_seq; b++)
            blocks[b] = r * blocks_per_seq + b;
        std::vector<int32_t> prompt(toks.begin() + starts[r], toks.begin() + starts[r] + lens[r]);
        drv.prefill(prompt, blocks);
    }

    std::vector<int> block_tables((size_t)n_rows * blocks_per_seq);
    for (int r = 0; r < n_rows; r++)
        for (int b = 0; b < blocks_per_seq; b++)
            block_tables[(size_t)r * blocks_per_seq + b] = r * blocks_per_seq + b;

    ArmResult out;
    out.steps.reserve(kDecodeSteps);
    for (int s = 0; s < kDecodeSteps; s++) {
        std::vector<int32_t> fed(n_rows);
        std::vector<int> pos(n_rows), ctx(n_rows);
        for (int r = 0; r < n_rows; r++) {
            // Teacher forcing: the token fed at position lens[r]+s is the TRUE
            // corpus token there, never a sampled one — so the row's history is
            // identical in every arm no matter what the model predicted.
            fed[r] = toks[starts[r] + lens[r] + s];
            pos[r] = lens[r] + s;
            ctx[r] = lens[r] + s + 1;
        }
        const std::vector<float>& logits = drv.step(fed, pos, ctx, block_tables);
        // Row 0's logits predict the token AFTER the one just fed.
        out.steps.push_back(score_row(logits.data(), vocab, toks[kPromptLen + s + 1]));
    }
    return out;
}

}  // namespace

// The instrument. One checkpoint, three arms, two comparisons:
//   control  M=1 vs M=1  — must be identically zero under IMP_DETERMINISTIC=1,
//                          otherwise the M=32 number is run-to-run noise.
//   measured M=1 vs M=32 — the W4A16 -> W4A4 activation step, priced.
TEST(BatchInvarianceTest, NativeNvfp4DecodeSoloVsBatched) {
    const std::string path = model_path();
    if (!fs::exists(path + "/config.json"))
        GTEST_SKIP() << "native-NVFP4 checkpoint not present at " << path << " (set "
                     << imp_test::kEnvModelNvfp4 << ")";
    const size_t free_mib = device_free_mib();
    if (free_mib < kNeededMiB)
        GTEST_SKIP() << "needs ~" << kNeededMiB << " MiB free, card has " << free_mib << " MiB";

    // Both arms must be bit-stable on their own before a difference between
    // them means anything. Must be set before model/engine creation.
    setenv("IMP_DETERMINISTIC", "1", 1);

    std::shared_ptr<Model> model = load_safetensors(path, /*load_mtp_head=*/false);
    ASSERT_NE(model, nullptr) << "failed to load " << path;
    ASSERT_TRUE(model->upload_weights_gpu(QType::F16, nullptr, 1ULL << 30));
    gemm_init();

    const int vocab = static_cast<int>(model->config().vocab_size);
    std::vector<int32_t> toks = model->tokenizer()->encode(kCorpus);
    // Deepest read: row 31 starts at kPromptLen/2+31 and walks kDecodeSteps on
    // top of its own prompt; row 0's target is one past its last fed token.
    ASSERT_GT(static_cast<int>(toks.size()), 2 * kPromptLen + kDecodeSteps + kBatchRows + 8)
        << "corpus tokenized to only " << toks.size() << " tokens";

    // ---- Arm set 1: shipping defaults (M=2..32 -> smallM W4A4) -------------
    ArmResult solo_a4, batched_a4;
    Comparison control{}, measured_a4{};
    int blocks_per_seq = 0;
    {
        RuntimeConfig rc;
        rc.runtime.deterministic = true;
        rc.server.prefix_cache = false;
        rc.gemm.nvfp4_smallm = true;
        set_pending_runtime_config(rc);

        Engine engine;
        ASSERT_TRUE(engine.init(model, batch_invariance_config()));
        GraphExecutor* ex = engine.executor();
        KVCache* kv = engine.kv_cache();
        ASSERT_NE(ex, nullptr);
        ASSERT_NE(kv, nullptr);

        const int block_size = kv->block_size();
        blocks_per_seq = (kPromptLen + kDecodeSteps + 2 + block_size - 1) / block_size;
        ASSERT_GE(kv->total_blocks(), kBatchRows * blocks_per_seq)
            << "KV pool too small for " << kBatchRows << " rows";

        TeacherForcedDriver drv(*ex, *kv, kBatchRows, blocks_per_seq, vocab);
        solo_a4 = run_arm(drv, toks, 1, blocks_per_seq, vocab);
        const ArmResult solo_repeat = run_arm(drv, toks, 1, blocks_per_seq, vocab);
        batched_a4 = run_arm(drv, toks, kBatchRows, blocks_per_seq, vocab);

        control = compare(solo_a4, solo_repeat);
        measured_a4 = compare(solo_a4, batched_a4);
        print_comparison("control M=1/M=1", solo_a4, solo_repeat, control);
        print_comparison("W4A4 M=1/M=32", solo_a4, batched_a4, measured_a4);
    }

    // ---- Arm set 2: a SECOND engine on the same config --------------------
    // The cross-engine baseline. A fresh engine re-plans its arena and can pick
    // different split-K / cache geometry, so M=1 is not bit-stable across inits
    // (the documented "NON-DETERMINISTIC across fresh contexts" boundary,
    // tests/refs/e2e_greedy_locks.h). Without this number, arm set 3's M=1
    // difference would be misread as an effect of the switch it flips.
    Comparison reinit{};
    {
        RuntimeConfig rc;
        rc.runtime.deterministic = true;
        rc.server.prefix_cache = false;
        rc.gemm.nvfp4_smallm = true;
        set_pending_runtime_config(rc);

        Engine engine;
        ASSERT_TRUE(engine.init(model, batch_invariance_config()));
        TeacherForcedDriver drv(*engine.executor(), *engine.kv_cache(), kBatchRows, blocks_per_seq, vocab);
        const ArmResult solo = run_arm(drv, toks, 1, blocks_per_seq, vocab);
        reinit = compare(solo_a4, solo);
        print_comparison("re-init M=1/M=1", solo_a4, solo, reinit);
    }

    // ---- Arm set 3: the same batch WITHOUT activation quantization --------
    // `gemm.nvfp4_smallm=false` routes 2..32 rows to the dequant/CUTLASS GEMM,
    // i.e. W4A16 at every M. It is the switch an operator would flip, and it
    // is the only way to tell "batching changed the GEMM shape" from "batching
    // quantized the activations": both arms below share the shape.
    ArmResult batched_w4a16;
    Comparison measured_w4a16{}, solo_vs_solo_cfg{};
    {
        RuntimeConfig rc;
        rc.runtime.deterministic = true;
        rc.server.prefix_cache = false;
        rc.gemm.nvfp4_smallm = false;
        set_pending_runtime_config(rc);

        Engine engine;
        ASSERT_TRUE(engine.init(model, batch_invariance_config()));
        TeacherForcedDriver drv(*engine.executor(), *engine.kv_cache(), kBatchRows, blocks_per_seq, vocab);
        const ArmResult solo = run_arm(drv, toks, 1, blocks_per_seq, vocab);
        batched_w4a16 = run_arm(drv, toks, kBatchRows, blocks_per_seq, vocab);

        solo_vs_solo_cfg = compare(solo_a4, solo);
        measured_w4a16 = compare(solo, batched_w4a16);
        print_comparison("smallm-off M=1/M=1", solo_a4, solo, solo_vs_solo_cfg);
        print_comparison("W4A16 M=1/M=32", solo, batched_w4a16, measured_w4a16);
    }

    // The control arm is the test's own calibration: two identical M=1 runs of
    // the same engine. Anything but exact agreement means the deterministic
    // mode did not hold and the measured arms are noise, not batching.
    EXPECT_EQ(control.top1_flips, 0) << "M=1 disagreed with ITSELF: deterministic mode is not "
                                        "holding, so the M=32 deltas are unattributable";
    EXPECT_LT(control.max_abs_lp, 1e-9) << "M=1 vs M=1 logprobs differ by " << control.max_abs_lp;

    // Second calibration, and a boundary worth knowing: re-initializing the
    // SAME configuration reproduces M=1 bit for bit (the arm above prints
    // zero), but turning smallM off — which removes a tenant from the engine
    // arena and touches no M=1 kernel — moves M=1 anyway. That is the
    // address/alignment boundary, measured, and it is why every batched
    // comparison here is taken WITHIN one engine. It must stay smaller than
    // the batch effect itself, or the arms below are comparing the wrong thing.
    EXPECT_LT(solo_vs_solo_cfg.mean_abs_lp, measured_a4.mean_abs_lp)
        << "flipping an unrelated config knob moved M=1 by " << solo_vs_solo_cfg.mean_abs_lp
        << ", as much as batching itself (" << measured_a4.mean_abs_lp
        << "): the solo baseline is too unstable for the batched numbers to mean anything";

    // Regression fences around the numbers this test printed on
    // Qwen3-14B-NVFP4 (docs/PERF.md "Batch invariance"). They are NOT an
    // invariance claim — imp guarantees none (docs/determinism.md) — they fail
    // if the gap grows by an order of magnitude, which is what a broken
    // activation-quantization step or a wrong smallM tile would do.
    EXPECT_LT(std::fabs(measured_a4.nll_delta), 0.15)
        << "mean NLL moved by " << measured_a4.nll_delta << " between M=1 and M=32";
    EXPECT_GT(measured_a4.top1_agreement, 0.75)
        << "greedy top-1 agreement between M=1 and M=32 fell to " << measured_a4.top1_agreement;
    EXPECT_LT(measured_a4.mean_abs_lp, 0.6)
        << "mean |delta logprob| between M=1 and M=32 is " << measured_a4.mean_abs_lp;

    // Same fence on the W4A16 batched arm. Deliberately NOT an ordering
    // assertion between the two: on the checkpoint measured here the two are
    // the same size, i.e. the batch effect is the batch SHAPE and not the
    // activation format (docs/PERF.md). Pinning an order would freeze that
    // reading of one model into a gate.
    EXPECT_LT(measured_w4a16.mean_abs_lp, 0.6)
        << "mean |delta logprob| between M=1 and M=32 without smallM is " << measured_w4a16.mean_abs_lp;

    // Absolute anchor, and the reason it is here: every assertion above is a
    // DIFFERENCE, so an error that hits both arms equally cancels. A planted
    // 2 % output-scale error in the smallM A4 kernel did exactly that — all
    // five deltas stayed inside their fences while every absolute NLL moved by
    // 0.028. Only a pinned NLL catches that class, and only on the pinned
    // checkpoint, so the anchor is skipped when the model is overridden.
    // Re-pin deliberately when a kernel change is meant to move numerics.
    if (path == kDefaultModel) {
        EXPECT_NEAR(solo_a4.mean_nll(), 3.4403, 0.02)
            << "teacher-forced M=1 NLL on " << kDefaultModel << " moved to " << solo_a4.mean_nll()
            << " (pinned 3.4403 on 2026-09-06, docs/PERF.md). A uniform shift like this is a "
               "kernel numerics change, which the solo-vs-batched deltas cannot see.";
    }
}

}  // namespace imp
