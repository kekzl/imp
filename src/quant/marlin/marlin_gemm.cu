// imp-native launcher for the vendored Marlin W4A16 kernel (NVFP4 weights,
// FP16 activations). Port of marlin_mm() from vLLM's marlin.cu (Apache-2.0)
// specialized to: a=fp16, c=fp16, b=fp4-e2m1, s=fp8 trick format,
// group_size=16 (group_blocks=1), no act_order, no zero-points, no bias,
// stages=4. Kernel instantiations live in marlin_kernels_fp4_*.cu.

#include "marlin_kernel.h"
#include "marlin_w4a16.h"
#include "core/logging.h"

#include <algorithm>

namespace imp {
namespace marlin_w4a16 {

namespace {

constexpr vllm::ScalarTypeId kAId = vllm::kFloat16.id();
constexpr vllm::ScalarTypeId kBId = vllm::kFE2M1f.id();
constexpr vllm::ScalarTypeId kCId = vllm::kFloat16.id();
constexpr vllm::ScalarTypeId kSId = vllm::kFE4M3fn.id();

using FnPtr = void (*)(MARLIN_KERNEL_PARAMS);

struct KernelEntry {
    int threads;
    int thread_m_blocks;
    int thread_n_blocks;
    int thread_k_blocks;
    bool m_block_size_8;
    FnPtr fn;
};

#define IMP_MARLIN_ENTRY(TH, TM, TNB, TKB, M8) \
    {TH,  TM, TNB,                             \
     TKB, M8, MARLIN_NAMESPACE_NAME::Marlin<kAId, kBId, kCId, kSId, TH, TM, TNB, TKB, M8, 4, 1, false>}

#define IMP_MARLIN_CFG(TH, TNB, TKB)                                                        \
    IMP_MARLIN_ENTRY(TH, 1, TNB, TKB, true), IMP_MARLIN_ENTRY(TH, 1, TNB, TKB, false),      \
        IMP_MARLIN_ENTRY(TH, 2, TNB, TKB, false), IMP_MARLIN_ENTRY(TH, 3, TNB, TKB, false), \
        IMP_MARLIN_ENTRY(TH, 4, TNB, TKB, false)

const KernelEntry kKernels[] = {
    IMP_MARLIN_CFG(256, 8, 8),   // thread_k=128, thread_n=128
    IMP_MARLIN_CFG(256, 16, 4),  // thread_k=64,  thread_n=256
    IMP_MARLIN_CFG(128, 8, 4),   // thread_k=64,  thread_n=128
    IMP_MARLIN_CFG(128, 4, 8),   // thread_k=128, thread_n=64
};

#undef IMP_MARLIN_CFG
#undef IMP_MARLIN_ENTRY

struct ThreadConfig {
    int thread_k;
    int thread_n;
    int num_threads;
};

// Ordered by priority (vLLM small/large_batch_thread_configs).
const ThreadConfig kSmallBatchConfigs[] = {{128, 128, 256}, {64, 128, 128}, {128, 64, 128}};
const ThreadConfig kLargeBatchConfigs[] = {{64, 256, 256}, {64, 128, 128}, {128, 64, 128}};

FnPtr find_kernel(int threads, int thread_m_blocks, int thread_n_blocks, int thread_k_blocks,
                  bool m_block_size_8) {
    for (const auto& e : kKernels) {
        if (e.threads == threads && e.thread_m_blocks == thread_m_blocks &&
            e.thread_n_blocks == thread_n_blocks && e.thread_k_blocks == thread_k_blocks &&
            e.m_block_size_8 == m_block_size_8)
            return e.fn;
    }
    return nullptr;
}

struct DeviceInfo {
    int sms = 0;
    int max_shared_mem = 0;
};

const DeviceInfo& device_info() {
    static DeviceInfo info = [] {
        DeviceInfo d;
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&d.sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&d.max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
        return d;
    }();
    return info;
}

// Shared-memory footprint of one pipeline (port of get_kernel_cache_size for
// our fixed case: fp16 A, 4-bit B, group_size 16, no act_order/zp/bias-only).
int kernel_cache_size(int thread_k, int thread_n, int thread_m_blocks) {
    constexpr int stages = 4;
    const int tb_m = thread_m_blocks * 16;
    const int sh_a_size = stages * tb_m * thread_k * 2;
    const int sh_b_size = stages * (thread_k * thread_n / 8) * 4;
    const int sh_red_size = tb_m * (thread_n + 8) * 2;
    const int sh_bias_size = thread_n * 2;
    int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
    tmp_size = std::max(std::max(sh_b_size, sh_red_size), tmp_size);
    const int tb_groups = MARLIN_NAMESPACE_NAME::div_ceil(thread_k, 16);
    const int sh_s_size = tb_groups * thread_n * 2 * stages;
    return tmp_size + sh_a_size + sh_s_size;
}

bool is_valid_config(const ThreadConfig& cfg, int thread_m_blocks, int prob_n, int prob_k,
                     int max_shared_mem) {
    if (prob_k % cfg.thread_k != 0 || prob_n % cfg.thread_n != 0)
        return false;
    if (cfg.thread_n < MARLIN_NAMESPACE_NAME::min_thread_n ||
        cfg.thread_k < MARLIN_NAMESPACE_NAME::min_thread_k)
        return false;
    if (cfg.num_threads < 128)
        return false;
    return kernel_cache_size(cfg.thread_k, cfg.thread_n, thread_m_blocks) <= max_shared_mem;
}

// One-time cudaFuncSetAttribute for every instantiation (must happen outside
// graph capture; prepare() calls this).
void ensure_attrs() {
    static bool done = [] {
        const auto& d = device_info();
        for (const auto& e : kKernels) {
            cudaFuncSetAttribute(e.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, d.max_shared_mem);
        }
        return true;
    }();
    (void)done;
}

}  // namespace

bool shape_supported(int N, int K) {
    // Cheapest tile family: every kernel needs K and N divisible by 64, and
    // at least one config must divide both. Scale groups need K % 16 == 0.
    if (N % 64 != 0 || K % 64 != 0 || K % 16 != 0)
        return false;
    return true;
}

size_t workspace_bytes() { return static_cast<size_t>(device_info().sms) * sizeof(int); }

size_t c_tmp_bytes(int max_m) {
    int m_block = std::min((max_m + 15) / 16 * 16, 64);
    return static_cast<size_t>(device_info().sms) * m_block * MARLIN_NAMESPACE_NAME::max_thread_n *
           sizeof(float);
}

void marlin_ensure_func_attrs() { ensure_attrs(); }

bool gemm(const MarlinWeight& W, const half* A, half* C, int M, int lda, int* locks, float* c_tmp,
          cudaStream_t stream) {
    if (M <= 0 || W.qweight == nullptr || locks == nullptr || c_tmp == nullptr)
        return false;
    const int prob_n = W.N;
    const int prob_k = W.K;
    if (!shape_supported(prob_n, prob_k) || lda % 8 != 0)
        return false;

    const auto& dev = device_info();
    const int max_shared_mem = dev.max_shared_mem;
    const int num_groups = prob_k / 16;

    // vLLM heuristic: atomicAdd reduce only pays for narrow-N deep-K shapes;
    // additionally it is nondeterministic in accumulation order, so imp keeps
    // it off entirely (runtime.deterministic must hold).
    const bool use_atomic_add = false;
    const bool use_fp32_reduce = true;

    int max_par_here = MARLIN_NAMESPACE_NAME::max_par;
    if (prob_n <= 4096)
        max_par_here = MARLIN_NAMESPACE_NAME::max_par * 8;

    const int4* A_ptr = reinterpret_cast<const int4*>(A);
    const int4* B_ptr = reinterpret_cast<const int4*>(W.qweight);
    int4* C_ptr = reinterpret_cast<int4*>(C);
    int4* C_tmp_ptr = reinterpret_cast<int4*>(c_tmp);
    const int4* s_ptr = reinterpret_cast<const int4*>(W.scales);

    int rest_m = M;
    int max_thread_m_blocks = 4;
    while (rest_m) {
        int par_count = rest_m / (max_thread_m_blocks * 16);
        if (par_count > max_par_here)
            par_count = max_par_here;
        const int prob_m_split = par_count > 0 ? par_count * (max_thread_m_blocks * 16) : rest_m;

        const int thread_m_blocks = std::min(MARLIN_NAMESPACE_NAME::div_ceil(prob_m_split, 16),
                                             max_thread_m_blocks);
        const bool m_block_size_8 = prob_m_split <= 8;

        // Auto thread config (port of determine_exec_config + the small-grid
        // override in marlin_mm).
        const ThreadConfig* configs = thread_m_blocks > 1 ? kLargeBatchConfigs : kSmallBatchConfigs;
        const int n_configs = 3;
        ThreadConfig chosen{-1, -1, -1};
        for (int i = 0; i < n_configs; i++) {
            if (is_valid_config(configs[i], thread_m_blocks, prob_n, prob_k, max_shared_mem - 512) &&
                find_kernel(configs[i].num_threads, thread_m_blocks, configs[i].thread_n / 16,
                            configs[i].thread_k / 16, m_block_size_8) != nullptr) {
                chosen = configs[i];
                break;
            }
        }
        if (chosen.thread_k != -1) {
            // Few CTAs in flight: prefer the narrow-N config so more blocks
            // split K.
            if (prob_n / chosen.thread_n *
                    MARLIN_NAMESPACE_NAME::div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <=
                dev.sms) {
                ThreadConfig narrow{128, 64, 128};
                if (is_valid_config(narrow, thread_m_blocks, prob_n, prob_k, max_shared_mem) &&
                    find_kernel(narrow.num_threads, thread_m_blocks, narrow.thread_n / 16,
                                narrow.thread_k / 16, m_block_size_8) != nullptr)
                    chosen = narrow;
            }
        }
        if (chosen.thread_k == -1) {
            if (max_thread_m_blocks > 1) {
                max_thread_m_blocks--;
                continue;
            }
            return false;
        }

        FnPtr kernel = find_kernel(chosen.num_threads, thread_m_blocks, chosen.thread_n / 16,
                                   chosen.thread_k / 16, m_block_size_8);

        const bool part_use_atomic_add = use_atomic_add &&
                                         MARLIN_NAMESPACE_NAME::div_ceil(prob_m_split, 64) * prob_n <= 2048;

        // clang-format off
        kernel<<<dev.sms, chosen.num_threads, max_shared_mem, stream>>>(
            A_ptr, B_ptr, C_ptr, C_tmp_ptr, /*bias=*/nullptr, /*a_scales=*/nullptr,
            s_ptr, W.d_global_scale, /*zp=*/nullptr, /*g_idx=*/nullptr,
            num_groups, prob_m_split, prob_n, prob_k, lda, locks,
            /*has_bias=*/false, part_use_atomic_add, use_fp32_reduce, max_shared_mem);
        // clang-format on

        A_ptr += static_cast<size_t>(prob_m_split) * (lda / 8);
        C_ptr += static_cast<size_t>(prob_m_split) * (prob_n / 8);
        rest_m -= prob_m_split;
    }
    return true;
}

}  // namespace marlin_w4a16
}  // namespace imp
