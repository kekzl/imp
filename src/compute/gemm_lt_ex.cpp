#include "compute/gemm_lt_ex.h"

#include "compute/gemm_internal.cuh"
#include "core/cublas_check.h"
#include "core/logging.h"

#include <cstdint>
#include <mutex>
#include <type_traits>
#include <unordered_map>

namespace imp {

namespace {

struct Key {
    long long stride_a = 0, stride_b = 0, stride_c = 0;
    uint64_t workspace = 0;
    int transa = 0, transb = 0, m = 0, n = 0, k = 0, lda = 0, ldb = 0, ldc = 0, batch = 0;
    int a_type = 0, b_type = 0, c_type = 0, compute = 0, pointer_array = 0;
    uint32_t align_a = 0, align_b = 0, align_c = 0, pad = 0;
    bool operator==(const Key&) const = default;
};
static_assert(std::has_unique_object_representations_v<Key>, "KeyHash hashes the raw bytes");

struct KeyHash {
    size_t operator()(const Key& k) const {
        const auto* p = reinterpret_cast<const unsigned char*>(&k);
        size_t h = 1469598103934665603ull;  // FNV-1a over the padding-free POD
        for (size_t i = 0; i < sizeof(Key); i++)
            h = (h ^ p[i]) * 1099511628211ull;
        return h;
    }
};

struct Entry {
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo{};
    size_t workspace = 0;
    bool ok = false;
};

std::mutex s_mutex;
std::unordered_map<Key, Entry, KeyHash> s_cache;

cudaDataType_t scale_type_of(cublasComputeType_t c) {
    switch (c) {
        case CUBLAS_COMPUTE_16F:
        case CUBLAS_COMPUTE_16F_PEDANTIC:
            return CUDA_R_16F;
        case CUBLAS_COMPUTE_32I:
        case CUBLAS_COMPUTE_32I_PEDANTIC:
            return CUDA_R_32I;
        case CUBLAS_COMPUTE_64F:
        case CUBLAS_COMPUTE_64F_PEDANTIC:
            return CUDA_R_64F;
        default:
            return CUDA_R_32F;
    }
}

// Largest power of two (<= 256) dividing every column start the batch touches.
uint32_t alignment(const void* p, int ld, long long stride, size_t elem) {
    uint64_t bits = reinterpret_cast<uintptr_t>(p) | (static_cast<uint64_t>(ld) * elem) |
                    (static_cast<uint64_t>(stride) * elem) | 256u;
    return static_cast<uint32_t>(bits & (~bits + 1));
}

size_t elem_size(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_64F:
            return 8;
        case CUDA_R_32F:
        case CUDA_R_32I:
            return 4;
        case CUDA_R_16F:
        case CUDA_R_16BF:
            return 2;
        default:
            return 1;
    }
}

Entry build(const Key& key, cublasLtHandle_t lt) {
    Entry e;
    const auto ta = static_cast<cublasOperation_t>(key.transa),
               tb = static_cast<cublasOperation_t>(key.transb);
    const auto compute = static_cast<cublasComputeType_t>(key.compute);
    if (cublasLtMatmulDescCreate(&e.op, compute, scale_type_of(compute)) != CUBLAS_STATUS_SUCCESS)
        return e;
    IMP_CUBLAS_CHECK_LOG(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    IMP_CUBLAS_CHECK_LOG(cublasLtMatmulDescSetAttribute(e.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    auto layout = [&](cublasLtMatrixLayout_t* l, int type, int rows, int cols, int ld, long long stride) {
        if (cublasLtMatrixLayoutCreate(l, static_cast<cudaDataType_t>(type), rows, cols, ld) !=
            CUBLAS_STATUS_SUCCESS)
            return false;
        if (key.batch > 1) {
            const int32_t count = key.batch;
            IMP_CUBLAS_CHECK_LOG(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                                                  &count, sizeof(count)));
            if (key.pointer_array) {
                const uint32_t mode = CUBLASLT_BATCH_MODE_POINTER_ARRAY;
                IMP_CUBLAS_CHECK_LOG(cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_BATCH_MODE,
                                                                      &mode, sizeof(mode)));
            } else {
                const int64_t s = stride;
                IMP_CUBLAS_CHECK_LOG(
                    cublasLtMatrixLayoutSetAttribute(*l, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &s,
                                                     sizeof(s)));
            }
        }
        return true;
    };
    const bool na = ta == CUBLAS_OP_N, nb = tb == CUBLAS_OP_N;
    if (!layout(&e.a, key.a_type, na ? key.m : key.k, na ? key.k : key.m, key.lda, key.stride_a) ||
        !layout(&e.b, key.b_type, nb ? key.k : key.n, nb ? key.n : key.k, key.ldb, key.stride_b) ||
        !layout(&e.c, key.c_type, key.m, key.n, key.ldc, key.stride_c))
        return e;
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS)
        return e;
    const uint64_t ws = key.workspace;
    IMP_CUBLAS_CHECK_LOG(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                              &ws, sizeof(ws)));
    const uint32_t aligns[3] = {key.align_a, key.align_b, key.align_c};
    const cublasLtMatmulPreferenceAttributes_t align_attr[4] = {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                                                                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                                                                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                                                                CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES};
    for (int i = 0; i < 4; i++)
        IMP_CUBLAS_CHECK_LOG(cublasLtMatmulPreferenceSetAttribute(pref, align_attr[i], &aligns[i < 3 ? i : 2],
                                                                  sizeof(uint32_t)));
    cublasLtMatmulHeuristicResult_t res{};
    int found = 0;
    const cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, e.op, e.a, e.b, e.c, e.c, pref, 1, &res,
                                                             &found);
    IMP_CUBLAS_CHECK_LOG(cublasLtMatmulPreferenceDestroy(pref));
    if (st == CUBLAS_STATUS_SUCCESS && found > 0) {
        e.algo = res.algo;
        e.workspace = res.workspaceSize;
        e.ok = true;
    }
    return e;
}

cublasStatus_t run(const Key& key, const void* alpha, const void* A, const void* B, const void* beta, void* C,
                   cudaStream_t stream) {
    cublasLtHandle_t lt = gemm_internal_cublaslt_handle();
    const Entry* e = nullptr;
    {
        std::lock_guard<std::mutex> lock(s_mutex);
        auto it = s_cache.find(key);
        if (it == s_cache.end())
            it = s_cache.emplace(key, build(key, lt)).first;
        e = &it->second;
    }
    if (!e->ok) {
        IMP_LOG_ERROR("lt_gemm: no cuBLASLt algo for m %d n %d k %d batch %d types %d/%d/%d compute %d",
                      key.m, key.n, key.k, key.batch, key.a_type, key.b_type, key.c_type, key.compute);
        return CUBLAS_STATUS_NOT_SUPPORTED;
    }
    return cublasLtMatmul(lt, e->op, alpha, A, e->a, B, e->b, beta, C, e->c, C, e->c, &e->algo,
                          e->workspace > 0 ? gemm_internal_workspace() : nullptr, e->workspace, stream);
}

}  // namespace

cublasStatus_t lt_gemm_strided_batched(cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                                       int k, const void* alpha, const void* A, cudaDataType_t a_type,
                                       int lda, long long stride_a, const void* B, cudaDataType_t b_type,
                                       int ldb, long long stride_b, const void* beta, void* C,
                                       cudaDataType_t c_type, int ldc, long long stride_c, int batch,
                                       cublasComputeType_t compute, cudaStream_t stream) {
    if (m == 0 || n == 0 || batch == 0)
        return CUBLAS_STATUS_SUCCESS;
    const bool batched = batch > 1;
    const Key key{.stride_a = batched ? stride_a : 0,
                  .stride_b = batched ? stride_b : 0,
                  .stride_c = batched ? stride_c : 0,
                  .workspace = gemm_internal_workspace() ? gemm_internal_workspace_size() : 0,
                  .transa = transa,
                  .transb = transb,
                  .m = m,
                  .n = n,
                  .k = k,
                  .lda = lda,
                  .ldb = ldb,
                  .ldc = ldc,
                  .batch = batch,
                  .a_type = a_type,
                  .b_type = b_type,
                  .c_type = c_type,
                  .compute = compute,
                  .align_a = alignment(A, lda, batched ? stride_a : 0, elem_size(a_type)),
                  .align_b = alignment(B, ldb, batched ? stride_b : 0, elem_size(b_type)),
                  .align_c = alignment(C, ldc, batched ? stride_c : 0, elem_size(c_type))};
    return run(key, alpha, A, B, beta, C, stream);
}

cublasStatus_t lt_gemm_batched(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
                               const void* alpha, const void* const* a, cudaDataType_t a_type, int lda,
                               const void* const* b, cudaDataType_t b_type, int ldb, const void* beta,
                               void* const* c, cudaDataType_t c_type, int ldc, int batch,
                               cublasComputeType_t compute, cudaStream_t stream) {
    if (m == 0 || n == 0 || batch == 0)
        return CUBLAS_STATUS_SUCCESS;
    // The matrix pointers live on the device: assume only element alignment.
    const Key key{.workspace = gemm_internal_workspace() ? gemm_internal_workspace_size() : 0,
                  .transa = transa,
                  .transb = transb,
                  .m = m,
                  .n = n,
                  .k = k,
                  .lda = lda,
                  .ldb = ldb,
                  .ldc = ldc,
                  .batch = batch,
                  .a_type = a_type,
                  .b_type = b_type,
                  .c_type = c_type,
                  .compute = compute,
                  .pointer_array = 1,
                  .align_a = static_cast<uint32_t>(elem_size(a_type)),
                  .align_b = static_cast<uint32_t>(elem_size(b_type)),
                  .align_c = static_cast<uint32_t>(elem_size(c_type))};
    return run(key, alpha, static_cast<const void*>(a), static_cast<const void*>(b), beta,
               static_cast<void*>(const_cast<void**>(c)), stream);
}

void lt_gemm_ex_cleanup() {
    std::lock_guard<std::mutex> lock(s_mutex);
    for (auto& [key, e] : s_cache) {
        if (e.a)
            IMP_CUBLAS_CHECK_LOG(cublasLtMatrixLayoutDestroy(e.a));
        if (e.b)
            IMP_CUBLAS_CHECK_LOG(cublasLtMatrixLayoutDestroy(e.b));
        if (e.c)
            IMP_CUBLAS_CHECK_LOG(cublasLtMatrixLayoutDestroy(e.c));
        if (e.op)
            IMP_CUBLAS_CHECK_LOG(cublasLtMatmulDescDestroy(e.op));
    }
    s_cache.clear();
}

}  // namespace imp
