// GPTAQ pairs (roadmap row 101, arXiv 2504.02692): the source pass dumps each weight input [rows, K]
// FP16 in call order; the export pass over the same corpus and chunking reads record n of
// (layer, kind) at its n-th call of that key, so x and xq are the same token row by row.

#include "exec/activation_calibrator.h"

#include "compute/gemm_lt_ex.h"
#include "core/logging.h"
#include "quant/calibration_hessian.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

namespace imp {

namespace {

constexpr char kDumpMagic[8] = {'I', 'M', 'P', 'X', 'D', 'M', 'P', '1'};

struct RecHeader {
    int32_t layer = 0;
    int32_t kind = 0;
    int64_t rows = 0;
    int64_t K = 0;
};

}  // namespace

struct ActivationCalibrator::PairState {
    VRAMAllocator* alloc = nullptr;
    FILE* f = nullptr;
    bool reading = false;
    int layer_begin = 0, layer_end = 0;  // read mode: accumulated layers, end 0 = all
    std::string path;
    struct Rec {
        int64_t offset = 0;  // first payload byte
        int64_t rows = 0, K = 0;
    };
    std::map<uint32_t, std::vector<Rec>> index;  // read mode: records per key, call order
    std::map<uint32_t, size_t> cursor;
    struct Acc {
        float* d_hq = nullptr;  // sum xq^T xq
        float* d_c = nullptr;   // sum x^T xq, row-major C[i*K+j] = sum x_i xq_j
        int64_t K = 0;
        uint64_t rows = 0;
    };
    std::map<uint32_t, Acc> acc;
    void* d_x = nullptr;  // the source rows of the current call
    size_t d_x_bytes = 0;
    std::vector<uint16_t> host;
    uint64_t records = 0, misaligned = 0;

    // Dump mode: one record, the input copied to the host first.
    bool dump(const RecHeader& h, const void* src, int64_t row_stride, cudaStream_t stream) {
        const size_t n = static_cast<size_t>(h.rows) * static_cast<size_t>(h.K);
        host.resize(n);
        return cudaMemcpy2DAsync(host.data(), h.K * 2, src, row_stride * 2, h.K * 2, h.rows,
                                 cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
               cudaStreamSynchronize(stream) == cudaSuccess && std::fwrite(&h, sizeof(h), 1, f) == 1 &&
               std::fwrite(host.data(), 2, n, f) == n;
    }
    // Read mode: the key's next record when its shape matches, else nullptr.
    const Rec* next(uint32_t key, int64_t rows, int64_t K) {
        const auto idx = index.find(key);
        const size_t c = cursor[key];
        if (idx == index.end() || c >= idx->second.size() || idx->second[c].rows != rows ||
            idx->second[c].K != K)
            return nullptr;
        cursor[key] = c + 1;
        return &idx->second[c];
    }
    // Read mode: the record's rows into d_x (grown on demand). Pageable H2D returns once the host
    // buffer is staged, so the next call may refill it.
    bool load(const Rec& r, cudaStream_t stream) {
        const size_t n = static_cast<size_t>(r.rows) * static_cast<size_t>(r.K), bytes = n * 2;
        if (bytes > d_x_bytes) {
            if (d_x)
                alloc->free(d_x);
            d_x = alloc->allocate(bytes, "calibration_pair");
            d_x_bytes = d_x ? bytes : 0;
        }
        host.resize(n);
        return d_x && fseeko(f, r.offset, SEEK_SET) == 0 && std::fread(host.data(), 2, n, f) == n &&
               cudaMemcpyAsync(d_x, host.data(), bytes, cudaMemcpyHostToDevice, stream) == cudaSuccess;
    }
    Acc* acc_for(uint32_t key, int64_t K, cudaStream_t stream) {
        auto it = acc.find(key);
        if (it == acc.end()) {
            Acc a;
            a.K = K;
            const size_t kk = static_cast<size_t>(K) * static_cast<size_t>(K) * sizeof(float);
            a.d_hq = static_cast<float*>(alloc->allocate(kk, "calibration_pair"));
            a.d_c = static_cast<float*>(alloc->allocate(kk, "calibration_pair"));
            if (a.d_hq)
                IMP_CUDA_CHECK_LOG(cudaMemsetAsync(a.d_hq, 0, kk, stream));
            if (a.d_c)
                IMP_CUDA_CHECK_LOG(cudaMemsetAsync(a.d_c, 0, kk, stream));
            it = acc.emplace(key, a).first;
        }
        return it->second.d_hq && it->second.d_c ? &it->second : nullptr;
    }
};

void ActivationCalibrator::PairDeleter::operator()(PairState* p) const {
    if (!p)
        return;
    if (p->f)
        std::fclose(p->f);
    if (p->alloc) {
        for (auto& [key, a] : p->acc) {
            if (a.d_hq)
                p->alloc->free(a.d_hq);
            if (a.d_c)
                p->alloc->free(a.d_c);
        }
        if (p->d_x)
            p->alloc->free(p->d_x);
    }
    delete p;
}

std::string ActivationCalibrator::enable_pair_dump(const std::string& path) {
    if (!alloc_)
        return "pair dump needs a device allocator";
    pair_.reset(new PairState);
    pair_->alloc = alloc_;
    pair_->path = path;
    pair_->f = std::fopen(path.c_str(), "wb");
    if (!pair_->f || std::fwrite(kDumpMagic, 1, sizeof(kDumpMagic), pair_->f) != sizeof(kDumpMagic))
        return "cannot write " + path;
    return "";
}

std::string ActivationCalibrator::enable_pair_read(const std::string& path, int layer_begin, int layer_end) {
    if (!alloc_)
        return "pair read needs a device allocator";
    if (layer_begin < 0 || (layer_end != 0 && layer_end <= layer_begin))
        return "pair layers [" + std::to_string(layer_begin) + ", " + std::to_string(layer_end) + ") empty";
    pair_.reset(new PairState);
    pair_->alloc = alloc_;
    pair_->path = path;
    pair_->reading = true;
    pair_->layer_begin = layer_begin;
    pair_->layer_end = layer_end;
    FILE* f = std::fopen(path.c_str(), "rb");
    pair_->f = f;
    char magic[8];
    if (!f || std::fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        std::memcmp(magic, kDumpMagic, sizeof(magic)) != 0)
        return path + ": not a pair dump";
    RecHeader h;
    while (std::fread(&h, sizeof(h), 1, f) == 1) {
        if (h.layer < 0 || h.rows <= 0 || h.K <= 0)
            return path + ": bad record " + std::to_string(pair_->records);
        const int64_t at = ftello(f);
        const auto bytes = static_cast<int64_t>(h.rows * h.K * 2);
        if (fseeko(f, bytes, SEEK_CUR) != 0)
            return path + ": truncated record " + std::to_string(pair_->records);
        const uint32_t key = (static_cast<uint32_t>(h.layer) * 256u) + static_cast<uint32_t>(h.kind);
        pair_->index[key].push_back({at, h.rows, h.K});
        pair_->records++;
    }
    IMP_LOG_INFO("calibration: pair dump %s, %llu records over %zu weight inputs", path.c_str(),
                 static_cast<unsigned long long>(pair_->records), pair_->index.size());
    return pair_->records > 0 ? "" : path + ": no records";
}

void ActivationCalibrator::accumulate_pair_(int layer, TensorKind kind, const Tensor& input,
                                            int64_t row_stride, cudaStream_t stream) {
    // WK, WV, W_UP read the input of WQ / W_GATE (find_hessian maps them back).
    if (kind == TensorKind::WK || kind == TensorKind::WV || kind == TensorKind::W_UP)
        return;
    PairState& p = *pair_;
    const int64_t rows = input.shape[0], K = input.shape[1];
    const uint32_t key = (static_cast<uint32_t>(layer) * 256u) + static_cast<uint32_t>(kind);
    if (!p.reading) {
        const bool ok = p.dump(RecHeader{layer, static_cast<int32_t>(kind), rows, K}, input.data, row_stride,
                               stream);
        (ok ? p.records : p.misaligned)++;
        return;
    }
    const PairState::Rec* rec = p.next(key, rows, K);
    // Outside [begin, end): the record is consumed for the alignment check, never loaded.
    if (rec && (layer < p.layer_begin || (p.layer_end != 0 && layer >= p.layer_end)))
        return;
    PairState::Acc* a = rec && p.load(*rec, stream) ? p.acc_for(key, K, stream) : nullptr;
    // Row-major [rows, K] is column-major [K, rows]: D = A B^T, D(i, j) = sum_r A[r][i] B[r][j].
    // D = Xq^T X stored column-major is C = X^T Xq row-major.
    const float one = 1.0f;
    const auto k = static_cast<int>(K), m = static_cast<int>(rows), ld = static_cast<int>(row_stride);
    const bool ok = a &&
                    lt_gemm_ex(CUBLAS_OP_N, CUBLAS_OP_T, k, k, m, &one, input.data, CUDA_R_16F, ld,
                               input.data, CUDA_R_16F, ld, &one, a->d_hq, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                               stream) == CUBLAS_STATUS_SUCCESS &&
                    lt_gemm_ex(CUBLAS_OP_N, CUBLAS_OP_T, k, k, m, &one, input.data, CUDA_R_16F, ld, p.d_x,
                               CUDA_R_16F, k, &one, a->d_c, CUDA_R_32F, k, CUBLAS_COMPUTE_32F,
                               stream) == CUBLAS_STATUS_SUCCESS;
    if (!ok) {
        if (p.misaligned++ < 4)
            IMP_LOG_WARN("calibration: layer %d kind %s rows %lld: %s", layer, tensor_kind_name(kind),
                         static_cast<long long>(rows),
                         rec ? "pair accumulate failed (VRAM, read or GEMM)" : "no matching dump record");
        return;
    }
    a->rows += static_cast<uint64_t>(rows);
}

std::string ActivationCalibrator::finish_pairs(const std::string& out_path, cudaStream_t stream) {
    if (!pair_)
        return "";
    PairState& p = *pair_;
    if (!p.reading) {
        const bool ok = std::fclose(p.f) == 0;
        p.f = nullptr;
        IMP_LOG_INFO("calibration: pair dump %s, %llu records, %llu failed", p.path.c_str(),
                     static_cast<unsigned long long>(p.records),
                     static_cast<unsigned long long>(p.misaligned));
        return ok && p.misaligned == 0 && p.records > 0 ? "" : "pair dump " + p.path + " incomplete";
    }
    size_t unread = 0;
    for (const auto& [key, recs] : p.index)
        unread += recs.size() - p.cursor[key];
    IMP_LOG_INFO("calibration: pairs %zu weight inputs, %llu misaligned calls, %zu dump records unread",
                 p.acc.size(), static_cast<unsigned long long>(p.misaligned), unread);
    if (p.misaligned != 0 || unread != 0)
        return "source and export calls do not align (same corpus, chunking and binary?)";
    if (p.acc.empty())
        return "no weight input in pair layers [" + std::to_string(p.layer_begin) + ", " +
               std::to_string(p.layer_end) + ")";
    HessianWriter w;
    if (std::string err = w.open(out_path, static_cast<uint32_t>(p.acc.size()), 2); !err.empty())
        return err;
    std::vector<float> hq, c;
    for (const auto& [key, a] : p.acc) {
        const size_t n = static_cast<size_t>(a.K) * static_cast<size_t>(a.K);
        hq.resize(n);
        c.resize(n);
        if (cudaMemcpyAsync(hq.data(), a.d_hq, n * sizeof(float), cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaMemcpyAsync(c.data(), a.d_c, n * sizeof(float), cudaMemcpyDeviceToHost, stream) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream) != cudaSuccess)
            return "pair D2H failed";
        const float inv = a.rows > 0 ? 1.0f / static_cast<float>(a.rows) : 0.0f;
        for (size_t i = 0; i < n; i++) {
            hq[i] *= inv;
            c[i] *= inv;
        }
        const auto kind = static_cast<TensorKind>(key % 256u);
        if (std::string err = w.add(static_cast<int>(key / 256u), tensor_kind_name(kind), a.K, a.rows,
                                    hq.data(), c.data());
            !err.empty())
            return err;
    }
    IMP_LOG_INFO("GPTAQ pairs: %zu entries to %s", p.acc.size(), out_path.c_str());
    return w.close();
}

std::string ActivationCalibrator::write_second_moments(const std::string& base, cudaStream_t stream) {
    if (std::string err = write_hessians(base + ".hess", stream); !err.empty())
        return err;
    return finish_pairs(base + ".gptaq", stream);
}

}  // namespace imp
