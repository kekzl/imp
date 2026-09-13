#include "memory/ssm_state.h"

#include <algorithm>
#include "memory/mem_account.h"
#include "memory/ssm_state_size.h"
#include "memory/vram_allocator.h"
#include "memory/vram_query.h"
#include "core/logging.h"
#include <cuda_runtime.h>

namespace imp {

SSMState::~SSMState() {
    if (lazy_) {
        // What was never committed leaves the ledger with the reservation;
        // what was committed leaves the audit table with the region.
        vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(region_.reserved() - committed_bytes()));
        MemAccount::instance().note("ssm_state", -static_cast<std::ptrdiff_t>(committed_bytes()));
        region_.reset();
        pool_ = nullptr;
        return;
    }
    if (pool_) {
        if (alloc_)
            alloc_->free(pool_);
        else
            IMP_CUDA_CHECK_LOG(cudaFree(pool_));
        pool_ = nullptr;
    }
}

bool SSMState::init(int n_ssm_layers, int max_sequences, int conv_channels, int conv_kernel, int n_heads,
                    int head_dim_ssm, int state_size, QType h_dtype, VRAMAllocator* alloc, int n_reserved,
                    Backend* lazy_backend, int n_reserved_lazy) {
    n_ssm_layers_ = n_ssm_layers;
    max_sequences_ = max_sequences;
    n_reserved_ = std::max(0, n_reserved);
    n_reserved_lazy_ = std::clamp(n_reserved_lazy, 0, n_reserved_);
    h_dtype_ = h_dtype;
    alloc_ = alloc;

    // Geometry and byte counts: memory/ssm_state_size.h, the ONE formula. The
    // plan charges the same header, so what is charged is what is taken.
    const SsmStateGeometry geom{n_ssm_layers_, conv_channels, conv_kernel,
                                n_heads,       head_dim_ssm, state_size, h_dtype_};
    conv_bytes_ = ssm_conv_bytes_per_layer(geom);
    h_bytes_ = ssm_h_bytes_per_layer(geom);
    per_layer_bytes_ = ssm_bytes_per_layer(geom);
    per_seq_bytes_ = ssm_bytes_per_slot(geom);
    slot_stride_bytes_ = per_seq_bytes_;
    total_bytes_ = ssm_pool_bytes(geom, max_sequences_, n_reserved_);

    if (lazy_backend && per_seq_bytes_ > 0 && init_lazy_(*lazy_backend, max_sequences_ + n_reserved_))
        return true;

    if (alloc_) {
        // No raw-cudaMalloc retry behind the allocator's back: that hatch is how state landed
        // past the headroom and spilled the KV pool to a fraction of its bandwidth (MEMORY.md
        // D14) - the allocator said no, the pool took the memory anyway, and nothing downstream
        // knew. A rejected pool is a refused configuration now, with the numbers and the lever.
        pool_ = alloc_->allocate(total_bytes_, "ssm_state");
    } else {
        cudaError_t err = cudaMalloc(&pool_, total_bytes_);
        if (err != cudaSuccess)
            pool_ = nullptr;
    }
    if (!pool_) {
        size_t free_bytes = 0, total_device = 0;
        if (cudaMemGetInfo(&free_bytes, &total_device) != cudaSuccess)
            free_bytes = 0;
        IMP_LOG_ERROR("%s", ssm_pool_failure_message(total_bytes_, max_sequences_, n_reserved_,
                                                     n_ssm_layers_, free_bytes)
                                .c_str());
        return false;
    }

    // Zero-initialize all state
    IMP_CUDA_CHECK_LOG(cudaMemset(pool_, 0, total_bytes_));

    IMP_LOG_INFO(
        "SSM state: %d layers x %d sequences = %.2f MiB "
        "(conv=%.1f KB, h=%.1f KB [%s] per layer)",
        n_ssm_layers_, max_sequences_, total_bytes_ / (1024.0 * 1024.0), conv_bytes_ / 1024.0,
        h_bytes_ / 1024.0, dtype_name(h_dtype_));
    return true;
}

// Reserve address space for every slot, commit only the reserved (spec
// verify) ones. False falls back to the fixed pool: a backend without VMM is
// a real answer, not a startup failure.
bool SSMState::init_lazy_(Backend& be, int n_slots) {
    const size_t granule = std::max<size_t>(be.granularity(), 256);
    const size_t stride = ((per_seq_bytes_ + granule - 1) / granule) * granule;
    const size_t reserve = stride * static_cast<size_t>(n_slots);
    auto res = be.acquire_growable(reserve, 0, 256, RegionTag::SsmState);
    if (!res) {
        IMP_LOG_WARN("SSM state: lazy pool unavailable (%s reserving %.0f MiB) — allocating every slot",
                     mem_error_name(res.error), reserve / (1024.0 * 1024.0));
        return false;
    }
    lazy_ = true;
    backend_ = &be;
    region_ = std::move(res.region);
    pool_ = region_.base();
    slot_stride_bytes_ = stride;
    committed_.assign(static_cast<size_t>(n_slots), false);
    n_committed_ = 0;
    // The whole reservation is charged by the plan and not yet backed.
    vram_reserved_uncommitted_add(static_cast<std::ptrdiff_t>(reserve));

    for (int i = 0; i < n_reserved_ - n_reserved_lazy_; ++i) {
        if (!ensure_slot(reserved_slot(i))) {
            IMP_LOG_WARN("SSM state: could not commit reserved slot %d of %d — allocating every slot", i,
                         n_reserved_);
            vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(reserve - committed_bytes()));
            MemAccount::instance().note("ssm_state", -static_cast<std::ptrdiff_t>(committed_bytes()));
            region_.reset();
            pool_ = nullptr;
            lazy_ = false;
            backend_ = nullptr;
            committed_.clear();
            n_committed_ = 0;
            slot_stride_bytes_ = per_seq_bytes_;
            return false;
        }
    }
    IMP_LOG_INFO(
        "SSM state: %d layers x %d sequences lazy, %.2f MiB reserved, %.2f MiB committed "
        "(%d reserved slot(s); %.1f MiB per slot, stride %.1f MiB; conv=%.1f KB, h=%.1f KB [%s] per layer)",
        n_ssm_layers_, max_sequences_, reserve / (1024.0 * 1024.0), committed_bytes() / (1024.0 * 1024.0),
        n_reserved_, per_seq_bytes_ / (1024.0 * 1024.0), stride / (1024.0 * 1024.0), conv_bytes_ / 1024.0,
        h_bytes_ / 1024.0, dtype_name(h_dtype_));
    return true;
}

bool SSMState::slot_committed(int slot) const {
    if (slot < 0 || slot >= max_sequences_ + n_reserved_)
        return false;
    return !lazy_ || committed_[static_cast<size_t>(slot)];
}

size_t SSMState::committed_bytes() const {
    return lazy_ ? static_cast<size_t>(n_committed_) * slot_stride_bytes_ : (pool_ ? total_bytes_ : 0);
}

bool SSMState::ensure_slot(int slot) {
    if (slot < 0 || slot >= max_sequences_ + n_reserved_)
        return false;
    if (!lazy_)
        return pool_ != nullptr;
    if (committed_[static_cast<size_t>(slot)])
        return true;
    // Refuse rather than spill: the commit itself would succeed into host memory on a full
    // card (#1103). The reading excludes every lazy pool's pending charge except this one's,
    // which is added back since it's what this commit draws on; what is left must cover the
    // slot above the headroom.
    size_t free_now = 0, total_now = 0;
    if (vram_budget_mem_get_info(&free_now, &total_now) && total_now > 0) {
        const size_t own_pending = region_.reserved() - committed_bytes();
        const size_t free_for_us = free_now + own_pending;
        const size_t floor = vram_allocator_headroom(total_now);
        if (free_for_us < floor + slot_stride_bytes_) {
            IMP_LOG_WARN(
                "SSM state: slot %d not committed, %.0f MiB free against %.0f MiB headroom + %.1f MiB slot",
                slot, free_for_us / (1024.0 * 1024.0), floor / (1024.0 * 1024.0),
                slot_stride_bytes_ / (1024.0 * 1024.0));
            return false;
        }
    }
    const size_t before = region_.committed();
    const MemError e = backend_->commit_range(region_, static_cast<size_t>(slot) * slot_stride_bytes_,
                                              slot_stride_bytes_);
    const size_t added = region_.committed() - before;
    if (e != MemError::Ok || added == 0) {
        IMP_LOG_WARN("SSM state: commit of slot %d failed (%s)", slot, mem_error_name(e));
        return false;
    }
    committed_[static_cast<size_t>(slot)] = true;
    ++n_committed_;
    vram_reserved_uncommitted_add(-static_cast<std::ptrdiff_t>(added));
    MemAccount::instance().note("ssm_state", static_cast<std::ptrdiff_t>(added));
    return true;
}

bool SSMState::decommit_slot(int slot) {
    if (!lazy_ || slot < 0 || slot >= max_sequences_ + n_reserved_ || !committed_[static_cast<size_t>(slot)])
        return false;
    const size_t before = region_.committed();
    const MemError e = backend_->decommit_range(region_, static_cast<size_t>(slot) * slot_stride_bytes_,
                                                slot_stride_bytes_);
    const size_t released = before > region_.committed() ? before - region_.committed() : 0;
    if (e != MemError::Ok || released == 0)
        return false;
    committed_[static_cast<size_t>(slot)] = false;
    --n_committed_;
    vram_reserved_uncommitted_add(static_cast<std::ptrdiff_t>(released));
    MemAccount::instance().note("ssm_state", -static_cast<std::ptrdiff_t>(released));
    return true;
}

void* SSMState::conv_state(int seq_id, int ssm_layer_idx) {
    char* base = static_cast<char*>(pool_);
    return base + seq_id * slot_stride_bytes_ + ssm_layer_idx * per_layer_bytes_;
}

void* SSMState::h_state(int seq_id, int ssm_layer_idx) {
    char* base = static_cast<char*>(pool_);
    return base + seq_id * slot_stride_bytes_ + ssm_layer_idx * per_layer_bytes_ + conv_bytes_;
}

void SSMState::reset_sequence(int seq_id, cudaStream_t stream) {
    if (!pool_ || !slot_committed(seq_id))
        return;
    char* base = static_cast<char*>(pool_) + seq_id * slot_stride_bytes_;
    IMP_CUDA_CHECK_LOG(cudaMemsetAsync(base, 0, per_seq_bytes_, stream));
}

}  // namespace imp
