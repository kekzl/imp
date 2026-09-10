#pragma once

#include "core/tensor.h"
#include "memory/backend.h"
#include <cuda_runtime.h>
#include <vector>

namespace imp {

class VRAMAllocator;

// Manages per-sequence, per-SSM-layer state for Mamba2 models.
// Two state types per (sequence, layer):
//   - conv_state: [conv_channels, conv_kernel] float (sliding window for causal conv1d)
//   - h_state:    [n_heads, head_dim_ssm, state_size] in h_dtype (SSM recurrent state)
//
// conv_state is always FP32 (small, needs precision for convolution).
// h_state dtype is configurable: FP32 (default) or FP16 (saves ~50% VRAM for h_state).
// SSM scan computes in FP32 regardless; FP16 h_state uses FP16 load/store only.
//
// Two backing modes. Fixed: one allocation for every slot at init. Lazy: the
// address space for every slot is reserved at init (costs nothing), and a
// slot's slab is committed the first time ensure_slot() is asked for it, so
// an idle server holds the state of the sequences it runs, not of
// max_batch_size. The slot stride is padded to the backend's commit granule
// so two slots never share a page; the payload (per_seq_bytes) is unchanged.
class SSMState {
public:
    SSMState() = default;
    ~SSMState();

    // Allocate state for the given configuration.
    // h_dtype: QType::F32 (default) or QType::F16 for h_state storage.
    // n_reserved: extra slots past max_sequences that the scheduler never
    // hands out — scratch for the multi-candidate speculative verify, which
    // runs W candidates as W sequences and needs W-1 slots inside the pool
    // (the batched scan addresses state by slot id, not by pointer). Reached
    // through reserved_slot(i); same slab layout, priced with the pool.
    // lazy_backend: a growable backend to reserve from instead of allocating
    // every slot up front. Reserved slots commit at init; live slots commit
    // on ensure_slot(). Null, or a backend that cannot grow, keeps the fixed
    // pool.
    [[nodiscard]] bool init(int n_ssm_layers, int max_sequences, int conv_channels, int conv_kernel,
                            int n_heads, int head_dim_ssm, int state_size, QType h_dtype = QType::F32,
                            VRAMAllocator* alloc = nullptr, int n_reserved = 0,
                            Backend* lazy_backend = nullptr);

    // Get pointers into the state pool for a given sequence and SSM layer index.
    void* conv_state(int seq_id, int ssm_layer_idx);
    void* h_state(int seq_id, int ssm_layer_idx);

    // Zero-initialize all state for a sequence (on new request). A no-op on a
    // slot that is not committed: there is nothing there to zero, and the
    // slot is zeroed or restored when it is next acquired.
    void reset_sequence(int seq_id, cudaStream_t stream);

    // Base pointer / size of one sequence's full state slab (all layers,
    // conv + h contiguous) — the unit copied by recurrent-state snapshots.
    void* seq_base(int seq_id) {
        return static_cast<char*>(pool_) + static_cast<size_t>(seq_id) * slot_stride_bytes_;
    }
    size_t per_seq_bytes() const { return per_seq_bytes_; }
    // Slot-to-slot stride in the pool. Equal to per_seq_bytes() on a fixed
    // pool; padded to the commit granule on a lazy one. The batched kernels
    // step by THIS, copies move per_seq_bytes().
    size_t slot_stride_bytes() const { return slot_stride_bytes_; }

    // The same per-layer layout applied to an arbitrary slab of per_seq_bytes()
    // — a snapshot scratch rather than a live sequence. The speculative verify
    // writes a second, mid-chunk state into one of these so a partial
    // acceptance can adopt it instead of re-forwarding to reach it.
    void* conv_state_in(void* slab, int ssm_layer_idx) const {
        return static_cast<char*>(slab) + static_cast<size_t>(ssm_layer_idx) * per_layer_bytes_;
    }
    void* h_state_in(void* slab, int ssm_layer_idx) const {
        return static_cast<char*>(slab) + static_cast<size_t>(ssm_layer_idx) * per_layer_bytes_ + conv_bytes_;
    }

    // Live slots [0, max_sequences()): the scheduler's pool. Reserved slots
    // sit past them; reserved_slot(i) is -1 when i is out of range.
    int max_sequences() const { return max_sequences_; }
    int n_reserved() const { return n_reserved_; }
    int reserved_slot(int i) const { return (i >= 0 && i < n_reserved_) ? max_sequences_ + i : -1; }
    int n_ssm_layers() const { return n_ssm_layers_; }
    QType h_dtype() const { return h_dtype_; }
    // Bytes of the recurrent h state per (sequence, layer). Read by the
    // diagnostics.dump_gdn_state_dir drift dump, which has to size a host
    // buffer for one layer's state without re-deriving the geometry.
    size_t h_bytes() const { return h_bytes_; }

    // Lazy pool: commit the slot's slab if it is not committed yet. False
    // when the card cannot spare it above the allocator headroom right now
    // (a VMM commit does not fail when the card is full, it spills, #1103),
    // or the backend refused. A fixed pool answers true for every valid slot.
    // Does not zero the slab: every acquisition is followed by
    // reset_sequence() or a snapshot restore on the engine stream, and a
    // memset here would race that restore.
    [[nodiscard]] bool ensure_slot(int slot);
    // Lazy pool: hand a committed slot's pages back to the driver (the
    // address stays reserved, so graphs that step to it by id stay valid).
    // Used once, after warmup, which had touched every slot capturing the
    // decode graphs. False when the slot was not committed or the backend
    // cannot decommit.
    bool decommit_slot(int slot);
    bool lazy() const { return lazy_; }
    bool slot_committed(int slot) const;
    int committed_slots() const { return n_committed_; }
    // Physical bytes held right now (lazy: committed slots x stride; fixed:
    // the whole pool) and the address space reserved for the ceiling.
    size_t committed_bytes() const;
    size_t reserved_bytes() const { return lazy_ ? region_.reserved() : committed_bytes(); }

private:
    bool init_lazy_(Backend& be, int n_slots);

    VRAMAllocator* alloc_ = nullptr;
    void* pool_ = nullptr;
    int n_ssm_layers_ = 0;
    int max_sequences_ = 0;
    int n_reserved_ = 0;
    QType h_dtype_ = QType::F32;
    size_t conv_bytes_ = 0;       // per (seq, layer) conv state
    size_t h_bytes_ = 0;          // per (seq, layer) h state
    size_t per_layer_bytes_ = 0;  // conv_bytes_ + h_bytes_
    size_t per_seq_bytes_ = 0;    // per_layer_bytes_ * n_ssm_layers_
    size_t slot_stride_bytes_ = 0;
    size_t total_bytes_ = 0;

    // Lazy pool only.
    bool lazy_ = false;
    Backend* backend_ = nullptr;
    Region region_;
    std::vector<bool> committed_;
    int n_committed_ = 0;
};

}  // namespace imp
