#include "model/host_expert_fit.h"

#include "memory/weight_snapshot.h"
#include "model/model.h"
#include "quant/dequant_gpu.h"

namespace imp {

size_t layer_host_expert_bytes(const TransformerLayer& L) {
    size_t bytes = 0;
    for (const Tensor* p : {&L.expert_gate_packed, &L.expert_up_packed, &L.expert_down_packed}) {
        if (!p->data || p->ndim < 3 || !dequant_gpu_supported(p->qtype))
            continue;
        bytes += static_cast<size_t>(p->shape[0]) * p->shape[1] * qtype_row_bytes(p->qtype, p->shape[2]);
    }
    for (const std::vector<Tensor>* vt : {&L.expert_w_gate, &L.expert_w_up, &L.expert_w_down})
        for (const Tensor& t : *vt)
            if (t.data && !t.on_device && t.ndim >= 2)
                bytes += t.nbytes();
    return bytes;
}

size_t model_host_expert_bytes(const Model& model) {
    size_t bytes = 0;
    for (int i = 0; i < model.n_layers(); ++i)
        bytes += layer_host_expert_bytes(model.layer(i));
    return bytes;
}

// The pinned table is mmap'd and only needs page cache it can lose, but the allocator, the CUDA
// driver and the request path still need room.
bool host_ram_holds(size_t bytes) {
    const size_t avail = host_mem_available_bytes();
    return avail == 0 || avail > bytes + host_ram_headroom_bytes();
}

bool host_expert_pool_caps(size_t bytes, int pool_mib) {
    return pool_mib > 0 && (static_cast<size_t>(pool_mib) << 20) < bytes;
}

}  // namespace imp
