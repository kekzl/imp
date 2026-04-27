#pragma once

// Loader helpers that keep tensor + qtype + sidecar-scales in lockstep so
// no loader can silently forget to populate one of them. Used by the GGUF,
// SafeTensors, and llm-compressor loaders.
//
// Once Stage G lands, the *_qtype mirror fields on TransformerLayer go
// away entirely; the helpers will collapse to single-Tensor assigns then.

#include "core/tensor.h"
#include "core/qtype.h"

namespace imp {

// Assign a parsed tensor to a layer slot AND its qtype mirror field.
// The tensor's own qtype is the canonical source; the mirror field is a
// deprecated copy that gets read by some legacy dispatch sites. This
// helper guarantees the two stay in sync.
inline void assign_quant(Tensor& slot, QType& slot_qtype, const Tensor& src) {
    slot = src;
    slot_qtype = src.qtype;
}

// Variant that also wires NVFP4 / FP8 sidecar pointers on the tensor
// itself. Used by SafeTensors / llm-compressor loaders where the
// per-tensor scales come in as separate parsed tensors.
inline void assign_quant_with_scales(Tensor& slot, QType& slot_qtype,
                                     const Tensor& src,
                                     void* scales,
                                     float tensor_scale = 1.0f) {
    slot = src;
    slot.scales = scales;
    slot.tensor_scale = tensor_scale;
    slot_qtype = src.qtype;
}

} // namespace imp
