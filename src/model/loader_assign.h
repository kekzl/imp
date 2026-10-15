#pragma once

// Loader helpers keeping tensor + qtype + sidecar-scales in lockstep, used by GGUF,
// SafeTensors, and llm-compressor loaders. Once Stage G lands the *_qtype mirror fields on
// TransformerLayer go away; helpers collapse to single-Tensor assigns.

#include "core/tensor.h"
#include "core/qtype.h"

namespace imp {

// Assign a parsed tensor to a layer slot. The tensor's own qtype field
// carries the type — Stage G removed the per-layer *_qtype mirrors.
inline void assign_quant(Tensor& slot, const Tensor& src) { slot = src; }

}  // namespace imp
