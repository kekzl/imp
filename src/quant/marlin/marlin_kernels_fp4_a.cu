// imp: explicit Marlin instantiations for the NVFP4 W4A16 path
// (a=fp16, b=fp4 e2m1, c=fp16, s=fp8-e4m3 trick format, group_blocks=1).
// Thread config: threads=256, thread_n=128, thread_k=128.
// Mirrors vLLM generate_kernels.py output (Apache-2.0).
// clang-format off
#include "marlin_kernel.h"
#include "marlin_template.h"

namespace MARLIN_NAMESPACE_NAME {

#define IMP_MARLIN_FP4_INST(TM, M8)                                        \
  template __global__ void                                                 \
  Marlin<vllm::kFloat16.id(), vllm::kFE2M1f.id(), vllm::kFloat16.id(),     \
         vllm::kFE4M3fn.id(), 256, TM, 8, 8, M8, 4, 1, false>( \
      MARLIN_KERNEL_PARAMS);

IMP_MARLIN_FP4_INST(1, true)
IMP_MARLIN_FP4_INST(1, false)
IMP_MARLIN_FP4_INST(2, false)
IMP_MARLIN_FP4_INST(3, false)
IMP_MARLIN_FP4_INST(4, false)

#undef IMP_MARLIN_FP4_INST

}  // namespace MARLIN_NAMESPACE_NAME
