# Vendored Marlin (Apache-2.0)

W4A16 GEMM kernel vendored from vLLM v0.27.1,
`csrc/libtorch_stable/quantization/marlin/` — originally
https://github.com/IST-DASLab/marlin (Elias Frantar), modified by Neural
Magic and the vLLM project. License: Apache-2.0 (`LICENSE.marlin`); the
surrounding repo is MIT, the two coexist as long as this notice and the
license text stay with the vendored files.

| file | status |
|---|---|
| `marlin_template.h`, `dequant.h`, `marlin_mma.h`, `marlin.cuh`, `marlin_dtypes.cuh` | verbatim |
| `marlin_repack_kernel.cuh` | kernel body verbatim, torch entry removed |
| `scalar_type.hpp` | verbatim except torch include -> local check shim |
| `marlin_kernel.h`, `marlin_kernels_fp4_*.cu` | imp-written (decl + explicit instantiations) |
| `marlin_gemm.cu`, `marlin_repack.cu`, `marlin_w4a16.h` | imp-written (launcher, repack driver, API) |

Update path: diff against the vLLM tag, re-copy the verbatim files, re-run
`test-quant --gtest_filter='*MarlinW4A16*'`.
