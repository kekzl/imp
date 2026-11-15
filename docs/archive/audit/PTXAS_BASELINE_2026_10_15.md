# PTXAS resource baseline, release binary

Source: `ghcr.io/kekzl/imp:latest` v0.53.0 (revision cb591619), `/usr/local/bin/imp-server`, sm_120a cubins (131 ELF images, 130 PTX images).
Tool: `cuobjdump -res-usage` and `cuobjdump -sass` from `imp:toolchain` (CUDA 13.4, V13.4.92). No build, no GPU.

| metric | value |
|---|---:|
| kernel instantiations (STO_ENTRY) | 1018 |
| distinct kernel base names | 466 |
| REG >= 240 | 11 |
| STACK > 0 (local frame; every one also has STL in SASS) | 84 |
| LOCAL > 0 | 0 |
| ACQBULK in SASS (griddepcontrol.wait) | 94 |
| PREEXIT in SASS (griddepcontrol.launch_dependents) | 205 |

Columns: REG registers/thread, STACK local frame bytes/thread, SHARED static smem bytes (dynamic smem not shown), LOCAL declared local bytes, ACQ = ACQBULK count, PRE = PREEXIT count, STL / LDL = local store / load instruction count in SASS.
STACK > 0 does not separate register spill from an indexed local array: `-Xptxas -v` spill-store counts need a build (see INVENTORY_CUDA.md, Deferred to build step).
Sorted by STACK desc, then REG desc. Names demangled by cu++filt, truncated to 200 chars.

## Kernels with STACK > 0 or REG >= 240

| kernel | REG | STACK | SHARED | LOCAL | ACQ | PRE | STL | LDL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imp::topk_finalize_rows_kernel(const imp::TopkRowArgs *, int)` | 44 | 1056 | 1024 | 0 | 0 | 0 | 35 | 75 |
| `imp::topk_partial_rows_kernel(const imp::TopkRowArgs *, int)` | 42 | 1056 | 1024 | 0 | 0 | 0 | 35 | 75 |
| `imp::topk_finalize_kernel(int, float, float, unsigned int, const int *, int, const float *, const float *, const float *, const int *, int *)` | 40 | 1056 | 1024 | 0 | 0 | 0 | 35 | 75 |
| `imp::topk_partial_kernel(const float *, int, int, float, float *, float *, float *, int *)` | 39 | 1056 | 1024 | 0 | 0 | 0 | 39 | 76 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)128, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int ` | 66 | 528 | 1024 | 0 | 0 | 0 | 42 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)128, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int ` | 54 | 528 | 1024 | 0 | 0 | 0 | 40 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)64, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 62 | 272 | 1024 | 0 | 0 | 0 | 26 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)64, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 54 | 272 | 1024 | 0 | 0 | 0 | 24 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)32, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 104 | 144 | 1024 | 0 | 0 | 0 | 16 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)32, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 64 | 144 | 1024 | 0 | 0 | 0 | 18 | 10 |
| `imp::paged_attention_gqa_kernel(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, int, float, int, int, int, int, int, int, float, const __half *)` | 61 | 128 | 1024 | 0 | 0 | 1 | 46 | 55 |
| `imp::paged_attention_decode_kernel_generic(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, int, int, float, int, int, int, int, float)` | 40 | 128 | 1024 | 0 | 0 | 1 | 29 | 47 |
| `imp::<unnamed>::ple_conv_shift_kernel(const __half *, const __half *, __half *, long, const int *, int, const int *, int, int)` | 40 | 128 | 0 | 0 | 0 | 0 | 15 | 6 |
| `imp::<unnamed>::plp_rmsnorm_q8_roundtrip_kernel(const __half *, const __half *, __half *, int, float, float)` | 38 | 128 | 1156 | 0 | 0 | 0 | 2 | 1 |
| `imp::rmsnorm_quantize_q8_1_kernel(const __half *, const __half *, imp::block_q8_1 *, float *, __half *, int, float, float)` | 36 | 128 | 1168 | 0 | 1 | 0 | 2 | 2 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)0, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 255 | 112 | 1024 | 0 | 0 | 0 | 53 | 38 |
| `void imp::gdn_scan_chunkwise_kernel<(int)128, (int)128, (int)64, float>(const float *, const __half *, const __half *, const float *, const float *, float *, T4 *, int, int, int, int, int)` | 255 | 80 | 1024 | 0 | 0 | 0 | 32 | 43 |
| `void imp::gdn_scan_chunkwise_kernel<(int)128, (int)128, (int)64, __half>(const float *, const __half *, const __half *, const float *, const float *, float *, T4 *, int, int, int, int, int)` | 255 | 80 | 1024 | 0 | 0 | 0 | 32 | 43 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)1, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 255 | 64 | 1024 | 0 | 0 | 0 | 41 | 43 |
| `imp::rowwise_topm_reduce_kernel(const float *, const int *, int, int, int *)` | 40 | 64 | 0 | 0 | 0 | 0 | 1 | 1 |
| `void apa::apa_kernel<(bool)0, __half, apa::FlatKV<__half>>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, ` | 168 | 56 | 1280 | 0 | 0 | 0 | 31 | 29 |
| `void apa::apa_kernel<(bool)0, __half, apa::PagedF16KV>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, floa` | 168 | 56 | 1280 | 0 | 0 | 0 | 31 | 29 |
| `void apa::apa_kernel<(bool)1, __half, apa::FlatKV<__half>>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, ` | 168 | 56 | 1280 | 0 | 0 | 0 | 10 | 13 |
| `void apa::apa_kernel<(bool)1, __half, apa::PagedF16KV>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, floa` | 168 | 48 | 1280 | 0 | 0 | 0 | 11 | 12 |
| `void imp::fmha_sm120_fa2_kernel_2cta<(int)128, (int)128, (bool)1, (bool)1, (int)64, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float,` | 128 | 40 | 1024 | 0 | 0 | 0 | 17 | 18 |
| `void imp::<unnamed>::gemm_fp16_kernel<(int)128, (int)2>(const __half *, const __half *, __half *, int, int, int, float, float)` | 244 | 32 | 5120 | 0 | 0 | 0 | 2 | 4 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<cute::tuple<int, int, int>>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120Arra` | 168 | 32 | 1024 | 0 | 0 | 0 | 19 | 22 |
| `void imp::<unnamed>::gemm_fp16_kernel<(int)64, (int)2>(const __half *, const __half *, __half *, int, int, int, float, float)` | 158 | 32 | 5120 | 0 | 0 | 0 | 2 | 4 |
| `void imp::paged_attention_splitk_fp8_tile_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, int, ` | 96 | 32 | 1024 | 0 | 0 | 1 | 2 | 6 |
| `void imp::paged_attention_splitk_fp8_tile_gqa_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 80 | 32 | 10240 | 0 | 0 | 1 | 2 | 4 |
| `void apa::quant_kv_kernel<(int)256, apa::PagedNvfp4KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 74 | 32 | 35072 | 0 | 0 | 0 | 16 | 256 |
| `imp::<unnamed>::ple_conv_add_kernel(const __half *, const __half *, const __half *, const __half *, long, const int *, int, __half *, int, int, int, int, int)` | 40 | 32 | 0 | 0 | 0 | 0 | 17 | 8 |
| `imp::post_decode_step_kernel(const int *, int *, int *, int *, int *, int *, int, int, const int *, int, const int *, int, int, int, int *, int *, int *, int *, int *, const unsigned char *, int, int,` | 39 | 24 | 0 | 0 | 0 | 0 | 3 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 16 | 1024 | 0 | 0 | 0 | 42 | 29 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)512>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 79 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)512>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 79 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::flash_attention_blackwell_kernel<(int)128, (int)96>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 64 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, in` | 64 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)256>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 64 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 63 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)96>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 56 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 16 | 1024 | 0 | 0 | 0 | 8 | 8 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 56 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)192>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 56 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)256>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 55 | 16 | 1024 | 0 | 0 | 0 | 1 | 12 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, int` | 54 | 16 | 1024 | 0 | 0 | 1 | 2 | 4 |
| `void imp::flash_attention_blackwell_kernel<(int)128, (int)64>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 53 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)128>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 53 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)16, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 50 | 16 | 1024 | 0 | 0 | 0 | 6 | 2 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)128>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 48 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)64>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 48 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, in` | 48 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)96>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, in` | 48 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, in` | 48 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)96>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, int` | 48 | 16 | 1024 | 0 | 0 | 1 | 2 | 4 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 44 | 16 | 1024 | 0 | 0 | 0 | 4 | 4 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)6>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)6>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `imp::dequant_q3k_kernel(const unsigned char *, __half *, int, int)` | 24 | 16 | 0 | 0 | 0 | 0 | 1 | 1 |
| `void ra2::attn_kernel<(int)256, (int)8, (int)2, (bool)0, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 255 | 8 | 1152 | 0 | 0 | 0 | 1 | 1 |
| `void imp::fmha_sm120_fa2_kernel_2cta<(int)128, (int)64, (bool)1, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, ` | 128 | 8 | 1024 | 0 | 0 | 0 | 4 | 3 |
| `void imp::fmha_sm120_kernel<(int)64, (int)256, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 128 | 8 | 1024 | 0 | 0 | 0 | 2 | 1 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)16, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 60 | 8 | 1024 | 0 | 0 | 0 | 4 | 2 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)3, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 48 | 8 | 1040 | 0 | 1 | 1 | 1 | 1 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)3, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 48 | 8 | 1040 | 0 | 1 | 1 | 1 | 1 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp32_kernel<(int)3>(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int)` | 48 | 8 | 1040 | 0 | 1 | 1 | 1 | 1 |
| `imp::<unnamed>::expert_tier_wait_kernel(const imp::TierMailbox *, const unsigned int *)` | 24 | 8 | 0 | 0 | 0 | 0 | 1 | 0 |
| `void ra2::attn_kernel<(int)256, (int)8, (int)2, (bool)1, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 250 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q5k_raw_kernel<(int)128, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 248 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q5k_raw_kernel<(int)128, (bool)1>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 248 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)1, (int)32, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 246 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)0, (int)32, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 242 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |

## All kernels

| kernel | REG | STACK | SHARED | LOCAL | ACQ | PRE | STL | LDL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `imp::topk_finalize_rows_kernel(const imp::TopkRowArgs *, int)` | 44 | 1056 | 1024 | 0 | 0 | 0 | 35 | 75 |
| `imp::topk_partial_rows_kernel(const imp::TopkRowArgs *, int)` | 42 | 1056 | 1024 | 0 | 0 | 0 | 35 | 75 |
| `imp::topk_finalize_kernel(int, float, float, unsigned int, const int *, int, const float *, const float *, const float *, const int *, int *)` | 40 | 1056 | 1024 | 0 | 0 | 0 | 35 | 75 |
| `imp::topk_partial_kernel(const float *, int, int, float, float *, float *, float *, int *)` | 39 | 1056 | 1024 | 0 | 0 | 0 | 39 | 76 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)128, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int ` | 66 | 528 | 1024 | 0 | 0 | 0 | 42 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)128, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int ` | 54 | 528 | 1024 | 0 | 0 | 0 | 40 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)64, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 62 | 272 | 1024 | 0 | 0 | 0 | 26 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)64, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 54 | 272 | 1024 | 0 | 0 | 0 | 24 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)32, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 104 | 144 | 1024 | 0 | 0 | 0 | 16 | 10 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)32, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 64 | 144 | 1024 | 0 | 0 | 0 | 18 | 10 |
| `imp::paged_attention_gqa_kernel(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, int, float, int, int, int, int, int, int, float, const __half *)` | 61 | 128 | 1024 | 0 | 0 | 1 | 46 | 55 |
| `imp::paged_attention_decode_kernel_generic(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, int, int, float, int, int, int, int, float)` | 40 | 128 | 1024 | 0 | 0 | 1 | 29 | 47 |
| `imp::<unnamed>::ple_conv_shift_kernel(const __half *, const __half *, __half *, long, const int *, int, const int *, int, int)` | 40 | 128 | 0 | 0 | 0 | 0 | 15 | 6 |
| `imp::<unnamed>::plp_rmsnorm_q8_roundtrip_kernel(const __half *, const __half *, __half *, int, float, float)` | 38 | 128 | 1156 | 0 | 0 | 0 | 2 | 1 |
| `imp::rmsnorm_quantize_q8_1_kernel(const __half *, const __half *, imp::block_q8_1 *, float *, __half *, int, float, float)` | 36 | 128 | 1168 | 0 | 1 | 0 | 2 | 2 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)0, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 255 | 112 | 1024 | 0 | 0 | 0 | 53 | 38 |
| `void imp::gdn_scan_chunkwise_kernel<(int)128, (int)128, (int)64, float>(const float *, const __half *, const __half *, const float *, const float *, float *, T4 *, int, int, int, int, int)` | 255 | 80 | 1024 | 0 | 0 | 0 | 32 | 43 |
| `void imp::gdn_scan_chunkwise_kernel<(int)128, (int)128, (int)64, __half>(const float *, const __half *, const __half *, const float *, const float *, float *, T4 *, int, int, int, int, int)` | 255 | 80 | 1024 | 0 | 0 | 0 | 32 | 43 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)1, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 255 | 64 | 1024 | 0 | 0 | 0 | 41 | 43 |
| `imp::rowwise_topm_reduce_kernel(const float *, const int *, int, int, int *)` | 40 | 64 | 0 | 0 | 0 | 0 | 1 | 1 |
| `void apa::apa_kernel<(bool)0, __half, apa::FlatKV<__half>>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, ` | 168 | 56 | 1280 | 0 | 0 | 0 | 31 | 29 |
| `void apa::apa_kernel<(bool)0, __half, apa::PagedF16KV>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, floa` | 168 | 56 | 1280 | 0 | 0 | 0 | 31 | 29 |
| `void apa::apa_kernel<(bool)1, __half, apa::FlatKV<__half>>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, ` | 168 | 56 | 1280 | 0 | 0 | 0 | 10 | 13 |
| `void apa::apa_kernel<(bool)1, __half, apa::PagedF16KV>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const apa::HeadScale *, const T2 *, T3, T2 *, apa::Dims, floa` | 168 | 48 | 1280 | 0 | 0 | 0 | 11 | 12 |
| `void imp::fmha_sm120_fa2_kernel_2cta<(int)128, (int)128, (bool)1, (bool)1, (int)64, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float,` | 128 | 40 | 1024 | 0 | 0 | 0 | 17 | 18 |
| `void imp::<unnamed>::gemm_fp16_kernel<(int)128, (int)2>(const __half *, const __half *, __half *, int, int, int, float, float)` | 244 | 32 | 5120 | 0 | 0 | 0 | 2 | 4 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cutlass::gemm::GroupProblemShape<cute::tuple<int, int, int>>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120Arra` | 168 | 32 | 1024 | 0 | 0 | 0 | 19 | 22 |
| `void imp::<unnamed>::gemm_fp16_kernel<(int)64, (int)2>(const __half *, const __half *, __half *, int, int, int, float, float)` | 158 | 32 | 5120 | 0 | 0 | 0 | 2 | 4 |
| `void imp::paged_attention_splitk_fp8_tile_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, int, ` | 96 | 32 | 1024 | 0 | 0 | 1 | 2 | 6 |
| `void imp::paged_attention_splitk_fp8_tile_gqa_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 80 | 32 | 10240 | 0 | 0 | 1 | 2 | 4 |
| `void apa::quant_kv_kernel<(int)256, apa::PagedNvfp4KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 74 | 32 | 35072 | 0 | 0 | 0 | 16 | 256 |
| `imp::<unnamed>::ple_conv_add_kernel(const __half *, const __half *, const __half *, const __half *, long, const int *, int, __half *, int, int, int, int, int)` | 40 | 32 | 0 | 0 | 0 | 0 | 17 | 8 |
| `imp::post_decode_step_kernel(const int *, int *, int *, int *, int *, int *, int, int, const int *, int, const int *, int, int, int, int *, int *, int *, int *, int *, const unsigned char *, int, int,` | 39 | 24 | 0 | 0 | 0 | 0 | 3 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 16 | 1024 | 0 | 0 | 0 | 42 | 29 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)512>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 79 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)512>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 79 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::flash_attention_blackwell_kernel<(int)128, (int)96>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 64 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, in` | 64 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)256>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 64 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 63 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)96>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 56 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 16 | 1024 | 0 | 0 | 0 | 8 | 8 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, i` | 56 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)192>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 56 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)256>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 55 | 16 | 1024 | 0 | 0 | 0 | 1 | 12 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, int` | 54 | 16 | 1024 | 0 | 0 | 1 | 2 | 4 |
| `void imp::flash_attention_blackwell_kernel<(int)128, (int)64>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 53 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)128>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 53 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)16, (int)128, (int)128, (int)3>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 50 | 16 | 1024 | 0 | 0 | 0 | 6 | 2 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)128>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 48 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::flash_attention_blackwell_kernel<(int)64, (int)64>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 48 | 16 | 1024 | 0 | 0 | 0 | 1 | 10 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, in` | 48 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_fp8_pipeline_kernel<(int)96>(const __half *, const unsigned char *, const unsigned char *, float *, const int *, const int *, int, int, int, int, float, float, int, in` | 48 | 16 | 1024 | 0 | 0 | 1 | 1 | 1 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, in` | 48 | 16 | 1024 | 0 | 0 | 1 | 1 | 2 |
| `void imp::paged_attention_splitk_int4_pipeline_kernel<(int)96>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, float *, const int *, const int *, int, int` | 48 | 16 | 1024 | 0 | 0 | 1 | 2 | 4 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 44 | 16 | 1024 | 0 | 0 | 0 | 4 | 4 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)6>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)6>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 16 | 1040 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)6>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 16 | 1024 | 0 | 0 | 0 | 1 | 1 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)6>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 16 | 1024 | 0 | 0 | 0 | 2 | 2 |
| `imp::dequant_q3k_kernel(const unsigned char *, __half *, int, int)` | 24 | 16 | 0 | 0 | 0 | 0 | 1 | 1 |
| `void ra2::attn_kernel<(int)256, (int)8, (int)2, (bool)0, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 255 | 8 | 1152 | 0 | 0 | 0 | 1 | 1 |
| `void imp::fmha_sm120_fa2_kernel_2cta<(int)128, (int)64, (bool)1, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, ` | 128 | 8 | 1024 | 0 | 0 | 0 | 4 | 3 |
| `void imp::fmha_sm120_kernel<(int)64, (int)256, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 128 | 8 | 1024 | 0 | 0 | 0 | 2 | 1 |
| `void imp::<unnamed>::smallM_kernel_v1<(int)16, (int)128, (int)256, (int)2>(const void *const *, const void *const *, const void *const *, const void *const *, void *const *, const float *, const int *` | 60 | 8 | 1024 | 0 | 0 | 0 | 4 | 2 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)3, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 48 | 8 | 1040 | 0 | 1 | 1 | 1 | 1 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)3, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 48 | 8 | 1040 | 0 | 1 | 1 | 1 | 1 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp32_kernel<(int)3>(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int)` | 48 | 8 | 1040 | 0 | 1 | 1 | 1 | 1 |
| `imp::<unnamed>::expert_tier_wait_kernel(const imp::TierMailbox *, const unsigned int *)` | 24 | 8 | 0 | 0 | 0 | 0 | 1 | 0 |
| `void ra2::attn_kernel<(int)256, (int)8, (int)2, (bool)1, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 250 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q5k_raw_kernel<(int)128, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 248 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q5k_raw_kernel<(int)128, (bool)1>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 248 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)1, (int)32, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 246 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)0, (int)32, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 242 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_ssd_kernel<(bool)0, (bool)0>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *` | 239 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_ssd_kernel<(bool)0, (bool)1>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *` | 239 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q6k_raw_kernel<(int)128, (bool)0>(const signed char *, const __half *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 235 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q6k_raw_kernel<(int)128, (bool)1>(const signed char *, const __half *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 235 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_ssd_kernel<(bool)1, (bool)0>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *` | 233 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_ssd_kernel<(bool)1, (bool)1>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *` | 233 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, float, (int)2, float, (int)4>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, int, cons` | 224 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, float, (int)2, __nv_bfloat16, (int)4>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, i` | 224 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, __half, (int)2, float, (int)4>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, int, con` | 224 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, __half, (int)2, __nv_bfloat16, (int)4>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, ` | 224 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)1, (int)32, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 222 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)256, (bool)1, (bool)1, (int)64, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 222 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, float, (int)2, float, (int)1>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, int, cons` | 222 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, float, (int)2, __nv_bfloat16, (int)1>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, i` | 222 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, __half, (int)2, float, (int)1>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, int, con` | 222 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)128, (int)128, __half, (int)2, __nv_bfloat16, (int)1>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, ` | 222 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)1, (bool)1, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 221 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)1, (bool)0, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 219 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)64, (int)64, float, (int)1, float, (int)1>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, int, const ` | 218 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gdn_scan_fused_kernel<(int)64, (int)64, __half, (int)1, float, (int)1>(const float *, const __half *, const __half *, const float *, const float *, T5 *, T3 *, int, int, int, int, int, const` | 218 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)2, (int)8, (int)2, float>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 212 | 0 | 25600 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)2, (int)8, (int)2, __half>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 210 | 0 | 25600 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gdn_scan_chunkwise_wy_kernel<(int)128, (int)128, (int)32>(const float *, const __half *, const __half *, const float *, const float *, float *, __half *, int, int, int, int, int)` | 204 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 201 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_mma_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const int *,` | 196 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)128, (int)128, (bool)1, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool` | 195 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)1, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 195 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 195 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q51_raw_kernel<(int)128, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 194 | 0 | 43008 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q51_raw_kernel<(int)128, (bool)1>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 194 | 0 | 43008 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)128, (int)128, (bool)1, (bool)1, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool` | 191 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)1, (bool)1, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 191 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)64, (bool)1, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 190 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)256, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 189 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)256, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 189 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)256, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 188 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q4k_raw_kernel<(int)128, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 187 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q4k_raw_kernel<(int)128, (bool)1>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 187 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_q8k128_kernel<(int)128, (bool)1>(const signed char *, const __half *, const signed char *, const __half *, __half *, int, int, int)` | 186 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)128, (bool)1, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 185 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)64, (bool)1, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 181 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)64, (bool)1, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 177 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::chunkpar::<unnamed>::gdn_chunkpar_intra_kernel<(int)128, (int)128>(const float *, const __half *, const __half *, const float *, const float *, float *, int, int, int, int, int, int)` | 174 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)128, (bool)1, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 172 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)128, (bool)1, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 170 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<cute::tuple<int, int, int, int>, cutlass::gemm::collective::CollectiveMma<cutlass::gemm::MainloopSm120TmaWarpSpecializedBlockScaled<(in` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::chunkpar::<unnamed>::gdn_chunkpar_pass_kernel<(int)128, (int)128, float>(float *, T3 *, __half *, int, int, int, int, int, int)` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::chunkpar::<unnamed>::gdn_chunkpar_pass_kernel<(int)128, (int)128, __nv_bfloat16>(float *, T3 *, __half *, int, int, int, int, int, int)` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)128, (int)128, (bool)0, (bool)0, (int)64, (bool)0, (bool)0, (bool)1>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)0, (bool)0, (int)64, (bool)0, (bool)0, (bool)1>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gdn_scan_chunkwise_kernel<(int)64, (int)64, (int)64, float>(const float *, const __half *, const __half *, const float *, const float *, float *, T4 *, int, int, int, int, int)` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gdn_scan_chunkwise_kernel<(int)64, (int)64, (int)64, __half>(const float *, const __half *, const __half *, const float *, const float *, float *, T4 *, int, int, int, int, int)` | 168 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)0, (bool)1, (int)8, (int)16>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 168 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::attn_kernel<(int)128, (int)12, (int)5, (bool)1, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 168 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)64, (bool)1, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 166 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::attn_kernel<(int)128, (int)12, (int)5, (bool)0, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 165 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)256, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 164 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)128, (int)64, (bool)1, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 163 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)64, (bool)1, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, ` | 163 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)128, (int)128, (bool)1, (bool)1, (int)64, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool` | 161 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)1, (bool)1, (int)64, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 161 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)128, (bool)1, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 158 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)512, (int)4, (int)2>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 157 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)128, (int)128, (bool)0, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool` | 156 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)0, (bool)0, (int)64, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 156 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)64, (bool)1, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 156 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_q8k128_kernel<(int)128, (bool)0>(const signed char *, const __half *, const signed char *, const __half *, __half *, int, int, int)` | 155 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cub::_V_300403_SM_1200::detail::radix_sort::DeviceRadixSortSingleTileKernel<cub::_V_300403_SM_1200::detail::radix_sort::policy_selector_from_types<float, int, unsigned int>, (cub::_V_300403_SM_12` | 153 | 0 | 34880 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)128, (bool)1, (bool)1, (int)64, (bool)1, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool,` | 153 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)128, (bool)1, (bool)0, (bool)1, (bool)1>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 153 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)2, (int)4, (int)4, float>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 152 | 0 | 13312 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)2, (int)4, (int)4, __half>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 152 | 0 | 13312 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)64, (bool)1, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 150 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)128, (bool)1, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 150 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)256, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 150 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)512, (int)4, (int)2>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 148 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)128, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 145 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)256, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 144 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)1, (int)8, (int)2, float>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 144 | 0 | 13312 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)1, (int)8, (int)2, __half>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 144 | 0 | 13312 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fa2_kernel<(int)64, (int)64, (bool)1, (bool)0, (int)64, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, ` | 143 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)128, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 140 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)128, (bool)1, (bool)0, (bool)1, (bool)1>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 140 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)96, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 138 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)128, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 137 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_kernel<(int)128, (bool)0, (bool)0, (bool)0>(const signed char *, const __half *, const float *, const signed char *, const __half *, __half *, int, int, int, const int *,` | 136 | 0 | 45056 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_kernel<(int)128, (bool)1, (bool)0, (bool)0>(const signed char *, const __half *, const float *, const signed char *, const __half *, __half *, int, int, int, const int *,` | 136 | 0 | 45056 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)128, (bool)1, (bool)0, (bool)1, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 135 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)128, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 135 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)64, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 134 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)128, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 133 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)128, (bool)1, (bool)0, (bool)1, (bool)1>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 133 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemm_q6k_fused_moe_prefill_tc_kernel(const unsigned char *, const __half *, __half *, const int *, const int *, int, int, unsigned long, int, int *)` | 128 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)128, (int)128>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)128, (int)256>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)128, (int)128, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)128, (int)256, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)256, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)96, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_moe_fused_kernel<(imp::QKType)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const int *, int, int, unsigned long, int)` | 128 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_v2_kernel<(int)3, __half>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, T2 *, float, int, int, int, int, i` | 128 | 0 | 1072 | 0 | 2 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_v2_kernel<(int)4, __half>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, T2 *, float, int, int, int, int, i` | 128 | 0 | 1088 | 0 | 2 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_v2_kernel<(int)6, float>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, T2 *, float, int, int, int, int, in` | 128 | 0 | 1120 | 0 | 2 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_v2_kernel<(int)6, __half>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, T2 *, float, int, int, int, int, i` | 128 | 0 | 1120 | 0 | 2 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)256, (int)4, (int)4>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 128 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_gqa_kernel<(int)256, (int)4>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __ha` | 128 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)256, (int)4, (int)4>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 128 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_mma_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const int *,` | 128 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_gqa_kernel<(int)256, (int)4>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, floa` | 128 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)0, (bool)0, (int)8, (int)16>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)1, (bool)0, (int)8, (int)16>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)1, (bool)1, (int)8, (int)16>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 128 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::attn_kernel<(int)64, (int)16, (int)6, (bool)1, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 128 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemm_q6k_moe_fused_kernel(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const int *, int, int, unsigned long, int)` | 127 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)128, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, in` | 127 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)64, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 127 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)96, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 127 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::attn_kernel<(int)64, (int)16, (int)6, (bool)0, __half, (int)1>(const unsigned char *, const float *, const unsigned char *, const ra2::HeadScale *, T5 *, ra2::Dims)` | 127 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_f16_narrow_prefill_kernel<(int)128>(imp::<unnamed>::PrefillArgs)` | 126 | 0 | 37888 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_fp8_gqa_kernel<(int)16, (int)2, (int)5>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int` | 125 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)64, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 124 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)64, (int)256>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 123 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_tc_kernel<(int)512>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const int *, const int ` | 123 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)128, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 122 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_v2_kernel<(int)2, __half>(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, T2 *, float, int, int, int, int, i` | 122 | 0 | 1056 | 0 | 2 | 1 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)64, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 120 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)64, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 120 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)128, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 118 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)96, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 118 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)32, (int)256, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 115 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_kernel<(bool)1>(const unsigned char *, const unsigned char *, float, const __half *, const unsigned char *, const unsigned char *, float, float *, int, int, int)` | 114 | 0 | 22528 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)64, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 113 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)96, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 112 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)1, (int)16, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 112 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)96, (bool)1, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 111 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q5k_raw_kernel<(int)32, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 111 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_fp8_gqa_kernel<(int)16, (int)2, (int)4>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int` | 111 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)64, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 110 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)96, (bool)1, (bool)1, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 109 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q6k_raw_kernel<(int)32, (bool)0>(const signed char *, const __half *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 109 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_gqa_kernel<(int)128, (int)4>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, floa` | 107 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)16, (int)512, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 106 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)32, (int)128, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 106 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)96, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 105 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_gqa_kernel<(int)128, (int)4>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __ha` | 105 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_gqa_kernel<(int)256, (int)3>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __ha` | 105 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)64, (int)128, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 103 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)1, (int)8, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 103 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_kernel<(bool)0>(const unsigned char *, const unsigned char *, float, const __half *, const unsigned char *, const unsigned char *, float, float *, int, int, int)` | 103 | 0 | 22528 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_gqa_kernel<(int)256, (int)3>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, floa` | 103 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::mmq_imma_q4k_raw_kernel<(int)32, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 102 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)128, (int)4, (int)4>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 102 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)16, (int)512, (bool)1>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 101 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)32, (int)64, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 100 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)1, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 100 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)128, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 99 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)128, (int)4, (int)4>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 99 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)64, (int)64, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int,` | 98 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_moe_fused_kernel<(imp::QKType)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const int *, int, int, unsigned long, int)` | 98 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_f16_narrow_prefill_kernel<(int)96>(imp::<unnamed>::PrefillArgs)` | 98 | 0 | 41984 | 0 | 1 | 1 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)128, (int)64, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 96 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_mxfp4_kernel<(int)128, (int)96, (bool)0, (bool)0, (bool)0, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int` | 96 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_tc_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const int *, const int ` | 96 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)1, (int)4, (int)4, float>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 96 | 0 | 7168 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)1, (int)4, (int)4, __half>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 96 | 0 | 7168 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)2, (int)2, (int)4, __half>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 96 | 0 | 7168 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_nvfp4_smallm_v2_multi_kernel<(int)6>(imp::<unnamed>::SmallMV2MultiArgs, const unsigned char *, const unsigned char *, int, int)` | 96 | 0 | 1120 | 0 | 2 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_fp8_gqa_kernel<(int)16, (int)2, (int)3>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int` | 96 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)0, (bool)1, (int)16, (int)8>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)1, (bool)1, (int)16, (int)8>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 96 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)128, (int)96, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 95 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmq_imma_q51_raw_kernel<(int)32, (bool)0>(const signed char *, const __half *, const float *, const unsigned char *, __half *, int, int, int, const int *, unsigned long)` | 95 | 0 | 25344 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)64, (int)64, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 94 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)32, (int)96, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 93 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)64, (int)96, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 93 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::conv_silu_split_kernel<(int)4>(const float *, const __half *, long, const __half *, const __half *, __half *, __half *, __half *, int, int, int)` | 90 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)64, (int)128>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 89 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)256, (int)4, (int)2>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 89 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)32, (int)128>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 88 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_kernel<(int)32, (int)64, (bool)0>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int, const __half *)` | 88 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)256, (int)4, (int)2>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 87 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)64, (int)64>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 86 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_fp8_gqa_kernel<(int)16, (int)2, (int)2>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int` | 85 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)32, (int)64>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 84 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fmha_sm120_fp8_kernel<(int)32, (int)256>(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float, bool, int, float, int)` | 83 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cub::_V_300403_SM_1200::detail::radix_sort::DeviceRadixSortOnesweepKernel<cub::_V_300403_SM_1200::detail::radix_sort::policy_selector_from_types<float, int, unsigned int>, (cub::_V_300403_SM_1200` | 80 | 0 | 37376 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)0, (int)16, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 80 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)0, (int)16, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 80 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_kernel<(int)32, (bool)0, (bool)0, (bool)0>(const signed char *, const __half *, const float *, const signed char *, const __half *, __half *, int, int, int, const int *, ` | 80 | 0 | 28928 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_q8k128_kernel<(int)32, (bool)0>(const signed char *, const __half *, const signed char *, const __half *, __half *, int, int, int)` | 80 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_q8k128_kernel<(int)32, (bool)1>(const signed char *, const __half *, const signed char *, const __half *, __half *, int, int, int)` | 80 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)64, (int)4, (int)4>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, ` | 80 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_gqa_kernel<(int)128, (int)3>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __ha` | 80 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_gqa_kernel<(int)256, (int)2>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __ha` | 80 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)512, (int)4, (int)1>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 80 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_gqa_kernel<(int)128, (int)3>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, floa` | 80 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_gqa_kernel<(int)256, (int)2>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, floa` | 80 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)0, (bool)0, (int)16, (int)8>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::ssm_scan_reg_kernel<(bool)1, (bool)0, (int)16, (int)8>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half ` | 80 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::mmq_imma_kernel<(int)32, (bool)0, (bool)0, (bool)1>(const signed char *, const __half *, const float *, const signed char *, const __half *, __half *, int, int, int, const int *, ` | 79 | 0 | 28928 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)64, (int)4, (int)4>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, i` | 78 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `imp::gemm_gate_fp32_rows_kernel(const __half *, const __half *, float *, int, int, int)` | 76 | 0 | 27648 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_f16_narrow_prefill_kernel<(int)64>(imp::<unnamed>::PrefillArgs)` | 74 | 0 | 33792 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)0, (int)8, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 72 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)0, (int)8, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 72 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_fp8_kernel<(int)512>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int, float, float, int, int, int,` | 72 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_kernel<(int)512>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 72 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)512, (int)4, (int)1>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 71 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)512, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 70 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_tc_kernel<(int)512>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const int *, const int *` | 70 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::sparse_score_blocks_kernel<imp::<unnamed>::ScoreMeanStd>(const __half *, const __half2 *, const int *, const int *, float *, int, int, int, int, int, int, int, float, int, int, in` | 70 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)0, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 69 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)512, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 69 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::sparse_score_blocks_kernel<imp::<unnamed>::ScoreCornerBound>(const __half *, const __half2 *, const int *, const int *, float *, int, int, int, int, int, int, int, float, int, int` | 69 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemm_qk_dp4a_dense_kernel<(imp::QKType)0, (int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int, int)` | 68 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)256, apa::FlatKV<__half>>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 65 | 0 | 35072 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)256, apa::PagedF16KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 65 | 0 | 35072 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_nvfp4_swiglu_residual_kernel(const unsigned char *, const unsigned char *, float, const __half *, const __half *, __half *, const __half *, int, int)` | 64 | 0 | 1104 | 0 | 1 | 1 | 0 | 0 |
| `imp::mtp_attn_kv_scan_rows_kernel(const __half *, const __half *, const __half *, __half *, const int *, const int *, long, int, int, int, float)` | 64 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::prompt_logprobs_rows_kernel(const float *, int, int, int, const int *, int, float *, int *, int *, float *)` | 64 | 0 | 5120 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)128, apa::PagedNvfp4KV>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 64 | 0 | 5120 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)256, apa::PagedNvfp4KV>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 64 | 0 | 9216 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)64, apa::PagedNvfp4KV>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 64 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_swiglu_residual_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const __half *, const __half *, __half *, const __half *, int, int)` | 64 | 0 | 1088 | 0 | 1 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_kernel<(int)512>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)128, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)128, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)512, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)512, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_tc_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const int *, const int ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_tc_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const int *, const int *` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_residual_reduce_kernel<(int)512>(const float *, const __half *, __half *, const int *, int, int, int, float, int, float, int, const __half *, const __half *, int, int, const ` | 64 | 0 | 1040 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)128, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)128, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_tc_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const int *, const int *` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemm_q8_0_dp4a_dense_kernel<(int)16, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int)` | 64 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_q8_0_dp4a_dense_kernel<(int)16, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int)` | 64 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_q8_0_dp4a_dense_kernel<(int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int)` | 64 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_q8_0_dp4a_dense_kernel<(int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int)` | 64 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_q8_0_dp4a_dense_kernel<(int)8, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int)` | 64 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_q8_0_dp4a_dense_kernel<(int)8, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, int, int, int)` | 64 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)128, (int)4, (int)2>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)256, (int)4, (int)1>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_fp8_gqa_kernel<(int)16, (int)2, (int)1>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_gqa_kernel<(int)128, (int)2>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __ha` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_nvfp4_multitok_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)128, (int)4, (int)2>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)256, (int)4, (int)1>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 64 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)256, ra2::PagedNvfp4KV>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 64 | 0 | 35200 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 63 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)64, (int)4, (int)2>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, ` | 63 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)64, (int)4, (int)2>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, i` | 63 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_nvfp4_multitok_gqa_kernel<(int)128, (int)2>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, floa` | 63 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `imp::<unnamed>::gemm_f16_narrow_smallm_kernel(imp::<unnamed>::NarrowArgs)` | 62 | 0 | 17412 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 62 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)4>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 62 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 62 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::fp8_head_mma_kernel<(int)1, (int)2, (int)4, __half>(const unsigned char *, const float *, const __half *, T4 *, int, int, int)` | 62 | 0 | 4096 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)256, ra2::PagedF16KV>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 62 | 0 | 35200 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 60 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)128, apa::PagedF16KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 57 | 0 | 18176 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_nvfp4_geglu_residual_kernel(const unsigned char *, const unsigned char *, float, const __half *, const __half *, __half *, const __half *, int, int)` | 56 | 0 | 1104 | 0 | 1 | 1 | 0 | 0 |
| `imp::mtp_attn_kv_scan_kernel(const __half *, const __half *, const __half *, __half *, int, int, int, int, int, float)` | 56 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::qknorm_rope_fused_fp16_kernel(__half *, __half *, const __half *, const __half *, const int *, int, int, int, float, float, float, int, bool, float, float, float, float, float, const float *, imp` | 56 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)128, apa::FlatKV<__half>>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 56 | 0 | 18176 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)128, apa::PagedNvfp4KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 56 | 0 | 18176 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_geglu_residual_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const __half *, const __half *, __half *, const __half *, int, int)` | 56 | 0 | 1088 | 0 | 1 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_fp8_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int, float, float, int, int, int, ` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_int4_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, ` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)64, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const i` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_int8_kernel<(int)256>(const __half *, const signed char *, const signed char *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, ` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)256, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)256, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const i` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)64, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const in` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_kernel<(int)64, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const in` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_tc_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const int *, const int *` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_nvfp4_tc_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, float *, const int *, const int *,` | 56 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::ssm_scan_kernel<(bool)0, (bool)0>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *, int, int, int` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::ssm_scan_kernel<(bool)1, (bool)0>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *, int, int, int` | 56 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)64, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const i` | 55 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)8>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)4>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)4>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 54 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemm_f16_narrow_prefill_kernel<(int)32>(imp::<unnamed>::PrefillArgs)` | 54 | 0 | 25600 | 0 | 1 | 1 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)256, ra2::FlatKV<__half>>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 54 | 0 | 35200 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_fp32_accum_to_fp16_kernel(const __half *, const __half *, float *, __half *, int, float, float)` | 53 | 0 | 1168 | 0 | 0 | 0 | 0 | 0 |
| `imp::ssm_conv1d_prefill_kernel(float *, const __half *, const __half *, const __half *, __half *, int, int, int, const int *, float *, const int *, const float *)` | 52 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 52 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)4, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 52 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)4, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 52 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp32_kernel<(int)4>(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int)` | 52 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)4, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 52 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)4, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 52 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `imp::<unnamed>::qsa_select_kernel(const int *, const float *, int, int *, int *, imp::QsaGeom)` | 50 | 0 | 2320 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)4>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 50 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemm_q6k_fused_moe_prefill_kernel(const unsigned char *, const __half *, __half *, const int *, int, int, unsigned long, int)` | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_fp32_kernel(const float *, const float *, float *, int, int)` | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_gate_topk_fused_kernel(const __half *, const __half *, int, int, int, int *, float *, bool, bool, const __half *)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_q8_0_moe_decode_kernel(const unsigned char *, const int *, const __half *, __half *, int, int, unsigned long, int, int)` | 48 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_q_kernel<(int)256, __half>(const T2 *, apa::Dims, float, unsigned char *, unsigned char *, float *)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_q_kernel<(int)64, __half>(const T2 *, apa::Dims, float, unsigned char *, unsigned char *, float *)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)128, apa::PagedF16KV>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 48 | 0 | 5120 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)256, apa::PagedF16KV>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 48 | 0 | 9216 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)64, apa::PagedF16KV>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 48 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)4>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)4>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 48 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)2>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)5>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)4>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)4>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_fp8_kernel<(int)256>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int, float, float, int, int, int,` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_int8_kernel<(int)128>(const __half *, const signed char *, const signed char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int,` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_int8_kernel<(int)256>(const __half *, const signed char *, const signed char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int,` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_int8_kernel<(int)64>(const __half *, const signed char *, const signed char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, ` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_int8_kernel<(int)96>(const __half *, const signed char *, const signed char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, ` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_kernel<(int)256>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)256, (imp::ScaleDtype)0>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_nvfp4_kernel<(int)256, (imp::ScaleDtype)1>(const __half *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, __half *, const ` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_residual_reduce_kernel<(int)256>(const float *, const __half *, __half *, const int *, int, int, int, float, int, float, int, const __half *, const __half *, int, int, const ` | 48 | 0 | 1040 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_int8_kernel<(int)128>(const __half *, const signed char *, const signed char *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, ` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_int8_kernel<(int)64>(const __half *, const signed char *, const signed char *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, f` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_int8_kernel<(int)96>(const __half *, const signed char *, const signed char *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, f` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_kernel<(int)192>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_kernel<(int)256>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)64>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_pipeline_kernel<(int)96>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::ssm_scan_kernel<(bool)0, (bool)1>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *, int, int, int` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::ssm_scan_kernel<(bool)1, (bool)1>(const __half *, const __half *, const __half *, const __half *, const float *, const float *, const float *, void *, __half *, const __half *, int, int, int` | 48 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)3, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 48 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)3, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 48 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)128, (int)4, (int)1>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float,` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_fp8_multitok_kernel<(int)128, (int)4>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, float` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)128, (int)4, (int)1>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, ` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_splitk_f16_multitok_kernel<(int)64, (int)4, (int)1>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, i` | 48 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)128, ra2::FlatKV<__half>>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 48 | 0 | 18304 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)128, ra2::PagedF16KV>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 48 | 0 | 18304 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)128, ra2::PagedNvfp4KV>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 48 | 0 | 18304 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)64, ra2::PagedNvfp4KV>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 48 | 0 | 9856 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_int4_kernel<(int)64>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, i` | 47 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_int4_kernel<(int)96>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, i` | 47 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_kernel<(int)192>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 47 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_kernel<(int)128>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 47 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `imp::quantize_fp16_mxfp8_cutlass_kernel(const __half *, unsigned char *, unsigned char *, int, int, int)` | 46 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_q_kernel<(int)128, __half>(const T2 *, apa::Dims, float, unsigned char *, unsigned char *, float *)` | 46 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_gate_up_fused_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const unsigned char *, const unsigned char *, float, const __half *, __half *, __half *, int, ` | 46 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gemv_nvfp4_qkv_fused_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const unsigned char *, const unsigned char *, float, const unsigned char *, const unsigned char *,` | 46 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::mtp_mrope_kernel<(bool)0>(__half *, int, int, int, float, int, int, int, int, int, int, float, float, float, float, float, const int *)` | 46 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_mrope_kernel<(bool)1>(__half *, int, int, int, float, int, int, int, int, int, int, float, float, float, float, float, const int *)` | 46 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_int4_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, ` | 46 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_kernel<(int)128>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 46 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_kernel<(int)96>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 46 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 45 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_scatter_fused_residual_kernel<(bool)1>(const __half *, const int *, const float *, const __half *, __half *, int, int, const __half *, const int *)` | 45 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_nvfp4_gate_up_fused_kernel(const unsigned char *, const unsigned char *, float, const unsigned char *, const unsigned char *, float, const __half *, __half *, __half *, int, int, imp::NvFP4N` | 44 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `imp::gemv_nvfp4_qkv_fused_kernel(const unsigned char *, const unsigned char *, float, const unsigned char *, const unsigned char *, float, const unsigned char *, const unsigned char *, float, const __` | 44 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `imp::gemv_nvfp4_residual_kernel(const unsigned char *, const unsigned char *, float, const __half *, __half *, const __half *, int, int, imp::NvFP4NormFoldOut)` | 44 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `imp::quantize_fp16_mxfp4_cutlass_kernel(const __half *, unsigned char *, unsigned char *, int, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)128, apa::FlatKV<__half>>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 44 | 0 | 5120 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)256, apa::FlatKV<__half>>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 44 | 0 | 9216 | 0 | 0 | 0 | 0 | 0 |
| `void apa::stats_kernel<(int)64, apa::FlatKV<__half>>(T2, apa::Dims, unsigned int *, float *, const int *, int)` | 44 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)4>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 44 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)3>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 44 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_residual_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const __half *, __half *, const __half *, int, int, imp::NvFP4NormFoldOut)` | 44 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::paged_attention_splitk_kernel<(int)64>(const __half *, const __half *, const __half *, float *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 44 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_kernel<(int)96>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 43 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `imp::gemv_nvfp4_kpar_fp32_kernel(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int, imp::NvFP4NormFoldIn)` | 42 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `imp::gemv_nvfp4_kpar_kernel(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 42 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `imp::gemv_q8_0_moe_gate_up_fused_kernel(const unsigned char *, const unsigned char *, const int *, const __half *, __half *, __half *, int, int, unsigned long, unsigned long, int, int)` | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_residual_fp16_kernel(__half *, const __half *, const __half *, __half *, int, float, float)` | 42 | 0 | 1168 | 0 | 1 | 1 | 0 | 0 |
| `imp::<unnamed>::expert_cache_gather_kernel(const imp::DevMissEntry *, const int *, unsigned long, unsigned long, unsigned long)` | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::gemm_f16_narrow_prefill_reduce_kernel(imp::<unnamed>::PrefillArgs)` | 42 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `imp::<unnamed>::gemv_fp8_block_moe_kernel(imp::Fp8BlockMoeArgs, int)` | 42 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void apa::kv_fingerprint_kernel<(int)128, apa::PagedF16KV>(T2, apa::Dims, float *, int, int *, bool)` | 42 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_multirow_fp32_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int, imp::NvFP4NormFoldIn)` | 42 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::gemv_nvfp4_multirow_kernel<(int)8>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 42 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)64, ra2::PagedF16KV>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 42 | 0 | 9856 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequantize_mxfp8_cutlass_fp16_kernel(const unsigned char *, const unsigned char *, const float *, __half *, int, int, int)` | 41 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `build_grouped_3x_staging_kernel(cute::tuple<int, int, int> *, cute::tuple<long, cute::C<(int)1>, cute::C<(int)0>> *, cute::tuple<long, cute::C<(int)1>, cute::C<(int)0>> *, cute::tuple<long, cute::C<(i` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `hidden_pool_sum_kernel(const __half *, int, int, float *)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_penalties_device_count_kernel(float *, const int *, const int *, int, int, float, float, float)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_penalties_kernel(float *, const int *, int, int, float, float, float)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::avg_pool_spatial_kernel(const __half *, __half *, int, int, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequantize_fp8_rows_kernel(const unsigned char *, const float *, __half *, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gdn_scan_decode_kernel(const float *, const float *, const float *, const __half *, const __half *, const float *, const float *, float *, __half *, const __half *, int, int, int, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gdn_scan_reference_kernel(const float *, const __half *, const __half *, const float *, const float *, float *, __half *, int, int, int, int, int, int, int, const int *)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_bf16_kernel(const __nv_bfloat16 *, const __nv_bfloat16 *, __nv_bfloat16 *, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_f16_moe_decode_kernel(const __half *, const int *, const __half *, __half *, int, int, unsigned long, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_fp16_kernel(const __half *, const __half *, __half *, int, int)` | 40 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `imp::gemv_gate_fp32_fp32input_kernel(const __half *, const float *, float *, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_gate_up_kernel(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, const __half *, __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_geglu_residual_kernel(const unsigned char *, const unsigned char *, const __half *, const __half *, __half *, const __half *, int, int)` | 40 | 0 | 1104 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_qkv_kernel(const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, const unsigned char *, const __half *, __half *, __half *,` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_swiglu_residual_kernel(const unsigned char *, const unsigned char *, const __half *, const __half *, __half *, const __half *, int, int)` | 40 | 0 | 1104 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_nvfp4_moe_gate_up_fused_kernel(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const unsigned char *, const float *, const int *, const __half *, __half *` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_q6k_kernel(const unsigned char *, const __half *, __half *, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_q6k_moe_decode_kernel(const unsigned char *, const int *, const __half *, __half *, int, int, unsigned long, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_q6k_moe_gate_up_fused_kernel(const unsigned char *, const unsigned char *, const int *, const __half *, __half *, __half *, int, int, unsigned long, unsigned long, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::kv_batched_copy_kernel(const imp::KVCache::CopyDesc *, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::mla_absorbed_decode_kernel(const __half *, const __half *, const __half *, __half *, float *, const int *, int, int, int, int, int, int, int, float)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_expert_trace_kernel(const int *, const float *, int *, float *, unsigned int *, unsigned long long, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_permute_scatter_kernel(const int *, int, int, int, int, const int *, int *, int *, int *, int *)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_weighted_sum_residual_kernel(const __half *, const float *, const __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mxfp4_promote_select_kernel(const float *, const float *, unsigned char *, int, int, int, int, int, int, int, bool, float, int)` | 40 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_attention_reduce_kernel(const float *, __half *, int, int, int, const __half *, imp::block_q8_1 *, float *)` | 40 | 0 | 4104 | 0 | 1 | 1 | 0 | 0 |
| `imp::quantize_fp16_mxfp4_strided_kernel(const __half *, int, unsigned char *, unsigned char *, int, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp16_to_q8_1_ggml_kernel(const __half *, imp::ggml_block_q8_1 *, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_fp16_kernel(const __half *, const __half *, __half *, int, float, float)` | 40 | 0 | 1168 | 0 | 1 | 1 | 0 | 0 |
| `imp::rmsnorm_fp16_warp_kernel(const __half *, const __half *, __half *, int, int, float, float)` | 40 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `imp::rowwise_argmax_reduce_kernel(const float *, const int *, int, int *)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::ssm_conv1d_decode_f32_silu_kernel(float *, const __half *, const __half *, const __half *, float *, int, int, const int *, long)` | 40 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `imp::ssm_conv1d_decode_kernel(float *, const __half *, const __half *, const __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::ssm_conv1d_prefill_f32_silu_kernel(float *, const __half *, const __half *, const __half *, float *, int, int, int, const int *, float *, const int *, const float *, const int *, long)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::swiglu_fp16_nvfp4_kernel(const __half *, const __half *, __half *, unsigned char *, unsigned char *, long)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `imp::swiglu_fp32_vec4_kernel(const float *, const float *, float *, long)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::swiglu_quantize_fp16_nvfp4_cutlass_kernel(const __half *, const __half *, __half *, unsigned char *, unsigned char *, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::topk_gating_kernel(const float *, int, int, int *, float *, bool, bool, const __half *)` | 40 | 0 | 1120 | 0 | 1 | 0 | 0 | 0 |
| `imp::<unnamed>::accum_abs_cols_experts_kernel(const __half *, const int *, long, long, double *, unsigned long *)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::encoder_mean_pool_kernel(const __half *, float *, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::expert_cache_resolve_kernel(imp::DevExpertLayer, const int *, int, int, int *, imp::DevMissEntry *, int *, imp::TierMailbox *, unsigned int *, int, int *)` | 40 | 0 | 29768 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::expert_stage_touched_kernel(const imp::DevExpertSrc *, const int *, char *, unsigned long, unsigned long, unsigned long, int, imp::StageSlots)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::expert_stage_touched_range_kernel(const imp::DevExpertSrc *, const int *, char *, unsigned long, unsigned long, unsigned long, int, int, int2, imp::StageSlots)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::gemv_nvfp4_gdn_input_kernel(imp::<unnamed>::GdnInputArgs)` | 40 | 0 | 1056 | 0 | 2 | 2 | 0 | 0 |
| `imp::<unnamed>::hc_mix_kernel(const __half *, const __half *, __half *, __half *, int, int, long)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_read_decode_kernel(const __half *, const __half *, const __half *, const __half *, __half *, __half *, __half *, int, int, int, int, float, float, int)` | 40 | 0 | 1312 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::kv_block_copy_kernel(char *, char *, char *, unsigned long, unsigned long, const unsigned long *, imp::<unnamed>::KVCopyPairs)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::mmq_splitk_finalize_kernel(const float *, int, int, __half *, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::pos_embed_add_kernel(__half *, const __half *, const int *, const float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::q8_split_kernel(const unsigned char *, signed char *, __half *, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::qsa_score_kernel(const __half *, const int *, const __half *, float *, int, imp::QsaGeom)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::shortconv_out_kernel(void *, const int *, long, const __half *, const __half *, __half *, int, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::shortconv_state_kernel(void *, const int *, long, const __half *, int, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::split_qkv_rope_kernel(const __half *, const int *, const int *, __half *, __half *, __half *, int, int, int, float)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::ssm_conv_tap_apply_kernel(float *, long, const __half *, int, int, const int *)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::verify_hist_count_kernel(const int *, int, int, int *)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::verify_penalties_kernel(float *, int, int, const int *, const int *, float, float, float)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::write_kv_cache_fp8_fused_kernel(const __half *, const __half *, const int *, const int *, __nv_fp8_e4m3 *, __nv_fp8_e4m3 *, float, int, int, int, int, int, int)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `imp::write_kv_cache_fused_kernel(const __half *, const __half *, const int *, const int *, __half *, __half *, int, int, int, int, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::write_kv_cache_int4_kernel(const __half *, const __half *, const int *, const int *, unsigned char *, unsigned char *, __half *, __half *, int, int, int, int, int, int, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::write_kv_cache_int8_kernel(const __half *, const __half *, const int *, const int *, signed char *, signed char *, __half *, __half *, int, int, int, int, int, int, int, int)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::write_kv_cache_rope_fused_kernel(const __half *, const __half *, const int *, const int *, __half *, __half *, int, int, int, int, int, int, int, int, float, float, int, bool, const float *)` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)64, apa::FlatKV<__half>>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 40 | 0 | 9728 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)64, apa::PagedF16KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 40 | 0 | 9728 | 0 | 0 | 0 | 0 | 0 |
| `void apa::quant_kv_kernel<(int)64, apa::PagedNvfp4KV>(T2, apa::Dims, const float *, const apa::HeadScale *, unsigned char *, int, int, const int *, int *)` | 40 | 0 | 9728 | 0 | 0 | 0 | 0 | 0 |
| `void cub::_V_300403_SM_1200::detail::radix_sort::DeviceRadixSortHistogramKernel<cub::_V_300403_SM_1200::detail::radix_sort::policy_selector_from_types<float, int, unsigned int>, (cub::_V_300403_SM_120` | 40 | 0 | 5120 | 0 | 0 | 0 | 0 | 0 |
| `void imp::embedding_lookup_vec_kernel<float>(const T1 *, const int *, T1 *, int)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<(int)1>(const __half *, const __half *, unsigned char *, unsigned char *const *, const int *, int, int, int, int, const __half *, const __hal` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<(int)3>(const __half *, const __half *, unsigned char *, unsigned char *const *, const int *, int, int, int, int, const __half *, const __hal` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_batched_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)4>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)1>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_fp32_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)4>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)4>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)1>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)4, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)4, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)1, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)1, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)0>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)1>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)2>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)3>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)4>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_fp32_kernel<imp::DequantTraits<(imp::DPQTag)5>>(const unsigned char *, const imp::block_q8_1 *, const float *, float *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)0>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)1>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)2>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)5>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)0>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)0>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)1>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)1>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)2>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)2>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)3>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)3>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)4>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)4>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)5>, (bool)0>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_kernel<imp::DequantTraits<(imp::DPQTag)5>, (bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)0>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)1>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)2>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)3>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_qkv_kernel<imp::DequantTraits<(imp::DPQTag)5>>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *,` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)2>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)4>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)7>, (int)1>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_decode_kernel<imp::DequantTraits<(imp::DPQTag)7>, (int)2>(const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, int, int, unsigned long, int, in` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)0>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)1>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_moe_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)7>>(const unsigned char *, const unsigned char *, const int *, const imp::block_q8_1 *, const float *, __half *, __half *, int, i` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)0>, (int)4>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)1>, (int)4>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)3>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)1>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_qkv_kernel<imp::DequantTraits<(imp::DPQTag)5>, (int)2>(const unsigned char *, const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_decode_mr_kernel<(int)16>(const unsigned char *, const unsigned char *, const float *, const int *, const __half *, __half *, int, int, unsigned long, unsigned long, int, int,` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_decode_mr_kernel<(int)32>(const unsigned char *, const unsigned char *, const float *, const int *, const __half *, __half *, int, int, unsigned long, unsigned long, int, int,` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_decode_mr_kernel<(int)4>(const unsigned char *, const unsigned char *, const float *, const int *, const __half *, __half *, int, int, unsigned long, unsigned long, int, int, ` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_decode_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, const float *, const int *, const __half *, __half *, int, int, unsigned long, unsigned long, int, int, ` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_gate_up_mr_kernel<(int)16>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const unsigned char *, const float *, const int *, const __half ` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_gate_up_mr_kernel<(int)32>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const unsigned char *, const float *, const int *, const __half ` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_gate_up_mr_kernel<(int)4>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const unsigned char *, const float *, const int *, const __half *` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_nvfp4_moe_gate_up_mr_kernel<(int)8>(const unsigned char *, const unsigned char *, const float *, const unsigned char *, const unsigned char *, const float *, const int *, const __half *` | 40 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmvq_kernel<(imp::MMVQTag)3>(const void *, const imp::ggml_block_q8_1 *, __half *, int, int)` | 40 | 0 | 1408 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_scatter_deterministic_kernel_impl<float>(const T1 *, const int *, const int *, const float *, float *, int, int, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_scatter_deterministic_kernel_impl<__half>(const T1 *, const int *, const int *, const float *, float *, int, int, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_scatter_fused_residual_kernel<(bool)0>(const __half *, const int *, const float *, const __half *, __half *, int, int, const __half *, const int *)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::paged_attention_decode_fp8_kernel<(int)128>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int, float, float, int, int, int,` | 40 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_fp8_kernel<(int)96>(const __half *, const unsigned char *, const unsigned char *, __half *, const int *, const int *, int, int, int, int, float, float, int, int, int, ` | 40 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_decode_kernel<(int)64>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, int, int, int, int, float)` | 40 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_residual_reduce_kernel<(int)128>(const float *, const __half *, __half *, const int *, int, int, int, float, int, float, int, const __half *, const __half *, int, int, const ` | 40 | 0 | 1040 | 0 | 0 | 1 | 0 | 0 |
| `void imp::paged_attention_residual_reduce_kernel<(int)64>(const float *, const __half *, __half *, const int *, int, int, int, float, int, float, int, const __half *, const __half *, int, int, const _` | 40 | 0 | 1040 | 0 | 0 | 1 | 0 | 0 |
| `void imp::rmsnorm_fp16_rowblock_kernel<(int)2>(const __half *, const __half *, __half *, int, float, float)` | 40 | 0 | 1156 | 0 | 1 | 1 | 0 | 0 |
| `void imp::rmsnorm_fp16_rowblock_nvfp4_kernel<(int)2, (bool)0>(const __half *, const __half *, __half *, unsigned char *, unsigned char *, int, float, float, const __half *, __half *, __half *)` | 40 | 0 | 1156 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)1, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)1, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)2, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp16_kernel<(int)2, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp32_kernel<(int)1>(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int)` | 40 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_kpar_mb_fp32_kernel<(int)2>(const unsigned char *, const unsigned char *, float, const __half *, float *, int, int)` | 40 | 0 | 1040 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)1, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)1, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)2, (bool)0>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_nvfp4_multirow_mb_kernel<(int)8, (int)2, (bool)1>(const unsigned char *, const unsigned char *, float, const __half *, __half *, int, int)` | 40 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::gemv_q8_0_q8_1_residual_masked_kernel<(bool)1>(const unsigned char *, const imp::block_q8_1 *, const float *, const unsigned int *, __half *, const __half *, int, int)` | 40 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::paged_attention_decode_f16_multitok_kernel<(int)64, (int)4, (int)1>(const __half *, const __half *, const __half *, __half *, const int *, const int *, int, int, int, int, float, ` | 40 | 0 | 1024 | 0 | 0 | 1 | 0 | 0 |
| `void imp::<unnamed>::smallm_v2_reduce_kernel<(bool)0>(const float *, __half *, int, int, int, float)` | 40 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::smallm_v2_reduce_kernel<(bool)1>(const float *, __half *, int, int, int, float)` | 40 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::sparse_update_key_minmax_layers_kernel<__half, (bool)0>(const T1 *, long, __half2 *, long, const int *, const int *, const int *, int, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::sparse_update_key_minmax_layers_kernel<__nv_fp8_e4m3, (bool)0>(const T1 *, long, __half2 *, long, const int *, const int *, const int *, int, int, int, int, int)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::sparse_update_key_minmax_layers_nvfp4_kernel<(bool)0>(const unsigned char *, long, const unsigned char *, long, __half2 *, long, const int *, const int *, const int *, int, int, i` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::sparse_update_key_minmax_layers_nvfp4_kernel<(bool)1>(const unsigned char *, long, const unsigned char *, long, __half2 *, long, const int *, const int *, const int *, int, int, i` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_kv_kernel<(int)64, ra2::FlatKV<__half>>(T2, ra2::Dims, const unsigned int *, const float *, unsigned char *, ra2::HeadScale *)` | 40 | 0 | 9856 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_q_kernel<(int)128, __half>(const T2 *, ra2::Dims, float, unsigned char *, float *)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_q_kernel<(int)256, __half>(const T2 *, ra2::Dims, float, unsigned char *, float *)` | 40 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)128, ra2::PagedF16KV>(T2, ra2::Dims, unsigned int *, float *)` | 40 | 0 | 1536 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)128, ra2::PagedNvfp4KV>(T2, ra2::Dims, unsigned int *, float *)` | 40 | 0 | 1536 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)256, ra2::PagedF16KV>(T2, ra2::Dims, unsigned int *, float *)` | 40 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)256, ra2::PagedNvfp4KV>(T2, ra2::Dims, unsigned int *, float *)` | 40 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)64, ra2::FlatKV<__half>>(T2, ra2::Dims, unsigned int *, float *)` | 40 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)64, ra2::PagedNvfp4KV>(T2, ra2::Dims, unsigned int *, float *)` | 40 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_penalties_rows_kernel(const imp::PenaltyRowArgs *, int)` | 39 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `imp::dequant_mxfp4_split_kernel(const unsigned char *, const unsigned char *, __half *, int, int, int)` | 39 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_kpar_fp32_kernel(const unsigned char *, const unsigned char *, const __half *, float *, int, int)` | 39 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_kpar_kernel(const unsigned char *, const unsigned char *, const __half *, __half *, int, int)` | 39 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_mxfp4_residual_kernel(const unsigned char *, const unsigned char *, const __half *, __half *, const __half *, int, int)` | 39 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_nvfp4_moe_decode_kernel(const unsigned char *, const unsigned char *, const float *, const int *, const __half *, __half *, int, int, unsigned long, unsigned long, int, int, int, const int *` | 39 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `imp::mirostat_v2_sample_kernel(const float *, int, float, float, unsigned int, int *, float *)` | 39 | 0 | 1068 | 0 | 0 | 0 | 0 | 0 |
| `imp::mxfp4_k_channel_mean_kernel(const __half *, float *, int, int, int)` | 39 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mxfp4_tile_mean_kernel(const __half *, float *, int, int, int, int)` | 39 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::accum_abs_cols_kernel(const __half *, long, long, long, double *, double *)` | 39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void apa::finalize_stats_kernel<(int)128>(const unsigned int *, const float *, int, float *, apa::HeadScale *, int, float, const int *, bool)` | 39 | 0 | 1536 | 0 | 0 | 0 | 0 | 0 |
| `void apa::finalize_stats_kernel<(int)256>(const unsigned int *, const float *, int, float *, apa::HeadScale *, int, float, const int *, bool)` | 39 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `void apa::finalize_stats_kernel<(int)64>(const unsigned int *, const float *, int, float *, apa::HeadScale *, int, float, const int *, bool)` | 39 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<(int)0>(const __half *, const __half *, unsigned char *, unsigned char *const *, const int *, int, int, int, int, const __half *, const __hal` | 39 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_dp4a_kpar_gate_up_kernel<imp::DequantTraits<(imp::DPQTag)3>>(const unsigned char *, const unsigned char *, const imp::block_q8_1 *, const float *, __half *, __half *, int, int)` | 39 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_mxfp4_multirow_fp32_kernel<(int)8>(const unsigned char *, const unsigned char *, const __half *, float *, int, int)` | 39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_mxfp4_multirow_kernel<(int)8>(const unsigned char *, const unsigned char *, const __half *, __half *, int, int)` | 39 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmvq_kernel<(imp::MMVQTag)2>(const void *, const imp::ggml_block_q8_1 *, __half *, int, int)` | 39 | 0 | 1408 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_topk_kernel<float>(const T1 *, int, int, int *)` | 39 | 0 | 3104 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_topw_pass1_kernel<float>(const T1 *, int, int, float *, int *)` | 39 | 0 | 3104 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_topw_pass1_kernel<__half>(const T1 *, int, int, float *, int *)` | 39 | 0 | 3104 | 0 | 0 | 0 | 0 | 0 |
| `void imp::rmsnorm_fp16_rowblock_nvfp4_kernel<(int)2, (bool)1>(const __half *, const __half *, __half *, unsigned char *, unsigned char *, int, float, float, const __half *, __half *, __half *)` | 39 | 0 | 1156 | 0 | 1 | 1 | 0 | 0 |
| `void imp::<unnamed>::build_swiglu_block_mask_kernel<(int)256>(const __half *, const __half *, unsigned int *, int, float)` | 39 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::quant_q_kernel<(int)64, __half>(const T2 *, ra2::Dims, float, unsigned char *, float *)` | 39 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::absmax_final_reduce_kernel(const float *, float *, int)` | 38 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_q8_0_kernel(const unsigned char *, const __half *, __half *, int, int)` | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::topp_sample_from_sorted_kernel(const float *, const int *, int, float, unsigned int, const int *, int *)` | 38 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::sparse_prefill_merge_rows_kernel(const float *, float *, int, int, int)` | 38 | 0 | 1408 | 0 | 0 | 0 | 0 | 0 |
| `imp::write_kv_cache_mxfp4_kv_kernel(const __half *, const __half *, const int *, const int *, unsigned char *, unsigned char *, unsigned char *, unsigned char *, int, int, int, int, int, int, int, int` | 38 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_fp8_e4m3_kernel<(bool)1>(const unsigned char *, const __half *, __half *, int, int, float, const float *)` | 38 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `void imp::mmvq_kernel<(imp::MMVQTag)0>(const void *, const imp::ggml_block_q8_1 *, __half *, int, int)` | 38 | 0 | 1408 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_argmax_kernel<float>(const T1 *, int, int *)` | 38 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::sparse_update_key_minmax_layers_kernel<__half, (bool)1>(const T1 *, long, __half2 *, long, const int *, const int *, const int *, int, int, int, int, int)` | 38 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::sparse_update_key_minmax_layers_kernel<__nv_fp8_e4m3, (bool)1>(const T1 *, long, __half2 *, long, const int *, const int *, const int *, int, int, int, int, int)` | 38 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::lora_gemv_b_kernel(const __half *, const float *, __half *, float, int, int)` | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_topk_kernel<__half>(const T1 *, int, int, int *)` | 37 | 0 | 3104 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemv_gate_fp32_kernel(const __half *, const __half *, float *, int, int)` | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::nvfp4_moe_native_quant_kernel(const __half *, void *const *, void *const *, const int *, int)` | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::ssm_conv1d_commit_kernel(float *, const __half *, int, int, int, long, const int *, const int *, long, const int *)` | 36 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::probe_silu_kernel(int, const __half *, const __half *, int, unsigned long long *)` | 36 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::write_kv_cache_nvfp4_kernel(const __half *, const __half *, const int *, const int *, unsigned char *, unsigned char *, unsigned char *, unsigned char *, int, int, int, int, int, int, int, int)` | 36 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_fp16_nvfp4_cutlass_moe_kernel<(int)2>(const __half *, const __half *, unsigned char *, unsigned char *const *, const int *, int, int, int, int, const __half *, const __hal` | 36 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::gemv_fp8_e4m3_kernel<(bool)0>(const unsigned char *, const __half *, __half *, int, int, float, const float *)` | 36 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `void imp::rope_forward_kernel<float>(T1 *, T1 *, const int *, int, int, int, int, int, float, float, int, bool, float, float, float, float, const float *, imp::MRopeParams)` | 36 | 0 | 5120 | 0 | 1 | 1 | 0 | 0 |
| `void imp::rope_forward_kernel<__half>(T1 *, T1 *, const int *, int, int, int, int, int, float, float, int, bool, float, float, float, float, const float *, imp::MRopeParams)` | 36 | 0 | 5120 | 0 | 1 | 1 | 0 | 0 |
| `void ra2::stats_kernel<(int)256, ra2::FlatKV<__half>>(T2, ra2::Dims, unsigned int *, float *)` | 36 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mmvq_kernel<(imp::MMVQTag)1>(const void *, const imp::ggml_block_q8_1 *, __half *, int, int)` | 35 | 0 | 1408 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)128, ra2::FlatKV<__half>>(T2, ra2::Dims, unsigned int *, float *)` | 35 | 0 | 1536 | 0 | 0 | 0 | 0 | 0 |
| `void ra2::stats_kernel<(int)64, ra2::PagedF16KV>(T2, ra2::Dims, unsigned int *, float *)` | 35 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `imp::argmax_reduce_rows_kernel(const imp::GreedyRowArgs *)` | 34 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `imp::embedding_lookup_fp16_device_kernel(const __half *, const int *, __half *, int)` | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp16_nvfp4_cutlass_kernel(const __half *, unsigned char *, unsigned char *, int, int, int)` | 34 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp16_nvfp4_cutlass_moe_kernel(const __half *, const int *, unsigned char *, unsigned char *const *, const int *, int, int, int, int)` | 34 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_nvfp4_from_absmax_kernel(const __half *, unsigned char *, unsigned char *, const float *, float *, long, long)` | 34 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_nvfp4_kernel(const __half *, unsigned char *, unsigned char *, float, long, long)` | 34 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::embedding_lookup_vec_kernel<__half>(const T1 *, const int *, T1 *, int)` | 34 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `void imp::embedding_lookup_vec_kernel<unsigned short>(const T1 *, const int *, T1 *, int)` | 34 | 0 | 0 | 0 | 1 | 1 | 0 | 0 |
| `imp::compute_sfa_offsets_kernel(const int *, long *, unsigned char **, unsigned char *, int, int)` | 32 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `imp::elementwise_add_fp32_kernel(float *, const float *, long)` | 32 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gelu_fp32_vec4_kernel(const float *, float *, long)` | 32 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::penalties_hist_rows_kernel(const imp::PenaltyRowArgs *, int, unsigned int *, int)` | 32 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `void imp::rmsnorm_fp16_rowblock_kernel<(int)1>(const __half *, const __half *, __half *, int, float, float)` | 32 | 0 | 1156 | 0 | 1 | 1 | 0 | 0 |
| `imp::mxfp4_tile_mean_paged_kernel(const unsigned char *, const unsigned char *, const int *, int, int, int, int, int, float *)` | 31 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::quantize_act_fast_kernel(const __half *, int, int, signed char *, __half *, float *)` | 31 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::calibrate_fp8_scales_per_expert_kernel(const __half *, const int *, float *, int)` | 30 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_fused_permute_deterministic_kernel(const int *, int, int, int, int *, int *, int *, int *)` | 30 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::shared_expert_gate_scale_kernel(const __half *, const __half *, __half *, int, int)` | 30 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_inject_add_kernel(__half *, const __half *, const __half *, int, int, long)` | 30 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_write_kernel(__half *, const __half *, const __half *, const __half *, int, int, long)` | 30 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::sparse_select_topk_kernel(const float *, const int *, const int *, int *, int *, int, int, int, int, int, int, int, int)` | 30 | 0 | 1040 | 0 | 0 | 0 | 0 | 0 |
| `void imp::rmsnorm_fp16_rowblock_nvfp4_kernel<(int)1, (bool)0>(const __half *, const __half *, __half *, unsigned char *, unsigned char *, int, float, float, const __half *, __half *, __half *)` | 30 | 0 | 1156 | 0 | 1 | 1 | 0 | 0 |
| `void imp::rmsnorm_fp16_rowblock_nvfp4_kernel<(int)1, (bool)1>(const __half *, const __half *, __half *, unsigned char *, unsigned char *, int, float, float, const __half *, __half *, __half *)` | 30 | 0 | 1156 | 0 | 1 | 1 | 0 | 0 |
| `imp::dequantize_nvfp4_moe_kernel(const unsigned char *, const unsigned char *, const float *, __half *, int, long, long, unsigned long, unsigned long)` | 29 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::bzero_sfa_active_kernel(unsigned char *, const long *, int, long)` | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequantize_nvfp4_kernel(const unsigned char *, const unsigned char *, float, __half *, long, long)` | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::pixel_shuffle_kernel(const __half *, __half *, int, int)` | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::qsa_prep_queries_kernel(const __half *, const int *, const __half *, __half *, __half *, imp::QsaGeom)` | 28 | 0 | 1552 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::split_heads_kernel(const __half *, __half *, __half *, __half *, int, int, int)` | 28 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::mtp_argmax_kernel<__half>(const T1 *, int, int *)` | 28 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::qsa_pool_blocks_kernel(const __half *, const __half *, __half *, int, const int *, imp::QsaGeom)` | 27 | 0 | 1552 | 0 | 0 | 0 | 0 | 0 |
| `imp::absmax_kernel(const __half *, long, float *)` | 26 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::causal_softmax_fp32_to_fp16_kernel(const float *, __half *, int, int, int, bool, int, const __half *)` | 26 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `imp::compact_alpha_active_kernel(const float *, const int *, float *, int *, int)` | 26 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::group_rmsnorm_fp16_kernel(const __half *, const __half *, __half *, int, int, float)` | 26 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_kv_gather_fp16_kernel(__half *, const __half *, const int *, int, int, int, int, const int *)` | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::perplexity_nll_kernel(const float *, const int *, int, int, int, double *, int *)` | 26 | 0 | 5128 | 0 | 0 | 0 | 0 | 0 |
| `imp::q_position_temperature_kernel(__half *, const int *, int, int, float, int)` | 26 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp16_to_fp8_dscale_kernel(const __half *, unsigned char *, int, const float *)` | 26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::softmax_2d_kernel(__half *, int)` | 26 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_grouped_rmsnorm_kernel(const __half *, const __half *, __half *, int, int, float)` | 26 | 0 | 1056 | 0 | 0 | 0 | 0 | 0 |
| `imp::vision_layernorm_kernel(const __half *, const __half *, const __half *, __half *, int, float)` | 26 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void cub::_V_300403_SM_1200::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<cub::_V_300403_SM_1200::detail::radix_sort::policy_selector_from_types<float, int, unsigned int>, unsigned int>(T2 *)` | 26 | 0 | 2208 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_typical_p_kernel(float *, int, float, bool)` | 25 | 0 | 10292 | 0 | 0 | 0 | 0 | 0 |
| `imp::causal_softmax_inplace_kernel(__half *, int, int, int, bool, int, const __half *)` | 24 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q6k_to_fp8_kernel(const unsigned char *, unsigned char *, int)` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q6k_v2_kernel(const unsigned char *, __half *, int)` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::fa2_amax_fp16_kernel(const __half *, long, float *)` | 24 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_topw_pass2_kernel(const float *, const int *, int, int, int *, float *)` | 24 | 0 | 3104 | 0 | 0 | 0 | 0 | 0 |
| `imp::mxfp8_scale_cols_kernel(__half *, const float *, int, int)` | 24 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_kv_gather_int4_to_fp16_kernel(__half *, const unsigned char *, const __half *, const int *, int, int, int, int, const int *)` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_kv_gather_int8_to_fp16_kernel(__half *, const signed char *, const __half *, const int *, int, int, int, int, const int *)` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::layernorm_kernel(const __half *, const __half *, const __half *, __half *, int, float)` | 24 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::write_slot_scales_kernel(float *, imp::<unnamed>::SlotScaleBatch)` | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::extract_patches_kernel(const __half *, __half *, int, int, int, int, int, int)` | 23 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::qsa_gather_kv_kernel(const int4 *, const int4 *, const int *, int, int, const int *, const int *, int, int4 *, int4 *, int, int *)` | 23 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::build_attn_ptr_arrays_kernel(const void **, const void **, void **, const char *, long, const char *, long, char *, long, int, int)` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gemma4v_tail_norm_kernel(const __half *, const __half *, const __half *, __half *, int, float, float)` | 22 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::gpt_oss_glu_fp32_vec4_kernel(const float *, const float *, float *, long)` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::lora_gemv_a_kernel(const __half *, const __half *, float *, int, int)` | 22 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_add_expert_bias_sorted_kernel(__half *, const __half *, const int *, int, long, int)` | 22 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mxfp4_attn_softmax_kernel(__half *, int, int, int, float, float, bool)` | 22 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_kv_gather_mxfp4_kv_to_fp16_kernel(__half *, const unsigned char *, const unsigned char *, const int *, int, int, int, int, const int *)` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_kv_gather_nvfp4_to_fp16_kernel(__half *, const unsigned char *, const unsigned char *, const int *, int, int, int, int, const int *)` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::residual_kv_write_multi_indirect_kernel(const __half *, const __half *, __half *, __half *, long, const int *, const int *, int)` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::rowwise_topm_partial_kernel(const float *, int, int, float *, int *)` | 22 | 0 | 3136 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_repeat_kernel(const __half *, __half *, int, int, long)` | 22 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::scaled_residual_kernel(__half *, const __half *, const __half *, long, int)` | 22 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::vision_rmsnorm_kernel(const __half *, const __half *, __half *, int, float)` | 22 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::vision_rmsnorm_opt_kernel(const __half *, const __half *, __half *, int, float)` | 22 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_gather_kernel_impl<float>(const T1 *, const int *, T1 *, int, int)` | 22 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_gather_kernel_impl<__half>(const T1 *, const int *, T1 *, int, int)` | 22 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::broadcast_add_bias_fp16_kernel(__half *, const __half *, int, int)` | 21 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q5k_kernel(const unsigned char *, __half *, int, int)` | 21 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::convert_nvfp4_to_mxfp4_scales_kernel(const unsigned char *, unsigned char *, unsigned char *, float, int, int, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_iq3_s_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_iq4_nl_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_iq4_xs_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q2k_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q4_0_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q4_1_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q4k_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q5_0_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q5_1_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q8_0_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_q8k_kernel(const unsigned char *, __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::embedding_lookup_q6k_device_kernel(const unsigned char *, const int *, __half *, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::embedding_lookup_q6k_kernel(const unsigned char *, const int *, __half *, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::embedding_lookup_q8_0_kernel(const unsigned char *, const int *, __half *, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::mla_assemble_kv_kernel(const __half *, const __half *, __half *, __half *, int, int, int, int, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mla_latent_cache_write_kernel(const __half *, const __half *, __half *, const int *, int, int, int, int, int, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_add_expert_bias_indexed_kernel(__half *, const __half *, const int *, long, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_gather_rows_kernel(const __half *, const int *, __half *, int, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::paged_kv_gather_fp8_to_fp16_kernel(__half *, const __nv_fp8_e4m3 *, const int *, float, int, int, int, int, const int *)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_add_residual_kernel(const __half *, const __half *, const __half *, __half *, int, float, float)` | 20 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::rowwise_argmax_partial_kernel(const float *, int, float *, int *)` | 20 | 0 | 3072 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::add_bias_kernel(__half *, const __half *, long, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::cls_pos_kernel(__half *, const __half *, const __half *, long, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::encoder_add_type_kernel(__half *, const __half *, int, int)` | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::layernorm_residual_kernel(const __half *, const __half *, const void *, bool, const void *, bool, __half *, int, float)` | 20 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::merge_heads_kernel(const __half *, __half *, int, int, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::ple_gate_value_kernel(const __half *, __half *, const __half *, int, int, float)` | 20 | 0 | 1056 | 0 | 0 | 0 | 0 | 0 |
| `imp::vision_rope2d_neox_kernel(__half *, const int *, const int *, int, int, int, float)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_scatter_kernel_impl<__half>(const T1 *, const int *, const int *, const float *, float *, int, int)` | 20 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::add_bias_3way_kernel(__half *, const __half *, int, __half *, const __half *, int, __half *, const __half *, int, int)` | 19 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_fp32_in_fp32_out_kernel(const float *, const __half *, float *, int, float, float)` | 19 | 0 | 1168 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_fp32_kernel(const float *, const float *, float *, int, float, float)` | 19 | 0 | 1168 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_residual_fp32_kernel(float *, const float *, const float *, float *, int, float, float)` | 19 | 0 | 1168 | 0 | 0 | 0 | 0 | 0 |
| `imp::vhead_tiled_to_grouped_f32_kernel(const float *, float *, int, int, int, int)` | 19 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::add_rmsnorm_inplace_kernel(const __half *, const __half *, __half *, const __half *, int, float, float)` | 18 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::advance_residual_state_multi_kernel(int *, int *, const int *, int, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::decode_pipeline_advance_kernel(int, const char *, unsigned long, int *, int *, int *, int *, int, const int *, const int *, int *, int, const int *)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_awq4_kernel(__half *, const int *, const int *, const __half *, int, int, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::embedding_lookup_q8_0_device_kernel(const unsigned char *, const int *, __half *, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::geglu_fp32_vec4_kernel(const float *, const float *, float *, long)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::kv_chunk_append_fp16_kernel(__half *, const __half *, const int *, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::penalties_hist_kernel(float *, const int *, int, const int *, int, int, float, float, float, unsigned int *)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::rope_q_only_fp16_kernel(__half *, const int *, int, int, float, float, int, bool, const float *)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::swiglu_fp16_kernel(const __half *, const __half *, __half *, long)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::attn_gate_split_interleaved_kernel<__half>(const T1 *, T1 *, T1 *, int, int, int, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::attn_gate_split_interleaved_kernel<__nv_bfloat16>(const T1 *, T1 *, T1 *, int, int, int, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::embedding_lookup_scalar_kernel<float>(const T1 *, const int *, T1 *, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::embedding_lookup_scalar_kernel<__half>(const T1 *, const int *, T1 *, int)` | 18 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::moe_scatter_kernel_impl<float>(const T1 *, const int *, const int *, const float *, float *, int, int)` | 18 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::advance_residual_state_kernel(int *, int *, int, int)` | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::argmax_reduce_kernel(const float *, const int *, int, int *)` | 17 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::residual_add_rmsnorm_kernel(__half *, const __half *, const __half *, __half *, int, float, float)` | 17 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::rmsnorm_fp32_in_fp16_out_kernel(const float *, const __half *, __half *, int, float, float)` | 17 | 0 | 1168 | 0 | 0 | 0 | 0 | 0 |
| `imp::argmax_partial_rows_kernel(const imp::GreedyRowArgs *, int)` | 16 | 0 | 1088 | 0 | 1 | 0 | 0 | 0 |
| `imp::axial_pos_add_kernel(__half *, const __half *, const int *, const int *, int, int, int)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::calibrate_quantize_fp8_kernel(const __half *, unsigned char *, const float *, float *, int)` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequant_gptq4_kernel(__half *, const int *, const int *, const __half *, const int *, int, int, int, imp::gptq::ZeroFormat)` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_add_logit_bias_kernel(float *, const __half *, long, int)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_kv_append_kernel(const __half *, const __half *, __half *, __half *, int, int, int, int)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_kv_append_rows_kernel(const __half *, const __half *, __half *, __half *, const int *, const int *, long, int, int)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp16_to_fp8_per_expert_kernel(const __half *, unsigned char *, const int *, const float *, int)` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp8_rows_kernel(const __half *, unsigned char *, int, float *)` | 16 | 0 | 1156 | 0 | 0 | 0 | 0 | 0 |
| `imp::softmax_to_pairs_device_kernel(const float *, int, float, const float *, float *, int *)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::add_vision_embeddings_kernel(__half *, const int *, const __half *, int, int, int, int, int)` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::fill_rows_kernel(__half *, const __half *, long, int)` | 16 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_q8_1_kernel<imp::GeGLUAct>(const __half *, const __half *, imp::block_q8_1 *, float *, int, T1)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_q8_1_kernel<imp::IdentityAct>(const __half *, const __half *, imp::block_q8_1 *, float *, int, T1)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_q8_1_kernel<imp::SwiGLUAct>(const __half *, const __half *, imp::block_q8_1 *, float *, int, T1)` | 16 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::convert_scales_sfatom_kernel(const unsigned char *, unsigned char *, int, int, int)` | 15 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::convert_scales_sfatom_moe_kernel(const unsigned char *, unsigned char *, int, int, int, unsigned long, unsigned long)` | 15 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gdn_rmsnorm_gated_silu_fp32inout_kernel(float *, const float *, const __half *, const __half *, float, int, int, int)` | 15 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_imbalance_kernel(const int *, unsigned int *, int, int)` | 15 | 0 | 1032 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_gate_attn_out_kernel(__half *, const __half *, int, int, int)` | 15 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_gated_v_broadcast_kernel(const __half *, const __half *, __half *, int, int, int)` | 15 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::hadamard_warp_kernel<(int)16>(const __half *, __half *, int, int)` | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ban_logits_if_kernel(float *, const int *, int, int, const int *)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `ban_logits_kernel(float *, const int *, int, int)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::add_bias_kernel(__half *, const __half *, int, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::add_tensors_kernel(const __half *, const __half *, __half *, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_min_p_kernel(float *, int, float)` | 14 | 0 | 1060 | 0 | 0 | 0 | 0 | 0 |
| `imp::argmax_partial_kernel(const float *, int, float *, int *)` | 14 | 0 | 1088 | 0 | 0 | 0 | 0 | 0 |
| `imp::build_mxfp4_scales_kernel(const unsigned char *, unsigned char *, unsigned char *, int, int, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::elementwise_add_store_fp16_kernel(const __half *, const __half *, __half *, long)` | 14 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `imp::fp32_to_fp16_rowscale_kernel(const float *, __half *, int, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gdn_rmsnorm_gated_silu_kernel(__half *, const __half *, const __half *, float, int, int, int)` | 14 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `imp::geglu_fp16_kernel(const __half *, const __half *, __half *, long)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::geglu_fp32_kernel(const float *, const float *, float *, long)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gelu_fp16_kernel(const __half *, __half *, long)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gpt_oss_glu_fp16_kernel(const __half *, const __half *, __half *, long)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gpt_oss_glu_fp32_kernel(const float *, const float *, float *, long)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::hadamard_128_kernel(const __half *, __half *, int, int)` | 14 | 0 | 1536 | 0 | 0 | 0 | 0 | 0 |
| `imp::hadamard_64_kernel(const __half *, __half *, int, int)` | 14 | 0 | 1280 | 0 | 0 | 0 | 0 | 0 |
| `imp::mla_reorder_q_kernel(__half *, int, int, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_permute_tile_hist_kernel(const int *, int, int, int, int *)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mul_tensors_kernel(const __half *, const __half *, __half *, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mxfp8_sf_from_row_scales_kernel(const float *, unsigned char *, float *, int, int, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::quantize_fp16_to_fp8_scaled_kernel(const __half *, unsigned char *, int, float)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::replace_vision_embeddings_v2_kernel(__half *, const int *, const __half *, int, int, int, int, int)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::residual_kv_write_indirect_kernel(const __half *, const __half *, __half *, __half *, const int *, int, int)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::softmax_max_kernel(const float *, int, float *)` | 14 | 0 | 1056 | 0 | 0 | 0 | 0 | 0 |
| `imp::softmax_sum_device_max_kernel(const float *, int, float, const float *, float *)` | 14 | 0 | 1056 | 0 | 0 | 0 | 0 | 0 |
| `imp::softmax_sum_device_max_single_block_kernel(const float *, int, float, const float *, float *)` | 14 | 0 | 1056 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::encoder_l2_normalize_kernel(float *, int)` | 14 | 0 | 1152 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::mtp_hc_add_broadcast_kernel(__half *, const __half *, int, int)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::swiglu_packed_rows_kernel(const __half *, __half *, int, int)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::fused_act_quantize_q8_1_kernel<imp::ReLUSqrAct>(const __half *, const __half *, imp::block_q8_1 *, float *, int, T1)` | 14 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void imp::hadamard_warp_kernel<(int)32>(const __half *, __half *, int, int)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::smallm_splitk_reduce_kernel<(bool)0>(const float *, __half *, int, int)` | 14 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::argmax_kernel(const float *, int, int *)` | 13 | 0 | 1088 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_concat_kernel(const __half *, const __half *, __half *, int, int)` | 13 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::gelu_erf_kernel(__half *, long)` | 13 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::softmax_rows_kernel(__half *, int)` | 13 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::absmax_reduce_kernel(const __half *, float *, int)` | 12 | 0 | 2048 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_dry_sparse_kernel(float *, const int *, const float *, int)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::dequantize_fp8_to_fp16_scaled_kernel(const unsigned char *, __half *, int, float)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::elementwise_add_fp16_kernel(__half *, const __half *, long)` | 12 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `imp::fp32_plus_fp16_to_fp16(const float *, const __half *, __half *, long)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::gelu_fp32_kernel(const float *, float *, long)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::mla_compact_attn_output_kernel(const __half *, __half *, int, int, int)` | 12 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_alpha_mul_kernel(const float *, const float *, float *, int)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_apply_per_expert_scale_kernel(float *, const int *, const __half *, int)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_compute_M_per_kernel(const int *, int *, int)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::parallel_residual_merge_fp16_kernel(__half *, const __half *, const __half *, long)` | 12 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::pipeline_advance_kernel(int *, int *)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::scale_fp16_kernel(__half *, __half, long)` | 12 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::sigmoid_mul_fp16_kernel(const __half *, const __half *, __half *, long)` | 12 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::swiglu_fp32_kernel(const float *, const float *, float *, long)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_sub_kernel(const __half *, const __half *, __half *, long)` | 12 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::q8_planes_to_f16_kernel(const signed char *, const __half *, __half *, long)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::ssm_conv_tap_stash_kernel(__half *, const __half *, int, int, const int *, const int *)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `void imp::<unnamed>::smallm_splitk_reduce_kernel<(bool)1>(const float *, __half *, int, int)` | 12 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::apply_logit_bias_kernel(float *, const int *, const float *, int)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::constrain_mask_allow_kernel(float *, const unsigned short *, const unsigned char *, const unsigned short *, int, int, bool)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::device_copy_v4_kernel(const uint4 *, uint4 *, unsigned long)` | 10 | 0 | 1024 | 0 | 1 | 1 | 0 | 0 |
| `imp::fp16_to_fp32_kernel(const __half *, float *, long)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::fp32_to_fp16_kernel(const float *, __half *, long)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::gelu_tanh_kernel(__half *, int)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_add_kernel(__half *, const __half *, int)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::mtp_add_shared_kernel(__half *, const __half *, int)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::nvfp4_fp32_to_fp16_kernel(const float *, __half *, long)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::nvfp4_moe_native_fill_scales_kernel(float *, int)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::scale_fp16_by_fp16ptr_kernel(__half *, const __half *, long)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::scale_fp32_by_fp16ptr_kernel(float *, const __half *, long)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::gdn_factor_clear_kernel(float *, long, int, const int *, int, int)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::gelu_tanh_kernel(__half *, long)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::grammar_mask_kernel(float *, const unsigned char *, int, int)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::mtp_add_rows_kernel(__half *, const __half *, int)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::penalty_hist_append_kernel(const char *, unsigned long, imp::PenaltyAppendArgs, int *)` | 10 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| `imp::<unnamed>::regex_mask_kernel(float *, const unsigned char *, int, int)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::residual_add_kernel(__half *, const __half *, long)` | 10 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::write_slot_idx_kernel(int *, imp::<unnamed>::SlotIdxBatch)` | 10 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::force_single_token_kernel(float *, int, int)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::init_cub_max_sum_kernel(float *)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::logit_softcap_fp32_kernel(float *, float, float, long)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::moe_expert_hist_kernel(const int *, unsigned int *, int, int, int)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::relu_sqr_inplace_fp16_kernel(__half *, long)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::sanitize_fp16_kernel(__half *, long)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::scale_fp32_kernel(float *, float, long)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::silu_inplace_fp16_kernel(__half *, long)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::softcap_fp16_kernel(__half *, long, float)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::softcap_fp32_kernel(float *, long, float)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_inject_weights_kernel(__half *, float, long)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::hc_silu_div_kernel(__half *, float, long)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::norm_fold_zero_kernel(unsigned long long *, int)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::qsa_init_scratch_bt_kernel(int *, int)` | 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `imp::<unnamed>::sparse_prefill_set_len_kernel(int *, int)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `imp::zero_float_kernel(float *, int)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::reset_stats_kernel<(int)128>(unsigned int *, const int *)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::reset_stats_kernel<(int)256>(unsigned int *, const int *)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void apa::reset_stats_kernel<(int)64>(unsigned int *, const int *)` | 8 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
| `void cub::_V_300403_SM_1200::detail::EmptyKernel<void>()` | 4 | 0 | 1024 | 0 | 0 | 0 | 0 | 0 |
