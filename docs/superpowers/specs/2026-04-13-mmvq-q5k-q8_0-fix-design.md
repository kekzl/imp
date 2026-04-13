# Design: Fix Q5_K + Q8_0 MMVQ Kernels for Gemma-4 Decode Quality

## Context

Gemma-4 26B-A4B produces correct first answers ("Paris", "Berlin", "I am Gemma") but decode degenerates after ~15 tokens. The Q4_K MMVQ kernel is verified correct and improves decode quality. Q5_K and Q8_0 MMVQ kernels have bugs that prevent them from being activated. Fixing both will enable all weight GEMMs (attention Q/K/V/O + shared expert + MoE experts) to use numerically-correct quantized matmul, matching llama.cpp's accumulation behavior.

## Bugs

### Q8_0: Struct pointer arithmetic with 34-byte blocks

`ggml_block_q8_0` is 34 bytes (half d + int8_t qs[32]). Pointer arithmetic `(ggml_block_q8_0*)W + kbx` produces correct byte offsets (kbx * 34) but the `qs` field at offset 2 is not 4-byte aligned for odd block indices. The current `load_int_unaligned()` workaround addresses the read but the struct pointer itself may still cause issues.

**Fix**: Replace struct pointer arithmetic with raw byte addressing. Read `d` and `qs` via explicit byte offsets (`bp + 0` for d, `bp + 2` for qs). Use `get_int_b2` (2-byte aligned reads) like llama.cpp does for Q8_0.

### Q5_K: Potentially misaligned qh access

The `qh` array pointer is cast to `int*` for 4-byte reads. While `qh` is at offset 16 in the Q5_K block (4-byte aligned when block starts aligned), the cast lacks validation. Additionally, the interaction between bq8_offset-based qh shifting and the int* cast may read wrong data.

**Fix**: Use `get_int_b4()` for qh reads (like llama.cpp). Verify alignment assumptions hold for all block offsets.

## Approach: Unit-Test First

### Step 1: Write GPU test (`tests/test_mmvq.cu`)

For each quant type (Q4_K, Q5_K, Q5_1, Q8_0):
1. Load real weight tensor from Gemma-4 GGUF (Layer 0)
2. Create known FP16 input (embedding norm output)
3. Compute reference via dp4a GEMV (verified correct)
4. Compute test via MMVQ kernel
5. Assert: max element-wise difference < 0.2% of dp4a magnitude

Test uses GTest framework (already in project). Weight data comes from loaded model (pattern: `test_e2e_models.cpp`).

### Step 2: Fix Q8_0 MMVQ

In `ggml_mmvq.cu`, replace `vec_dot_q8_0_q8_1`:
- Remove struct pointer arithmetic: use `const uint8_t* bp = (const uint8_t*)vbq + kbx * 34`
- Read d: `half d_h; memcpy(&d_h, bp, 2); float d = __half2float(d_h);`
- Read qs: use byte offset `bp + 2` with 2-byte aligned reads

### Step 3: Fix Q5_K MMVQ

In `ggml_mmvq.cu`, secure `qh` access in `vec_dot_q5_K_q8_1`:
- Replace raw `int*` cast with `get_int_b4()` helper
- Verify: Q5_K block starts aligned → qh at offset 16 is always 4-byte aligned

### Step 4: Enable all MMVQ types and test Gemma-4

- Add Q5_K, Q5_1, Q8_0 back to `use_mmvq` condition in `gemm_dispatch`
- Test: "What is the capital of France?" → coherent answer > 30 tokens
- Test: "Explain quantum computing" → thematically relevant > 30 tokens
- Test: Stop token `<turn|>` (106) generated correctly

## Success Criteria

1. All 4 MMVQ kernels pass GPU unit test (match dp4a < 0.2%)
2. Gemma-4 produces coherent multi-sentence answers with `--chat-template gemma`
3. No crashes, no NaN, no misaligned address errors
4. Performance: tg > 50 tok/s maintained

## Files

| File | Change |
|------|--------|
| `tests/test_mmvq.cu` | New: GPU unit test for all MMVQ kernels |
| `src/compute/ggml_mmvq.cu` | Fix Q8_0 + Q5_K vec_dot implementations |
| `src/graph/executor_kernels.cu` | Enable Q5_K/Q5_1/Q8_0 in use_mmvq condition |
| `CMakeLists.txt` | Add test_mmvq.cu to test sources |
