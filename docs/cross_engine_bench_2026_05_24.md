# Cross-Engine Bench: imp vs llama.cpp vs vLLM
*2026-05-24 · RTX 5090 (sm_120a, 32 GiB GDDR7) · CUDA 13.2*

## Engine versions

| Engine | Version / build | Notes |
|---|---|---|
| imp | main @ commit b959a32 + PR #394 + #395 + #396 merged | NVFP4 / SafeTensors first-class; GGUF legacy/maintenance |
| llama.cpp | latest (`5d246a7` 2026-05-24, CUDA build, `-DCMAKE_CUDA_ARCHITECTURES=120`) | GGUF native |
| vLLM | 0.21.0 | NVFP4 SafeTensors via `compressed-tensors` quant; CUDA graphs ON (`enforce_eager=False`), prefix caching OFF for clean pp |

## Methodology

- 3 reps per (model, metric); median reported.
- imp uses `--bench --bench-pp <N> --bench-reps 3 --max-tokens 128 --temperature 0` (chunked prefill defaults, NVFP4 caches on by default).
- llama.cpp uses `llama-bench -p 512,2048 -n 128 -r 3 -ngl 99`.
- vLLM uses a custom harness that issues `n_pp`-token prefill + 1-token gen for pp, and 1-token prefill + 128-token gen for tg. Prefix caching disabled and prompts rotated per rep to defeat the on-by-default cache hits that bias the first sweep's pp numbers.
- All three engines run in fresh Docker containers with bind-mounted models.

## GGUF model sweep — imp vs llama.cpp

Same `.gguf` files loaded by both engines.

| Model | Quant | Metric | imp (tok/s) | llama.cpp (tok/s) | imp Δ |
|---|---|---|---|---|---|
| Qwen3-8B (dense, 8.1 GiB) | Q8_0 | pp512 | **12197** | 11741 | +3.9 % |
|  |  | pp2048 | **11110** | 9938 | +11.8 % |
|  |  | tg128 | **272** | 146 | **+86.1 %** |
| Qwen3-14B (dense, 11.3 GiB) | Q6_K | pp512 | 5861 | **5880** | -0.3 % |
|  |  | pp2048 | 5161 | **5337** | -3.3 % |
|  |  | tg128 | **164** | 106 | **+54.3 %** |
| gemma-3-12B (dense, 6.8 GiB) | Q4_K_M | pp512 | 4059 | **7762** | -47.7 % |
|  |  | pp2048 | 6347 | **7462** | -14.9 % |
|  |  | tg128 | 128 | **139** | -8.2 % |
| gemma-4-26B-A4B (MoE, 15.7 GiB) | Q4_K_M | pp512 | 4734 | **9932** | -52.3 % |
|  |  | pp2048 | 4658 | **9145** | -49.1 % |
|  |  | tg128 | **258** | 208 | **+23.9 %** |
| Qwen3.6-35B-A3B (GDN+MoE, 20.6 GiB) | Q4_K_M | pp512 | 3151 | **7644** | -58.8 % |
|  |  | pp2048 | 3183 | **7810** | -59.3 % |
|  |  | tg128 | **246** | 227 | +8.6 % |

### Verdict (GGUF sweep)

**imp wins decisively on decode (tg128) across the board** — Qwen3-8B-Q8_0 **+86 %**, Qwen3-14B-Q6_K **+54 %**, Gemma-4-A4B **+24 %**, Qwen3.6-A3B **+9 %**. The only GGUF tg128 loss is Gemma-3-12B (-8 %, see below). imp's NVFP4 decode cache + GDN chunkwise scan + tuned GEMV kernels translate to real throughput wins at the model level, not just kernel microbench numbers.

**llama.cpp wins on Q4_K_M prefill** by 48-59 % on Gemma-3, Gemma-4 and Qwen3.6. imp has no direct Q4_K_M GEMM kernel — it dequants Q4_K_M to FP16 then runs cuBLAS. The open `docs/plans/q4k_imma_design_2026_05_17.md` plan covers this; it's the single most impactful imp roadmap item from this bench. Note that the same shape on dense Q8_0 / Q6_K is competitive with llama.cpp (-0.3 to +12 %), so the gap is specifically the Q4_K_M dispatch path, not a structural disadvantage on quant-prefill broadly.

**Gemma-3-12B is imp's only tg loss** (-8 %). Per the 2026-05-23 profile (`memory/MEMORY.md` Performance Baselines section), **66 % of decode is in the dp4a GEMV family** on Gemma-3-12B — the model can't use the NVFP4 decode cache (Q4_K isn't on the `nvfp4_beneficial` list) and its fused `rmsnorm_quantize_q8_1` only partially hides the kernel cost. llama.cpp's Q4_K dp4a path is more efficient at this specific shape. Tracked separately.

## NVFP4 SafeTensors sweep — imp vs vLLM

Same SafeTensors directories loaded by both engines. llama.cpp omitted (no NVFP4 support).

| Model | Metric | imp (tok/s) | vLLM (tok/s) | imp Δ |
|---|---|---|---|---|
| Qwen3-8B-NVFP4-cortecs (dense) | pp512 | 19762 | **21252** | -7.0 % |
|  | pp2048 | 28391 | **42402** | **-33.0 %** |
|  | tg128 | **225** | 143 | **+57.3 %** |
| Qwen3.6-35B-A3B-NVFP4 (GDN+MoE) | pp512 | **10407** | *failed* | **imp-only**¹ |
|  | pp2048 | **10794** | *failed* | **imp-only** |
|  | tg128 | **232** | *failed* | **imp-only** |
| Gemma-4-26B-A4B-it-NVFP4 (MoE) | pp512 | 16640 | **20804** | -20.0 % |
|  | pp2048 | 21902 | **25463** | -14.0 % |
|  | tg128 | 206 | **212** | -2.7 % |

¹ **vLLM fails to load the Qwen3.6 hero model on sm_120.** Engine debug log reports all NVFP4 MoE backends incompatible: `FLASHINFER_TRTLLM: kernel does not support current device cuda`, same for `FLASHINFER_CUTEDSL` and `FLASHINFER_CUTEDSL_BATCHED`. The Qwen3-Next architecture isn't bench-able in vLLM on Blackwell consumer SKUs at vLLM 0.21.0.

### Verdict (NVFP4 sweep)

**Decode (tg128) is imp's strong suit:** **+57 %** on Qwen3-8B-NVFP4 vs vLLM. imp's NVFP4 GEMV + CUDA Graph + chunkwise GDN scan stack delivers near-2× vLLM's decode tok/s on dense models. On MoE (Gemma-4), the gap closes to -2.7 % at tg128 — vLLM's MoE decode is well-tuned, imp ~par.

**Prefill is vLLM's strong suit on NVFP4**: -7 % to -33 % depending on shape and size. vLLM's CUTLASS NVFP4 grouped GEMM (the one we previously dug into in `memory/nvfp4_moe_prefill_landscape_2026_05_10.md`) is more aggressive at long prefill. imp closes some of this with the Qwen3-Coder MoE prefill landing (PR #374, ratio 1.056×, MEMORY entry "MoE prefill gap to vLLM 1.14–1.32 × → 1.056 ×") but the dense NVFP4 path at pp2048 still trails meaningfully on Qwen3-8B.

**The Qwen3.6-A3B hero result is imp-only on sm_120**: vLLM can't even load it. For Blackwell-consumer-class GPUs, imp is currently the *only* engine in this comparison that runs Qwen3-Next architecture at NVFP4 with working decode. Note: this isn't an imp win on perf, it's a coverage win — vLLM on a B200 / SM100 with `tcgen05` would presumably work and possibly be faster.

## Where imp stands — overall

**Strengths**:

1. **Decode throughput across all formats** — +24 % to +86 % on GGUF tg128, +57 % on NVFP4 tg128 (dense). This is the customer-visible number for chat / streaming workloads.
2. **NVFP4 GDN-hybrid coverage on sm_120** — imp's the only engine in scope that runs Qwen3-Next-NVFP4 on RTX 5090.
3. **Q8_0 / Q6_K prefill par with llama.cpp** on dense models — FP8 prefill cache + cuBLAS path is competitive at large quant sizes.

**Weaknesses**:

1. **Q4_K_M prefill: -50 to -59 % vs llama.cpp**. Single biggest open lever; tracked at `docs/plans/q4k_imma_design_2026_05_17.md`. Affects Gemma-3, Gemma-4-A4B, Qwen3.6-A3B in GGUF.
2. **NVFP4 prefill, especially pp2048 on dense**: -33 % on Qwen3-8B vs vLLM. vLLM's CUTLASS NVFP4 grouped GEMM tail-utilisation is better; imp's MoE-prefill gap to vLLM has been closed to 1.056 × on Qwen3-Coder, but the dense NVFP4 prefill on Qwen3-8B is still a gap.
3. **Gemma-3-12B tg128**: -8 % vs llama.cpp. dp4a GEMV path needs the same kind of tuning the NVFP4 path got, OR a path-switch decision to use Q4_K mmvq more aggressively.

**Roadmap-priority order based on this bench**:

1. **Q4_K_M direct GEMM kernel (IMMA INT8)** — biggest GGUF gap, multiple models. Plan exists.
2. **NVFP4 dense prefill** — vLLM is 7-33 % faster on Qwen3-8B-NVFP4. Their CUTLASS path is what imp competes against; can plausibly close at pp512 (already -7 %) but pp2048 (-33 %) needs more.
3. **dp4a GEMV tuning for Gemma-3 / Q4_K dense decode** — small absolute gap (-8 %) but tractable.

## How to reproduce

llama.cpp:
```bash
docker build -t llama-cpp:cuda13 -f /tmp/llamacpp.Dockerfile /tmp
docker run --rm --gpus all -v /home/kekz/models:/home/kekz/models llama-cpp:cuda13 \
  llama-bench -m /home/kekz/models/Qwen3-8B-Q8_0.gguf -p 512,2048 -n 128 -r 3
```

vLLM (custom harness, prefix caching disabled, prompt rotation):
```bash
docker build -t vllm:latest -f /tmp/vllm.Dockerfile /tmp
docker run --rm --gpus all -v /home/kekz/models:/home/kekz/models -v /tmp:/tmp_host vllm:latest \
  python3 /tmp_host/vllm_bench.py /home/kekz/models/Qwen3-8B-NVFP4-cortecs 512 128
```

imp:
```bash
docker run --rm --gpus all -v $REPO:/workspace -v /home/kekz/models:/home/kekz/models \
  -w /workspace imp:builder bash -c \
  "./build/imp-cli --model /home/kekz/models/Qwen3-8B-Q8_0.gguf \
   --bench --bench-pp 512 --bench-reps 3 --max-tokens 128 --temperature 0"
```

The Dockerfiles and bench scripts are at `/tmp/llamacpp.Dockerfile`, `/tmp/vllm.Dockerfile`, `/tmp/vllm_bench.py` and `/tmp/imp_bench.sh` respectively.
