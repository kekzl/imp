# SM120 Real Performance Plan (RTX 5090)

> **Supersedes** `docs/blackwell-tc-unlock-plan.md` (basierte auf falscher tcgen05-Annahme — see `memory/sm120_real_perf_levers_2026_05_04.md`).

**Goal:** Concrete SM120-realistic performance unlocks for imp. NO tcgen05/TMEM (Hardware fehlt auf Consumer-Blackwell). Alle Hebel basieren auf was die SM120 PTX/SASS tatsächlich kann.

**Architecture:** Five-lever rollout, prioritized by ROI. Hebel #1 (SSM dispatch fix) löst direkt den Mamba2-Multi-Chunk-Hang in 2-4h Aufwand. Hebel #2-5 sind progressive throughput-uplifts.

**Tech Stack:** CUDA 13.2.1, CUTLASS 4.4.2, sm_120a (Consumer-Blackwell PTX), `mma.sync.aligned.kind::mxf4nvf4.block_scale` als peak NVFP4 path.

---

## Cross-Refs

- `memory/sm120_real_perf_levers_2026_05_04.md` — authoritative Hardware-Capability-Reference
- `memory/sass_audit_120a_no_tcgen05_2026_05_04.md` — SASS-Audit + Korrekturen
- `memory/cuda_arch_120a_2026_05_04.md` — Build-Switch (notwendig)
- `memory/nemotron_h_moe_imp_broken_2026_05_04.md` — der konkrete Use-Case der Hebel #1 motiviert

---

## Levers (Impact-priorisiert)

| # | Hebel | Erwarteter Gewinn | Aufwand | Status |
|---|---|---|---|---|
| 1 | **SSM-Layer in cutlass_nvfp4_cache aufnehmen** → fixt Mamba2-Multi-Chunk-Hang | Multi-Chunk-Prefill funktioniert; Decode +50-100% | 2-4h | this plan |
| 2 | **NVFP4 KV-Cache mit HW absmax intrinsics** | +30% memory throughput KV-bound paths | 2-3 Tage | TBD |
| 3 | **CLC-Persistent-Kernel für continuous batching** | +10-20% bei multi-user | 1-2 Tage | TBD |
| 4 | **CUTLASS Tile-Tuning für 228 KiB SMEM** | +20-40% bei großen GEMMs (langer prompt) | 1-2 Tage benchmark | TBD |
| 5 | **Online-Softmax SFU-exp2 micro-opt** | +5-15% softmax-bound paths | 4-8h | TBD |

---

## Lever 1: SSM-Layer in cutlass_nvfp4_cache aufnehmen

### Why first

`executor_kernels.cu:1820-1870` zeigt: NVFP4-quantisierte Weights gehen via `cutlass_nvfp4_cache`-lookup → `gemm_nvfp4_cutlass_sm120()` (fast Sm120 cooperative kernel mit block-scaled MMA). Wenn der lookup miss → slow-fallback `gemm_nvfp4()` (dequant entire weight to FP16 + cuBLAS GEMM).

`executor_pre_dequant.cu:735-753` excludet `ssm_in`/`ssm_out` von der NVFP4-Cache-Population. Das war für die ALTE NVFP4-Cache-Architektur (dequant cache) sinnvoll. Aber die **`cutlass_nvfp4_cache`** ist eine separate Cache (für CUTLASS-Format-Weights). Beide Codepfade behandeln NVFP4-Weights als FP4-quantisiert — der Math-Pfad ist äquivalent. Die Exclusion ist überkonservativ.

Resultat: Mamba2-Layer in_proj/out_proj GEMMs (Shapes [10304, 2688] und [4096, 2688]) fallen auf slow-fallback × 17 Layer × Multi-Chunk-Prefill = 5min timeout. Fix ist Mini-Patch.

### Architecture

Trenne die SSM-exclusion in zwei separate Concerns:
- `nvfp4_exclude_ptrs` → controls dequant cache + tier promotion (keep exclusion if accuracy-needed)
- `cutlass_nvfp4_cache` → CUTLASS-Format-cache, MMA-path-only, no accuracy difference

Add SSM tensors zu `cutlass_nvfp4_cache` separately, **even if** they remain in `nvfp4_exclude_ptrs`.

### Files

- **Modify:** `src/graph/executor_pre_dequant.cu` — populate cutlass_nvfp4_cache for SSM tensors even when excluded from nvfp4_exclude_ptrs
- **No new tests:** correctness via `scripts/validate_safetensors.py --smoke --model Nemotron-3-Nano-30B-A3B-NVFP4`, success criterion: 600-token prompt completes in <30s (was 300s timeout)

### Task 1.1: Locate cutlass_nvfp4_cache population code

- [ ] **Step 1: Find where cutlass_nvfp4_cache is populated**

```bash
grep -nE "cutlass_nvfp4_cache.*\[|cutlass_nvfp4_cache.*emplace|cutlass_nvfp4_cache.*insert|build_cutlass_nvfp4|populate.*cutlass" /home/kekz/github.com/kekzl/imp/src/graph/executor_pre_dequant.cu
```

Expected: find the loop that populates the cache from model weights. Note line numbers.

- [ ] **Step 2: Map exclusion logic**

```bash
grep -n "nvfp4_exclude_ptrs" /home/kekz/github.com/kekzl/imp/src/graph/executor_pre_dequant.cu
```

Verify: `nvfp4_exclude_ptrs.insert(L.ssm_in.data)` at line ~745, `nvfp4_exclude_ptrs.insert(L.ssm_out.data)` at line ~750, AND check whether the population loop tests against this set.

### Task 1.2: Add SSM tensors to cutlass_nvfp4_cache

The exact change depends on Step 1.1 findings. Two possible patterns:

**Pattern A (likely):** the population loop iterates all weights and skips if `nvfp4_exclude_ptrs.count(ptr)`. Add a separate populate-loop that includes excluded SSM tensors specifically into cutlass_nvfp4_cache only.

**Pattern B:** the cache is populated tier-by-tier with separate exclusion sets. Add SSM weights with explicit cutlass_nvfp4 tier assignment.

- [ ] **Step 1: Read the populate-loop structure** (line numbers from 1.1)

- [ ] **Step 2: Modify to add SSM tensors to cutlass_nvfp4_cache**

Concrete edit (will adjust based on actual code structure — placeholder is the conceptual change):

```cpp
// After the regular nvfp4_exclude_ptrs population, add SSM tensors to
// cutlass_nvfp4_cache anyway (CUTLASS path is math-equivalent to slow-fallback;
// the original exclusion was for the dequant cache, not the CUTLASS cache).
{
    int n_ssm_cutlass = 0;
    for (int i = 0; i < cfg.n_layers; i++) {
        const auto& L = model_->layer(i);
        if (L.ssm_in.data && L.ssm_in.qtype == QType::NVFP4) {
            // populate cutlass_nvfp4_cache from L.ssm_in
            // ... same logic as regular weight population ...
            n_ssm_cutlass++;
        }
        if (L.ssm_out.data && L.ssm_out.qtype == QType::NVFP4) {
            // populate cutlass_nvfp4_cache from L.ssm_out
            n_ssm_cutlass++;
        }
    }
    if (n_ssm_cutlass > 0)
        IMP_LOG_INFO("CUTLASS NVFP4 cache: included %d SSM projections (separate from nvfp4_exclude_ptrs)", n_ssm_cutlass);
}
```

- [ ] **Step 3: Build**

```bash
cd /home/kekz/github.com/kekzl/imp
docker build -t imp:lever1 . 2>&1 | tail -10
```

- [ ] **Step 4: Boot Nemotron + check log**

```bash
docker stop imp-nemo-test 2>/dev/null
docker run -d --rm --name imp-nemo-lever1 \
  --network autoflow_default --network-alias llm \
  -e IMP_MODEL=/models/Nemotron-3-Nano-30B-A3B-NVFP4 \
  -e IMP_HOST=0.0.0.0 -e IMP_PORT=8080 -e IMP_MODELS_DIR=/models \
  -v /home/kekz/models:/models --gpus all -p 8080:8080 imp:lever1
until curl -sf http://localhost:8080/health 2>/dev/null | grep -q '"model_loaded":true'; do sleep 5; done
docker logs imp-nemo-lever1 2>&1 | grep -E "CUTLASS NVFP4|GDN/SSM" | head -5
```

Expected: log shows `CUTLASS NVFP4 cache: included N SSM projections` with N > 0.

- [ ] **Step 5: The 600-token Nemotron test (the actual unlock-test)**

```bash
PROMPT=$(printf 'Tell me the capital of these countries one by one in a list format. %.0s' {1..60})
START=$(date +%s)
RESULT=$(curl -s -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Nemotron-3-Nano-30B-A3B-NVFP4\",\"messages\":[{\"role\":\"user\",\"content\":$(jq -Rs <<<"$PROMPT")}],\"temperature\":0.0,\"seed\":42,\"max_tokens\":80}" \
  --max-time 60)
END=$(date +%s)
echo "elapsed: $((END-START))s"
echo "$RESULT" | jq -r '.choices[0] | "finish=\(.finish_reason) clen=\(.message.content // "" | length) HEAD=\((.message.content // "")[0:200])"'
docker logs imp-nemo-lever1 2>&1 | grep -E "slow dequant-to-FP16|completion 1/1" | tail -5
```

**Acceptance**: elapsed < 30s (was 300s timeout), finish_reason=stop or length, content non-empty + coherent. AND `slow dequant-to-FP16` warning fires 0× during this request (was firing per Mamba2 layer per chunk).

- [ ] **Step 6: Smoke validation on Nemotron**

```bash
cd /home/kekz/github.com/kekzl/imp
docker stop imp-nemo-lever1
IMP_DOCKER_IMG=imp:lever1 IMP_MODELS_DIR=/home/kekz/models \
  python3 scripts/validate_safetensors.py --smoke --model Nemotron-3-Nano-30B-A3B-NVFP4
cat MODEL_VALIDATION_SUMMARY.csv
```

**Acceptance**: phase 4 battery improves vs baseline (was 1/4, target ≥3/4).

- [ ] **Step 7: Smoke validation on Qwen3.6 (no-regression check)**

```bash
IMP_DOCKER_IMG=imp:lever1 IMP_MODELS_DIR=/home/kekz/models \
  python3 scripts/validate_safetensors.py --smoke --model Qwen3.6-35B-A3B-NVFP4
```

**Acceptance**: decode_tok_per_s ≥ baseline (Qwen3.6 GDN-Layer also gehen jetzt durch CUTLASS — should be **faster**, not slower).

- [ ] **Step 8: Promote build to imp:latest**

```bash
docker tag imp:lever1 imp:latest
```

- [ ] **Step 9: Restart autoflow**

```bash
docker start autoflow-llm-1
cd /home/kekz/github.com/kekzl/autoflow && docker compose start worker
```

- [ ] **Step 10: Final commit**

```bash
cd /home/kekz/github.com/kekzl/imp
git add src/graph/executor_pre_dequant.cu
git commit -m "fix(nvfp4): populate cutlass_nvfp4_cache for SSM tensors (Mamba2 multi-chunk fix)"
git tag lever1-ssm-cutlass-cache
```

### Lever 1 Acceptance Criteria

- [ ] Boot logs: `CUTLASS NVFP4 cache: included N SSM projections` with N > 0
- [ ] Nemotron-H 600-token prompt: completes in < 30s (was 300s timeout)
- [ ] `slow dequant-to-FP16 fallback` warning: 0× per Nemotron-H prefill (was multiple times per layer per chunk)
- [ ] Nemotron-H phase 4 battery: ≥3/4 (was 1/4)
- [ ] Qwen3.6-NVFP4 decode: same or better than baseline (no regression on GDN models)

---

## Lever 2: NVFP4 KV-Cache mit HW absmax intrinsics (Sketch)

After Lever 1 fixes the GEMM dispatch path, KV-cache quantization becomes the next bandwidth bottleneck. SM120 has `F2FP.SATFINITE.E2M1.F32.PACK_AB_MERGE_C` for hardware FP4 saturation — KV-write path can use this directly. Currently imp uses FP8 KV (per memory). NVFP4 KV would halve KV-bandwidth pressure.

Files: `src/memory/kv_cache.cu`, KV-write hot-path in attention. Target: 30% KV-bandwidth reduction → measurable on long-context decode.

## Lever 3: CLC-Persistent-Kernel (Sketch)

Cluster Launch Control persistent kernel pattern (already 167× CCTL in current binary, partial use). Full persistent grid + work-stealing dispatcher for continuous-batching server-mode. Files: `src/runtime/engine.cpp` (continuous-batching loop), CUTLASS `KernelTmaWarpSpecializedCooperativePersistent` schedule.

## Lever 4: CUTLASS Tile-Tuning für 228 KiB SMEM (Sketch)

SM120 has 228 KiB SMEM/CTA. Current CUTLASS tiles (`<128,128,128>` per example 79a) might not maximize SMEM. Audit + bench larger tiles: `<256,128,128>`, `<128,256,128>`, `<128,128,256>`. Files: `src/compute/gemm_cutlass_*.cu`. Profile-driven, 1-2 days benchmarking.

## Lever 5: Online-Softmax SFU-exp2 micro-opt (Sketch)

SM120 SFU is improved over Ada. The 2269× `MUFU.EX2` we counted are softmax-related. Switching to vectorized SFU dispatch could give +5-15% softmax-bound paths.

---

## Open Risks

1. **Lever 1 — accuracy regression for SSM with CUTLASS path?** Original exclusion comment said "4-bit degrades quality on 9B+ models". Need to verify that going through CUTLASS doesn't somehow worsen this — but the math is identical (same FP4 weights, same Block-Scaling), so theoretical risk is zero. Validate via Phase 5 determinism + battery checks in step 6/7.

2. **Lever 1 — performance regression on Qwen3.6?** Currently Qwen3.6 GDN tensors also excluded (linear_attn maps to gdn_gate/ssm_in/ssm_out paths). Lever 1 will affect them too. **Should be a positive**, but test in step 7.

3. **CUTLASS cache memory overhead.** Adding SSM tensors to cutlass_nvfp4_cache adds metadata storage. For Nemotron-H: 17 NVFP4-Mamba2 layers × 2 projections × ~few KB metadata = trivial.
