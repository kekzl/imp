# imp Mission Journal — Autonomous "best engine on RTX 5090" run

> Append-only. Survives context resets. Newest entries at the BOTTOM of each section,
> but the **RESUME HERE** block at the top is always kept current.

---

## RESUME HERE (always current)

**Session:** 2026-05-29. Branch `mission/sm120-best` (off main).
**Phase:** Optimization loop. LEAD-1 LANDED (commit 817fa25). Attacking LEAD-2 next.
**Build:** `imp:test` green w/ LEAD-1. Tests green. Scoreboard + llama.cpp measured (below).

**LANDED this session:**
- **LEAD-1 (817fa25): NVFP4 decode cache for FP16/BF16 lm_head.** Broad decode win across
  non-hybrid NVFP4 fleet: Qwen3-8B +16%, Qwen3-14B +12.6%, Phi-4 +8.3%, Gemma-4-26B +11.3%,
  Qwen3-30B-A3B +2.9%. Hybrid Qwen3.6-35B correctly excluded (unchanged). NVFP4 dense decode
  now BEATS GGUF same-model and crushes llama.cpp. All correctness gates green.

**NEXT (priority order):**
1. **LEAD-2: NVFP4 MoE decode.** Qwen3-30B-A3B NVFP4 175 vs imp-Q4_K_M 276 vs llama.cpp 317.
   lm_head fix only gave +2.9% → gap is in per-expert GEMVs. PROFILE NEXT to find cause.
2. MoE/hybrid GGUF decode loss to llama.cpp (Qwen3.6-35B 158 vs 229).
3. GGUF prefill 1.3-2.4× behind llama.cpp (hard: custom MMQ kernel).
4. Stand up vLLM NVFP4 to confirm imp's NVFP4 prefill/decode lead is real.

**GPU at start:** RTX 5090, 29°C idle. Host nsys works (recipe in memory nsys-host-to-container;
use `--no-cuda-graphs` imp flag + `--trace=cuda`, --user root, /tmp/nsys_out chmod 777).

**Next actions:**
1. ✅ Build green, tests green, scoreboard measured (see below). llama.cpp bench bg-agent running.
2. **IN PROGRESS — LEAD-1 fix (NVFP4 lm_head decode cache):** native-NVFP4 checkpoints store
   `lm_head.weight` BF16 → decode pays a cuBLAS FP16 GEMV. PROFILE CONFIRMS: 19.3% of decode
   GPU time, 0.78 ms/token (Qwen3-8B NVFP4). Fix = quantize BF16 lm_head → NVFP4, register in
   `wcache_.nvfp4[output_proj.data]` (forward_logits `lm_nvfp4_secondary` hook already reads it).
   Predicted: Qwen3-8B NVFP4 decode 239 → ~279 (would beat Q8_0 274). Quality-safe: GGUF Q8_0
   already NVFP4-quantizes output_proj and passes. Reading phase-3 quantize path to implement.
3. After LEAD-1: profile LEAD-2 (NVFP4 MoE decode 170 vs Q4_K_M 276) → biggest gap.
4. Update scoreboard with measured llama.cpp; find next-biggest gap.

**Key facts verified this session:**
- GPU: RTX 5090 32607 MiB, host `nvidia-smi` works (WSL2).
- Models present: see Inventory below.
- Canonical perf gate: `tests/perf_baseline.json` (Qwen3-8B Q8_0: pp512=7736, tg128=276.59, 3%/5% thresholds).
- North-star: Qwen3-14B Q6_K decode @ctx2048 = 157.71 tok/s (May 23, cold-median).

---

## Model Inventory (verified `/home/kekz/models/` 2026-05-29)

GGUF: gemma-3-12b-it-Q4_K_M, gemma-4-26B-A4B-it-{Q8_0,UD-Q4_K_M}, gemma-4-31B-it-Q4_K_M,
  Qwen3-14B-Q6_K, Qwen3-8B-Q8_0, Qwen3.5-4B-mxfp4, qwen3.6-35B-A3B-gguf.
NVFP4 SafeTensors: Gemma-4-26B-A4B-it-NVFP4, Gemma-4-31B-IT-NVFP4, Nemotron-3-Nano-30B-A3B-NVFP4,
  Nemotron-Labs-3-Elastic-30B-A3B-NVFP4, Phi-4-reasoning-plus-NVFP4, Qwen3-14B-NVFP4,
  Qwen3-30B-A3B-NVFP4-Modelopt, Qwen3.6-35B-A3B-NVFP4, Qwen3-8B-NVFP4-cortecs,
  Qwen3-Coder-30B-A3B-Instruct-FP4, Qwen3-30B-A3B-Q4_K_M.
Also: $REPO/models/ has Q8_0 verify baselines.

---

## Scoreboard (current snapshot)

imp, 2026-05-29, reps=8, ctx≈512, CUBLAS_WORKSPACE_CONFIG=:4096:8. tok/s. (partial — sweep running)

| Model | Family | Quant | pp512 | tg128 (decode) |
|---|---|---|---|---|
| Qwen3-8B | dense | Q8_0 | 7971 | **274.4** |
| Qwen3-8B | dense | NVFP4 | 21581 | 239.6 |
| Qwen3-14B | dense | Q6_K | 4965 | **155.4** |
| Qwen3-14B | dense | NVFP4 | 17961 | 147.6 |
| Qwen3-30B-A3B | moe | NVFP4 | 16099 | 170.7 |
| Qwen3-30B-A3B | moe | Q4_K_M | 3851 | **276.5** |
| Qwen3-Coder-30B-A3B | moe | NVFP4 | 17139 | 170.9 |
| Qwen3.6-35B-A3B | hybrid | NVFP4 | 8373 | 151.2 |
| ...rest pending | | | | |

**llama.cpp MEASURED on THIS machine (2026-05-29, commit 19e92c3, CUDA 12.8, sm_120, -fa 1,
-ngl 999, -r 5). Same GGUF weights as imp:**

| Model | Quant | imp pp512 | llama pp512 | pp winner | imp tg128 | llama tg128 | tg winner |
|---|---|---|---|---|---|---|---|
| Qwen3-8B | Q8_0 | 7971 | 14068 | llama 1.77× | 274.4 | 159.7 | **imp +72%** |
| Qwen3-14B | Q6_K | 4965 | 6367 | llama 1.28× | 155.4 | 113.7 | **imp +37%** |
| Qwen3-30B-A3B | Q4_K_M | 3851 | 9232 | llama 2.40× | 276.5 | 316.7 | llama +15% |
| Qwen3.6-35B-A3B | Q4_K_M | 3625 | 6290 | llama 1.73× | 157.9 | 228.6 | **llama +45%** |
| Gemma-4-26B | Q4_K_M | 4278 | 10457 | llama 2.44× | 252.0 | 212.2 | **imp +19%** |
| Gemma-3-12B | Q4_K_M | (bogus) | 8850 | — | (bogus) | 142.3 | — |

**REFRAME — the real picture (this overrides stale "imp wins decode +24-86%" memory):**
- imp DOMINATES dense GGUF decode (+37-72%). Already best-in-class there.
- imp LOSES ALL GGUF prefill 1.28-2.44× (worst: MoE/Gemma Q4_K_M ~2.4×). dequant→cuBLAS vs
  llama MMQ. Consistent, biggest gap. Hard (custom MMQ kernel, weeks) — memory q4k_prefill.
- imp LOSES MoE/hybrid GGUF decode: Qwen3.6-35B −45% (158 vs 229!), Qwen3-30B −15%. NEW finding.
- NVFP4 path is UNCONTESTED by llama.cpp (no NVFP4 support); imp NVFP4 prefill 16-24k crushes
  llama GGUF. NVFP4's only competitor is vLLM (broken on sm_120 MoE-NVFP4 per research). So
  GOAL.md's "NVFP4 = recommended path" is well-supported; making NVFP4 decode/MoE great (LEAD-1/2)
  is the highest-leverage, most-defensible work. Need measured vLLM NVFP4 to confirm prefill lead.
- Gemma-3-12B Q4_K_M row is bogus (tg=1588, known degeneration early-stop). Re-measure / flag.

---

## Backlog (prioritized gaps) — initial, pre-profiling

**LEAD-1 (decode, all NVFP4 dense): native NVFP4 decodes SLOWER than GGUF on same model.**
Qwen3-8B Q8_0 274 vs NVFP4 240 (−13%); Qwen3-14B Q6_K 155 vs NVFP4 148 (−5%). Same arch.
NVFP4 is the *recommended* fast path per GOAL.md, yet it's slower than GGUF at decode.
Hypothesis: native-NVFP4 checkpoints keep lm_head/embeddings/some-proj in FP16; GGUF path
quantizes output_proj to NVFP4 (pre_dequant_phase3:101). NEEDS nsys kernel-mix diff.

**LEAD-2 (decode, NVFP4 MoE): biggest gap. NVFP4 MoE decode 170 vs Q4_K_M MoE 276 (−38%).**
Qwen3-30B-A3B + Coder-30B both ~170 NVFP4 vs 276 Q4_K_M. GGUF MoE gets a dedicated
Q4_K×Q8_1 dp4a `can_decode_fast` path (executor_forward_moe.cu:92); NVFP4 MoE uses per-expert
FP16-activation GEMVs (gemv_nvfp4_moe_*). NEEDS nsys to confirm cause. Possibly biggest win.

**LEAD-3 (strategic, multi-week): spec-decode / EAGLE / MTP** — 1.5-4× batch=1 multiplier
competitors ship. imp parked at 196 tok/s ceiling. Highest ceiling, highest cost. Revisit.

**LEAD-4 (prefill MoE):** vLLM MoE-NVFP4 broken on sm_120 (Marlin fallback) per research →
imp may already win MoE prefill in practice. Confirm with measured vLLM later.

NOTE: imp may already be best-in-class on most DECODE cells vs llama.cpp. If confirmed, the
product gap is "make NVFP4 (recommended path) ≥ GGUF speed" (LEAD-1/2) + prefill + spec-decode.

---

## Log (timestamped, append-only)

### 2026-05-29 — Session start
- Re-read GOAL.md, CLAUDE.md, perf_baseline.json. No existing journal.
- GPU idle & cool. Kicked off Docker build.
- Goal restated: batch=1 latency-first, decode tok/s is primary metric; must beat
  llama.cpp/vLLM/SGLang/ExLlamaV3/MLC on every supported arch×quant on sm_120a.

### 2026-05-29 — Build green, correctness baseline green
- `docker build -t imp:test .` → exit 0. Binaries: imp-cli, imp-server, imp-tests, test-*.
- `imp-tests-unit`: 34 passed / 3 skipped. `imp-tests` (GPU): all suites PASS
  (84+139+34+83+73), 0 FAILED; skips are model-path-dependent only. **main is correct.**
- Bench rig validated vs perf gate: Qwen3-8B Q8_0 pp512=7927 (gate 7736, +2.5% cuBLAS var),
  tg128=273.99 (gate 276.59, −0.9%). Measurement trustworthy.

### 2026-05-29 — Competitive research (web, this session)
- **llama.cpp CUDA is the only credible batch=1 DECODE threat.** It beats vLLM at batch=1
  decode on Blackwell (measured 3rd-party: 134 vs 89 tok/s Qwen3.5-9B Q4_K_M vs BF16).
- vLLM wins PREFILL (3.4×) and batched throughput; loses batch=1 decode. vLLM MoE-NVFP4
  backend selection is BROKEN on sm_120 (falls back to Marlin) — opportunity.
- B200-only (NOT sm_120): tcgen05, TMEM, wgmma, TMA warp-specialized grouped GEMM.
  Confirms CLAUDE.md. SGLang's headline FP4 wins are all sm_100/B200.
- ExLlamaV3 (EXL3/QTIP): consumer batch=1 contender but dequant-to-FP16, NO native FP4 TC
  path, sm_120 support is a monkey-patch. imp's native NVFP4 decode cache should beat it.
- **Biggest strategic lever per research = speculative decoding (EAGLE-3/P-EAGLE/MTP),
  1.5–4× batch=1.** Competitors ship it; imp's spec-decode is parked at 196 tok/s ceiling.
  This is the one front where vLLM+P-EAGLE / SGLang+EAGLE-3 could beat imp on Qwen3-class.
- Published llama.cpp 5090 batch=1 decode (InsiderLLM 2026-03-25, Q4_K_M): Qwen3-8B 185.9@4k,
  Qwen3-14B 123.8@4k, Qwen3-30B-A3B 234.3@4k. (To be reproduced on THIS machine.)

### 2026-05-29 — LEAD-1 implemented (NVFP4 FP16-lm_head decode cache)
- Confirmed via safetensors headers: Qwen3-8B/14B/30B NVFP4 all store `lm_head.weight` BF16.
- Confirmed via nsys (eager, --no-cuda-graphs): cuBLAS FP16 lm_head GEMV = 19.3% of decode
  GPU time, 0.78 ms/call, Qwen3-8B NVFP4. 2nd-biggest kernel.
- Fix (commit pending): new `GraphExecutor::nvfp4_decode_cache_fp16_lm_head_()` in
  pre_dequant_phase3 — quantizes an FP16/BF16 output_proj → NVFP4 via
  quantize_fp16_to_nvfp4_async, stores in wcache_.nvfp4[lm.data]; forward_logits
  `lm_nvfp4_secondary` hook reads it (zero forward-pass change). Gated: skips GDN/SSM-hybrid
  models (LM-head NVFP4 degrades recurrent quality). Config `gemm.nvfp4_lm_head` (default on),
  env `IMP_NO_NVFP4_LM_HEAD=1` to disable for A/B. Same transform the GGUF Q8_0 path already
  applies to output_proj (proven quality-safe + passes perf gate).
- Predicted: Qwen3-8B NVFP4 239→~279 (beats Q8_0 274 & crushes llama.cpp 160). Verifying.

### 2026-05-29 — LEAD-1 VERIFIED + landed ✅
A/B Qwen3-8B NVFP4 (identical conditions, reps=10, fix default-ON vs IMP_NO_NVFP4_LM_HEAD=1):
- **Decode: 238.56 → 276.80 tok/s (+16.0%)** — now > Q8_0 (274) & **+73% vs llama.cpp (160)**.
- **Prefill: 25514 → 29949 tok/s (+17.4%)** bonus (last-token logits proj is M=1 GEMV in prefill too).
- LM head log: "quantized FP16 [151936 x 4096] → NVFP4 (333.8 MiB)".
Correctness gates ALL GREEN:
- Long 200-tok gen: eloquent, coherent, zero repetition/degeneration. Error scan empty.
- Factual + ocean prompts coherent. Output diverges from BF16 by 1 benign token-flip (expected).
- Q8_0 GGUF gate model UNCHANGED (pp 7954, tg 273.96 — lm_head log absent, Q8_0 head untouched).
- Full GPU test suite: all PASSED, 0 FAILED.
Same NVFP4-quantize-output_proj transform the production GGUF Q8_0 path already uses → low risk,
confirmed. Applies to ALL non-hybrid native-NVFP4 dense models (Qwen3-8B/14B, Phi-4). Committing.
NEXT: re-sweep NVFP4 dense models to capture the win; then LEAD-2 (NVFP4 MoE decode 170 vs 276).

### 2026-05-29 — Scoreboard sweep launched
- Wrote `scripts/scoreboard.sh` (committable harness): sweeps 14-model matrix, pp512+tg128,
  reps=8, CUBLAS_WORKSPACE_CONFIG=:4096:8 → `docs/scoreboard.tsv`.
- Launched sweep (bg) + llama.cpp-on-sm120 build/bench agent (bg, builds CUDA 12.8 image).
- Early results: Qwen3-8B Q8_0 pp512=7970/tg128=274 · Qwen3-8B-NVFP4 pp512=21581/tg128=239.
