# Model Validation Report

_Generated: 2026-05-04 21:57:51_  
_Mode: A (reduced scope — no BF16 reference, no NVFP4 calibration)_  
_Engine: imp (sm_120f), CUDA 13.2, deterministic_gemm=true, CUDA Graphs=on_  

Phase legend: 0=load+tokenizer, 3=graph replay (32x byte-identical), 4=20-prompt battery (T=0 seed=42), 5=degeneracy gates, 6=perf smoke. Phases 1, 2, 5c are out of scope (see report header).

## Verdicts

| Model | Verdict | Failure phase | Failure reason |
|---|---|---|---|
| `Qwen3.6-35B-A3B-NVFP4` | **FAIL** | 4_or_5 | battery passed 3/4; logit_health_ok=True; det3=True; graph_replay=4/4 |

---

## Qwen3.6-35B-A3B-NVFP4
**Verdict:** FAIL  
**Failure phase:** 4_or_5  
**Failure reason:** battery passed 3/4; logit_health_ok=True; det3=True; graph_replay=4/4  

### Config
- Path: `$IMP_MODELS_DIR/Qwen3.6-35B-A3B-NVFP4`
- Arch: `Qwen3_5MoeForConditionalGeneration`
- Param count (≈): 50.1B
- Weight files: 3 (25.04 GB)
- Config keys: `{}`

### Phase 0 (load + tokenizer probe)
- {
  "weight_files": 3,
  "weight_bytes": 25043516968,
  "tokenizer_probe_strings": 6,
  "tokenizer_probe_failures": 0,
  "tokenizer_probe_failure_examples": []
}

### Phase 3 (CUDA Graph 32x replay)
```json
{
  "replays": 4,
  "identical_to_first": 4,
  "first_output_steady": "The moon is Earth's only natural satellite.\n\n",
  "second_output_steady": "The moon is Earth's only natural satellite.\n\n",
  "warmup_request_1_output": "The moon is Earth's only natural satellite.\n\n",
  "warmup_request_2_output": "The moon is Earth's only natural satellite.\n\n",
  "first_request_visibly_degenerate": false,
  "median_elapsed_s": 0.6745860576629639
}
```

### Phase 4 (battery)
- Passed: 3 / 4
- Failures:
  - prompt 1 `factual_capital`: first-5-tokens='The capital of France is'
    output head: `'<think>The user is asking for the capital of France.</think>\nThe capital of France is Paris.\n\n\n'`

### Phase 5 (degeneracy)
```json
{
  "long_gen_4gram_rep_rate": null,
  "logit_health_ok": true,
  "determinism_3x_byte_identical": true,
  "determinism_outputs_head": [
    "<think>The user is asking for the capital of France.</think>\nThe capital of France is Paris.\n\n\n",
    "<think>The user is asking for the capital of France.</think>\nThe capital of France is Paris.\n\n\n",
    "<think>The user is asking for the capital of France.</think>\nThe capital of France is Paris.\n\n\n"
  ],
  "server_died_at_prompt": null,
  "phase5c_drift_status": "INCOMPLETE \u2014 no BF16 reference (Mode A)",
  "phase5c_ppl_status": "INCOMPLETE \u2014 imp-server has no logprobs-of-arbitrary-text endpoint"
}
```

### Phase 6 (perf smoke)
```json
{
  "vram_used_mb_before": 25943,
  "vram_used_mb_peak": 25943,
  "ttft_s_by_prompt_tokens": {
    "17": 1.1648776531219482,
    "60": 0.5916860103607178,
    "29": 1.3486251831054688
  },
  "decode_tok_per_s_by_prompt_tokens": {
    "17": 37.10773841099796,
    "60": 30.421540622578537,
    "29": 87.49651235807589
  },
  "ttft_caveat": "non-streaming end-to-end latency; not strict TTFT"
}
```

### Per-prompt detail
| # | name | check | tokens (in/out) | elapsed (s) | logits ok |
|---|---|---|---|---|---|
| 1 | factual_capital | ❌ first-5-tokens='The capital of France is' | 17/25 | 0.67 | ✅ |
| 2 | factual_arithmetic | ✅ first-5-tokens='4' | 17/96 | 1.16 | ✅ |
| 3 | code_completion | ✅ regex=fibonacci\s*\(\s*n\s*-\s*1\s*\)\s*\+\s*fibonacci\s*\(\s*n\s*-\s*2\s*\) | 60/18 | 0.59 | ✅ |
| 4 | instruction_primary_colors | ✅ matched={'blue', 'red', 'yellow'} via ['red', 'yellow', 'blue'] | 29/118 | 1.35 | ✅ |

---
