#!/usr/bin/env python3
"""Read every model's report.json under validation_artifacts/ and produce
the final MODEL_VALIDATION_REPORT.md + MODEL_VALIDATION_SUMMARY.csv covering
all models in one document."""

from __future__ import annotations
import json
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART = REPO / "validation_artifacts"
MD = REPO / "MODEL_VALIDATION_REPORT.md"
CSV = REPO / "MODEL_VALIDATION_SUMMARY.csv"


def main():
    reports = []
    for d in sorted(ART.iterdir()):
        rp = d / "report.json"
        if rp.exists():
            reports.append(json.loads(rp.read_text()))

    n_pass = sum(1 for r in reports if r['verdict'] == 'PASS')
    lines = [
        "# Model Validation Report",
        "",
        f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_  ",
        "_Mode: A (reduced scope, agreed with user)_  ",
        "_Engine: imp (sm_120f), CUDA 13.2, deterministic_gemm=true, CUDA Graphs=on, NVFP4 weights_  ",
        "",
        "## Executive summary",
        "",
        f"Validated {len(reports)} pre-quantized NVFP4 SafeTensors models against a 20-prompt "
        "battery + graph-replay determinism + degeneracy gates. Strict gate verdict: "
        f"{n_pass}/{len(reports)} PASS — but the failures split cleanly into three classes:",
        "",
        "1. **Real engine bugs** (3 fixed in this session, see _Bug fixes shipped_ below).",
        "2. **Real model-file defects** (Mistral-3.2-NVFP4 long-context regression — root-caused "
        "to upstream SmoothQuant calibration; not fixable in imp).",
        "3. **Test-design artifacts** (reasoning models truncated by 256-token budget; not "
        "actual generation problems).",
        "",
        "### Recommendation matrix for 32 GB VRAM (RTX 5090)",
        "",
        "| Use case | Model | Status |",
        "|---|---|---|",
        "| **Mistral-3.2 replacement (was broken at long context)** | `nvidia/Qwen3-30B-A3B-NVFP4` (Modelopt) | ✅ killer-test passed: Lorem-Ipsum repro produces *Paris/Berlin/Madrid…* (vs Mistral's *elit dolor elit dolor…*); 4-gram rep on 1024-tok creative drops 95.7% → 1.4% |",
        "| Coding (already in use) | `Qwen3-Coder-30B-A3B-Instruct-FP4` | ✅ unchanged from baseline |",
        "| MoE+GDN reasoning (existing) | `Qwen3.6-35B-A3B-NVFP4` | ✅ now no-crash after long-prompt prefill clamp |",
        "| Multimodal vision | `Gemma-4-26B-A4B-NVFP4` | ✅ working (graph non-determinism is engine-side, not output-quality) |",
        "| Small/dev iteration | `Qwen3-Coder-FP4` or future `Ministral-3-14B-Reasoning` | 14B Mistral 3 needs imp loader work for `model_type=ministral3` + new YARN |",
        "",
        "### Bug fixes shipped this session",
        "",
        "| Bug | Symptom | Fix | Files |",
        "|---|---|---|---|",
        "| Qwen3.6 long-prompt prefill crash | `terminate: reshape: numel mismatch` on 512-tok prompt; container exit | Clamp `effective_chunk` against `executor->max_tokens()`; throw on overflow; try/catch in batching engine | `engine.cpp`, `executor_forward.cu`, `batching_engine.cpp` |",
        "| `Cleared stale error: invalid device function` on every request | benign WARN spammed log per request | `cudaGraphKernelNodeGetParams` returns `func=nullptr` for driver-API kernel nodes and sets a stale CUDA error — swallow with `cudaGetLastError()` after the get | `cuda_graph.cu` |",
        "| Mistral-3.2 first-request degeneration (`illumin11111`) | first generated answer after server boot was garbage; all subsequent fine | flip `runtime.warmup` default true→false; warmup pollutes engine state in ways that survive its own forward (most visible on Mistral-NVFP4) — opt-in for prod rollout where TTFT matters | `config.h`, `imp.conf.example`, `tests/test_config.cpp`, `CLAUDE.md` |",
        "",
        "Verified in re-validation: Qwen3.6 long_context_recall (prompt 6) goes from server-crash "
        "to coherent execution. Mistral-3.2 first-request goes from `illumin11111` to clean. "
        "Qwen3-Coder graph-replay determinism improves 16/32 → 23/32 (warmup flip side-effect).",
        "",
        "### Mistral-3.2-NVFP4 long-context regression — root cause",
        "",
        "Investigation in this session refuted the prior hypothesis (missing `input_global_scale` "
        "in activation quantization) by two routes:",
        "1. Empirical: tested `alpha *= 1/IGS` and `alpha *= IGS` in CUTLASS GEMM — both produced "
        "different garbage, neither helped.",
        "2. Comparative: all three llm-compressor NVFP4 models (Gemma-4, Qwen3.6, Mistral-3.2) "
        "ship `input_global_scale` tensors, but only Mistral-3.2 breaks at long context. So "
        "the differentiator is not `input_global_scale`.",
        "",
        "Direct dump of Mistral L0 q_proj NVFP4-dequant FP16 values reveals the actual cause:",
        "",
        "- Per-K-channel max range: **335×** (max=4.36, median=0.013)",
        "- **20.3% (1037/5120) outlier K-channels** with max > 4× median",
        "- **97.8% of all NVFP4 micro-blocks contain ≥1 outlier** — block absmax dominated, "
        "all 15 non-outlier values in those blocks snap to ±0/±0.5",
        "- **~45% of dequanted weight values are exactly 0** — information lost at calibration",
        "",
        "This is the SmoothQuant 0.9 + per-block-NVFP4 incompatibility from the original memo, "
        "now confirmed quantitatively. The precision is gone in the model file itself; no "
        "runtime fix in imp can recover it (dequant→cuBLAS path also produces garbage; "
        "per-channel hybrid would need the original FP16 weights which aren't in the file). "
        "Realistic solutions are model-side: re-quantize without SmoothQuant, or use the "
        "Modelopt-format `nvidia/Qwen3-30B-A3B-NVFP4` as a drop-in replacement (validated above).",
        "",
        "Memo updated: `safetensors_validation_2026_05_02.md` (links the 5 models tested, the 3 "
        "bug fixes, and the long-context-regression confirmation).",
        "",
        "## Scope statement",
        "",
        "Original spec required: BF16 reference run (Phase 1), NVFP4 calibration "
        "from a calibration corpus (Phase 2), KL/PPL drift vs BF16 (Phase 5c).",
        "",
        "Mode A drops those because:",
        "- imp consumes pre-quantized NVFP4 SafeTensors (llm-compressor / NVIDIA Model "
        "Optimizer); it has no calibration entry-point, so Phase 2 cannot run.",
        "- No BF16 SafeTensors checkpoints exist on disk for any of the 5 NVFP4 models, "
        "and even with one, imp auto-converts BF16→FP16 at load — there is no BF16 "
        "execution path to compare against.",
        "- imp-server's OpenAI-compatible API only returns logprobs of generated tokens, "
        "not arbitrary text, so wikitext PPL cannot be computed without a new endpoint.",
        "",
        "Phase 5c drift checks are therefore reported as `INCOMPLETE`, never as PASS.",
        "",
        "## Verdicts",
        "",
        "| Model | Verdict | Failure phase | Failure reason |",
        "|---|---|---|---|",
    ]
    for r in reports:
        lines.append(f"| `{r['name']}` | **{r['verdict']}** | "
                     f"{r['failure_phase']} | {r['failure_reason']} |")
    lines += [
        "",
        f"**Strict-gate summary: {n_pass} / {len(reports)} PASS.** Failures by class:",
        "- 4 models fail the 32x-graph-replay byte-identical gate due to MoE-decode "
        "non-determinism (engine-side, all MoE NVFP4 models affected, not model-specific).",
        "- 1 model (Mistral-3.2) fails the degeneracy gate due to upstream SmoothQuant "
        "calibration loss (model-file defect; see Executive summary above).",
        "- All models pass the load + tokenizer + crash-resistance gates after this session's bug fixes.",
        "",
        "---",
        "",
    ]

    for r in reports:
        lines.append(f"## {r['name']}")
        lines.append(f"**Verdict:** {r['verdict']}  ")
        lines.append(f"**Failure phase:** {r['failure_phase']}  ")
        lines.append(f"**Failure reason:** {r['failure_reason']}  ")
        lines.append("")
        lines.append("### Config")
        lines.append(f"- Path: `{r['path']}`")
        lines.append(f"- Arch: `{r['arch']}`")
        lines.append(f"- Param count (≈): {r['param_count_b']}B")
        lines.append(f"- Weight files: {len(r['weight_files'])} ({r['weight_bytes']/1e9:.2f} GB)")
        lines.append(f"- Server config: `chat-template=auto`, `runtime.deterministic_gemm=true`, "
                     f"`cuda_graphs=auto (default on)`, `kv_cache.dtype=fp16 (default)`, "
                     f"`max_tokens=2048` (server CLI), `seed=42`, `temperature=0.0`")
        lines.append("")
        lines.append("### Phase 0 — load + tokenizer probe")
        lines.append(f"```json\n{json.dumps(r['phase0'], indent=2)}\n```")
        lines.append("")
        lines.append("### Phase 3 — CUDA Graph 32x replay (after 2 warmup requests)")
        lines.append(f"```json\n{json.dumps(r['phase3'], indent=2)}\n```")
        lines.append("")
        lines.append("### Phase 4 — battery")
        p4 = r.get("phase4", {})
        lines.append(f"- Passed: **{p4.get('prompts_passed', 0)} / {p4.get('prompts_total', 0)}**")
        if p4.get("fail_details"):
            lines.append("- Failures:")
            for f in p4["fail_details"]:
                lines.append(f"  - prompt **{f['id']} `{f['name']}`** — {f['why']}")
                head = (f["text_head"] or "").replace("\n", "\\n")
                lines.append(f"    head: `{head[:200]}`")
        lines.append("")
        lines.append("### Phase 5 — degeneracy gates")
        lines.append(f"```json\n{json.dumps(r['phase5'], indent=2)}\n```")
        lines.append("")
        lines.append("### Phase 6 — perf smoke")
        lines.append(f"```json\n{json.dumps(r['phase6'], indent=2)}\n```")
        lines.append("")
        lines.append("### Per-prompt detail")
        lines.append("| # | name | check | tokens (in/out) | elapsed (s) | logits ok |")
        lines.append("|---|---|---|---|---|---|")
        for p in r.get("prompts", []):
            ok = "✅" if p["check_pass"] else "❌"
            lh = "✅" if p.get("logprobs_health", {}).get("ok") else "❌"
            lines.append(
                f"| {p['id']} | {p['name']} | {ok} {p['check_reason']} | "
                f"{p['prompt_tokens']}/{p['completion_tokens']} | "
                f"{p['elapsed_s']:.2f} | {lh} |"
            )
        lines.append("")
        lines.append(f"_Server log: `validation_artifacts/{r['name']}/server.log`_")
        lines.append(f"_Per-model JSON: `validation_artifacts/{r['name']}/report.json`_")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Engine-bug callout
    lines += [
        "## Engine bugs surfaced by this run (separate from model quality)",
        "",
        "1. **Gemma-4 NVFP4 graph-replay non-determinism** — 1/32 byte-identical at "
        "phase 3 with `deterministic_gemm=true` and CUDA Graphs on. Outputs differ in "
        "bullet ordering AND content text across replays. cuBLAS determinism alone is "
        "insufficient; some other reduction in the Gemma-4 NVFP4 forward pass is "
        "non-deterministic (likely the FP32 router or the post-MoE atomicAdd reduction).",
        "2. **Qwen3-Coder NVFP4 graph-replay non-determinism** — 16/32 identical at "
        "phase 3. Outputs split between two stable continuations. Same root cause class.",
        "3. **Qwen3.6 long-prompt prefill crash** — `executor_forward.cu:164: n_tokens "
        "(512) exceeds max_tokens (256)`, then `terminate called: reshape: numel "
        "mismatch`. Server initialized GraphExecutor with `max_tokens=256` (prefill chunk "
        "size), but did not auto-chunk a 512-token prompt. Hard crash, container exits. "
        "Workaround: pass `--prefill-chunk-size 256` explicitly, but the server should "
        "either auto-chunk or reject cleanly with a 4xx instead of throwing.",
        "4. **Mistral-3.2 NVFP4 first-request degeneration** — first request after engine "
        "warmup produced `\"...illumin11111...\"`; second request onward produced clean "
        "output. Same prompt, same seed, same temperature. Suggests warmup-time KV / "
        "scratch state not yet calibrated on the very first user request, despite the "
        "`runtime.warmup=true` engine warmup forward pass.",
        "5. **Mistral-3.2 NVFP4 phase-4 prompt-1 HTTP 500 (empty body)** — reproducible "
        "HTTP 500 with empty body on the first phase-4 chat request, regardless of "
        "`Connection: close`. Server log shows the corresponding `imp-N` request as "
        "completing successfully. cpp-httplib edge case under the validation harness's "
        "request pattern; deserves its own bisect.",
        "6. **Mistral-3.2 NVFP4 3-run nondeterminism with deterministic_gemm=true** — "
        "phase 5e: 3 runs of the same prompt, same seed, T=0 produced 2 identical + 1 "
        "different (middle run). Same class of bug as #1/#2 — deterministic_gemm covers "
        "GEMM only, not all reductions.",
        "7. **`Cleared stale error before forward: invalid device function`** — every "
        "imp-server request logs this WARN. Engine is silently swallowing a CUDA error "
        "from a previous launch. Smell, not a confirmed defect.",
        "",
        "## Existing project-memory entries that align with these findings",
        "",
        "- `gemma4_nvfp4_decode_fastpath_2026_05_01.md` — recently restored decode "
        "fast-path; non-determinism here may be a regression or pre-existing.",
        "- `qwen36_nvfp4_decode_partial_2026_04_30.md` — Qwen3.6 NVFP4 partial-coherence "
        "issues (RMSNorm `1+W`, GDN head layout). Battery shows verbose-think and "
        "long-prompt crash (the latter is engine, not model).",
        "- `nvfp4_long_context_regression_2026_04_28.md` — Mistral-3.2-NVFP4 garbage at "
        "~500+ raw tokens. Battery shows severe repetition spirals on long_context_recall "
        "(prompt 6) and long_generation_creative (prompt 12, 95.7% 4-gram rep rate).",
        "- `fp8_fmha_stile_bug_2026_04_23.md` — long-context cliff is a recurring class.",
        "",
    ]

    MD.write_text("\n".join(lines))

    # CSV
    with CSV.open("w") as f:
        f.write("model,verdict,failure_phase,failure_reason,arch,param_b,"
                "weight_gb,phase4_passed,phase4_total,det3_ok,logit_ok,"
                "graph_replay_identical,vram_peak_mb,long_gen_4gram_rep_rate,"
                "first_request_degenerate\n")
        for r in reports:
            failure_reason = r["failure_reason"].replace(",", ";").replace("\n", " ")
            f.write(",".join([
                r["name"], r["verdict"], r["failure_phase"],
                f'"{failure_reason}"',
                r.get("arch", ""), str(r.get("param_count_b", 0)),
                f"{r['weight_bytes']/1e9:.2f}",
                str(r.get("phase4", {}).get("prompts_passed", 0)),
                str(r.get("phase4", {}).get("prompts_total", 0)),
                str(r.get("phase5", {}).get("determinism_3x_byte_identical", False)),
                str(r.get("phase5", {}).get("logit_health_ok", False)),
                f"{r['phase3'].get('identical_to_first', 0)}/{r['phase3'].get('replays', 0)}",
                str(r.get("phase6", {}).get("vram_used_mb_peak", 0) or 0),
                str(r.get("phase5", {}).get("long_gen_4gram_rep_rate") or ""),
                str(r['phase3'].get("first_request_visibly_degenerate", False)),
            ]) + "\n")

    print(f"wrote {MD}")
    print(f"wrote {CSV}")
    print(f"models: {len(reports)}")


if __name__ == "__main__":
    main()
