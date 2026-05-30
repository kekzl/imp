# imp — Audit: Performance & Agent-Readiness (2026-05-31)

Read-only Audit. Keine Code-Änderung. Belege: **[M]** = frisch gemessen (diese Session),
**[P]** = profilbelegt (nsys diese Session oder dokumentiertes ncu/nsys), **[C]** = Code-Stelle, **[H]** = Hypothese.

Mess-Setup: Host RTX 5090 (sm_120), CUDA 13.3, Binary `build-ciq/imp-cli` (29.05., CompileIQ-ACF-Build —
absolute Zahlen ±ein paar %, die qualitativen Schlüsse sind robust). `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

---

## Executive Summary

1. **Die zentrale Audit-Prämisse ist veraltet.** "MoE-Prefill ~1258 tok/s, 20× hinter vLLM" stimmt nicht mehr.
   Frisch gemessen liefert Qwen3-Coder-30B-A3B-NVFP4 **16.5–18.2k tok/s Prefill** (pp512–pp4096). Der "1258"-Wert
   stammt aus `archive/vllm_comparison_2026_05_10.md` (pp512, **vor** dem CUTLASS-3.x-grouped-GEMM-Umbau, alter
   `NVFP4→FP16-dequant + cuBLAS-grouped`-Pfad). Dieser wurde mit ~42× abgelöst; der **reale Rest-Gap zu vLLM ist
   ~1.15–1.42×**, nicht 20×. Das größte einzelne Prefill-Optimierungsziel aus dem Briefing existiert so nicht mehr.
2. **Der echte NVFP4-Prefill-Hebel ist die Attention, nicht der grouped GEMM.** Der MoE-Expert-GEMM läuft fused
   (Dequant in der MMA) und ist nahe Roofline. Der adressierbare Rest ist der teil-materialisierte Attention-Pfad
   (FA2 greift nur teilweise).
3. **Agent-Backend ist überraschend reif.** Serving (continuous batching, prefix-cache, paged-KV, cancellation,
   rate-limit, Prometheus) ist weitgehend produktionsreif. Constrained Output **funktioniert** (entgegen erster
   Vermutung — `apply_mask` ist im Sampling verdrahtet). Echte Lücken: synthetisches `/v1/messages`-Streaming,
   kein Per-Request-Spec-Decode, Determinismus-Vorbehalt bei MoE, kein Prompt-Caching-Header.

---

# Teil A — Performance

## A.0 Verifikation der Prämisse (frische Messung)

Qwen3-Coder-30B-A3B-Instruct-FP4, 3 reps, Default-Pfad (`executor_forward_moe_cutlass.cu:112` device-args):

| Metrik | pp512 | pp2048 | pp4096 | decode tg256 |
|---|---|---|---|---|
| tok/s **[M]** | 16.535 | 17.206 | 18.229 | 290–295 |

- Fallback-Pfad `IMP_NVFP4_DEVICE_ARGS=0` (host-args CUTLASS 3.x): pp2048 = **12.815 tok/s [M]** — auch das ist
  nicht 1258. Der echte ~77-tok/s-Wert ist der *legacy serielle* Pfad, längst tot (`executor_forward_moe_cutlass.cu:306`:
  "Prefill n=120: ~2750 vs legacy ~77 — 35× win").
- Decode-Moat bestätigt: tg256 ≈ 290 tok/s mit Graphs (höher als die im Briefing genannten 200). **[M]**
- vLLM-Vergleichszahl 25.513 ist die einzige nicht selbst-verifizierte Größe; selbst gegen sie ist imp ~0.65–0.72×.

## A.1 Prefill-Kernel-Breakdown (nsys, prefill-isoliert, Graphs off)

Profil: ~2800-Token-Prompt + `--max-tokens 1` (isoliert Prefill). **Wichtiger Fallstrick zuerst:**

- `convert_scales_sfatom_kernel` erscheint mit **21.7 %** — aber mit **identisch 37.249 Instanzen / ~47 ms in
  *beiden* Läufen** (1 vs. 256 Decode-Steps). Das ist **einmalige Cache-Bau-Arbeit beim Modell-Laden**, kein
  Per-Prefill-Kostenpunkt. Im Server (einmal laden) amortisiert → **nicht jagen** (deckt sich mit
  `docs/decode-gap-analysis-2026-05-29.md:14`). **[P]**

Bereinigter Per-Prefill-Mix (NVFP4 MoE):

| Kernel | Anteil | Klasse | Beleg |
|---|---|---|---|
| CUTLASS NVFP4 grouped GEMM (`GroupProblemShape`) | ~19 % | Expert-FFN, ~Roofline | [P] nsys |
| `fmha_sm120_fa2_kernel<128>` | ~13 % | FA2-Attention (greift partiell) | [P] nsys |
| `causal_softmax_fp32_to_fp16` + nvjet-cuBLAS-MMAs | ~18 % | **materialisierte Attention (Nicht-FA2)** | [P] nsys |
| MoE quant/scatter/permute/gating/scale | je ~1 % | Routing-Overhead, klein | [P] nsys |

**Befund:** FA2 (`fmha_sm120_fa2_kernel`) *und* der alte materialisierte Pfad (`causal_softmax_fp32_to_fp16` + cuBLAS
QK^T/PV) laufen gleichzeitig → Attention ist nur teilweise auf FA2. Das deckt sich mit der dokumentierten
NVFP4-pp2048-Analyse: "CUTLASS NVFP4 GEMM 39 % (competitive w/ vLLM) + Attention ~37 %"
(`docs/MISSION_JOURNAL.md:314`) und dem Roofline-Detail "FFN-Shapes ~100 % FP4-Roofline, Attention-Shapes ~34 %"
(`docs/plans/nvfp4_pp2048_analysis_2026_05_25.md:35`).

## A.2 Dequant fused vs. separater Pass — Hypothese aus dem Briefing geklärt

- **NVFP4-MoE-Prefill: Dequant ist FUSED** in die block-scaled MMA (`mma.sync ...kind::mxf4nvf4.block_scale...m16n8k64`,
  `src/compute/gemm_grouped_nvfp4_smallM.cu:81`; Pfad `executor_forward_moe_cutlass.cu:32-295`). Kein separater
  FP16-Pass. Die Briefing-Arbeitshypothese ("grouped-GEMM Dequant-Pfad als Ursache") trifft auf NVFP4 **nicht** zu. **[C]**
- **GGUF-Q4_K-MoE-Prefill: Dequant IST ein separater Pass** (`dequant_gpu()` → FP16 → `gemm_moe_batched()`,
  4.55 B/elem vs. llama.cpp MMQ 0.55 B/elem = 8.3× Bandbreiten-Overhead, `docs/plans/q4k_prefill_analysis_2026_05_25.md:20-31`).
  Das ist der reale "separate dequant"-Schmerz — aber bei GGUF, das laut Projekt-Policy *Legacy* ist (NVFP4 ist Priorität).
- Fallback `try_run_moe_nvfp4_dequant_batch_prefill_` (`executor_forward_moe_batch.cu:54`) macht separaten Dequant
  (Variable `slow_down_act`, Z.116) — feuert nur, wenn die device-args-Vorbedingungen fehlen. **[C]**

## A.3 CUDA-13.2/13.3-Features — Einbaubarkeit

| Feature | Anwendbar? | Wo / Effekt | Beleg |
|---|---|---|---|
| `cublasLtMatmulGrouped` NVFP4 device-shapes | **Nein** | 0 grouped-Algos auf sm_120 bis cuBLAS 13.4 (nur CC 10.x/11.0) | `docs/sm120.md:86`, `archive/cublas_13_4_sm120_no_movement_2026_05_09.md` [C] |
| `cub::DeviceTopK` (Expert-Routing/Sampling) | **Prüfen [H]** | Aktuell `topk_gating_kernel` (1 Block/Token) — im Prefill nur ~1 %, im Decode/Sampling evtl. mehr. Kein vorheriger Test dokumentiert. | nsys [P] |
| Grouped GEMM + CUDA Graphs | teilweise blockiert | CUTLASS `MoEProblemShape` trägt `IsMoEScheduler=false`-Stub für sm120; Capture-Hang unter `CaptureModeGlobal` | `archive/moe_prefill_graph_capture_analysis_2026_05_10.md`, `prefill_graph_blockers_2026_05_14.md` [C] |
| CUDA 13.2→13.3 Instruktionen | kein Gewinn | 0 von 247 sm_120a-Instruktionen geflippt | `docs/MISSION_JOURNAL.md:413` [C] |

## A.4 CUDA Graphs / Scheduling — Prämisse "nur Decode" widerlegt

- **Prefill IST graphifiziert** (default `runtime.prefill_graph=true`). Effekt frisch messbar: pp2048 17.206 (Graph)
  vs. 14.787 (Graph off) = **~+16 % [M]**. Die Briefing-Frage "werden Graphs nur im Decode genutzt?" → nein.
- Offen: Graphs für Nicht-Fast-Path-MoE-Decode (GGUF) bleiben blockiert (D2H-Expert-Routing, `docs/sm120.md:79`).

## A.5 Priorisierte Befundtabelle (Teil A)

| # | Befund | Beleg | Erwarteter Speedup | Aufwand | Decode-Risiko | Adressiert MoE-Prefill-Lücke? |
|---|---|---|---|---|---|---|
| A1 | **FA2-Abdeckung im Prefill erhöhen** — materialisierter `causal_softmax`+cuBLAS-QK^T/PV-Pfad läuft parallel zu FA2; vollständige FA2-Umstellung über mehr Seq-Längen/Head-Dims | [P] nsys (~18 % Nicht-FA2-Attention) + [C] `attention.fmha_*` | pp +10–20 % (long-ctx) | M | mittel (Attention ändert sich; FA2 ist parity-getestet) | Ja (Haupt-Hebel NVFP4) |
| A2 | **GGUF-Q4_K-MoE-Prefill: in-SMEM-MMQ statt dequant→cuBLAS** | [C] 8.3× BW-Overhead, doku-belegt | pp +30–50 % GGUF | XL (2–3 Wo) | niedrig | nur GGUF (Legacy-Prio) |
| A3 | **Kleine-M grouped-GEMM-Effizienz** — Attention-Shapes ~34 % Roofline bei small-M; Tile/Scheduler-Tuning | [P] doku-Roofline | pp +5–10 % | L | mittel | teilweise |
| A4 | `convert_scales_sfatom` als Init **nicht** als Prefill-Kosten behandeln (Anti-Befund) | [P] identische Instanzzahl | 0 (Klarstellung) | — | — | nein |
| A5 | `cub::DeviceTopK` für Routing/Sampling evaluieren | [H] | gering (Routing ~1 % Prefill) | S (Messung) | niedrig | nein |
| A6 | Grouped-GEMM-CUDA-Graph-Capture entblocken (`IsMoEScheduler`) | [C] | pp +10–15 % | XL | mittel | teilweise |

> **Markiert als MoE-Prefill-Lücke:** A1 ist der mit Abstand beste Hebel — die Lücke ist *Attention*, nicht der
> grouped GEMM. Der grouped GEMM ist bereits nahe Roofline und fused.

---

# Teil B — Agent-Betrieb

Status pro Achse. **Korrektur vorab:** Der erste Sub-Audit meldete Constrained Output als "toten Code"
(`apply_mask` nie aufgerufen). **Das ist falsch** — selbst verifiziert: `apply_mask()` wird im Sampling an
`src/exec/executor.cu:122/124, 220/222, 360/362` aufgerufen (drei Sampling-Pfade), FSM-Fortschritt via
`constraint_manager.cpp:167`. Constrained Output ist funktional.

| # | Achse | Status | Code-Beleg | Agent-Impact | Aufwand |
|---|---|---|---|---|---|
| 1 | **Structured/Constrained Output** | **funktioniert (Core)** | FSM `json_constrain.cu`/`schema_constrain.cu`; Maske im Sampling `executor.cu:122-362`; Server `handlers.cpp:731` (`response_format`) | kritisch | — |
| 1b | … Regex/`pattern` + GBNF-Grammar | **fehlt** | kein `pattern`-Parsing in `json_schema.h`; kein GBNF-Compiler | wichtig | M |
| 1c | … Masking-Overhead | unbelegt | kein Benchmark/Kommentar | nice | S (messen) |
| 2 | **Tool Calling** (Definition, tool_choice auto/required/named, Dialekte) | **teilweise→vollständig** | `tool_call.cpp:6-150`, `chat_template.cpp:1093`; Vektor-Return `tool_call.cpp:142` (Mehrfach-Parsing vorhanden) | kritisch | — |
| 2b | … Streaming von Tool-Call-Deltas | **fehlt** | Streaming-Pfad segmentiert Tool-Args nicht | wichtig | M |
| 2c | … Argument-Validierung gegen input_schema | **fehlt** | lenientes `json::parse`, keine Schema-Prüfung | wichtig (Halluzination) | M (FSM aus 1 wiederverwenden) |
| 3 | **Anthropic `/v1/messages`** (Blocks, tool_use/result, stop_reason, multi-block) | **vollständig** | `anthropic.cpp:19-394`, Handler `handlers.cpp:3449` | kritisch | — |
| 3b | … **Streaming** | **synthetisch** | `handlers.cpp:3572` "Phase 2 synthetic" — volle Generierung dann SSE-Replay → TTFT = volle Latenz | **kritisch für Agents** | M |
| 3c | … `thinking`-Blocks | **fehlt** | `anthropic.cpp:102` — reasoning verworfen | wichtig | S |
| 3d | … prompt-caching-Header (`cache_control`) | **fehlt** | kein Parsing/Tracking | wichtig | M |
| — | OpenAI `/v1/chat/completions`-Streaming | **echt** | `pop_token`-Loop `handlers.cpp:1371/1759/2762/2932` | — | — |
| 4 | **Prefix/Prompt-Caching über Turns** | **vollständig** | content-addressed `kv_cache_manager.h:70-283`, Scheduler-Reuse `scheduler.cpp:70`, LRU-Eviction, Metrik `imp_tokens_cached_total` | kritisch | — |
| 5 | **KV-Cache lange Loops** | **vollständig** | paged block_size=16 `kv_cache.h:12`; StreamingLLM `kv_cache_manager.h:124`; FP8/INT8/NVFP4-KV | wichtig | — |
| 6 | **Reliability** (cancel/timeout/OOM) | **vollständig** | cancel `batching_engine.cpp:93`; timeout `handlers.cpp:1360`; OOM-try/catch `batching_engine.cpp:108`; KV-exhaustion early-cancel `scheduler.cpp:48` | kritisch | — |
| 7 | **Concurrency/Scheduling** | **vollständig (in-flight batching)** | continuous batching `scheduler.cpp:17-126`, SJF gegen HoL `scheduler.cpp:27`; single-worker (kein Thread-Parallelismus) | kritisch | — |
| 7b | … p50/p99-Latenz | **teilweise** | nur Single-Gauges `handlers.h:61` (kein Histogramm) | wichtig | S |
| 7b' | … Aggregat-Durchsatz skaliert kaum mit N | [doku] | ~130 tok/s flat (`server_batching_throughput_ceiling`) | wichtig (Single-User-5090: evtl. Non-Goal) | L |
| 8 | **Speculative Decoding (MTP)** | **teilweise — nicht im Serving** | Head+Forward da (`mtp_forward.cu`), Engine-API `engine.h:142`; **kein Per-Request-Flag**, Draft-Verify-Loop nicht im Decode verdrahtet | wichtig | L (zudem K=1-Acceptance ~25-30 %, [[mtp_diagnosis]]) |
| 9 | **Determinismus** | **teilweise / Vorbehalt** | greedy argmax deterministisch `sampling.cu:41`; **MoE-Routing + top-k via atomics → nicht reproduzierbar** (`moe_routing.cu`); Qwen3.6-35B doku-belegt non-det @temp0 | wichtig (Eval/Test) | M |
| 10 | **Observability** | **vollständig (bis auf Histogramm)** | Prometheus `/metrics` `handlers.cpp:3154`; logprobs `request.h:72`; queue_depth; per-Request JSONL | wichtig | — |
| 11 | **Session/State** | **stateless** (Prefix-Pin als Ersatz) | `handlers.cpp:2426`; `pin_prefix` `kv_cache_manager.h:99`; kein Session-Store | nice | M |
| 12 | **Multi-Model-Serving** | **fehlt/teilweise** | 1 Modell/Instanz; Reload-POST deferred `handlers.cpp:267` | wichtig (Tool-Router-Modell) | L |
| 13 | **Container-Manager/Sandbox** | **fehlt komplett** | keine exec/sandbox/docker-socket-Infrastruktur; "tools" = nur Output-Formatierung | kritisch f. autonome Tool-Exec | XL |
| 14 | **Security/Limits** | **vollständig** | API-Key konstant-Zeit `main.cpp:138`; per-IP rate-limit `handlers.h:108`; payload-cap 100 MiB `main.cpp:76`; max-concurrent 429 `main.cpp:113`; timeout 300s | wichtig | — |
| 14b | … Input-Token-Längen-Limit | **fehlt** | keine Token-Count-Validierung vor Prefill | nice | S |

---

# Schlussteil

## Top-5-Maßnahmen (Impact/Aufwand)

1. **`/v1/messages` echtes Token-Streaming** (B-3b). Synthetisches Streaming bedeutet TTFT = volle Generierung —
   für interaktive Agent-Loops gegen den Anthropic-Endpoint ein harter Nachteil. OpenAI-Pfad streamt bereits echt,
   das Muster ist also vorhanden. *Aufwand M, Impact kritisch.*
2. **FA2-Abdeckung im Prefill vervollständigen** (A1). Der einzige große, decode-neutrale Prefill-Hebel; Attention,
   nicht grouped GEMM, ist die Rest-Lücke zu vLLM. FA2 ist parity-getestet. *Aufwand M, Impact hoch.*
3. **Tool-Argument-Validierung gegen `input_schema`** (B-2c) — die vorhandene Schema-FSM (B-1) auf Tool-Call-Bodies
   anwenden. Verhindert halluzinierte/kaputte Tool-Argumente, der häufigste Agent-Loop-Bruch. *Aufwand M (Reuse).*
4. **Determinismus-Schalter** (B-9): optionaler atomic-freier Routing-/Top-k-Pfad für `temperature=0`, für
   reproduzierbare Agent-Evals. *Aufwand M, Impact wichtig (Testbarkeit).*
5. **p50/p99-Latenz-Histogramm + prompt-caching-Header** (B-7b, B-3d): günstige Observability/Kosten-Sichtbarkeit
   für Multi-Agent-Last. *Aufwand S–M.*

> Container-Manager/Sandbox (B-13) ist der größte fehlende Baustein für *autonome* Tool-Ausführung, aber XL und
> eine eigene Produktentscheidung — bewusst nicht in Top-5.

## Decode-Schutz (Regressions-Guards vor jeder Prefill-Optimierung)

- **Gate vor jedem Merge:** `make verify-fast` gegen `tests/perf_baseline.json` (3 % Decode / 5 % Prefill).
- **A/B nur über Decode** (`tg256`), isoliert + 60–120 s GPU-Cooldown, 10 reps, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
  Prefill-pp variiert bis 2.6× über Container-Restarts — niemals als alleiniges Signal.
- **Graphs ON *und* OFF** messen (Graph-Replay kann silent Fallback verstecken — `check-degeneration`-Skill).
- **Kohärenz-Check** nach Forward-/Attention-/Routing-Änderungen (kein Repetition-Loop/Token-Stuck).
- Baseline-Modelle für die Decode-Wand: Qwen3.6-35B-A3B-NVFP4 (Moat), Qwen3-14B-NVFP4 (dense), Qwen3-Coder-30B (MoE).

## Implementierungs-Befunde (2026-05-31, Tasklist-Abarbeitung)

### T2 (FA2-Abdeckung) — gemessen, kein sicherer globaler Flip [M]
FA2 wird in `executor_attention.cu:811-817` (chunked) und `:849` (non-chunked) per
`attention.fmha_prefill_threshold` gegated (Auto-Default = S-Matrix-VRAM-Cap+1,
`executor_workspace_buffers.cu:267`). `causal_softmax_fp32_to_fp16` kommt aus `attention_cublas.cu`
(materialisierter Pfad unter dem Threshold). pp512 A/B default vs FA2-always (`fmha_prefill_threshold=1`),
decode durchweg neutral (FA2 berührt Decode nicht):

| Modell | default | FA2-always | Δ |
|---|---|---|---|
| Qwen3-30B-A3B | 13737 | 18154 | +32% |
| Qwen3-Coder-30B | 15742 | 17465 | +11% |
| Qwen3.6-35B-A3B | 10469 | 10575 | +1% |
| Qwen3-14B (dense) | 18178 | 15321 | −16% |
| Gemma-4-26B (MoE) | 22297 | 16732 | −25% |

→ Crossover ist modellabhängig; **kein decode-neutraler globaler Default-Flip möglich** (Gemma-4/dense
regredieren → Perf-Gate). Sicherer Win heute: per-Modell `attention.fmha_prefill_threshold=1` für
Qwen3-30B/Coder. Echte Lösung = gemessene per-(arch,seqlen)-Crossover-Heuristik (Folge-Arbeit, zoo-weit
zu validieren). pp≥2048 nutzt FA2 bereits (Default ≈ FA2-always).

### T11 (vLLM-Gegenmessung) — gemessen [M], korrigiert die Doku-Annahme
vLLM 0.21.0 (`vllm/vllm-openai`), Qwen3-Coder-30B-A3B-NVFP4, echter FlashInfer/CUTLASS-NVFP4-Pfad
(NICHT Marlin: `FlashInferCutlassNvFp4LinearKernel` + `FLASHINFER_CUTLASS` MoE), prefix-caching aus,
`--max-concurrency 1`, `--random-output-len 1`, best-of-3 (WSL2-bimodal → kühlerer Modus):

| Prompt-Len | vLLM prefill tok/s | imp prefill tok/s | vLLM-Vorsprung |
|---:|---:|---:|---:|
| 512  | ~21.200 | 16.500 | 1.28× |
| 2048 | ~33.300 | 17.200 | 1.94× |
| 4096 | ~29.200 | 18.200 | 1.60× |

→ Die "20×"-Prämisse ist endgültig widerlegt, **aber der reale Prefill-Gap (1.3–1.9× zugunsten vLLM) ist
GRÖSSER als die in den Memos genannten 1.15–1.42×.** vLLM gewinnt Prefill auf jeder Länge über batched
Prefill-GEMMs; Decode wurde nicht gemessen (imp-Decode-Moat unberührt). Konsistent mit A1/A.5: der Hebel
ist Prefill-GEMM/Attention-Effizienz, nicht der Routing-/Dequant-Pfad.

### Build-/Verifikations-Status (Tasklist-Batch, Branch `feat/agent-readiness-batch`)
8 Code-Tasks (T1,3,4,5,6,7,8,9) implementiert + **Docker-Build grün** (CUDA 13.3), Unit 37/37, GPU-Tests
73 passed/0 failed. Decode-Moat-Check: neues Binary tg256=253.0 vs build-ciq 256.9 (−1.5 %, im Rauschen) —
**decode-neutral** ✓. Determinismus-Flag (T4) verifiziert: `runtime.deterministic=true` liefert bit-identische
Tokens über Läufe (Qwen3-Coder, temp=0). Output kohärent, keine Degeneration.

## Offene Fragen / fehlende Messungen

1. **vLLM-Gegenmessung auf dieser Box** für Qwen3-Coder-NVFP4 (pp2048/pp4096), apples-to-apples — die 25.513
   ist die einzige nicht selbst-verifizierte Zahl; der reale Rest-Gap (~1.4×?) sollte frisch bestätigt werden.
2. **ncu fehlt auf dem Host** (`ncu: command not found`) — für Roofline/Occupancy/Stall der Top-Prefill-Kernels
   (FA2, grouped GEMM) muss ncu via Container gemountet werden (Recipe in [[decode_frontier_reconfirmed]]).
3. **FA2-Threshold:** FA2 feuerte bereits bei ~2800 Tokens, aber `causal_softmax` lief parallel — warum die
   Aufteilung? (Head-Dim ≠ 128? Chunk-Grenzen?) Klärt den genauen Umfang von A1.
4. **`cub::DeviceTopK`** im Sampling/Routing: lohnt nur, wenn `topk_gating`/Sampling im Decode messbar Zeit kostet — ncu nötig.
5. **MTP im Serving:** lohnt der Draft-Verify-Loop bei ~25–30 % Acceptance überhaupt? (Doku sagt: bei NVFP4 nein.)
