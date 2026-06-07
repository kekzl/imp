# Testsuite-Audit & Refactor-Plan — Re-Audit 2026-06-06

Commit-Basis: `7a94d81f` (post-#574) · Vorgänger: [`docs/TEST_AUDIT.md`](../docs/TEST_AUDIT.md)
(2026-06-04, Phase-1-Gap-Analyse; deren Top-10-Programm wurde via PRs #527–#539
umgesetzt). Dieses Dokument ist das **Delta-Re-Audit** nach PRs #538–#574
(gpt-oss, LoRA, IQ4, chunked-NLL-Umbau, CUTLASS grouped GEMM, Issue-Sweep)
plus die im 06-04-Audit nicht abgedeckten Dimensionen (Perf-Varianz,
Stale/Dead, Laufzeit/Split).

Hinweis Phase 0: `AGENTS.md` und `AUDIT.md` existieren im Repo nicht
(Audit-Quellen waren CLAUDE.md, DISPATCH.md, docs/TEST_AUDIT.md, CMakeLists,
Makefile, scripts/verify.sh, tests/refs/README.md). `DISPATCH.md` beschreibt
den geschelvten cuTile-FA2-Track und betrifft die Suite nicht.

Klassifikation (wie 06-04): **A** = unabhängige Referenz (fp64/Format-Spez,
committeter Generator, dokumentierte Toleranz) · **A−** = echte Referenz, aber
benigne Daten / unbegründete oder nicht assertete Toleranz · **B** =
tautologisch (imp-vs-imp, Round-trip) · **C** = Smoke/strukturell.

---

## 1. Coverage-Matrix (Modul × Testtyp)

Stand nach #574. „CI" = läuft im GitHub-CI (kein GPU-Runner → nur unit);
„lokal" = braucht GPU und/oder Modell.

| Modul | unit | integration | e2e | numeric-correctness | perf | ungetestet (wertvollste Lücken) |
|---|---|---|---|---|---|---|
| core (tensor/config) | ✓ (strukturell) | — | — | — | — | reshape/slice-Numerik (unverändert seit 06-04) |
| compute/attention | ✓ | ✓ | ✓ greedy-locks | **A**: crosspath fp64-Golden (6 Prefill-Pfade paarweise, `test_attention_crosspath.cu`); **A**: paged-Oracle als TYPED_TEST über 6 KV-Dtypes inkl. INT4/INT8 (`test_attention_paged_oracle.cu`, R8/#582) | bench-only | `attention_blackwell.cu` [Routing-Tabelle: **geschlossen #577** `test_routing_decision.cpp`; attention sinks (gpt-oss #572): **A geschlossen, #584** `test_gpt_oss_sinks_ref.cu`] |
| compute/sonstige Kernel | ✓ | — | — | A (RoPE/RMSNorm/softmax/reduce vs CPU; **YaRN-Langseq fp64 #584** `test_gpt_oss_yarn_ref.cu`) | — | GPT_OSS_GLU-Aktivierung, Sampling-Numerik (nur „token in vocab") |
| quant | ✓ | ✓ | ✓ | **A**: `test_gguf_dequant_ref.cu` (Q8_0/Q6_K/Q4_K/IQ4_NL/IQ4_XS als TYPED_TEST vs fp64-Format-Spez, Byte-LCG, no-NaN-Guard); **A**: `test_nvfp4_outlier_ref.cu` (adversariale Fixtures); **A**: `test_gpt_oss_mxfp4_convert_ref.cu` (R1.1/#576, bit-exakt); **A**: `test_cutlass_grouped_ref.cu` (R1.2/#576, fp64-CPU-Ref auf identischen Quant-Bits) | bench-only | — |
| memory/KV | ✓ | ✓ | ✓ prefix-cache E2E (#538-Ship-Gate, 4/4 aktiv) | A (FP8-KV-Kalibrierung) | — | Eviction+Refill-Output-Stabilität, INT8/INT4/NVFP4-KV-Genauigkeitsbänder, vram-Allocator-Budget |
| model/loader/tokenizer | ✓ inkl. fault-injection (#535-Familie) | ✓ | ✓ | A (Merges, Jinja; **Harmony-Render-Golden vs HF #584** `test_gpt_oss_harmony_golden.cpp`) | — | hf_hub |
| exec | ✓ teilweise | ✓ | ✓ | **A** (grouped GEMM via `test_cutlass_grouped_ref.cu`, s. §2) | — | `executor_ffn.cu` isoliert, `executor_lora.cu`-Kernel (nur E2E via test_lora) [grouped-vs-fallback-Routing: **geschlossen #577** `test_routing_decision.cpp`] |
| lora (neu #571) | — | — | ✓ `test_lora.cpp` (A−: zero-B/nonzero-B-Identität) | — | — | Kernel-Isolation, Multi-Adapter, Rank-Grenzfälle |
| runtime | ✓ (think/stop, scheduler, json-FSM inkl. $ref/$defs #562) | ✓ | ✓ determinism-E2E (#542) | — | graphs-Gate in verify.sh | ConditionalRunner, Request-Lifecycle/Abort, Warmup-Token-Typ |
| vision | ✓ CPU-Preprocessing (#564) | ✓ GPU-Encoder+Projector Frozen-Golden (R9/#583: SigLIP+gemma4v, committetes 64² PNG → Projector-Spots, f16-Klasse ≤1e-2 rel + NaN/Inf-Guard, mmproj standalone ohne LM) | manuell (gemma-3/4 VL) | — | — | Vision-RoPE/Norm-Einzelkernel isoliert (Golden lockt nur den Encoder-Output) |
| api/server | ✓ SSE-Utils, anthropic-Transform, stream_pipeline (echte Quelldateien einkompiliert, nicht Mock!) | ✓ relaunch | ✓ | — | — | `handlers.cpp` selbst (nur dessen Utils getestet), Abort-Pfad |
| api/HTTP (Python) | ✓ Mock (Contract/Errors/Lifecycle) | ✓ real (modellgebunden) | ✓ | — | TTFT/Decode-Gates (real-only) | rekursives json_schema auf API-Ebene; /v1/messages-Streaming-E2E |

Gegenüber 06-04: ~45 % direkt → spürbar besser in attention/quant/runtime/api
(6 der 10 Risiken geschlossen, s. §6); vision-GPU jetzt per Frozen-Golden
abgedeckt (R9/#583), exec-Isolation unverändert blind.

## 2. Hot-Path-Kernels — Tiefe der Correctness-Tests

| Kernel | Tests | Klasse | Befund |
|---|---|---|---|
| NVFP4 quantize→dequant→GEMV | `test_nvfp4_outlier_ref.cu` + `tests/refs/gen_nvfp4_outlier_golden.py` | **A** | fp64-Referenz aus Formatdefinition (E2M1+UE4M3+1/512-Floor), 4 adversariale Verteilungen (Gemma-Kollaps-Klasse #514/#516), harter no-NaN/Inf-Guard. Vorbildlich. |
| NVFP4 block-scaled MMA (mxf4nvf4) | `test_mxf4nvf4_qkt_validate.cu` | A− | E2M1-exakte Inputs → Referenz verlustfrei, aber benignes Datenregime; Präzisionskante nie exerziert. `_probe` = Smoke. |
| NVFP4-GEMV-Loop | `test_nvfp4_gemv_kpar_loop.cu` | B | Bewusst dokumentiertes Negativ-Resultat (Repro mathematisch äquivalent); Regressions-Riegel, kein Correctness-Beweis. OK so. |
| **CUTLASS grouped GEMM (pp512-10×-Pfad #574)** | `test_cutlass_grouped_3x_nvfp4.cu` + `test_cutlass_grouped_ref.cu` (R1.2/#576) | **A** | grouped-vs-per-expert (B, Staging) PLUS unabhängige fp64-CPU-Referenz auf den identischen Quant-Bits, die die GPU konsumiert (Quant-Fehler kancelt → f16-Akkumulationsklasse ≤1e-2, gemessen ~4.9e-4). Boundary-Verteilungen M=0/1/200, single-active-expert. |
| FP8-E4M3-Encoder | `test_cutlass_nvfp4_alpha.cu` | A− | kanonische Bit-Grenzwerte exakt gepinnt (448/overflow/240-Cliff); Prefill-Teil Smoke. |
| fp8 FMHA | `test_fmha_fp8.cu` | A− | CPU-Referenz korrekt, aber Toleranz bewusst NICHT assertet („characterized", README §2 — e4m3 auf kurzen Zeilen 0.58–0.71 rel, #512-Mechanismus). Qualitätsgate liegt per Policy bei den E2E-Locks. Füllung teils noch `%13`-Muster. |
| FA2-Prefill (alle 6 Pfade inkl. cuBLAS-legacy) | `test_attention_crosspath.cu` + fp64-Golden | **A** | Paarweise f16-Klassen-Agreement 1e-2 (gemessen ~4e-4 post-#528), realistisches Heavy-Tail-Datenregime, bit-identische LCG-Fixtures Python↔C++. Der „Killer-Assert" aus Risiko #1 existiert. |
| FA2 chunked continuation (#553/#568) | `test_attention_chunked.cu` + `test_chunked_prefill.cu` | A−/E2E | Kernel-Test + teacher-forced-NLL-Gates (seit #553 NLL statt Byte-Equality — korrekt wegen Re-Download-Quant-Drift). hd=256-Paritäts-Locks via #570. |
| Paged decode F16 | `test_attention_paged_oracle.cu` | **A** | fp64-Ref aus Original-f16-K/V; kv_len=333 absichtlich nicht block-aligned. |
| Paged decode FP8/INT8/INT4 | dito (Envelopes; seit R8/#582 als TYPED_TEST über alle 6 KV-Dtypes) | A− | Quant-Pfade „characterized" mit ASSERTETEN eingefrorenen Envelopes (Korrektur 06-07: INT4/INT8-Envelopes existierten schon seit PR #534 — der „keine"-Erstbefund war stale). Risiko #6 von 06-04 geschlossen. |
| Paged NVFP4-TC | `test_attention_paged_nvfp4_tc*.cu` | C lokal | Launch+SASS-Guard; Numerik via Offline-Microbench + E2E (synthetische Daten NaN-en beide Pfade — dokumentiert). |
| GGUF-Dequant inkl. IQ4 (#561) | `test_gguf_dequant_ref.cu` | **A** | s. §1. Edge-Cases (d=0, NaN-d, Max-Scale) drin. |
| MMVQ/dp4a | `test_mmvq.cu`, `test_gemm_dp4a.cu` | B (Diagnose) | dp4a-vs-MMVQ ohne harten Threshold; die echte Referenz läuft über `test_gguf_dequant_ref.cu` (GEMV ≤2.5e-2 mit gemessenem Envelope). |
| MoE-Routing | `test_moe.cu` | A− | CPU-top-k-Referenz, aber hartkodierte eindeutige Logits — keine Tie-Cases. Executor-Test = not-NaN. |
| **gpt-oss MXFP4-Experten + sinks (#572)** | — | **C→A für Sinks/Harmony/YaRN (#584)** | Sinks: fp64-Softmax-Ref + Eviction-Geometrie (`test_gpt_oss_sinks_ref.cu`); Harmony: HF-Render-Golden exakt (`test_gpt_oss_harmony_golden.cpp`); YaRN-Langseq fp64 bis 131071 + Inversions-Sensitivität (`test_gpt_oss_yarn_ref.cu`). MXFP4-Konverter: **A geschlossen #576** (`test_gpt_oss_mxfp4_convert_ref.cu`, bit-exakt). |
| GDN/SSM | `test_gdn.cu`, `test_ssm.cu` | A− | CPU-Delta-Rule-Scan-Referenz vorhanden; Toleranz nur implizit (EXPECT_NEAR), Daten synthetisch-benign. |

Hinweis zum Audit-Prompt: „TMA warp-spec grouped GEMM" existiert auf sm_120
nicht (kein TMA-WS/tcgen05); der grouped-GEMM-Pfad ist CUTLASS `mma.sync` —
oben entsprechend bewertet.

## 3. Quant-Correctness: Goldens & Toleranzen

- **`tests/refs/` ist das funktionierende Schema** (Erbe Phase 2): committete
  numpy-Generatoren, bit-exakte Regeneration, Toleranz-Policy mit Begründung
  in `tests/refs/README.md` (f16-Klasse ≤1e-2 rel gemessen ~4e-4; fp8
  characterized-not-blessed; NVFP4 ≤1e-1 + E2E-Locks; Generator-Crosscheck
  ≤1e-9). Drei Konsumenten: crosspath, nvfp4_outlier, e2e_greedy_locks.
- **Lücke (Stand 06-07 weitgehend geschlossen via #576/#584):** grouped GEMM
  und gpt_oss_mxfp4_convert haben jetzt class-A-Referenzen nach der Policy;
  YaRN/Harmony-Goldens committet. Verbleibend: mxf4nvf4-Validate (benigne
  Daten); FP8-FMHA-Fixtures nutzen teils noch das `%13`-Füllmuster, das #525
  vakuös machte.
- GGUF-Dequant-Toleranzen sind dokumentiert UND begründet (Dequant ≤1e-3 =
  reine f32-Rundung; GEMV ≤1e-2; dp4a/MMVQ 2.5e-2 mit Aktivierungs-Quant-
  Rauschen, gemessener Envelope) — Soll-Zustand.

## 4. Determinismus

- **DISABLED_-Inventar (3):** 2× `test_determinism_e2e.cpp`
  (cross-context auf GDN-Hybrid — dokumentierte, berechtigte Grenze aus #542)
  · 1× `test_attention_fmha_mxfp4.cu:141 DISABLED_BasicHD256` („requires
  large shared memory; disabled pending smem optimization" — begründet, aber
  ohne Issue-Ref).
- **Exact-Equal-Asserts sind bewusst und korrekt eingesetzt:** Greedy-Locks
  (`test_e2e_greedy_lock.cpp:170`) laufen 2× fresh-context, sodass ein
  Atomics-Flip selbst der Befund ist; PPL-Bit-Identität ist das Deliverable
  des deterministic mode (#542); prefix-cache fresh-vs-hit-Token-Equality ist
  das Ship-Gate (#538). Chunked-Prefill nutzt seit #553 NLL-Toleranz statt
  Byte-Equality (richtige Reaktion auf Logit-Tie-/Quant-Drift-Klasse).
- **MoE-Atomics:** Qwen3.6-Nondeterminismus ist dokumentiert und über
  `[runtime] deterministic` + DetEvalE2ETest abgedeckt; ein quantifizierender
  „Atomics-Spread-Probe" (N=20, Spread-Bound) fehlt weiterhin (Honorable
  Mention 06-04).
- ~51 GTEST_SKIP-Pfade, alle legitim gemustert (Modell/HW/Build-Feature).
  Keine Seed-Variation in irgendeinem Test (alles seed=42) — bei reinem
  Greedy-Fokus akzeptabel, für Sampling-Numerik ein Loch (s. §1).

## 5. Perf-Tests vs. Messvarianz

- `verify.sh`-Gate: Median aus Trials mit 15 s Cooldown, decode 3 % hart;
  **prefill ist WARN-only** — bewusste und richtige Antwort auf die
  2.6×-Container-Restart-Varianz von cuBLAS. Decode-Tagesdrift (8–15 %,
  #526) wird NICHT erkannt: Das Gate vergleicht gegen die Baseline ohne
  Clock-/Power-Plausibilisierung (`nvidia-smi`-Sampling WÄHREND des Benches,
  healthy = ~2850 MHz SM / 13801 MHz mem / ~500 W) → an gedrückten Tagen
  false-positive Regression möglich.
- **Drei Baseline-Dateien, drei Schemata** (legacy implizit / v1 / north_star
  ohne Versionsfeld); `perf_baseline_chunked.json` 29 Tage alt;
  `north_star` wird von keinem Target außer `verify-north-star` gelesen.
- `tests/api/test_perf_regression.py` gated TTFT-p95/Decode-p50 hart gegen
  Keys `["ttft"]`/`["throughput"]` in `perf_baseline.json` — **die dort gar
  nicht existieren** (verify.sh-Schema). Real-only-markiert, fällt im
  Mock-Lauf raus, würde aber bei realem Lauf gegen fehlende Keys laufen →
  prüfen/reparieren.
- Alle 11 `*Bench*`-GTests (ctest -L perf) sind print-only-Diagnose ohne
  Asserts — okay, solange das so dokumentiert ist; sie sollten nicht den
  Eindruck eines Gates erwecken.

## 6. Gaps vs. bekannte Prioritäten

| Priorität (Prompt) | Stand 06-06 |
|---|---|
| FA2-Prefill-Coverage inkl. legacy cuBLAS-Pfad | **GESCHLOSSEN** — crosspath testet alle 6 Pfade paarweise (A-Klasse); chunked continuation via #553/#570 NLL-/Paritäts-Locks |
| /v1/messages-Streaming real | **OFFEN auf E2E-Ebene.** Seit #564 sind Envelope/Reasoning-Split/anthropic-Transform unit-getestet (gegen die echten Server-Quelldateien, nicht Mock — gut). Echtes Streaming-Verhalten E2E weiterhin nur OpenAI-seitig; Anthropic-SSE-E2E fehlt. |
| Tool-Arg-Schema-Validierung | Teilweise: Pass-through in `test_tools.py`; FSM-Ebene inkl. $ref/$defs in `test_json_constrain.cu` (#562). **API-Ebene rekursives Schema ungetestet.** |
| Constrained output | Gut abgedeckt (whole-token-FSM #517/#519-Kette + #562). |
| **Neu seit 06-04, ungetestet:** | executor_lora-Kernel. [gpt_oss_mxfp4_convert: **geschlossen #576** (bit-exakte fp64-Ref); grouped-vs-fallback-Routing: **geschlossen #577**; attention sinks, Harmony-Parität vs HF, YaRN-Langsequenz: **geschlossen #584**] |
| Offene 06-04-Risiken: | #2 teilweise (mode-2-from-scratch weiter ohne externe Oracle), #6 **geschlossen** (INT4/INT8-paged-Envelopes existierten seit PR #534 — Erstbefund stale; seit R8/#582 TYPED_TEST über alle KV-Dtypes), #9 zurückgestellt (begründet — MTP dead end), #10 teilweise (fault injection ✓, Unicode-Roundtrip ✓ via fixtures+robustness; NUL-Klasse via SSE-Tests ✓). |

## 7. Stale / Dead

| Fund | Beleg | Bewertung |
|---|---|---|
| `tests/golden/*.txt` (5 Dateien) | **null Konsumenten** (grep über tests/scripts/tools/CI); 2× Mistral-Small = gelöschtes Modell; Makefile `test-golden` verweist selbst auf pytest | **tot → löschen** (low-risk) |
| `tests/api/test_outputs/` (21 committete .txt) | Lauf-Artefakte von `test_repetition_compare.sh` (schreibt dorthin), einmalig vor Monaten committet | **Artefakte → löschen + .gitignore** (low-risk) |
| `tests/refs/gen_reference.py` | null Nutzung; Infrastruktur für künftige HF-Layer-Vergleiche | schlafend — behalten, im README als dormant markieren |
| `tests/fixtures/` | **NICHT tot** (Agent-Erstbefund falsch): `scripts/validate_safetensors.py:628,892` konsumiert beide Dateien | behalten |
| Stale-Kommentar „FMHA sm120 requires sm_90+ (WMMA fallback)" | `test_attention_fmha_sm120.cu:84` | irreführend auf sm_120a-only-Engine → fixen (low-risk) |
| Kommentar „same fate as the SM100-only tcgen05 family" (`tests/bench/mmq_q4k_imma_bench.cu:30`) | Agent-Erstbefund „dead commentary" — geprüft: im Kontext korrekt (tcgen05 IST sm_100-only, genau deshalb fehlt es auf sm_120) | behalten |
| `test_engine_chat.sh`, `test_server_concurrent.sh`, `tests/api/elchtest.sh` | von keinem Target/CI aufgerufen | manuelle Werkzeuge → im Kopf der Skripte als manuell dokumentieren, nicht löschen |
| TurboQuant/BitDecoding/FP4-PV-Reste | sauber entfernt; nur Kommentar-Verweis in CMakeLists (dokumentarisch) | ✓ kein Handlungsbedarf |
| Vermeintliche Duplikate (mmq_q4k×5, mxf4nvf4×3, attention_mxfp4×2) | geprüft: Pyramide unit→layout→bench bzw. paged-vs-FMHA | **keine Duplikate** |
| `tests/bench/*.cu` in `IMP_COMPUTE_SOURCES` (CMakeLists ~196–207) | nur unter `IMP_BUILD_TESTS OR IMP_BUILD_BENCH`, kommentiert (P3/P5-Entscheid) | akzeptiert; kein Prod-Leak im Default-Release ohne Tests |

## 8. Laufzeit & Parallelisierbarkeit, GPU/Host-Trennung

- **Split ist auf Binary-Ebene sauber:** 8 Binaries, ctest-Labels
  unit/gpu/perf; unit-Binaries sind reine .cpp ohne Device-Code (cuda_fp16.h
  nur für Host-Konvertierung). CI (kein GPU-Runner) baut + skippt Test-Job;
  ctest läuft ohne `-j` — für GPU-Tests korrekt (VRAM/Context-Konkurrenz),
  für die unit-Label-Teilmenge wäre `-j` gefahrlos.
- **Fragilität (GESCHLOSSEN #580, R5):** der unit/gpu-Split von `test-e2e`
  hing an einem hartkodierten gtest_filter-String in CMakeLists
  (`_unit_e2e_filter`) — Test-Umbenennung verschob Tests still ins falsche
  Label. CPU- und GPU-Tests sind in `test_e2e.cpp`/`test_continuous_batching.cpp`
  verschränkt (StubModelTest teilt eine Fixture über CPU- und GPU-Subtests),
  also kein sauberes eigenes Binary ohne Fixture-Duplizierung → stattdessen
  Filter behalten + Guard-Test `guard_e2e_lane_split`
  (`scripts/check_e2e_lane_split.sh`), der per `--gtest_list_tests` prüft, dass
  der Filter EXAKT die eingefrorene CPU-Menge auflöst (37 Tests) — eine
  Umbenennung schlägt jetzt laut fehl statt still zu verschieben. Außerdem:
  `gtest_discover_tests()` registrierte ALLE Tests nochmal einzeln neben den
  Label-Aggregaten → `ctest` ohne Label führte alles doppelt aus (gemessen:
  1215 ctest-Einträge); entfernt → jetzt 14 Einträge (3 unit + 1 Guard + 6 gpu
  + 4 perf), jeder Test läuft genau einmal.
- **Kein per-Test-Timeout in CMake** (nur CI-global 120 s); lange E2E-Tests
  (Modell-Load + 128 Tokens) blockieren den sequentiellen Lauf.
- **Gemessen (2026-06-06, RTX 5090, ohne Modelle):** Gesamtsuite 1.154 Tests;
  7 von 8 Binaries zusammen < 11 s, aber **test-attention allein 241 s** —
  die Makefile-Annahme „GPU tests < 30s" ist stale. Treiber sind die
  paged-/crosspath-Oracle-Sweeps; ein `-L unit`-Lauf bleibt < 1 s. (Makefile-
  Kommentar `test-gpu` korrigiert, #580; Stand 06-07 sind es ~1.202 Tests.)
- **Python-API-Suite hängt an nichts:** `run_mock_tests.sh` (CPU-fähig,
  Mock) wird weder von CI noch von verify.sh aufgerufen — die einzige
  CPU-CI-fähige Contract-Suite läuft nur manuell.
- Modell-Gating über `IMP_TEST_MODEL*`-Env-Vars, dezentral in den Testdateien
  kopiert. **GESCHLOSSEN #581 (R6):** zentrale Registry `tests/test_models.h`
  (header-only) hält die Env-Var-Namen (`imp_test::kEnv*`) + Accessoren
  (`env_path`/`env_path_or`) an EINER Stelle; die ~14 Konsumenten ziehen den
  Namen jetzt aus der Registry statt aus kopierten String-Literalen (die
  GTEST_SKIP-Aufrufe bleiben am Call-Site, weil GTEST_SKIP aus der
  *aufrufenden* Funktion zurückkehrt). Die hartkodierten
  `/models/...`-Fallback-Pfade (degeneration/api_generate/relaunch/lora/chunked)
  sind KEIN Defekt: sie matchen den Container-Mount `-v $(PWD)/models:/models`
  aus dem Makefile und skippen sauber, wenn die Datei fehlt — bleiben daher als
  call-site-sichtbare Literale erhalten. `test_mtp_forward.cpp` kodierte einen
  Host-Pfad (`/home/kekz/models/...`), der im Container nie existierte → auf
  `/models/...`-Container-Stil normalisiert (#581), skippt jetzt konsistent.

---

## Priorisierung (Impact × Aufwand)

**P1 — fängt echte Bugs, moderater Aufwand**
1. `gpt_oss_mxfp4_convert.cu`-Referenztest (Format-Spez-fp64 wie
   `test_gguf_dequant_ref.cu`; MXFP4-Nibble-Order war GERADE ein realer Bug
   im Issue-Sweep #560-Familie!). Aufwand: S–M.
2. CUTLASS-grouped-GEMM gegen unabhängige Referenz (per-expert-fp32-CPU oder
   dense-cuBLAS-fp16-Pfad statt desselben Adapters). Der neue 10×-Prefill-
   Pfad verdient mehr als B-Klasse. Aufwand: M.
3. Paged-INT4/INT8-Oracle nach dem Muster von `test_attention_paged_oracle.cu`
   (Methodik existiert, nur Dtypes ergänzen). Aufwand: S–M.
4. `attention_dispatch`-Routing-Tabellen-Test ((hd,seq,dtype)→Pfad, reine
   Host-Logik; #493 war genau diese Klasse). Aufwand: S.
5. `test_perf_regression.py`-Baseline-Key-Mismatch klären (greift ins Leere
   oder bricht beim ersten Real-Lauf). Aufwand: S.

**P2 — Velocity/Robustheit**
6. Python-Mock-Suite in CI verdrahten (`pytest -m "not perf and not tools"`,
   CPU-only — der einzige CI-fähige Contract-Check). Aufwand: S.
7. ~~Attention-sinks-Unit (Sink-Logit-Shift vs CPU-Softmax-Referenz) +
   Harmony-Template-Golden vs HF-Tokenizer-Output.~~ **DONE (P2.7, #584):**
   `test_gpt_oss_sinks_ref.cu` (gpt-oss Sink-Logit vs fp64-Softmax-Referenz,
   inkl. StreamingLLM-Slot-Eviction-Geometrie), `test_gpt_oss_harmony_golden.cpp`
   (imp-Jinja-Render vs HF `apply_chat_template`-Golden, exakt), plus
   `test_gpt_oss_yarn_ref.cu` (YaRN-Langsequenz-Parität bis pos 131071 vs fp64,
   sensitiv auf die #547 rope_freq_scale-Inversion). Generatoren+Goldens in
   `tests/refs/` (gen_harmony_golden.py, gen_yarn_rope_golden.py).
8. Decode-Gate um Clock-/Power-Plausibilisierung ergänzen (#526-Klasse:
   bei mem-clock < 13801 MHz WARN statt FAIL). Aufwand: S.
9. ~~unit_e2e_filter aus Test-Namen-Kopplung lösen + doppelte
   ctest-Registrierung bereinigen.~~ **DONE (R5, #580):** Filter behalten +
   Guard-Test `guard_e2e_lane_split` (rename-fest via `--gtest_list_tests`-
   Abgleich gegen die eingefrorene CPU-Menge); `gtest_discover_tests` entfernt
   (1215→14 ctest-Einträge, kein Doppellauf mehr).

**P3 — Hygiene (low-risk, sofort ausführbar)**
10. `tests/golden/` löschen (tot), `tests/api/test_outputs/` löschen +
    .gitignore, irreführende Skip-Message in `test_attention_fmha_sm120.cu`
    fixen, Manual-Skripte als manuell markieren,
    `tests/refs/gen_reference.py` als dormant markieren.

**Nicht tun:** Spec-decode-Exactness (#9 von 06-04) bleibt zurückgestellt
(MTP = bewiesener Dead End bei aktueller Präzision); Bench-Tests nicht in
Gates verwandeln (Varianz); keine Seed-Sweeps für Greedy-Pfade.

---

## Phase 2 — Refactor-Plan (Vorschlag, pro Item Risiko/Aufwand)

| # | Item | Risiko | Aufwand |
|---|---|---|---|
| R1 | **Referenz-First für neue Hot-Paths:** P1.1–P1.3 als ein Paket „class-A-Anker für #572/#574-Pfade" — Generatoren nach `tests/refs/`-Schema (committet, bit-exakt, Toleranz begründet) | niedrig (nur neue Tests) | M |
| R2 | **Routing-/Host-Logik-Units:** P1.4 attention_dispatch + grouped-vs-fallback-Entscheid in `executor_forward` als CPU-Tests (kein GPU nötig → CI-Lane) | niedrig | S |
| R3 | **CI-Lane Python-Mock** (P2.6): neuer CI-Step nach Build, `pytest tests/api -m "not perf and not tools"` gegen mock_server; läuft ohne GPU | niedrig (CI-only) | S |
| R4 | **Perf-Gate härten** (P1.5 + P2.8): test_perf_regression-Keys reparieren oder Suite auf verify.sh-Schema umstellen; verify.sh sampelt clocks.mem/power während des Benches und degradiert FAIL→WARN bei depressed-host-Signatur; Baseline-Dateien bekommen einheitliches `schema_version`-Feld | mittel (Gate-Semantik ändert sich — Abstimmung, weil CI-Verhalten betroffen) | M |
| R5 | **test-e2e-Split entkoppeln** (P2.9): Stub-Unit-Tests in eigenes Binary; Label-Aggregate behalten; gtest_discover_tests-Doppelregistrierung auf Label-Aggregate reduzieren — **ERLEDIGT (#580):** eigenes Binary verworfen (CPU/GPU-Tests teilen Fixtures, nicht trennbar ohne Duplizierung); stattdessen Filter behalten + Guard `guard_e2e_lane_split` (`scripts/check_e2e_lane_split.sh`, rename-fest), `gtest_discover_tests` entfernt (1215→14 ctest-Einträge, kein Doppellauf), Makefile-`<30s`-Kommentar gefixt | mittel (Runner-Umbau, Makefile/CI/verify.sh-Pfade anfassen) | M |
| R6 | **Modell-Env-Registry:** ein `tests/test_models.h` (Header-only) mit den Env-Vars + Accessoren statt kopiertem Muster; mechanische Migration — **ERLEDIGT (#581):** `tests/test_models.h` (`imp_test::kEnv*` + `env_path`/`env_path_or`), ~14 Dateien migriert (Env-Namen aus Registry, SKIP bleibt am Call-Site, /models-Fallbacks erhalten), `test_mtp_forward` Host-Pfad → Container-Pfad normalisiert | niedrig (mechanisch, semantikerhaltend) | M (Fleißarbeit) |
| R7 | **Hygiene-Batch** (P3.10) | minimal | S |
| R8 | **Parametrisierung Quant×Arch** (Prompt-Wunsch): NICHT als große Matrix empfohlen — die Greedy-Lock-/NLL-E2E-Suite parametrisiert bereits über die real vorhandenen Modelle, und eine synthetische Arch-Matrix (LLaMA/Mistral/DeepSeek/…) ohne Gewichte testet nur Loader-Pfade, die `test_e2e_models`/Loader-Tests schon abdecken. Stattdessen: TYPED_TEST über KV-Dtypes im paged-Oracle (R1) und über Quant-Formate im Dequant-Ref — dort ist Parametrisierung billig und echt. | niedrig | S–M |
| R9 | **Vision-GPU-Golden** (einziger komplett blinder GPU-Bereich): ein eingefrorenes SigLIP-Encoder-Golden (kleines committetes Bild → Projector-Output-Spots, Toleranz f16-Klasse) — **ERLEDIGT (#583)**: `tests/test_vision_golden.cu` + `tests/refs/vision_encoder_golden.h`, deckt SigLIP **und** gemma4v ab (committetes 64² PNG, mmproj standalone ohne LM), ≤1e-2 rel + 5e-3 abs + NaN/Inf-Guard, sauberer SKIP ohne Modell, `make test-vision` (Dump-Mode regeneriert) | niedrig | M |

Empfohlene Reihenfolge: R7 (sofort) → R2+R3 (billig, CI-Wirkung) → R1 (Kern) →
R4 → R6 → R5 → R9. R8 in R1 integrieren.

— Phase 1+2 Ende. Phase 3 nur nach Freigabe; als low-risk markiert und damit
ausführbar ohne weitere Freigabe: **R7 (Hygiene-Batch)**.
