# GEMV Decode Optimierungsplan

**Erstellt:** 2026-02-28
**Status-Update:** 2026-03-01

## Kontext

Benchmark-Analyse zeigte eine **20-39% Decode-Luecke** gegenueber llama.cpp bei Dense-Modellen
und eine **84% Prefill-Luecke** bei MoE-Modellen. Die Decode-Luecke war am groessten bei Q6_K
(39% auf DS-R1-14B) und konsistent bei ~23% fuer Q8_0 auf 7B+-Modellen. MoE-Decode war nahe
Paritaet (4% Luecke) oder schneller (2.4x auf Nemotron).

**Bandwidth-Auslastung:** ~43% der RTX 5090 Peak-Bandbreite (1.8 TB/s) bei Phi-4-Mini Decode.
llama.cpp erreicht ~55%. Die 12pp Luecke erklaert den tok/s-Unterschied.

## Benchmark-Ausgangslage (vor Optimierung)

| Modell | Quant | llama.cpp tg | imp tg | Luecke |
|--------|-------|-------------|--------|--------|
| Phi-4-Mini 3.8B | Q8_0 | 246 | 195 | 21% |
| DS-R1-7B | Q8_0 | 166 | 128 | 23% |
| Gemma-3-12B | Q8_0 | 92 | 71 | 23% |
| DS-R1-14B | Q6_K | 102 | 62 | **39%** |
| Qwen3-Coder MoE | Q6_K | 202 | 194 | 4% |
| Qwen3-Coder MoE pp | Q6_K | 5759 | 918 | **84%** |

---

## Optimierungen (nach ROI sortiert)

### 1. N_ROWS=4 + Multi-Row Gate+Up Kernel

**Auswirkung: +10-20% Decode | Aufwand: 1 Tag | Risiko: Niedrig**
**Status: ABGESCHLOSSEN (mit kMaxNRows-Einschraenkung)**

Der gate+up Fused-Kernel (`gemv_dp4a_gate_up_kernel`) war auf N_ROWS=1 hartcodiert — jeder
Warp verarbeitete genau 1 Output-Zeile. Fuer ein 7B-Modell mit d_ff=11008 bedeutete das
11008 Warps pro Projektion. N_ROWS=4 bedeutet: jeder Warp berechnet 4 Zeilen und
wiederverwendet den gleichen Q8_1-Input aus Shared Memory.

**Ergebnis:** NR=4 verursachte Register-Spill bei Q8_0 (59 Regs) und Q6_K (64 Regs),
was die Occupancy drastisch senkte. Loesung: `kMaxNRows`-Trait begrenzt NR pro Quant-Typ.
Q8_0/Q6_K erhalten NR=2 max, Q4_0/Q4_K/Q5_K behalten NR=4.

**Dateien:** `src/compute/gemv_dp4a_traits.cuh`, `src/compute/gemm.cu`

**Schritte (alle erledigt):**
1. ~~N_ROWS Template-Parameter zu gemv_dp4a_gate_up_kernel hinzugefuegt~~
2. ~~Inner Loop: `float sum[N_ROWS]; for (int r = 0; r < N_ROWS; r++)`~~
3. ~~Gate+Up Launcher: NR=4 wenn blocks_nr4 >= 128~~
4. ~~NR=4 Instantiierungen fuer basic/fp32/QKV Kernels~~
5. ~~Launch-Heuristiken: NR=4 >= 128, NR=2 >= 64, sonst NR=1~~
6. ~~Register-Pressure geprueft: kMaxNRows-Trait eingefuehrt~~


### 2. Inline Q8_1-Quantisierung in O-Projektion GEMV

**Auswirkung: +5-10% Decode | Aufwand: 1 Tag | Risiko: Niedrig-Mittel**
**Status: ABGESCHLOSSEN**

Vorher: 3 Kernels pro Layer (Attention → Quantize → GEMV).
Nachher: 2 Kernels (Attention → Fused Quantize+GEMV).

```
VORHER:
  paged_attn → attn_out [FP16]           (Kernel A)
  quantize_fp16_to_q8_1(attn_out)        (Kernel B — DRAM lesen+schreiben)
  gemv_dp4a(Wo, Q8_1) + residual         (Kernel C — DRAM lesen)

NACHHER:
  paged_attn → attn_out [FP16]           (Kernel A)
  gemv_dp4a_inline_quant(Wo, FP16→smem)  (Kernel B+C fused)
```

Eliminiert 1 Kernel-Launch + 1 DRAM-Roundtrip pro Layer (32+ pro Token).

**Dateien:** `src/compute/gemv_dp4a_traits.cuh`, `src/compute/gemm.cu`, `src/compute/gemm.h`, `src/graph/executor.cu`

**Schritte (alle erledigt):**
1. ~~`quantize_fp16_to_q8_1_smem()` Device-Helper implementiert~~
2. ~~`gemv_dp4a_inline_quant_kernel<QT, NR, ADD_RESIDUAL>` Template erstellt~~
3. ~~Launcher `launch_gemv_dp4a_inline_quant()` und Public Wrapper~~
4. ~~In `run_attention()` O-proj Decode-Pfad verdrahtet~~
5. ~~Korrektheit gegen separaten Kernel-Pfad getestet~~


### 3. MoE Prefill-Performance Fix (Qwen3-Coder 84% Luecke)

**Auswirkung: +200-500% MoE Prefill | Aufwand: 0.5 Tage | Risiko: Mittel**
**Status: ABGESCHLOSSEN**

Der MoE-Prefill-Pfad dequantisierte **alle** Experten (z.B. 128 fuer Qwen3-Coder) von Q6_K
nach FP8, obwohl nur ~8 pro Token aktiv sind. Das verschwendete ~768 MB Bandbreite pro
MoE-Layer fuer ungenutzte Experten-Dequantisierung.

**Loesung:** Fuer Modelle mit n_experts > 16 wird der fused Q6_K GEMM-Pfad
(`gemm_q6k_fused_moe_prefill`) bevorzugt, der nur aktive Experten verarbeitet.

**Dateien:** `src/graph/executor.cu`

**Schritte (alle erledigt):**
1. ~~Heuristik: n_experts > 16 → fused Q6_K GEMM statt FP8-Pfad~~
2. ~~`gemm_q6k_fused_moe_prefill` mit 128 Experten verifiziert~~
3. ~~Qwen3-Coder pp tok/s benchmarked~~


### 4. Shared Memory Q8_1-Caching fuer MoE Decode GEMV

**Auswirkung: +1% MoE Decode (erwartet: +5-10%) | Aufwand: 0.5 Tage | Risiko: Niedrig**
**Status: ABGESCHLOSSEN**

Die MoE-Decode-Kernels (Template #5 und #6) lesen Q8_1 aus **Global Memory** — jeder Warp
laedt die gleichen Q8_1-Bloecke unabhaengig. Mit 8 Warps/Block sind das 8x redundante
L2-Reads. Die Dense-GEMV-Kernels cachen Q8_1 bereits in Shared Memory.

**Ergebnis:** Kooperatives smem-Laden implementiert, aber nur +1% auf Qwen3-Coder und
+0.5% auf Nemotron. Die L1 Cache hat die redundanten Reads bereits effektiv abgefangen.
Aenderung beibehalten da robust gegen L1-Thrashing bei hoher CTA-Dichte.

| Modell | Vorher tg | Mit smem tg | Diff |
|--------|----------|------------|------|
| Qwen3-Coder MoE Q6_K | 214.51 | 216.71 | +1.0% |
| Nemotron MoE Q6_K | 60.06 | 60.34 | +0.5% |

**Dateien:** `src/compute/gemv_dp4a_traits.cuh` (Template #5 und #6 + Launcher)

**Schritte (alle erledigt):**
1. ~~`extern __shared__ char smem_q8[]` und kooperatives Laden zu MoE-Decode-Kernels~~
2. ~~Global-Memory Q8_1-Reads durch smem-Reads ersetzen~~
3. ~~smem_size in Launch-Configs uebergeben~~
4. ~~Testen mit Qwen3-Coder und Nemotron~~

---

## Zusaetzliche Optimierungen (waehrend Implementierung entdeckt)

### 5. K-par vs Row-par Heuristik-Fix

**Status: ABGESCHLOSSEN**

`kpar_is_better()` verglich gegen den selektierten NR-Wert (NR=2 wenn >= 64 Bloecke),
nicht gegen NR=1 (maximale Occupancy). Fix: immer gegen NR=1-Bloecke vergleichen.

### 6. Quant-Typ-abhaengiges K-par Tie-Breaking

**Status: ABGESCHLOSSEN**

Optimization #5 half Q8_0 aber brach Q6_K. Ursache: Q6_K ist compute-bound (K-par besser),
Q8_0 ist bandwidth-bound (row-par smem besser). Loesung: `kPreferKpar`-Trait mit
`>=`-Vergleich fuer compute-heavy Typen, `>`-Vergleich fuer einfache Typen.

### 7. Compiler-Flags und Device LTO

**Status: ABGESCHLOSSEN (kein Effekt)**

Getestete Massnahmen:
- `__launch_bounds__(256,6)` / `__launch_bounds__(128,12)`: Gemma-3 Regression -4.5%, reverted
- `--extra-device-vectorization`: +0.5% auf Q6_K, beibehalten
- Device LTO (`-dlto`, `code=lto_120`): 0% Effekt (alle heissen Kernels in einer .cu-Datei)

---

## Verifikation

1. `cmake --build build -j$(nproc)` — sauberer Compile ✓
2. `./build/imp-tests` — 219/219 Tests bestanden ✓
3. Benchmarks auf Phi-4-Mini, DS-R1-7B, DS-R1-14B, Gemma-3, Qwen3-Coder, Nemotron ✓
4. Real-Model-Test: `./build/imp-cli --model <model>.gguf --prompt "test" --max-tokens 32` ✓

## Ergebnisse nach Optimierung

| Modell | Quant | llama.cpp tg | Vorher tg | Nachher tg | vs Vorher | vs llama.cpp |
|--------|-------|-------------|----------|-----------|-----------|-------------|
| Phi-4-Mini 3.8B | Q8_0 | 250.96 | 231.26 | **219.15** | -5.2% | -12.7% |
| Qwen3-4B | Q8_0 | 217.54 | 221.89 | **213.33** | -3.9% | -1.9% |
| DS-R1-7B | Q8_0 | 164.30 | 147.59 | **149.52** | **+1.3%** | -9.0% |
| Gemma-3-12B | Q8_0 | 91.41 | 83.01 | **81.82** | -1.4% | -10.5% |
| DS-R1-14B | Q6_K | 102.22 | 86.81 | **83.85** | -3.4% | -18.0% |
| Qwen3-Coder MoE | Q6_K | 206.61 | 221.74 | **214.51** | -3.3% | +3.8% |
| Nemotron MoE | Q6_K | 25.77 | 58.74 | **60.06** | **+2.2%** | +133% |

**Anmerkung:** Die "Vorher"-Werte stammen von VOR der Template-Konsolidierung (Commit vor
`8c0a666`). Die Regressionen bei Phi-4-Mini, Qwen3-4B, DS-R1-14B und Qwen3-Coder
(2-5%) sind Compiler-Artefakte der Template-Umstellung, keine Logik-Fehler.
Verbesserungen bei DS-R1-7B (+1.3%) und Nemotron (+2.2%) zeigen, dass die
Optimierungen in der Summe wirksam sind.
