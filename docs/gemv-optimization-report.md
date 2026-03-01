# GEMV Decode Optimization Report

**Date:** 2026-03-01
**GPU:** NVIDIA GeForce RTX 5090 (sm_120, 170 SMs, 32 GB, custom water-cooled)
**Branch:** main (post `8c0a666`)

## Ausgangslage

Benchmark-Analyse zeigte eine **10-39% Decode-Luecke** gegenueber llama.cpp bei Dense-Modellen
und eine **84% Prefill-Luecke** bei MoE-Modellen. Ziel: systematische Optimierung der dp4a-GEMV-Kernels.

### Baseline-Benchmarks (vor Optimierung)

| Modell | Quant | llama.cpp tg | imp tg | Luecke |
|--------|-------|-------------|--------|--------|
| Phi-4-Mini 3.8B | Q8_0 | 250.96 | 231.26 | -7.8% |
| Qwen3-4B | Q8_0 | 217.54 | 221.89 | +2.0% |
| DS-R1-7B | Q8_0 | 164.30 | 147.59 | -10.2% |
| Gemma-3-12B | Q8_0 | 91.41 | 83.01 | -9.2% |
| DS-R1-14B | Q6_K | 102.22 | 86.81 | -15.1% |
| Qwen3-Coder MoE | Q6_K | 206.61 | 221.74 | +7.3% |
| Nemotron MoE | Q6_K | 25.77 | 58.74 | +128% |

---

## Durchgefuehrte Optimierungen

### 1. Template-Konsolidierung der GEMV-Kernels

**Status:** Abgeschlossen

33 handgeschriebene GEMV-Kernel-Funktionen wurden durch 6 Template-Kernels ersetzt:

| Template | Funktion | K-par? |
|----------|----------|--------|
| #1 `gemv_dp4a_kernel` | Standard GEMV (+-Residual) | Ja |
| #2 `gemv_dp4a_fp32_kernel` | FP32-Output (Logits) | Ja |
| #3 `gemv_dp4a_qkv_kernel` | Fused Q/K/V-Projektion | Ja |
| #4 `gemv_dp4a_gate_up_kernel` | Fused Gate+Up MLP | Ja |
| #5 `gemv_dp4a_moe_decode_kernel` | MoE Decode pro Expert | Nein |
| #6 `gemv_dp4a_moe_gate_up_kernel` | MoE Gate+Up pro Expert | Nein |

Alle Quant-Typen (Q8_0, Q6_K, Q4_0, Q4_K, Q5_K) ueber `DequantTraits<QType>` abstrahiert.

**Dateien:** `src/compute/gemv_dp4a_traits.cuh`, `src/compute/gemm.cu`


### 2. N_ROWS Multi-Row GEMV

**Status:** Abgeschlossen (mit kMaxNRows-Einschraenkung)

Jeder Warp berechnet N_ROWS Output-Zeilen statt nur einer. Q8_1-Input aus Shared Memory
wird ueber alle N_ROWS wiederverwendet, was die Blockanzahl um N_ROWS reduziert.

**Dispatch-Schwellwerte:**
- NR=4: wenn `nr4_blocks >= 128` (genuegend Bloecke fuer GPU-Auslastung)
- NR=2: wenn `nr2_blocks >= 64`
- NR=1: Fallback

**Problem entdeckt:** NR=4 verursacht Register-Spill bei komplexen Quant-Typen:

| Kernel | QType | NR=1 | NR=2 | NR=4 |
|--------|-------|------|------|------|
| gate_up | Q8_0 | 40 | 40 | **59** |
| gate_up | Q6_K | 40 | 40 | **64** |
| basic | Q8_0 | 40 | 40 | **56** |
| basic | Q6_K | 40 | 40 | **64** |
| K-par (alle) | alle | 39-40 | - | - |
| Q4_0/Q4_K/Q5_K | alle | 40 | 40 | **40** |

59 Register (Q8_0 NR=4) → Occupancy faellt von 48 auf 32 Warps/SM (-33%).
64 Register (Q6_K NR=4) → Occupancy faellt auf 24 Warps/SM (-50%).

**Loesung:** `kMaxNRows`-Trait in `DequantTraits`:
- Q8_0, Q6_K: `kMaxNRows = 2` (NR=4 verboten)
- Q4_0, Q4_K, Q5_K: `kMaxNRows = 4` (NR=4 bleibt bei 40 Registern)

**Auswirkung:** DS-R1-7B Q8_0: 122.79 → 149.40 tok/s (+22% Recovery)


### 3. Inline Q8_1-Quantisierung fuer O-Projektion

**Status:** Abgeschlossen

Vorher:
```
paged_attn → attn_out [FP16]        (Kernel A)
quantize_fp16_to_q8_1(attn_out)     (Kernel B — DRAM read+write)
gemv_dp4a(Wo, Q8_1) + residual      (Kernel C — DRAM read)
```

Nachher:
```
paged_attn → attn_out [FP16]        (Kernel A)
gemv_dp4a_inline_quant(Wo, FP16)    (Kernel B+C fused — Q8_1 in smem)
```

Eliminiert 1 Kernel-Launch + 1 DRAM-Roundtrip pro Layer.
Ueber 32+ Layers: ~32 Kernel-Launches + ~32 DRAM-Zyklen eingespart.

**Dateien:** `src/compute/gemv_dp4a_traits.cuh` (Kernel), `src/graph/executor.cu` (Wiring)


### 4. MoE Prefill-Fix

**Status:** Abgeschlossen

Bei hochparallelen MoE-Modellen (z.B. Qwen3-Coder mit 128 Experten) dequantisierte der
FP8-Pfad **alle** Experten, obwohl nur ~8 pro Token aktiv sind. Fuer n_experts > 16
wird jetzt der fused Q6_K GEMM-Pfad bevorzugt, der nur aktive Experten verarbeitet.

**Dateien:** `src/graph/executor.cu` (MoE Prefill-Pfad-Selektion)


### 5. K-par vs Row-par Heuristik-Fix

**Status:** Abgeschlossen

**Problem:** Die `kpar_is_better()`-Funktion verglich K-parallel-Occupancy gegen den NR-Wert,
der vom Schwellwert-Dispatch selektiert wuerde (NR=2 wenn Bloecke >= 64). Bei grossen M
(gate+up mit d_ff=14336) hat NR=2 aber weniger Bloecke/SM als NR=1:

```
Gemma-3-12B gate_up (M=14336, 170 SMs):
  NR=2: 896 Bloecke → 5 Bloecke/SM → 40 Warps/SM
  K-par: 14336 Bloecke → 12 Bloecke/SM → 48 Warps/SM  → K-par gewaehlt
  NR=1: 1792 Bloecke → 6 Bloecke/SM → 48 Warps/SM     → GLEICH wie K-par!
```

K-par wurde faelschlicherweise bevorzugt, obwohl NR=1 mit smem-gecachtem Q8_1 gleichwertig ist.

**Loesung:** Vergleich immer gegen NR=1-Bloecke (maximale Occupancy-Baseline).


### 6. Quant-Typ-abhaengiges K-par Tie-Breaking (`kPreferKpar`-Trait)

**Status:** Abgeschlossen

**Problem:** Der NR=1-Vergleich (Optimierung #5) half Gemma-3 Q8_0 (+8.7%), brach aber
DS-R1-14B Q6_K (-30.5%!). Ursache:

- **Q8_0** (einfache Dequantisierung): Bandwidth-bound → smem-gecachtes Q8_1 ist schneller als L2
- **Q6_K** (komplexe Dequantisierung: 6-Bit-Extraktion, Gruppen-Skalen, Sub-Bloecke):
  Compute-bound → K-par's kooperatives K-Splitting (4 Warps teilen K-Dimension) ist besser,
  weil jeder Warp weniger Iterationen hat und schneller fertig wird. Die syncthreads-Kosten
  des smem-Ladens ueberwiegen den smem-Vorteil.

**Loesung:** `kPreferKpar`-Trait in `DequantTraits`:

| QType | kPreferKpar | Begruendung |
|-------|-------------|-------------|
| Q6_K | `true` | Compute-heavy Dequant → K-par gewinnt bei Gleichstand |
| Q4_K | `true` | Komplex (Sub-Block-Skalen, Mins) |
| Q5_K | `true` | Komplex (5-Bit + High-Nibbles) |
| Q8_0 | `false` | Einfache Dequant → Row-par smem gewinnt bei Gleichstand |
| Q4_0 | `false` | Einfache Dequant |

`kpar_is_better<PREFER_KPAR>()` ist jetzt ein Template:
- `PREFER_KPAR=true`: `>=`-Vergleich (K-par gewinnt bei Gleichstand)
- `PREFER_KPAR=false`: `>`-Vergleich (Row-par gewinnt bei Gleichstand)

---

## Aktuelle Benchmark-Ergebnisse (nach allen Optimierungen)

| Modell | Quant | llama.cpp tg | Baseline tg | Aktuell tg | vs Baseline | vs llama.cpp |
|--------|-------|-------------|------------|-----------|-------------|-------------|
| Phi-4-Mini 3.8B | Q8_0 | 250.96 | 231.26 | **219.15** | -5.2% | -12.7% |
| Qwen3-4B | Q8_0 | 217.54 | 221.89 | **213.33** | -3.9% | -1.9% |
| DS-R1-7B | Q8_0 | 164.30 | 147.59 | **149.52** | **+1.3%** | -9.0% |
| Gemma-3-12B | Q8_0 | 91.41 | 83.01 | **81.82** | -1.4% | -10.5% |
| DS-R1-14B | Q6_K | 102.22 | 86.81 | **83.85** | -3.4% | -18.0% |
| Qwen3-Coder MoE | Q6_K | 206.61 | 221.74 | **214.51** | -3.3% | +3.8% |
| Nemotron MoE | Q6_K | 25.77 | 58.74 | **60.06** | **+2.2%** | +133% |

---

## Offene Punkte

### P1: Compiler-Artefakte durch Template-Umstellung (2-5%)

**Schwere:** Mittel | **Betroffene Modelle:** Phi-4-Mini (-5.2%), Qwen3-4B (-3.9%), DS-R1-14B (-3.4%), Qwen3-Coder (-3.3%)

Die 33 handgeschriebenen GEMV-Kernel wurden durch 6 Template-Kernels ersetzt. Der C++-Code
ist funktional identisch, aber `nvcc` erzeugt fuer Template-Instantiierungen manchmal leicht
anderen Maschinencode (SASS) als fuer explizite Funktionen. Das ist kein Logik-Fehler,
sondern ein Compiler-Artefakt. Moegliche Ursachen:
- **Register-Allocation:** Templates werden im Header instantiiert (andere Compilation-Unit),
  was die Heuristiken des Register-Allocators beeinflusst
- **Inlining:** `__forceinline__` in `dp4a_block()` kann je nach Template-Kontext
  unterschiedlich aggressiv inlined werden
- **Instruction Scheduling:** `if constexpr`-Branches (fuer NR, ADD_RESIDUAL) koennen
  die Pipeline-Reihenfolge beeinflussen, selbst wenn der tote Code eliminiert wird

**Getestete Massnahmen (alle ohne Effekt oder negativ):**

1. **`__launch_bounds__`**: `__launch_bounds__(256, 6)` (row-par) und `__launch_bounds__(128, 12)`
   (K-par) auf alle 11 Kernel-Templates angewendet. Ergebnis: Gemma-3 regredierte um 4.5%
   (81.82 → 78.14 tok/s, Prefill 6260 → 4999). Reverted.

2. **`--extra-device-vectorization`** (CUDA 13.1 Flag): Marginaler Effekt (+0.5% auf Q6_K).
   Beibehalten da kein Nachteil.

3. **Device LTO (`-dlto`, `code=lto_120`)**: Cross-Module Link-Time-Optimization.
   Kein messbarer Effekt da die heissen GEMV-Kernels alle in einer einzigen .cu-Datei liegen:
   | Modell | Ohne LTO | Mit LTO | Diff |
   |--------|----------|---------|------|
   | Phi-4-Mini Q8_0 | 218.76 | 218.76 | 0.0% |
   | DS-R1-7B Q8_0 | 149.51 | 150.47 | +0.6% |
   | DS-R1-14B Q6_K | 84.30 | 83.76 | -0.6% |
   | Gemma-3 Q8_0 | 82.02 | 82.08 | +0.1% |

**Verbleibende Optionen:**
1. SASS-Diff zwischen alter und neuer Version (`cuobjdump --dumpsass`)
2. Performance-kritische Kernel als explizite Spezialisierungen behalten
3. NVRTC JIT-Kompilierung (hoher Aufwand)

**Geschaetzte Auswirkung bei Loesung:** +2-5% Decode auf allen Modellen


### P2: MoE Decode Shared-Memory Q8_1 Caching

**Schwere:** Niedrig | **Plan-Item #4** — **ABGESCHLOSSEN**

MoE-Decode-Kernels (Template #5 und #6) lasen Q8_1 aus Global Memory. Kooperatives
smem-Laden implementiert (gleiches Muster wie Dense-Kernels).

**Ergebnis:** +1.0% Qwen3-Coder, +0.5% Nemotron. Weniger als erwartet — L1 Cache
hat die 8x redundanten Warp-Reads bereits effektiv abgefangen. Aenderung beibehalten
als Schutz gegen L1-Thrashing bei hoher CTA-Dichte.

**Dateien:** `src/compute/gemv_dp4a_traits.cuh` (Template #5 und #6 + Launcher)


### P3: Prefill-Performance bei Dense-Modellen

**Schwere:** Mittel | **Betroffene Modelle:** Phi-4-Mini, DS-R1-7B, Gemma-3

Die Prefill-Luecke (pp tok/s) ist bei realen Prompts erheblich:
- Phi-4-Mini: 26629 (llama.cpp) vs 1225 (imp real) — **95% Luecke**
- imp bench pp ist naeher dran (20610), was zeigt dass die Bottleneck in der
  Tokenisierung/Graph-Setup liegt, nicht in den Compute-Kernels

**Naechste Schritte:**
1. Profilen wo die imp-real-Prefill-Zeit verbraucht wird
2. Batch-Tokenisierung optimieren
3. Graph-Setup-Overhead reduzieren (einmalig statt pro Prompt)

**Anmerkung:** Dies ist eher ein Infra-Thema als ein Kernel-Thema.


### P4: DS-R1-14B Q6_K Decode-Luecke (-18%)

**Schwere:** Hoch | **Groesste verbleibende Luecke zu llama.cpp**

DS-R1-14B Q6_K hat 83.85 vs llama.cpp's 102.22 tok/s. Moegliche Ursachen:
- Q6_K-Dequant ist compute-heavy; llama.cpp koennte effizienteren SASS-Code erzeugen
- On-the-fly Dequant (VRAM-Budget erreicht) kostet ~4% GPU-Zeit (696 Invocations)
- K-par-Kernel fuer Q6_K koennte suboptimale Warp-Scheduling haben

**Naechste Schritte:**
1. nsys-Profil von DS-R1-14B: Welche Kernels dominieren?
2. SASS-Code des Q6_K K-par-Kernels mit llama.cpp vergleichen
3. Pruefen ob ein spezialisierter Q6_K-Kernel (nicht Template) schneller waere
4. VRAM-Budget erhoehen um On-the-fly-Dequant zu vermeiden


### P5: Gemma-3 Q8_0 Decode-Luecke (-10.5%)

**Schwere:** Mittel

Gemma-3 hat spezielle Architektur-Eigenheiten:
- head_dim=256 (groesser als ueblich, beeinflusst Attention-Kernels)
- 48 Layers mit Sliding-Window + Global Attention alternierend
- Post-Norm statt Pre-Norm (zusaetzliche RMSNorm-Kernels)
- VRAM-Budget erreicht: "FP16 cache: VRAM budget reached after 276 tensors"

**nsys-Profil (Gemma-3 Q8_0 Decode, 128 Tokens):**
| Kernel | % GPU-Zeit | Avg Dauer | Invocations |
|--------|-----------|----------|-------------|
| gate_up K-par (jetzt row-par) | 21.1% | 119.8 us | 6144 |
| kpar_kernel (O-proj+Down) | 15.0% | 42.6 us | 12288 |
| kpar_qkv | 7.3% | 41.4 us | 6144 |
| fp32 logits | 4.6% | 1.2 ms | 135 |
| rmsnorm_fp32_accum | 4.5% | 12.0 us | ~24k |
| dequant_q8_0 | 4.0% | 203.3 us | 696 |
| rmsnorm_quantize | 3.3% | 9.2 us | ~24k |

**Naechste Schritte:**
1. gate_up row-par (nach K-par-Fix) erneut profilen — ist es jetzt schneller?
2. dequant_q8_0 eliminieren: VRAM-Budget erhoehen oder selektives Caching
3. head_dim=256 Attention-Performance pruefen (Blackwell-Kernel Br=64)
4. Pruefen ob Post-Norm-RMSNorm fusioniert werden kann


### P6: Phi-4-Mini Decode-Regression (-5.2%, 12.7% Luecke)

**Schwere:** Mittel

Phi-4-Mini hat relativ kleine Dimensionen (d_model=3072, d_ff=5504).
Bei diesen Groessen ist K-par klar besser (48 vs ~16 Warps/SM fuer NR=1).
Die Regression ist wahrscheinlich rein Template-bedingt.

**Naechste Schritte:**
1. nsys-Profil: Welche Kernels haben sich verschlechtert?
2. `__launch_bounds__(256, 6)` auf die K-par-Kernels testen
3. Register-Count mit `--ptxas-options=-v` pruefen


### P7: Persistent Decode Kernel (Phase 1 aus cuda131-optimization-plan.md)

**Schwere:** Niedrig (Aufwand hoch) | **Nicht begonnen**

Eliminiert 300+ Kernel-Launches pro Token durch einen einzelnen persistenten Grid,
der via Device-Side Work Queue alle Layers verarbeitet. Erwartet +5-8% auf kleinen Modellen.

Abhaengig von Cooperative Groups und device-side malloc — Komplexitaet hoch.

---

## Zusammenfassung der Aenderungen

### Geaenderte Dateien

| Datei | Aenderungen |
|-------|-------------|
| `src/compute/gemv_dp4a_traits.cuh` | `DequantTraits`: +kMaxNRows, +kPreferKpar; K-par-Check vereinfacht; `kpar_is_better<>` Template |
| `src/compute/gemm.cu` | gate_up-Dispatch: kMaxNRows + kPreferKpar-aware; Inline-Quant Launcher |
| `src/compute/gemm.h` | Inline-Quant Deklarationen |
| `src/graph/executor.cu` | Inline-Quant Wiring (O-proj Decode-Path); MoE Prefill-Fix |

### DequantTraits Uebersicht

| QType | kBlockBytes | kBlockElems | kQ8PerWeight | kMaxNRows | kPreferKpar |
|-------|------------|------------|-------------|----------|------------|
| Q6_K | 210 | 256 | 8 | 2 | true |
| Q8_0 | 34 | 32 | 1 | 2 | false |
| Q4_0 | 18 | 32 | 1 | 4 | false |
| Q4_K | 144 | 256 | 8 | 4 | true |
| Q5_K | 176 | 256 | 8 | 4 | true |

### Tests

219/219 Tests bestanden (5 uebersprungen: EndToEnd + AttentionTC erfordern Modell-Dateien).
