# tests/refs/ — independent reference goldens

Committed generators + versioned goldens for class-A tests (independent
ground truth). Rules (TEST_AUDIT.md §4):

1. **No magic constants.** Every golden value in a test traces to a committed
   generator in this directory. The generator header states tool versions,
   seed, and date; regenerating must reproduce the committed file bit-exactly
   (or the test that consumes it fails loudly).
2. **Tolerance policy** (state source + tolerance + why, per test):
   - f32-score-chain attention (cuBLAS FP32-S, FA2-f16): **≤ 1e-2 rel**
     vs an fp64 reference (inputs are f16-rounded; ~2^-11/element over a
     128-term dot plus f16 P/V rounding). MEASURED to hold even on sharp
     near-one-hot softmax data.
   - fp8-class attention (fp8 FMHA, FA2 e4m3 QK): the audit's 5e-2 single-op
     bound is **provably not meetable on short rows** (measured 0.58-0.71 rel
     on mild data at 2-24 keys: e4m3 score noise with no averaging — the
     #512 mechanism). These paths are CHARACTERIZED (envelope assert +
     printed stats), never blessed at 5e-2; E2E token-level locks carry the
     real quality gate.
   - WMMA attention (fmha_sm120, flash_attention_blackwell): **≤ 1e-2 rel**
     vs fp64, same class as the f32 chain (f16 QK with f32 accumulators +
     f32 softmax + f16 P). The 0.15-0.86 rel measured at the suite's birth
     was the #528 in-place float→half S/P-tile compaction race (half row r
     aliases the bytes of float row r/2); post-fix both paths measure ~4e-4
     on all configs and are held strict.
   - NVFP4: **≤ 1e-1 rel** single-op, plus E2E locks.
   - fp64-vs-fp64 generator cross-checks (numpy vs C++ double loops):
     **≤ 1e-9 rel** — summation-order ulps only; anything larger means the
     two implementations compute different math.
3. **Realistic data, bit-identical across languages.** Kernel fixtures are
   generated from integer LCGs mapped through f32 multiply-only transforms
   (no libm), so numpy and C++ produce bit-identical f16 inputs. References
   are computed FROM the f16-rounded values (the GPU sees exactly the same
   bits). Distributions mimic QK-normed model activations: heavy-tailed
   (cubed uniform), amplitude ~±8 with 1/256 outliers to ~±64 — the regime
   where e4m3 loses mantissa bits and the old benign `%13` fills proved
   vacuous (#525: a size_t underflow in that fill poisoned the suites
   silently for weeks).

Generators:

| file | golden | consumed by |
|---|---|---|
| `gen_attention_crosspath_golden.py` | `attention_crosspath_golden.h` | `tests/test_attention_crosspath.cu` |
| `gen_nvfp4_outlier_golden.py` | `nvfp4_outlier_golden.h` | `tests/test_nvfp4_outlier_ref.cu` |
| `gen_yarn_rope_golden.py` | `yarn_rope_golden.h` | `tests/test_gpt_oss_yarn_ref.cu` |
| `gen_harmony_golden.py` | `harmony_golden.h` | `tests/test_gpt_oss_harmony_golden.cpp` |
| `tests/test_vision_golden.cu` (dump mode = its own generator) | `vision_encoder_golden.h` | `tests/test_vision_golden.cu` |
| `gen_reference.py` | — (none committed) | **dormant** (TEST_AUDIT.md §7) — HF layer-by-layer dump infra for future use; no test consumes it |

gpt-oss YaRN goldens (P2.7): the generator reimplements YaRN-scaled RoPE
(corr-dim ramp, freq blend, mscale) in fp64 from the YaRN/HF semantics for the
gpt-oss-20b params (factor=32, orig_ctx=4096), pinning cos/sin at positions up
to 131071. The test re-derives the same math, cross-checks it against the
golden (**1e-9 rel**), then judges imp's `rope_forward` kernel against the
verified fp64 ref (**f16 class, 3e-2 rel** — fast-math `__cosf`/`__sinf`
argument reduction at pos≈131k costs a few ULPs of phase, measured 1.2e-2)
and asserts SENSITIVITY to the
#547 rope_freq_scale inversion (kernel must NOT match the 1024×-wrong ref).

Harmony goldens (P2.7): render strings from the REAL HF reference
(`transformers` `apply_chat_template`, model `openai/gpt-oss-20b`). The C++
test renders the committed `tests/fixtures/gpt_oss_chat_template.jinja` through
imp's own jinja engine and compares EXACTLY (the one strftime_now `Current
date:` line is normalized in both — documented in the test). Needs
`transformers`, only the tokenizer/template (no weights):
`docker run --rm -v $PWD:/work -v /home/kekz/models:/models -w /work
python:3.12-slim sh -c "pip install -q transformers jinja2 && MODEL=/models/gpt-oss-20b python3 tests/refs/gen_harmony_golden.py"`.

Vision encoder golden (R9 / #583) is the one exception to rule 1's "independent
generator": there is no fp64 oracle for the SigLIP / gemma4v encoder + projector
tail, so this is a **stability lock**, not an external truth. The generator is
the test itself in dump mode (`IMP_VISION_GOLDEN_DUMP=1`), which prints the
golden arrays measured on the current build; the header records the producing
commit + model file sizes so a drift is attributable. Tolerance is the **f16
class (≤ 1e-2 rel + 5e-3 abs floor)** on projector-output spot values, per-token
L2 norms, and the global mean, PLUS a hard no-NaN/Inf guard over the full
embedding (the real #489/#514-class assert). Regenerate via `make test-vision
IMP_VISION_GOLDEN_DUMP=1` and paste the emitted blocks; needs the RTX 5090 +
the gemma-3/gemma4v mmproj GGUFs (encoded standalone, no LM).

NVFP4 outlier goldens (risk #2): the generator reimplements NVFP4's two-level
dequant (E2M1 + UE4M3 micro-scale + f32 tensor-scale, incl. the 1/512 floor)
in fp64 from the format definition — never from imp. Adversarial weight
distributions (Gaussian, 64× outliers, a single 512× outlier, all-tiny) probe
the per-tensor-scale floor that collapsed Gemma mode-2 (#514/#516). The test
asserts the NVFP4 **1e-1 rel** class tolerance on quantize→dequant and
quantize→dequant→GEMV (`gemv_nvfp4_kpar`) spot values, PLUS a hard
no-NaN/Inf guard on every distribution (the real Gemma-class assert).

Run a generator inside any container with numpy (the host stays clean):

```
docker run --rm -v $PWD:/work -w /work python:3.12-slim \
  sh -c "pip install -q numpy && python3 tests/refs/gen_attention_crosspath_golden.py"
```
