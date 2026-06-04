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
   - WMMA attention (fmha_sm120, flash_attention_blackwell): measured
     0.15-0.86 rel on mild realistic data despite f32 accumulators — under
     investigation (cross-path divergence issue); characterized like fp8.
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

Run a generator inside any container with numpy (the host stays clean):

```
docker run --rm -v $PWD:/work -w /work python:3.12-slim \
  sh -c "pip install -q numpy && python3 tests/refs/gen_attention_crosspath_golden.py"
```
