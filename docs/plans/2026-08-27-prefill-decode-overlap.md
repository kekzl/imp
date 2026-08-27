# Prefill/decode overlap - plan and ledger

Goal: run the paced prefill chunk CONCURRENT with the in-flight batched
decode step instead of serially between two steps. Attribution that prices
it (step_timing 2026-08-27, 32 streams): outside-step 1.2-1.5 ms/step of
which prefill-block 0.8-1.0 ms while ingest is pending; bare turnaround
34 us. Upside is bounded by that block plus decode-idle absorption
(~2-4% aggregate on burst ingest) plus TTFT.

## What already exists

- `GreenContextManager`: SM partitioning fails on sm_120 (LIMITATIONS.md),
  but the FALLBACK ships two priority streams (prefill low, decode high)
  with distinct memSyncDomains and `is_available()==true` - built for
  exactly this overlap.
- `Workspace` dual-slot swap (03d50457, 2026-03): slot 0 prefill, slot 1
  decode; `allocate_decode_workspace(stream, max_batch)` already takes
  max_batch and the green-gated warmup already passes
  `config_.max_batch_size` ("Concurrent prefill/decode overlap enabled").
  The consumer never caught up: `step_decode_forward` uses slot 1 only at
  `valid_decode.size()==1` (the pre-continuous-batching shape); batched
  decode resizes and squats slot 0 - which is why the drain-before-prefill
  serializer exists.

## Shared-state collisions (the actual work)

| shared | users | resolution |
|---|---|---|
| slot-0 shared/persistent workspace | prefill + batched decode | decode batch<=decode_max_batch_ runs in slot 1 (flag-gated) |
| `smallm_xq_` + tags (+ producer fusion) | batched decode; prefill chunks with M<=32 also route there | prefill under overlap declines smallm/producer-xq (one executor member, three gates) - CUTLASS serves small chunks |
| `qscratch_` fp8_act/d_act_scale(+maxes)/q8_1/d8 | fp8 sidecars in batched decode; prefill fp8/dp4a paths | duplicated at decode size (~<1 MiB), swapped by `use_workspace` |
| `qscratch_.cutlass_act_*` | prefill CUTLASS act-quant; spec-VERIFY chunks (bs=1) | overlap only enqueues at decode batch >= 2 (no verify there) |
| `d_sample_result_` parity slots | decode batch sampling; prefill last-chunk sample | dedicated prefill sample buffer (always, 1 KB) |
| KV pool / GDN state | disjoint requests (PREFILLING vs DECODING) | no cross-stream ordering needed; block alloc host-side |
| cuBLAS handle | both, single-threaded enqueue | safe: setStream serializes on host |

## DECLINE gates (logged once, #1755 style)

`runtime.prefill_overlap=true` (default false; implies green-context
streams) AND decode workspace allocated at max_batch AND !has_moe AND
NVFP4-native (no GGUF decode overlay: `wcache_.nvfp4` empty) AND decode
batch >= 2 this step. Anything else: serial path, unchanged.

## Phases

1. This PR: flag + slot-1 batched decode + drain relaxation + the three
   collision fixes + gates. Measure: 32-stream burst A/B (aggregate +
   TTFT), byte isolation 32-concurrent-vs-serial (the #1780 gate), degen.
2. If it pays: widen gates (GGUF via per-slot q8 scratch already done;
   MoE workspace audit), revisit bs==1+verify.
3. Green-context SM partitioning stays dead on sm_120 (documented).

## Ledger

- 2026-08-27, phase 1 BUILT AND MEASURED NEUTRAL — default stays OFF, with
  the measurement in the config comment (the #1755 convention).
  - Gate bug on the way: `!wcache_.nvfp4.empty()` is NOT a GGUF predicate
    (the map is populated on native-NVFP4 models as the secondary cache);
    the real one iterates the registry for `dequant_gpu_supported(source)`.
  - Short prompts (the conc_client shape), 4 waves x 3 alternating
    trials/arm, mbs=32/seq4096 pinned: OFF 1771.3 vs ON 1777.7 tok/s
    median, pairs -0.0/-0.3/+1.5% — inside wave noise.
  - Heavy ingest (~1000-token prompts, streaming TTFT), same discipline:
    OFF 789.7 vs ON 790.6 median (pairs +0.1/-1.4/+0.6%); TTFT p50
    3.70 vs 3.74 s. The workload the overlap was built for shows nothing.
  - Mechanism: green-context SM partitioning is dead on sm_120
    (LIMITATIONS.md), so both streams contend for the same SMs. The
    decode step's idle is us-grained inter-node gaps a concurrent
    low-priority stream cannot fill; the paced serial prefill was never
    wasted time, it is GPU work that costs the same wall concurrently and
    stretches the in-flight decode step by roughly its own duration.
  - Kept (default off, all gates logged): the per-slot quant scratches,
    slot-1 batched decode, the dedicated prefill sample slot and the
    ENTERED/DECLINED gate chain — the experiment is one `--set` away on
    hardware where partitioning is real. degen_suite 50/0 on the ON arm;
    0 CUDA errors across all ON runs.
