---
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
---

# Performance

**This file is the single source of truth for every number about imp.** The
README embeds a generated extract of it; nothing else in the tree states a
throughput figure without linking here.

Methodology, and what makes a number admissible at all, is in
[`internals/BENCHMARKING.md`](internals/BENCHMARKING.md). Read that before
quoting anything below, because two of the figures here are routinely
misread.

## Read this before the tables

[PROV: commit=2230e1c2 date=2026-08-13 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
       cuda=13.3 path=gguf-dp4a cmd=`make verify-fast` n=6 note=six quiet runs
       across two days, used to set the gate thresholds in #1400]

**Decode on this host moves several percent between sessions, with nothing
changed.** The same tree measured 287.63 tok/s on one day and 276.92 the next at
healthy clocks; six quiet runs spanned 278.59-289.77. Three processes *within*
one session agree to 0.16 %. That is why the regression gate sits at 8 % and not
at 3 %: below 8 % it was reporting the host, not the change.

**Prefill varies more than decode**, because cuBLAS algo selection is re-timed
per process. Under host-resident MoE experts the same arm varies about 15 %
between two runs. A single prefill number is not a measurement; a paired,
alternating A/B is.

**A red gate is not a regression until it reproduces.** Under container load the
same build reads 4-8 % low.

## The pinned gate

The only figures re-measured on every push. `make verify-fast` runs them locally,
because CI has no GPU runner.

<!-- PERF:BEGIN -->
| metric | value | threshold |
|---|---|---|
| decode tg128 | **287.19 tok/s** | 8 % |
| prefill pp128 | 4885.13 tok/s | 8 % |
| prefill pp512 | **12406.87 tok/s** | 8 % |
| prefill pp4096 | 15324.7 tok/s | 8 % |
| peak VRAM (own) | 20716 MiB | 10 % |

[PROV: commit=1e4fad60 date=2026-07-26 hw=RTX5090 model=Qwen3-8B-Q8_0 quant=Q8_0
       cuda=13.3 path=gguf-dp4a cmd=`make verify-fast` n=5x5]
<!-- PERF:END -->

Pinned 2026-07-26, thresholds widened 2026-08-13 (#1400). Median of 5 trials x 5
reps with a 15 s cooldown; `tg128` comes from the `pp512` run so the gate matches
what it measures; speculation off since #1011. Since #1214 the gate takes the
median across **three independent processes** and prints the spread.

`own_peak_mb` is peak device memory this *process* allocates after engine init.
It excludes the CUDA primary context (~1680 MiB) and any neighbour process, and
it is byte-identical across repeat runs, which is why its threshold can stay at
10 % while throughput needs 8 %.

## Why the prefill pin is lower than the 2026-07-15 one

`pp512` was 14515 before 2026-07-26 and is 12406.87 now. **Nothing regressed.**
Until #1061, `imp-cli --bench` left prefix caching on, so repeated bench reps
were partly measuring cache hits on top of prefill. One-shot runs now disable it,
because a single-generation process never re-sees its own prefix. Confirmed by
bisect (first differing commit `d8bc45a8`) and by forcing the old behaviour back
on, which reproduces the old band. The current figure is the honest prefill cost.

This is the single most misread number in the repo. Anyone comparing against a
pre-2026-07-26 figure is comparing two different measurements.

## Competitive standing

Per-model numbers, the models each named, live in
[`BENCHMARKS.md`](BENCHMARKS.md). The one-line summary, current as of the
2026-07-12 sweep:

- batch=1 decode leads llama.cpp on every hero measured, by 13-48 % on dense GGUF
- MoE single-sequence prefill leads vLLM
- cross-engine perplexity parity has been measured, not assumed

Where imp loses is in [`LIMITATIONS.md`](LIMITATIONS.md), and it is in the README
too, on purpose.

## Reproducing any of this

```
make build          # the image you measure; make dev is not it
nvidia-smi          # must show no compute processes, and check `docker ps` too
make verify-fast    # ~90 s, the gate
```

On WSL2 `nvidia-smi` alone does **not** tell you the card is free: a container
can hold it without appearing there. Check `docker ps` as well.

To refresh the baseline after a change that intentionally moves performance:
`scripts/gen_perf_baseline.sh`, and say so in the PR. A baseline refreshed
without that sentence is indistinguishable from a regression that was papered
over.

## MoE host offload

Only relevant when a MoE model's experts do not fit in VRAM. GGUF experts have a
working host path; NVFP4 experts are refused at load rather than served with
their GEMMs skipped.

<!-- markdownlint-disable -->
| arm | decode |
|---|---|
| experts resident | 311.24 tok/s |
| all 48 layers host-resident, LRU cache | 48.3 tok/s |
| staging buffer only (no cache) | 6.63 tok/s |

[PROV: commit=1e4fad60 date=2026-08-11 hw=RTX5090 model=Qwen3-30B-A3B-Q4_K_M
       quant=Q4_K_M cuda=13.3 path=moe-host-offload
       cmd=`imp-cli --bench --set moe.force_host_experts=48 --bench-reps 3` n=5
       note=warm; a cold run of the same build reads 20.99 rather than 49.60]

**Warm and cold differ by 2.4x on this path**, so a figure from it has to say
which it is. Cache capacity is the lever: `moe.expert_cache_budget_pct` moves
the same model 10.51 to 51.86 tok/s across 5 % to 50 %. The default stays 15
because on a model that genuinely does not fit, the same VRAM is what the KV
pool wants.
