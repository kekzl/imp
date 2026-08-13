---
layer: L1
audience: operators
verified: 2026-08-13
commit: 81ffa573
---

# Troubleshooting

Symptom, cause, fix. Ordered by how often each one actually happens.

## The model loads and answers nonsense

**Repeated tokens, or a fluent answer that contradicts itself.**

Check `stderr` for `packed NVFP4 weight reached the generic cuBLAS path`. That
means weights reached a kernel that cannot read them and the multiply was
skipped, so the layer contributed nothing. Since #1403 an NVFP4 MoE checkpoint
whose experts do not fit is refused at load instead, so if you see this on a
current build it is a new defect and worth an issue.

**Fluent but wrong is the harder case.** Partial corruption does not look like
corruption: the model stays grammatical. Do not judge output quality from one
short prompt; run the degeneration battery (`tools/analysis/degen_suite.py`).

## Decode is far slower than the numbers in the docs

In order of likelihood:

1. **Something else is on the GPU.** On WSL2 `nvidia-smi` does **not** show a
   container holding the card. Check `docker ps` as well.
2. **CUDA graphs did not capture.** The log prints `Resolved dispatch: ...
   graphs=1` when they did. `graphs=0` typically costs 2-3x on decode.
3. **The NVFP4 decode cache is partial.** Look for
   `NVFP4 decode caches: FULL (n/n MoE layers)`. Anything less means some
   tensors decode from their slower source, and a partial cache also aborts
   graph capture.
4. **VRAM spilled to host.** A successful allocation proves nothing on
   WSL2/WDDM: the driver oversubscribes into host memory and returns success.
   The tell is bandwidth, roughly 1530 GB/s resident against 237 GB/s spilled,
   so the symptom is a ~6.5x cliff rather than an error. Reduce
   `runtime.max_seq_len` or the KV dtype.
5. **You are comparing against a number that was measured differently.** The
   prefill pin changed meaning on 2026-07-26 when one-shot runs stopped hitting
   the prefix cache. See [`PERF.md`](PERF.md).
6. **It is the host, not the change.** Decode on this box moves several percent
   between sessions. A single slow reading is not a regression until it
   reproduces in a paired A/B.

## The perf gate is red

A red gate is **not** a regression until it reproduces. Before investigating:

- re-run it; the host's own between-session movement is several percent
- check `docker ps` and `nvidia-smi` for a co-tenant
- if it still fails, A/B against `main` with alternating rounds, not two runs
  twenty minutes apart

If the change genuinely and intentionally moves performance, refresh the
baseline with `scripts/gen_perf_baseline.sh` **and say so in the PR**. A refresh
without that sentence is indistinguishable from a regression papered over.

## A request hangs, or streams nothing until the end

Almost always a reverse proxy buffering the response. Set `proxy_buffering off`
(nginx) or the equivalent; see [`DEPLOYMENT.md`](DEPLOYMENT.md).

If there is no proxy: a very large grammar can take a long time to build its
first token mask. Grammars are compiled and memoised, but the first mask inside a
deeply nested state is the expensive one.

## Requests fail after the first one

`finish_reason: "cancelled"`, or the second identical request returns nothing.

This is usually the KV pool being too small to hold one full-length sequence: the
load succeeds, and every full-length request is then cancelled at admission.
Since v0.23.0 that condition is reported at load time for an operator-set
`max_seq_len`. Reduce `runtime.max_seq_len`, or pick a smaller KV dtype.

Also check `kv_cache.swa_snapshot_mb`: a value **below one snapshot size**
silently disables prefix caching, which is worse than setting it to zero. It
warns since #1092.

## The server answers 503

Either the model is suspended (`POST /admin/resume`), or the server was started
without `--model` and the request resolved to no model. `GET /v1/models` shows
what is available.

## An image is ignored, or refused

- `400 vision_unavailable` means the checkpoint loaded **text-only**: imp does
  not recognise its vision tower. This is a refusal on purpose, so you do not get
  a confident description of a picture the model never saw.
- `400` on an `image_url` means it could not be fetched or decoded. Dropping it
  silently would shift every later image onto the wrong placeholder.
- Check `MODELS.md` for which checkpoints can actually see.

## Constrained output comes back unconstrained

Since v0.23.0 this should be impossible: a pattern imp cannot compile is a `400`.
If you get free text at HTTP 200 with a `response_format` set, that is a bug
worth reporting, and it was the exact defect fixed in #1256.

## Build or test problems

- `build/` and `build-dev/` are root-owned by the container. Remove them with
  `make dev-clean` or a throwaway container, never `sudo`.
- `make test-unit` runs a **different binary** from the CI lane. Green there is
  not green in CI. The CI lane is `ctest -L unit`, which is `make dev-test`.
- `build-dev/` carries whichever branch was last compiled in it; `git checkout`
  does not rebuild.
- A `--gtest_filter` naming a value-parameterised suite without wildcards matches
  zero tests and reports `PASSED`. `DetEvalE2ETest.*` matches nothing;
  `*DetEvalE2ETest*` is correct.

More: [`CONTRIBUTING.md`](../CONTRIBUTING.md) and
[`internals/BENCHMARKING.md`](internals/BENCHMARKING.md).
