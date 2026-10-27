# DISPATCH_launch 2026-10-27

Append-only. Launch prep for Show HN + r/LocalLLaMA. Deliverables: repo changes on branch `launch/prep-2026-10-27`, texts in `docs/launch/` (gitignored, local only). Nothing posted; release, repo metadata, Discussions, GHCR tags wait for the owner.

## 2026-10-27 Phase 0: inventory

| Item | State |
|---|---|
| main | `2e58825c`, 3 dependabot commits after `v0.59.0` (`ca8780ef`) |
| `ghcr.io/kekzl/imp:latest` | v0.59.0, revision `ca8780ef`, created 2026-10-25, CUDA 13.4.3; pulls without a local build |
| brief vs tree | brief assumes v0.33.x, a "60-second quickstart" heading and a README range "-26 % to +141 %": none exist; README already documents Flash-Next and `scripts/stage-model.sh` |
| Discussions | already enabled (`hasDiscussionsEnabled: true`); `gh repo edit --enable-discussions` not needed |
| social preview | exists: `docs/social-preview.html` -> `.png` (1280 x 640), numbers from the 2026-10-11 sweep |
| `docs/launch/` | new, gitignored (owner decision 2026-10-27): drafts stay local; numbers traced in its `CLAIMS.md` |

## 2026-10-27 Phase 1: claim audit (before measurement)

Table: `docs/launch/CLAIMS.md` (local) section Phase 1. Four findings: README ~102 tok/s stale and conflicting with README:27; description 1.4x llama.cpp compares NVFP4 with Q4_K_M; Flash-Next speed only in a PROV comment; brief LOC/commit counts wrong (2750 commits, 340804 lines).

## 2026-10-27 Phase 2: re-measurement

| Item | Result |
|---|---|
| protocol | image `ghcr.io/kekzl/imp:latest` v0.59.0 for every imp number; 3 fresh-server trials per arm; PERF_LOG entries 2026-10-27; raw `docs/audit/launch_2026_10_27/` |
| 91-94 vs ~102 | both stale: same workload today reads 96.3 (spec off) / 116.2 (MTP, `--max-batch 1`); the README `docker run` (batch auto 32, no MTP) reads 96.5 |
| 1.3x vLLM | 1.38x one user without speculation, 1.40x at 32 users; with MTP on both vLLM leads one user 124.4 vs 116.2 (new LIMITATIONS row) |
| 1.4x llama.cpp | replaced by 1.23x on the same Q4_K_M file (77.1 vs 62.9) |
| Flash-Next | server 92.1 tok/s, TTFT 392 ms, ~68 GiB host RAM, 25.7 GiB card, ready 38-43 s; bench tg128 116.29 reproduces the README's 114.90-115.53 (that number was the bench, not the server) |
| not reproduced | CHANGELOG [0.59.0] server decode 148.4-150.3 tok/s: 6 client variants read 90.2-93.9 (`fn_client_ab.txt`); left as is, not quoted |
| incident | a `scripts/check-release.sh` run without `SKIP_VERIFY=1` started the GPU gate during the llama.cpp 1-stream arm of trial 1 (29.4 tok/s); discarded, re-run (63.8) |
| tooling | `tools/analysis/launch_server_bench.sh`, `tools/analysis/seq_stream_client.py`; `engine3_conc.sh` takes several `IMP_SET` pairs and `VLLM_SPEC` |
| found | `scripts/stage-model.sh` exits without `make build` (default `imp:test`) and reads 0 GiB free on a fresh output dir: both fixed; `hf://` into a bind-mounted `~/models` needs `--user` (README says so) |

## 2026-10-27 Phases 3-9

| Phase | Result | Commit |
|---|---|---|
| 3 README | first screen: scope sentence, 27B three-engine table with workload, MTP loss beside it, "Where imp loses", Flash-Next line, "Models larger than VRAM" section; 159 -> 145 lines (gate `tools/filesize_thresholds.toml` readme = 159 kept) | `7ac69531` |
| 4 quickstart | `ghcr.io/kekzl/imp:latest` (v0.59.0) pulls anonymously (token endpoint 200, tags to 0.59.0); README install = clone, `stage-model.sh`, `docker run`; 3 bugs in `stage-model.sh` fixed (default image, df on a fresh dir, compressed-tensors NVFP4 sent to imp-quantize); clean shell (`env -i`): 19.2 GiB in 5 min 17 s, ready 14 s, chat answer; transcript `launch_2026_10_27/quickstart_clean_shell.txt`. `--model hf://` into a bind-mounted `~/models` fails without `--user` (uid 1001 vs 1000), README says so | `6487f00a` |
| 5 demo | `docs/assets/demo.tape` + `demo-chat.sh` + `vhs.Dockerfile`; recorded here: `demo.gif` 18.0 s, 759340 bytes, Flash-Next, nvidia-smi 27069 / 32607 MiB, 193 tokens at 92.9 tok/s | `7ac69531` |
| 6 community bench | `scripts/bench-community.sh`, RTX 5090 block in `launch_2026_10_27/community_rtx5090.md`; Discussions already on | `b75aaffc` |
| 7 release + metadata | drafts only: `docs/launch/RELEASE_NOTES.md` (v0.59.1, patch: no engine change since v0.59.0), `REPO_METADATA.md` (316-character description), `docs/assets/social-preview.png` | `7941f7d3` |
| 8 launch texts | `SHOW_HN.md` (titles 72 / 67 characters, first comment 323 words, 10 questions), `REDDIT_LOCALLLAMA.md` | this commit |
| 9 verification | every number in README top, SHOW_HN, REDDIT, release notes, description, social preview found in `CLAIMS.md` (rest: ports, card names, versions, CSS); relative links and anchors of README and the launch docs resolve (checker self-tested on a bad anchor and a missing file); GGUF sha256 `c600de0300ae8a0e` | this commit |

Findings outside the scope, not fixed here:
- `GET /v1/models` answers 500 when one directory under `--models-dir` is unreadable for uid 1001 (a mode-700 scratch dir in `~/models`): the whole list fails, the web UI with it.
- CHANGELOG [0.59.0] Flash-Next server 148.4-150.3 tok/s does not reproduce with the in-tree client (90.2-93.9).

## 2026-10-27 Owner decision

`docs/launch/` goes to `.gitignore`: CLAIMS, SHOW_HN, REDDIT, DISCUSSION_BENCH, RELEASE_NOTES, REPO_METADATA stay local; `scripts/docs_lint.py` exclusion reverted (gitignored paths are skipped, #1698).

## 2026-10-27 Correction: MTP row refuted

The Phase 2 row "with MTP on both vLLM leads one user 124.4 vs 116.2" does not hold. `tools/analysis/mtp_vs_vllm.sh` (identical real chat prompts, 3 starts per engine): imp 126.49 vs vLLM 116.84 tok/s at k=2, imp 128.00 vs 100.87 at k=1; MTP acceptance equal, imp's step 9-13 % cheaper. The 124.4 / 116.2 pair came from `conc_client.py` template completions (arm tag in every prompt, 3 waves including the first); the same shape with a warm wave read 134.7 vs 125.1. LIMITATIONS row removed, README / BENCHMARKS / CHANGELOG corrected. PERF_LOG entry of the same date.
