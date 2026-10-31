<!--
layer: L1
audience: operators
verified: 2026-10-31
commit: daca1665
-->

# Models larger than the card

Mixture-of-experts models use a few experts per token.

- When the experts do not fit in VRAM, imp keeps them in pinned host RAM, caches the busiest on the card (least recently used out) and copies a missing expert over PCIe when a token needs it.
- Nothing to configure: imp decides at load.

## Qwen3.8-Flash-Next on one RTX 5090

[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): 123.6 GiB, 56.25 GiB of experts.

| | measured |
| --- | --- |
| one user, 6 real prompts x 400 tokens | **91.8 tokens/s** (89.5-95.9 across the prompts, 91.8-92.0 across 3 starts), first token after 388 ms |
| reading a 512-token prompt (v0.59.0) | 1338.18 tokens/s (1331.51-1341.98) |
| host RAM taken at load | ~67 GiB (`MemAvailable` 74.7-75.9 -> 8.4-8.7 GiB on a 78 GiB host) |
| card memory in use, start to ready | 25.6 GiB of 32; 45-50 s (76 s on the first start, which records the memory plan) |

<!-- [PROV: commit=ed4422b7 date=2026-10-31 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` n=3 server starts note=v0.60.0 tree 0937bac4, harness_md5=7ade1eb1, docs/audit/launch_2026_10_31/fn.txt; rows one user, host RAM, card memory] -->
<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=
       `imp-cli --bench --bench-pp 512 --bench-reps 5 --max-tokens 128 --bench-prompt-file novel_prose_2026.txt` n=3 server starts, 3 bench processes note=40 GiB row: CHANGELOG [0.50.0] #2621] -->

## What you need

- a 32 GB card or bigger, the host RAM above free, 123.6 GiB of disk
- with less RAM imp reads missing experts from disk instead (at a 40 GiB container limit: 46.9..60.0 tokens/s, v0.50.0, [`CHANGELOG.md`](../CHANGELOG.md))
- speed depends on PCIe (test card: PCIe 5.0 x16) and on how often a conversation changes topic

## Run it

```bash
STAGE_DIR=~/models/Qwen3.8-Flash-Next-NVFP4 scripts/stage-model.sh nvidia/Qwen3.8-Flash-Next-NVFP4 ~/models/Qwen3.8-Flash-Next-NVFP4
docker run -d --name imp-server --gpus all -v ~/models:/models -v imp-cache:/home/imp/.cache/imp \
  -p 127.0.0.1:8090:8090 ghcr.io/kekzl/imp:latest --model /models/Qwen3.8-Flash-Next-NVFP4 --host 0.0.0.0 --port 8090
```

**Ready** when `docker logs imp-server` prints `Server listening on http://0.0.0.0:8090`. Load times, KV growth, batching and a 27-turn chat: [`QUICKSTART.md`](QUICKSTART.md#qwen38-flash-next).
