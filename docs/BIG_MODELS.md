<!--
layer: L1
audience: operators
verified: 2026-10-30
commit: a4070e6a
-->

# Models larger than the card

Mixture-of-experts models use a few experts per token.

- When the experts do not fit in VRAM, imp keeps them in pinned host RAM, caches the busiest on the card (least recently used out) and copies a missing expert over PCIe when a token needs it.
- Nothing to configure: imp decides at load.

## Qwen3.8-Flash-Next on one RTX 5090

[`nvidia/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4): 123.6 GiB, 56.25 GiB of experts.

| | measured |
| --- | --- |
| one user, 6 real prompts x 400 tokens | **92.1 tokens/s** (89.2-96.4 across the prompts, 91.8-92.2 across 3 starts), first token after 392 ms |
| reading a 512-token prompt | 1338.18 tokens/s (1331.51-1341.98) |
| host RAM taken at load | ~68 GiB (`MemAvailable` 76.6-77.0 -> 8.9-9.2 GiB on a 78 GiB host) |
| card memory in use, start to ready | 25.7 GiB of 32; 38-43 s (50 s on the first start, which records the memory plan) |

<!-- [PROV: commit=ca8780ef date=2026-10-27 hw=RTX5090 model=Qwen3.8-Flash-Next-NVFP4 quant=NVFP4 cuda=13.4.3 path=host-resident-experts cmd=`CACHE_VOL=imp-launch-cache tools/analysis/launch_server_bench.sh` +
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
