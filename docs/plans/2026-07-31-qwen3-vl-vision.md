# Qwen3-VL vision — measured inventory and build plan

Roadmap gap 2. Step 0 (the text tower loads and generates, `degen_suite` 34/34)
shipped 2026-07-31; this plan covers what is left. Everything below is read off
the staged `Qwen3-VL-4B-Instruct` checkpoint and this tree, not from
recollection — the previous one-line assessment of this gap was wrong about
where two thirds of the work sits.

## What the checkpoint actually contains

315 tensors under `model.visual.` (all BF16):

| Tensor | Count | Shape |
|---|---:|---|
| `patch_embed.proj.weight` / `.bias` | 1 / 1 | `[1024, 3, 2, 16, 16]` → loaded as `[1024, 1536]` / `[1024]` |
| `pos_embed.weight` | 1 | `[2304, 1024]` (a 48×48 grid) |
| `blocks.N.norm1.{weight,bias}` | 24 | `[1024]` |
| `blocks.N.attn.qkv.{weight,bias}` | 24 | `[3072, 1024]` / `[3072]` — **fused** |
| `blocks.N.attn.proj.{weight,bias}` | 24 | `[1024, 1024]` / `[1024]` |
| `blocks.N.norm2.{weight,bias}` | 24 | `[1024]` |
| `blocks.N.mlp.linear_fc1.{weight,bias}` | 24 | `[4096, 1024]` / `[4096]` |
| `blocks.N.mlp.linear_fc2.{weight,bias}` | 24 | `[1024, 4096]` / `[1024]` |
| `merger.norm.{weight,bias}` | 1 | `[1024]` |
| `merger.linear_fc1.{weight,bias}` | 1 | `[4096, 4096]` / `[4096]` |
| `merger.linear_fc2.{weight,bias}` | 1 | `[2560, 4096]` / `[2560]` |
| `deepstack_merger_list.N.norm.{weight,bias}` | 3 | `[4096]` ← **not 1024** |
| `deepstack_merger_list.N.linear_fc1.{weight,bias}` | 3 | `[4096, 4096]` / `[4096]` |
| `deepstack_merger_list.N.linear_fc2.{weight,bias}` | 3 | `[2560, 4096]` / `[2560]` |

Config: `depth 24`, `hidden_size 1024`, `num_heads 16`, `intermediate_size 4096`,
`patch_size 16`, `temporal_patch_size 2`, `spatial_merge_size 2`,
`num_position_embeddings 2304`, `out_hidden_size 2560`,
`hidden_act gelu_pytorch_tanh`, `deepstack_visual_indexes [5, 11, 17]`.
Text side: `rope_scaling {mrope_interleaved: true, mrope_section: [24, 20, 20]}`.

Two shapes carry information worth stating:
- `merger.linear_fc1` takes **4096** = `spatial_merge_size² × 1024`, so the 2×2
  spatial merge is a plain concatenation of four patch vectors.
- The main merger's `norm` is `[1024]` but each deepstack merger's `norm` is
  `[4096]`. So the main merger normalises **before** the concat and the deepstack
  mergers **after** it. Do not copy one from the other. (Confirmed in
  `modeling_qwen3_vl.py`: one flag, `use_postshuffle_norm`, selects exactly
  this — the shape difference was not a coincidence.)

## What imp already has

More than the old assessment implied:

- **Wrapper handling** — nested `text_config`, `model.language_model.` stripping,
  `model.visual.*` skipping. Now generic via `ModelConfig::multimodal_wrapper`.
- **Kernels** — `vision_layernorm_kernel` (LayerNorm *with* bias, which is what
  `norm1`/`norm2` need), `gelu_tanh_kernel` (exactly `gelu_pytorch_tanh`),
  attention and GEMM paths in `vision_encoder.cu`.
- **The >4-D loader drop is fixed**, so `patch_embed.proj.weight` now arrives as
  the `[1024, 1536]` matrix a GEMM wants.

## What is genuinely new

1. **Dynamic token count.** `vision_encoder.{h,cu}` takes
   `[3, image_size, image_size]` and sizes every buffer off one fixed
   `num_patches`. Qwen3-VL has no `image_size` at all. Buffers must be sized per
   image (or to a configured maximum), and attention must run over a variable
   token count. This is the only piece that is genuinely encoder work.
2. **Position-embedding interpolation.** `pos_embed` is a fixed 48×48 grid; a
   non-48×48 patch grid needs it resampled. HF gathers precomputed indices and
   sums them with precomputed weights (`pos_embed(interp_indices) *
   interp_weights`), i.e. bilinear expressed as a gather + weighted sum.

2b. **The encoder also has its own 2-D RoPE** — missed in the first draft of this
   plan and found only by reading the model code. `rotary_pos_emb(position_ids)`
   feeds `apply_rotary_pos_emb_vision` inside every block, *in addition to* the
   learned `pos_embed` above. Not free, but not new either: imp's gemma4v
   encoder already does "2D axial NEOX RoPE" (`vision_model.h`), so the building
   block exists. Attention also runs packed with `cu_seqlens`, which is the
   variable token count of piece 1 seen from the kernel side.
3. **Patch embed over the temporal axis.** `temporal_patch_size 2` means a still
   image is repeated along t before the projection; the flattened `[1024, 1536]`
   matrix already expects `3·2·16·16` inputs, so the preprocessor must build
   patches in that order.
4. **Merger.** 2×2 concat → norm → fc1 → gelu → fc2 → `[*, 2560]`. Straight
   GEMMs; note the norm placement above.
5. **DeepStack.** Blocks 5/11/17 tap their hidden state through their own
   merger. **Resolved against `modeling_qwen3_vl.py` rather than guessed** — and
   the two index spaces are different, which is the trap:
   - the vision-side taps are blocks `[5, 11, 17]` (`deepstack_visual_indexes`);
   - the LM-side injection is at **layers 0, 1, 2** — `layer_idx in
     range(len(deepstack_visual_embeds))`, i.e. the first three LM layers in
     order, *not* layers 5/11/17;
   - and it is an **add at image-token positions only**:
     `hidden[visual_pos_masks] += deepstack_embed[i]` (`_deepstack_process`).
     So the LM needs a per-token mask of which positions came from the image.
6. **3-axis M-RoPE in the main forward.** Today the section split exists **only**
   in the MTP draft head (`mtp_forward.cu`), hardcoded to `[11, 11, 10]` for
   `rope_dim == 64`, with an explicit "imp doesn't load this from config yet",
   and — by its own comment — only ever exercised where all three positions are
   equal, i.e. equivalent to plain partial-rope. Needed: read `mrope_section`
   from config, carry per-token `(t, h, w)` ids, and apply the split in
   `rope_forward` **and** `qknorm_rope_fused` (the decode fast path). This
   touches the hot path for every model, so it needs a text-only invariant test:
   with all three positions equal the output must be bit-identical to today.
7. **Image preprocessing.** Resize to a multiple of `patch_size × spatial_merge_size`
   (32) within the model's pixel bounds. **The sizing half is done** —
   `qwen_smart_resize`, ported from transformers' `smart_resize` with its
   ties-to-even rounding and its asymmetric clamp branches, 7 unit tests.
   Remaining: resample + normalise + patchify into the `3·2·16·16 = 1536` order
   the flattened `patch_embed` expects. Note the bounds in
   `preprocessor_config.json` are PIXEL COUNTS despite being spelled
   `shortest_edge` / `longest_edge` — 65536 = 256² and 16777216 = 4096².

## Status (2026-07-31, end of day)

| # | Piece | Where | Status |
|---|---|---|---|
| 0 | Text tower loads and generates | — | ✅ #1163 (`degen_suite` 34/34) |
| — | Loader keeps >4-D tensors | `safetensors_loader.cpp` | ✅ #1164 |
| 7a | `smart_resize` | `vision/image_processor` | ✅ #1166 |
| 3+7b | `patchify` (`C,T,ph,pw`, merge-block order) | `vision/image_processor` | ✅ #1167 |
| — | Vision tensor name mapping | `vision/qwen3vl_vision_map` | ✅ #1168 |
| — | `vision_config` parse | `vision/qwen3vl_vision_config` | ✅ #1169 |
| — | Tower load (315 tensors, shape-checked) | `vision/qwen3vl_vision_load` | ✅ #1170 |
| 1+2+2b+4 | Grid math, upload, encoder forward | `vision/qwen3vl_{vision_grid,vision_upload,encoder}` | ✅ #1171 |
| 6a | M-RoPE kernel + config read | `compute/rope`, `model_config` | ✅ #1172 |
| 6b | Per-token `(t, h, w)` positions | `model/mrope_positions` | ✅ #1173 |
| 5 | DeepStack injection at LM layers 0/1/2 | `vision/deepstack_inject` | ✅ #1176 |
| 7c | Image-placeholder expansion | `model/image_placeholders` | ✅ #1177 |
| — | End-to-end wiring (engine, CLI, M-RoPE positions) | `runtime/engine_qwen3vl` | ✅ #1178 |
| — | Images over `/v1/chat/completions` | `imp-server/handlers_chat_*` | ✅ #1179 |
| — | Prefix cache salted with the image | `model/image_placeholders` | ✅ #1180 |

**The path works.** `imp-cli --model Qwen3-VL-4B-Instruct --image cat.jpg
--prompt "Describe this image in one sentence."` answers *"A striped tabby cat
with green eyes sitting on a …"*; the pizza photo gets *"A freshly baked pizza
with a golden crust, melted cheese …"*; and a text-only prompt on the same model
still answers *"Red, blue, green."*

The server path landed too (#1179), and #1180 closed the prefix-cache hole it
exposed: the cache is addressed by token ids and every image token carries the
same id, so two requests with different pictures shared a prefix until block
hashes were salted with the image content.

**Still open**, tracked as the remainder of gap 2 in [`../roadmap.md`](../roadmap.md):

- **One image per request.** Every `image_url` part is parsed but written into the
  same buffer, so the last one silently wins, and exactly one vision block is
  attached to the first user message. `expand_image_placeholders` and
  `qwen_build_mrope_positions` already take per-image counts and grids (with
  tests); the plumbing between the request parser and them does not.
- **Videos.** `temporal_patch_size` exists but only as a still-image repeat.
- **The interactive CLI and `imp_generate`** still take the mmproj-only branch, so
  `/image` in a chat session loads a picture the prompt never references.

## Corrections to this plan, found while building it

- **Piece 6 is interleaved, not sectioned.** Qwen3-VL sets
  `mrope_interleaved: true`: the axes alternate across the frequency spectrum
  (`T,H,W,T,H,W,…` and then a `T` tail past `3·section[axis]`), where Qwen2-VL
  takes three contiguous blocks. Different dimensions get different angles, so
  the two are not interchangeable — both are implemented.
- **`mrope_section` lives under two different keys.** `rope_scaling` in the
  Qwen2-VL generation, `rope_parameters` in Qwen3-VL. Reading one leaves a
  multimodal model on single-axis RoPE with no error at all.
- **An image costs `max(rows, cols)` positions, not its token count.** A 2×3
  image is six tokens and three positions.
- **The LM-side vision token order is raster over the MERGED grid**, while the
  encoder consumes patches in merge-block order. They agree because four
  consecutive encoder tokens form one merged token in raster order — but the
  two orders are written differently in the reference and are easy to conflate.

## Verification

**Correction to the original plan: an end-to-end oracle was not enough, and a
better one was available.** The plan below still holds for the final check, but
the encoder is verified against a from-scratch double-precision reimplementation
of `modeling_qwen3_vl.py`, written in the test
(`tests/test_qwen3vl_encoder.cu`). End-to-end tells you *something* is wrong; the
reference tells you *what*. The HF source is a `curl` away and should be read,
not recalled.

Two results from mutation-checking that reference are worth keeping:

- **A reference test can be structurally blind.** The first version could not
  catch a swapped RoPE axis. Small random gains made every LayerNorm shrink its
  activations until the attention logits sat within ±0.01 and softmax came out
  near-uniform — the rotary embedding had no influence on the output at all.
  Realistic magnitudes (norm gain near 1, Xavier linears) fixed it. Check the
  magnitudes of a synthetic fixture, not just its structure.
- **The two GELU variants are not distinguishable in FP16.** The block MLP's
  tanh-GELU and the mergers' erf-GELU differ by at most `4.7e-4`, below one FP16
  ulp at magnitude 1 (`9.8e-4`). The code follows upstream on the strength of
  the reference; no test covers it, and the test says so.

The end-to-end oracle still applies to the finished path: a VL model that
describes a test image correctly is strong evidence, because a wrong encoder
produces unrelated text rather than slightly-off text. This is the same standard that carried the MoE quantizer
bisection on 2026-07-31 (coherent vs cross-script garbage), and it is decisive
in practice.

Fixtures already in the tree: `tests/fixtures/vision_test_64.png` plus the
gemma-3 test images under `~/models/gemma-3-4b-vl/` (`test_bus.jpg`,
`test_cat.jpg`, `test_pizza.jpg`) — a model that names the bus, the cat and the
pizza is passing. `IMP_VISION_GOLDEN_DUMP` can pin encoder outputs afterwards to
catch regressions, but it is a self-golden and proves nothing about correctness
on its own.
