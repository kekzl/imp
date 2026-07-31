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
  mergers **after** it. Do not copy one from the other.

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
   non-48×48 patch grid needs it resampled. HF interpolates bilinearly.
3. **Patch embed over the temporal axis.** `temporal_patch_size 2` means a still
   image is repeated along t before the projection; the flattened `[1024, 1536]`
   matrix already expects `3·2·16·16` inputs, so the preprocessor must build
   patches in that order.
4. **Merger.** 2×2 concat → norm → fc1 → gelu → fc2 → `[*, 2560]`. Straight
   GEMMs; note the norm placement above.
5. **DeepStack.** Blocks 5/11/17 tap their hidden state through their own merger.
   **Open question — do not guess:** where the three results enter the LM. The
   HF implementation adds them to the text hidden state at the first layers, but
   that must be read off `modeling_qwen3_vl.py` before implementing, not
   inferred from the config.
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
   (32) within the model's pixel bounds; `preprocessor_config.json` is staged.

## Suggested order

3 → 1 → 2 → 4 gets an image to `[*, 2560]` embeddings. Then 6, then 5. Piece 6
is the one that can regress existing models, so it should land on its own with
the invariant test rather than inside a large vision commit.

## Verification

**No reference implementation is required, and none is available here** (no
torch on this host, no llama.cpp Qwen3-VL mmproj staged). The oracle is
end-to-end: a VL model that describes a test image correctly is strong evidence
the encoder is right, because a wrong encoder produces unrelated text rather
than slightly-off text. This is the same standard that carried the MoE quantizer
bisection on 2026-07-31 (coherent vs cross-script garbage), and it is decisive
in practice.

Fixtures already in the tree: `tests/fixtures/vision_test_64.png` plus the
gemma-3 test images under `~/models/gemma-3-4b-vl/` (`test_bus.jpg`,
`test_cat.jpg`, `test_pizza.jpg`) — a model that names the bus, the cat and the
pizza is passing. `IMP_VISION_GOLDEN_DUMP` can pin encoder outputs afterwards to
catch regressions, but it is a self-golden and proves nothing about correctness
on its own.
