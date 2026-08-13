---
layer: L2
audience: kernel-devs
verified: 2026-08-13
commit: 81ffa573
---

# Gemma-4 vision (gemma4v) encoder — implementation spec

Reverse-engineered from llama.cpp master (`tools/mtmd/models/gemma4v.cpp`, `clip.cpp`,
`mtmd-image.cpp`). imp implements the gemma3 SigLIP path; gemma4v is a structurally
different encoder. This doc is the reference for porting it.

## GGUF facts (ggml-org/gemma-4-26B-A4B-it-GGUF mmproj)
- `clip.vision.projector_type = gemma4v`, projection_dim=2816, hidden=1152,
  block_count=27, head_count=16 (d_head=72), ffn=4304, eps=1e-6.
- image_size=224, patch_size=16, image_mean=[0,0,0], image_std=[1,1,1].
- Tensors: `mm.input_projection.weight` ne=[1152,2816] **[in,out] standard orient**,
  `v.patch_embd.weight` [16,16,3,1152] (NO bias), `v.position_embd.weight` ne=[1152,10240,2]
  (two axial tables: [:,:,0]=x/col, [:,:,1]=y/row), `v.std_bias` [1152], `v.std_scale` [1152].
- ABSENT vs gemma3: no `mm.soft_emb_norm`, no `v.post_ln`, no `v.patch_embd.bias`.
- n_merge = 3 (pool kernel), align = patch_size*n_merge = 48.

## Forward pass (exact)
```
# preprocess (host): variable aspect ratio, single image, no tiling
align=48; min_pixels=252*48²=580608; max_pixels=280*48²=645120
(w_bar,h_bar)=calc_size_preserved_ratio(W,H,align,min,max)  # both multiples of 48
img = bilinear_resize(orig,w_bar,h_bar); img/=255 (RGB)     # mean0/std1
n_px=w_bar/16; n_py=h_bar/16  (multiples of 3); n_patches=n_px*n_py

# graph
inp_raw = inp_raw*2 - 1                                       # → [-1,1]
inp = conv2d(patch_embd 16x16 s16, inp_raw) → [1152, n_patches]   # NO bias
pos_x[i]=i%n_px; pos_y[i]=i/n_px
inp += get_rows(pos_embd[:,:,0], pos_x) + get_rows(pos_embd[:,:,1], pos_y)
kq_scale=1.0; rope_theta=100
for il in 0..26:                          # RMSNorm blocks (NOT LayerNorm)
    h = rms_norm(x)*ln1_w
    Q=h·q_w+q_b; K=h·k_w+k_b; V=h·v_w+v_b      # 16 heads, d_head=72
    Q = concat(rope_neox(Q[:36],pos_x,θ100), rope_neox(Q[36:],pos_y,θ100))   # axial 2D RoPE
    K = concat(rope_neox(K[:36],pos_x,θ100), rope_neox(K[36:],pos_y,θ100))
    V = rms_norm(V)                            # gemma4v-only V-norm, no weight
    A = softmax(KᵀQ * kq_scale=1.0)·V          # no mask (bidirectional)
    x = x + (A·o_w+o_b)
    h2 = rms_norm(x)*ln2_w
    x = x + ffn_gelu_tanh(h2)                  # up/gate/down + biases
# NO post_ln
# pooler
g = avg_pool2d([n_px,n_py,1152], k=3, stride=3) → [1152, (n_px/3)*(n_py/3)]
g = g * sqrt(1152)
g = (g - std_bias) * std_scale                 # per-channel affine
# embedder
g = rms_norm(g)                                # no weight
out = mm_input_proj_w · g                      # [1152→2816], plain mul_mat, no bias
# → [2816, n_tokens], n_tokens≈252..280
```

## Risks / ambiguities (verify on hardware)
1. **kq_scale=1.0** — relies on HF checkpoint pre-scaling Q. If wrong, logits off ~8.5×.
2. d_head=72 (1152/16), RoPE half-dim=36.
3. ffn_op = gelu_pytorch_tanh (gemma) — confirm.
4. `Gemma4ClippableLinear` clamp only if per-weight min/max scalars present (this GGUF: none → plain).
5. Token count is DYNAMIC (252–280), not a fixed mm_tokens_per_image.

Source: llama.cpp master `tools/mtmd/models/gemma4v.cpp` + `clip.cpp`.
