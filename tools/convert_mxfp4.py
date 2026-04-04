#!/usr/bin/env python3
"""Convert HuggingFace model weights to MXFP4 GGUF format.

MXFP4: 4-bit FP (E2M1) with UE8M0 block scales, 32 elements per block.
Tensor-core-native on Blackwell (sm_120) via CUTLASS block-scaled ops.

Usage:
    python tools/convert_mxfp4.py --model Qwen/Qwen3-4B --output qwen3-4b-mxfp4.gguf
"""

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# MXFP4 E2M1 quantization table (4-bit: values 0-15)
# E2M1: sign(1) + exp(2) + man(1), bias=1
# Values: 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0 (and negatives)
E2M1_TABLE = np.array([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,  # positive
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative
], dtype=np.float32)


def float_to_e2m1(val: float) -> int:
    """Quantize a float to FP4 E2M1 (4 bits). Returns 0-15."""
    sign = 1 if val < 0 else 0
    aval = abs(val)
    # Find nearest E2M1 value (positive magnitudes only: 0-7)
    pos_table = E2M1_TABLE[:8]
    idx = np.argmin(np.abs(pos_table - aval))
    return idx | (sign << 3)


def float_to_ue8m0(val: float) -> int:
    """Encode a positive float as UE8M0 (pure exponent, 8 bits).
    UE8M0 = 2^(bits - 127). Rounds up to next power of 2."""
    if val <= 0:
        return 0
    exp = math.ceil(math.log2(max(val, 1e-38)))
    bits = exp + 127
    return max(0, min(254, bits))


def ue8m0_to_float(bits: int) -> float:
    """Decode UE8M0 to float."""
    return 2.0 ** (bits - 127)


def quantize_mxfp4_block(values: np.ndarray) -> tuple[bytes, int]:
    """Quantize 32 float values to MXFP4 block (17 bytes).

    Returns (block_bytes, scale_byte).
    Block format: [16 bytes packed E2M1 | 1 byte UE8M0 scale]
    """
    assert len(values) == 32

    # Compute block scale: UE8M0 = ceil_pow2(absmax / 6.0)
    absmax = float(np.max(np.abs(values)))
    if absmax == 0:
        # All zeros: scale=0, all nibbles=0
        return bytes(17), 0

    # Scale such that max representable E2M1 value (6.0) covers absmax
    scale_float = absmax / 6.0
    scale_byte = float_to_ue8m0(scale_float)
    scale_val = ue8m0_to_float(scale_byte)

    if scale_val == 0:
        return bytes(17), 0

    # Quantize each value: val / scale → nearest E2M1
    nibbles = []
    for v in values:
        scaled = v / scale_val
        nibbles.append(float_to_e2m1(scaled))

    # Pack 32 nibbles into 16 bytes (2 nibbles per byte, lo|hi)
    packed = bytearray(16)
    for i in range(16):
        lo = nibbles[2 * i] & 0xF
        hi = nibbles[2 * i + 1] & 0xF
        packed[i] = lo | (hi << 4)

    # Append scale byte
    block = bytes(packed) + bytes([scale_byte])
    return block, scale_byte


def quantize_tensor_mxfp4(tensor: torch.Tensor) -> bytes:
    """Quantize a 2D tensor [N, K] to MXFP4 GGUF format (vectorized).

    Returns raw bytes: N * (K/32) blocks of 17 bytes each.
    """
    assert tensor.ndim == 2
    N, K = tensor.shape
    assert K % 32 == 0, f"K={K} must be multiple of 32"

    data = tensor.float().numpy()
    n_blocks = N * (K // 32)

    # Reshape to [n_blocks, 32]
    blocks = data.reshape(n_blocks, 32)

    # Compute per-block scales: UE8M0 = ceil_pow2(absmax / 6.0)
    absmax = np.max(np.abs(blocks), axis=1)  # [n_blocks]
    scale_float = absmax / 6.0
    # UE8M0 encoding: pure exponent byte
    scale_float = np.maximum(scale_float, 1e-38)
    log2_scale = np.ceil(np.log2(scale_float)).astype(np.int32)
    scale_bytes = np.clip(log2_scale + 127, 0, 254).astype(np.uint8)
    # Decode scales back for quantization
    scale_vals = np.ldexp(1.0, (scale_bytes.astype(np.int32) - 127))  # 2^(byte-127)
    scale_vals[absmax == 0] = 1.0  # avoid div by zero

    # Quantize: val/scale → nearest E2M1
    scaled = blocks / scale_vals[:, None]  # [n_blocks, 32]

    # E2M1 quantization: find nearest value in table
    pos_table = E2M1_TABLE[:8]  # [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    signs = (scaled < 0).astype(np.uint8)
    abs_scaled = np.abs(scaled)

    # Vectorized nearest-neighbor: broadcast [n_blocks, 32, 1] vs [8]
    diffs = np.abs(abs_scaled[:, :, None] - pos_table[None, None, :])
    indices = np.argmin(diffs, axis=2).astype(np.uint8)  # [n_blocks, 32]
    nibbles = indices | (signs << 3)  # [n_blocks, 32], 0-15

    # Pack pairs of nibbles into bytes: [n_blocks, 16]
    lo = nibbles[:, 0::2]  # even indices
    hi = nibbles[:, 1::2]  # odd indices
    packed = (lo | (hi << 4)).astype(np.uint8)  # [n_blocks, 16]

    # Build output: [16 bytes data | 1 byte scale] per block
    result = np.zeros((n_blocks, 17), dtype=np.uint8)
    result[:, :16] = packed
    result[:, 16] = scale_bytes

    return result.tobytes()


# ============================================================================
# GGUF Writer (minimal, for MXFP4 type)
# ============================================================================

GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian (bytes: G G U F)
GGUF_VERSION = 3

# GGML types
GGML_TYPE_MXFP4 = 31

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_BOOL = 1
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT32_ARRAY = 5  # for token types


class GGUFWriter:
    def __init__(self):
        self.kv_data = bytearray()
        self.kv_count = 0
        self.tensors = []  # (name, shape, type, data_bytes)

    def add_string(self, key: str, value: str):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<I', GGUF_TYPE_STRING))
        self._write_string_val(value)
        self.kv_count += 1

    def add_uint32(self, key: str, value: int):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<II', GGUF_TYPE_UINT32, value))
        self.kv_count += 1

    def add_int32(self, key: str, value: int):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<Ii', GGUF_TYPE_INT32, value))
        self.kv_count += 1

    def add_float32(self, key: str, value: float):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<If', GGUF_TYPE_FLOAT32, value))
        self.kv_count += 1

    def add_string_array(self, key: str, values: list[str]):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<I', GGUF_TYPE_ARRAY))
        self.kv_data.extend(struct.pack('<IQ', GGUF_TYPE_STRING, len(values)))
        for v in values:
            self._write_string_val(v)
        self.kv_count += 1

    def add_uint8_array(self, key: str, values: bytes):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<I', GGUF_TYPE_ARRAY))
        self.kv_data.extend(struct.pack('<IQ', GGUF_TYPE_UINT8, len(values)))
        self.kv_data.extend(values)
        self.kv_count += 1

    def add_int32_array(self, key: str, values: list[int]):
        self._write_key(key)
        self.kv_data.extend(struct.pack('<I', GGUF_TYPE_ARRAY))
        self.kv_data.extend(struct.pack('<IQ', GGUF_TYPE_INT32, len(values)))
        for v in values:
            self.kv_data.extend(struct.pack('<i', v))
        self.kv_count += 1

    def add_tensor(self, name: str, shape: tuple, ggml_type: int, data: bytes):
        self.tensors.append((name, shape, ggml_type, data))

    def write(self, path: str):
        n_tensors = len(self.tensors)

        # Build tensor info
        tensor_info = bytearray()
        data_offset = 0
        for name, shape, ggml_type, data in self.tensors:
            # Align data to 32 bytes
            alignment = 32
            padding = (alignment - (data_offset % alignment)) % alignment
            data_offset += padding

            # Name
            tensor_info.extend(struct.pack('<Q', len(name)))
            tensor_info.extend(name.encode('utf-8'))
            # ndims
            tensor_info.extend(struct.pack('<I', len(shape)))
            # shape (GGUF uses reverse order: innermost first)
            for d in reversed(shape):
                tensor_info.extend(struct.pack('<Q', d))
            # type
            tensor_info.extend(struct.pack('<I', ggml_type))
            # offset
            tensor_info.extend(struct.pack('<Q', data_offset))

            data_offset += len(data)

        # Write file
        with open(path, 'wb') as f:
            # Header
            f.write(struct.pack('<I', GGUF_MAGIC))
            f.write(struct.pack('<I', GGUF_VERSION))
            f.write(struct.pack('<Q', n_tensors))
            f.write(struct.pack('<Q', self.kv_count))

            # KV data
            f.write(self.kv_data)

            # Tensor info
            f.write(tensor_info)

            # Align to 32 bytes before tensor data
            pos = f.tell()
            alignment = 32
            padding = (alignment - (pos % alignment)) % alignment
            f.write(b'\x00' * padding)

            # Tensor data
            for _, _, _, data in self.tensors:
                # Align
                pos = f.tell()
                padding = (alignment - (pos % alignment)) % alignment
                f.write(b'\x00' * padding)
                f.write(data)

        print(f"Wrote {path} ({os.path.getsize(path) / 1024 / 1024:.1f} MB)")

    def _write_key(self, key: str):
        self.kv_data.extend(struct.pack('<Q', len(key)))
        self.kv_data.extend(key.encode('utf-8'))

    def _write_string_val(self, val: str):
        encoded = val.encode('utf-8')
        self.kv_data.extend(struct.pack('<Q', len(encoded)))
        self.kv_data.extend(encoded)


# ============================================================================
# HuggingFace → MXFP4 GGUF Converter
# ============================================================================

import os

# Weight name mapping: HuggingFace → GGUF
HF_TO_GGUF = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
    # Qwen3.5 multimodal: language_model prefix
    "model.language_model.embed_tokens.weight": "token_embd.weight",
    "model.language_model.norm.weight": "output_norm.weight",
    "model.language_model.lm_head.weight": "output.weight",
}

def hf_layer_to_gguf(name: str) -> str:
    """Map HuggingFace layer weight name to GGUF name."""
    # model.layers.{i}.{rest} → blk.{i}.{gguf_name}
    parts = name.split('.')
    # Handle both model.layers.{i}.* and model.language_model.layers.{i}.*
    layer_idx = None
    rest_start = None
    if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers':
        layer_idx = 2
        rest_start = 3
    elif len(parts) >= 5 and parts[0] == 'model' and parts[1] == 'language_model' and parts[2] == 'layers':
        layer_idx = 3
        rest_start = 4
    if layer_idx is not None:
        layer = parts[layer_idx]
        rest = '.'.join(parts[rest_start:])
        mapping = {
            # Standard attention
            'self_attn.q_proj.weight': 'attn_q.weight',
            'self_attn.k_proj.weight': 'attn_k.weight',
            'self_attn.v_proj.weight': 'attn_v.weight',
            'self_attn.o_proj.weight': 'attn_output.weight',
            'self_attn.q_norm.weight': 'attn_q_norm.weight',
            'self_attn.k_norm.weight': 'attn_k_norm.weight',
            # GDN (Gated DeltaNet) — Qwen3.5
            'linear_attn.in_proj_qkv.weight': 'attn_qkv.weight',
            'linear_attn.in_proj_z.weight': 'attn_gate.weight',
            'linear_attn.out_proj.weight': 'ssm_out.weight',
            'linear_attn.in_proj_a.weight': 'ssm_alpha.weight',
            'linear_attn.in_proj_b.weight': 'ssm_beta.weight',
            'linear_attn.conv1d.weight': 'ssm_conv1d.weight',
            'linear_attn.A_log': 'ssm_dt.weight',
            'linear_attn.dt_bias': 'ssm_dt.bias',
            'linear_attn.norm.weight': 'ssm_norm.weight',
            # FFN
            'mlp.gate_proj.weight': 'ffn_gate.weight',
            'mlp.up_proj.weight': 'ffn_up.weight',
            'mlp.down_proj.weight': 'ffn_down.weight',
            # Layer norms
            'input_layernorm.weight': 'attn_norm.weight',
            'post_attention_layernorm.weight': 'ffn_norm.weight',
        }
        if rest in mapping:
            return f"blk.{layer}.{mapping[rest]}"
    if name in HF_TO_GGUF:
        return HF_TO_GGUF[name]
    return None


def add_tokenizer(writer: GGUFWriter, model_path: str):
    """Extract tokenizer data from HuggingFace model and write to GGUF."""
    from transformers import AutoTokenizer
    import json

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Tokenizer type
    writer.add_string("tokenizer.ggml.model", "gpt2")  # BPE

    # Pre-tokenizer type (qwen2 for Qwen3 models)
    if hasattr(tokenizer, 'backend_tokenizer'):
        pre = tokenizer.backend_tokenizer.pre_tokenizer
        if pre is not None:
            pre_str = str(type(pre).__name__).lower()
            if 'byte' in pre_str:
                writer.add_string("tokenizer.ggml.pre", "qwen2")
            else:
                writer.add_string("tokenizer.ggml.pre", "default")

    # Vocabulary: extract token strings
    vocab_size = tokenizer.vocab_size
    if hasattr(tokenizer, 'get_vocab'):
        vocab_dict = tokenizer.get_vocab()
        # Sort by token ID
        tokens = [''] * vocab_size
        for tok_str, tok_id in vocab_dict.items():
            if tok_id < vocab_size:
                tokens[tok_id] = tok_str
        # Extend with added tokens
        added = getattr(tokenizer, 'added_tokens_encoder', {})
        max_id = max(list(vocab_dict.values()) + list(added.values())) if added else vocab_size - 1
        while len(tokens) <= max_id:
            tokens.append(f'[unused{len(tokens)}]')
        for tok_str, tok_id in added.items():
            if tok_id < len(tokens):
                tokens[tok_id] = tok_str
        writer.add_string_array("tokenizer.ggml.tokens", tokens)

        # Token types: 1=normal, 3=control/special
        token_types = [1] * len(tokens)
        all_special = set(tokenizer.all_special_ids) if hasattr(tokenizer, 'all_special_ids') else set()
        for sid in all_special:
            if sid < len(token_types):
                token_types[sid] = 3
        writer.add_int32_array("tokenizer.ggml.token_type", token_types)

    # BPE merge rules
    if hasattr(tokenizer, 'backend_tokenizer'):
        model = tokenizer.backend_tokenizer.model
        if hasattr(model, 'get_vocab') and hasattr(model, '__class__'):
            # Try to get merges from tokenizer.json
            try:
                tokenizer_json_path = Path(model_path) / "tokenizer.json"
                if not tokenizer_json_path.exists():
                    # Try huggingface cache
                    from transformers.utils import cached_file
                    tokenizer_json_path = cached_file(model_path, "tokenizer.json")
                with open(tokenizer_json_path, 'r') as f:
                    tok_json = json.load(f)
                merges = tok_json.get('model', {}).get('merges', [])
                if merges:
                    writer.add_string_array("tokenizer.ggml.merges", merges)
                    print(f"  Tokenizer: {len(tokens)} tokens, {len(merges)} merges")
            except Exception as e:
                print(f"  Warning: could not load merges: {e}")

    # Special tokens
    bos_id = getattr(tokenizer, 'bos_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    if bos_id is not None:
        writer.add_uint32("tokenizer.ggml.bos_token_id", bos_id)
    if eos_id is not None:
        # Handle list of eos tokens
        if isinstance(eos_id, list):
            writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id[0])
        else:
            writer.add_uint32("tokenizer.ggml.eos_token_id", eos_id)
    if pad_id is not None:
        writer.add_uint32("tokenizer.ggml.padding_token_id", pad_id)
    writer.add_uint32("tokenizer.ggml.add_bos_token", 0)

    # Chat template
    chat_template = getattr(tokenizer, 'chat_template', None)
    if chat_template:
        writer.add_string("tokenizer.chat_template", chat_template)
        print(f"  Chat template: {len(chat_template)} chars")


def hadamard_matrix(n: int) -> torch.Tensor:
    """Generate normalized Hadamard matrix of size n (must be power of 2)."""
    if n == 1:
        return torch.ones(1, 1)
    h_half = hadamard_matrix(n // 2)
    return torch.cat([
        torch.cat([h_half,  h_half], dim=1),
        torch.cat([h_half, -h_half], dim=1)
    ], dim=0)


def block_hadamard_rotate(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    """Apply block-diagonal Walsh-Hadamard rotation along K dimension.

    Each contiguous block of block_size columns is multiplied by
    (1/sqrt(block_size)) * H_{block_size}. This spreads outlier magnitudes
    across the block, improving uniform quantization quality.
    """
    N, K = weight.shape
    if K % block_size != 0:
        return weight  # can't rotate, return as-is
    H = hadamard_matrix(block_size).to(weight.dtype).to(weight.device)
    H = H / math.sqrt(block_size)
    # [N, K/bs, bs] × [bs, bs] → [N, K/bs, bs]
    w = weight.reshape(N, K // block_size, block_size)
    w = torch.einsum('nbk,mk->nbm', w, H)
    return w.reshape(N, K)


def convert_model(model_path: str, output_path: str, use_hadamard: bool = False):
    """Convert HuggingFace model to MXFP4 GGUF."""
    from transformers import AutoConfig, AutoModelForCausalLM
    import json

    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    # Qwen3.5 and other multimodal models nest text config
    text_cfg = getattr(config, 'text_config', config)

    # Load model weights lazily — process and discard each tensor individually
    # to avoid OOM on large models (27B+ needs ~54 GB in BF16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    )
    # Force garbage collection of unused model parts after state_dict extraction
    import gc

    writer = GGUFWriter()

    # Architecture metadata
    arch = "llama"
    model_type = getattr(text_cfg, 'model_type', getattr(config, 'model_type', ''))
    arch_map = {
        'llama': 'llama', 'mistral': 'mistral', 'qwen2': 'qwen3',
        'qwen3': 'qwen3', 'qwen3_5_text': 'qwen3', 'qwen3_5': 'qwen3',
        'gemma': 'gemma', 'gemma2': 'gemma',
    }
    arch = arch_map.get(model_type, 'llama')

    writer.add_string("general.architecture", arch)
    writer.add_string("general.name", getattr(config, '_name_or_path', model_path))
    writer.add_uint32(f"{arch}.context_length", getattr(text_cfg, 'max_position_embeddings', 4096))
    writer.add_uint32(f"{arch}.embedding_length", text_cfg.hidden_size)
    writer.add_uint32(f"{arch}.feed_forward_length", getattr(text_cfg, 'intermediate_size', text_cfg.hidden_size * 4))
    writer.add_uint32(f"{arch}.block_count", text_cfg.num_hidden_layers)
    writer.add_uint32(f"{arch}.attention.head_count", text_cfg.num_attention_heads)
    writer.add_uint32(f"{arch}.attention.head_count_kv", getattr(text_cfg, 'num_key_value_heads', text_cfg.num_attention_heads))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", getattr(text_cfg, 'rms_norm_eps', 1e-6))

    # GDN / SSM config (Qwen3.5 hybrid)
    if hasattr(text_cfg, 'linear_num_value_heads'):
        n_v_heads = text_cfg.linear_num_value_heads
        v_head_dim = getattr(text_cfg, 'linear_value_head_dim', 128)
        n_k_heads = getattr(text_cfg, 'linear_num_key_heads', n_v_heads)
        k_head_dim = getattr(text_cfg, 'linear_key_head_dim', v_head_dim)
        ssm_inner = n_v_heads * v_head_dim
        ssm_state = v_head_dim  # state_size = head_dim
        ssm_groups = n_k_heads  # group_count = n_key_heads
        ssm_dt_rank = n_v_heads  # dt_rank = n_value_heads
        conv_kernel = getattr(text_cfg, 'linear_conv_kernel_dim', 4)
        writer.add_uint32(f"{arch}.ssm.inner_size", ssm_inner)
        writer.add_uint32(f"{arch}.ssm.state_size", ssm_state)
        writer.add_uint32(f"{arch}.ssm.group_count", ssm_groups)
        writer.add_uint32(f"{arch}.ssm.time_step_rank", ssm_dt_rank)
        writer.add_uint32(f"{arch}.ssm.conv_kernel", conv_kernel)
        print(f"  SSM config: inner={ssm_inner} state={ssm_state} groups={ssm_groups} dt_rank={ssm_dt_rank} conv={conv_kernel}")

    # Partial RoPE (Qwen3.5: only 25% of head_dim gets rotary encoding)
    partial_rot = getattr(text_cfg, 'partial_rotary_factor', 0)
    if partial_rot > 0 and partial_rot < 1.0:
        hd = getattr(text_cfg, 'head_dim', 128)
        rope_dim = int(hd * partial_rot)
        writer.add_uint32(f"{arch}.rope.dimension_count", rope_dim)
        print(f"  Partial RoPE: factor={partial_rot} → rope_dim={rope_dim}")

    # RoPE
    rope_theta = getattr(text_cfg, 'rope_theta', None)
    if rope_theta is None:
        rope_params = getattr(text_cfg, 'rope_parameters', {})
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get('rope_theta', None)
    if rope_theta is not None:
        writer.add_float32(f"{arch}.rope.freq_base", float(rope_theta))

    # Tokenizer
    print("Extracting tokenizer...")
    add_tokenizer(writer, model_path)

    # Vocab size
    writer.add_uint32(f"{arch}.vocab_size", text_cfg.vocab_size)

    # Extract state dict and free the model to save ~50 GB RAM
    state_dict = model.state_dict()
    del model
    gc.collect()

    # Derive head_dim from Q projection shape (handles QK-norm models)
    q_proj_key = 'model.layers.0.self_attn.q_proj.weight'
    # For GDN hybrids, first attention layer might not be layer 0
    if q_proj_key not in state_dict:
        for i in range(text_cfg.num_hidden_layers):
            k = f'model.layers.{i}.self_attn.q_proj.weight'
            if k in state_dict:
                q_proj_key = k
                break
    head_dim = getattr(text_cfg, 'head_dim', 0)
    if q_proj_key in state_dict:
        q_out_dim = state_dict[q_proj_key].shape[0]
        head_dim = q_out_dim // text_cfg.num_attention_heads
    if head_dim > 0:
        writer.add_uint32(f"{arch}.attention.key_length", head_dim)
        writer.add_uint32(f"{arch}.attention.value_length", head_dim)
        print(f"  head_dim={head_dim}")

    # Hadamard metadata
    had_attn_bs = 0
    had_ffn_bs = 0
    if use_hadamard:
        had_attn_bs = head_dim if 'head_dim' in dir() else 128
        had_ffn_bs = 128
        # Find largest power of 2 ≤ block size
        for bs in [had_attn_bs, had_ffn_bs]:
            assert bs > 0 and (bs & (bs - 1)) == 0, f"Hadamard block size {bs} must be power of 2"
        writer.add_uint32("mxfp4.hadamard_block_size_attn", had_attn_bs)
        writer.add_uint32("mxfp4.hadamard_block_size_ffn", had_ffn_bs)
        print(f"  Hadamard: attn_bs={had_attn_bs}, ffn_bs={had_ffn_bs}")

    n_quantized = 0
    n_fp16 = 0

    # Process weights one at a time, freeing each after quantization to stay under RAM limit
    keys = list(state_dict.keys())
    for hf_name in keys:
        gguf_name = hf_layer_to_gguf(hf_name)
        if gguf_name is None:
            print(f"  skip: {hf_name}")
            del state_dict[hf_name]
            continue

        tensor = state_dict.pop(hf_name)  # pop to free immediately
        shape = tuple(tensor.shape)

        # Decide: quantize 2D weight matrices, keep norms/embeddings as FP16
        is_weight_matrix = tensor.ndim == 2 and 'norm' not in hf_name and 'embed' not in hf_name
        K = shape[-1] if tensor.ndim >= 2 else 0
        can_quantize = is_weight_matrix and K % 32 == 0

        if can_quantize:
            # For large tensors (>50M elements), process in row chunks to avoid OOM
            N = shape[0]
            chunk_rows = max(1, min(N, 50_000_000 // max(shape[-1], 1)))
            if N > chunk_rows:
                chunks = []
                for start in range(0, N, chunk_rows):
                    end = min(start + chunk_rows, N)
                    t = tensor[start:end].float()
                    if use_hadamard:
                        is_attn = 'attn' in gguf_name
                        bs = had_attn_bs if is_attn else had_ffn_bs
                        if K % bs == 0:
                            t = block_hadamard_rotate(t, bs)
                    chunks.append(quantize_tensor_mxfp4(t))
                    del t
                data = b''.join(chunks)
                del chunks
            else:
                t = tensor.float()
                if use_hadamard:
                    is_attn = 'attn' in gguf_name
                    bs = had_attn_bs if is_attn else had_ffn_bs
                    if K % bs == 0:
                        t = block_hadamard_rotate(t, bs)
                data = quantize_tensor_mxfp4(t)
                del t
            del tensor
            writer.add_tensor(gguf_name, shape, GGML_TYPE_MXFP4, data)
            n_quantized += 1
            bits_per_weight = len(data) * 8 / (shape[0] * shape[1])
            tag = "MXFP4+H" if use_hadamard else "MXFP4"
            print(f"  {tag}: {gguf_name} {list(shape)} → {len(data)/1024/1024:.1f} MB ({bits_per_weight:.2f} bpw)")
            del data
        else:
            # FP16
            fp16_data = tensor.half().numpy().tobytes()
            del tensor
            writer.add_tensor(gguf_name, shape, 1, fp16_data)  # GGMLType::F16 = 1
            n_fp16 += 1
            print(f"  FP16:  {gguf_name} {list(shape)} → {len(fp16_data)/1024/1024:.1f} MB")
            del fp16_data
        gc.collect()

    print(f"\nQuantized {n_quantized} tensors to MXFP4, {n_fp16} tensors kept as FP16")
    writer.write(output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace model to MXFP4 GGUF')
    parser.add_argument('--model', required=True, help='HuggingFace model name or path')
    parser.add_argument('--output', required=True, help='Output GGUF file path')
    parser.add_argument('--hadamard', action='store_true',
                        help='Apply block-diagonal Hadamard rotation before quantization (improves quality)')
    args = parser.parse_args()

    convert_model(args.model, args.output, use_hadamard=args.hadamard)


if __name__ == '__main__':
    main()
