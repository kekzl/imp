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
    """Quantize a 2D tensor [N, K] to MXFP4 GGUF format.

    Returns raw bytes: N * (K/32) blocks of 17 bytes each.
    """
    assert tensor.ndim == 2
    N, K = tensor.shape
    assert K % 32 == 0, f"K={K} must be multiple of 32"

    data = tensor.float().numpy()
    blocks_per_row = K // 32
    result = bytearray()

    for row in range(N):
        for blk in range(blocks_per_row):
            values = data[row, blk * 32 : (blk + 1) * 32]
            block_bytes, _ = quantize_mxfp4_block(values)
            result.extend(block_bytes)

    return bytes(result)


# ============================================================================
# GGUF Writer (minimal, for MXFP4 type)
# ============================================================================

GGUF_MAGIC = 0x46475547  # 'GGUF'
GGUF_VERSION = 3

# GGML types
GGML_TYPE_MXFP4 = 31

# GGUF value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT8 = 0


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
}

def hf_layer_to_gguf(name: str) -> str:
    """Map HuggingFace layer weight name to GGUF name."""
    # model.layers.{i}.{rest} → blk.{i}.{gguf_name}
    parts = name.split('.')
    if len(parts) >= 4 and parts[0] == 'model' and parts[1] == 'layers':
        layer = parts[2]
        rest = '.'.join(parts[3:])
        mapping = {
            'self_attn.q_proj.weight': 'attn_q.weight',
            'self_attn.k_proj.weight': 'attn_k.weight',
            'self_attn.v_proj.weight': 'attn_v.weight',
            'self_attn.o_proj.weight': 'attn_output.weight',
            'mlp.gate_proj.weight': 'ffn_gate.weight',
            'mlp.up_proj.weight': 'ffn_up.weight',
            'mlp.down_proj.weight': 'ffn_down.weight',
            'input_layernorm.weight': 'attn_norm.weight',
            'post_attention_layernorm.weight': 'ffn_norm.weight',
        }
        if rest in mapping:
            return f"blk.{layer}.{mapping[rest]}"
    if name in HF_TO_GGUF:
        return HF_TO_GGUF[name]
    return None


def convert_model(model_path: str, output_path: str):
    """Convert HuggingFace model to MXFP4 GGUF."""
    from transformers import AutoConfig, AutoModelForCausalLM
    import json

    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float32, device_map="cpu"
    )

    writer = GGUFWriter()

    # Architecture metadata
    arch = "llama"  # default, most models use LLaMA architecture
    if hasattr(config, 'model_type'):
        arch_map = {
            'llama': 'llama', 'mistral': 'mistral', 'qwen2': 'qwen3',
            'qwen3': 'qwen3', 'gemma': 'gemma', 'gemma2': 'gemma',
        }
        arch = arch_map.get(config.model_type, 'llama')

    writer.add_string("general.architecture", arch)
    writer.add_string("general.name", getattr(config, '_name_or_path', model_path))
    writer.add_uint32(f"{arch}.context_length", getattr(config, 'max_position_embeddings', 4096))
    writer.add_uint32(f"{arch}.embedding_length", config.hidden_size)
    writer.add_uint32(f"{arch}.feed_forward_length", getattr(config, 'intermediate_size', config.hidden_size * 4))
    writer.add_uint32(f"{arch}.block_count", config.num_hidden_layers)
    writer.add_uint32(f"{arch}.attention.head_count", config.num_attention_heads)
    writer.add_uint32(f"{arch}.attention.head_count_kv", getattr(config, 'num_key_value_heads', config.num_attention_heads))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", getattr(config, 'rms_norm_eps', 1e-6))
    writer.add_uint32(f"{arch}.vocab_size", config.vocab_size)

    # RoPE
    if hasattr(config, 'rope_theta'):
        writer.add_float32(f"{arch}.rope.freq_base", config.rope_theta)

    # Quantize and add tensors
    state_dict = model.state_dict()
    n_quantized = 0
    n_fp16 = 0

    for hf_name, tensor in state_dict.items():
        gguf_name = hf_layer_to_gguf(hf_name)
        if gguf_name is None:
            print(f"  skip: {hf_name}")
            continue

        shape = tuple(tensor.shape)

        # Decide: quantize 2D weight matrices, keep norms/embeddings as FP16
        is_weight_matrix = tensor.ndim == 2 and 'norm' not in hf_name and 'embed' not in hf_name
        K = shape[-1] if tensor.ndim >= 2 else 0
        can_quantize = is_weight_matrix and K % 32 == 0

        if can_quantize:
            data = quantize_tensor_mxfp4(tensor)
            writer.add_tensor(gguf_name, shape, GGML_TYPE_MXFP4, data)
            n_quantized += 1
            bits_per_weight = len(data) * 8 / tensor.numel()
            print(f"  MXFP4: {gguf_name} {list(shape)} → {len(data)/1024/1024:.1f} MB ({bits_per_weight:.2f} bpw)")
        else:
            # FP16
            fp16_data = tensor.half().numpy().tobytes()
            writer.add_tensor(gguf_name, shape, 1, fp16_data)  # GGMLType::F16 = 1
            n_fp16 += 1
            print(f"  FP16:  {gguf_name} {list(shape)} → {len(fp16_data)/1024/1024:.1f} MB")

    print(f"\nQuantized {n_quantized} tensors to MXFP4, {n_fp16} tensors kept as FP16")
    writer.write(output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert HuggingFace model to MXFP4 GGUF')
    parser.add_argument('--model', required=True, help='HuggingFace model name or path')
    parser.add_argument('--output', required=True, help='Output GGUF file path')
    args = parser.parse_args()

    convert_model(args.model, args.output)


if __name__ == '__main__':
    main()
