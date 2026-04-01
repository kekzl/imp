#!/usr/bin/env python3
"""Create a minimal MXFP4 GGUF file for testing the loader."""

import struct
import math
import sys
import os
import numpy as np

MAGIC = 0x46554747  # 'GGUF' little-endian
VERSION = 3
MXFP4_TYPE = 31
FP16_TYPE = 1

V, D, FF, HL, NH = 32, 64, 128, 1, 4
np.random.seed(42)

E2M1_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0])

def quantize_mxfp4(data):
    N, K = data.shape
    blocks = data.reshape(-1, 32)
    n = len(blocks)
    absmax = np.max(np.abs(blocks), axis=1)
    sf = np.maximum(absmax / 6.0, 1e-38)
    log2_sf = np.ceil(np.log2(sf)).astype(np.int32)
    sb = np.clip(log2_sf + 127, 0, 254).astype(np.uint8)
    sv = np.ldexp(1.0, sb.astype(np.int32) - 127)
    sv[absmax == 0] = 1.0
    scaled = blocks / sv[:, None]
    signs = (scaled < 0).astype(np.uint8)
    diffs = np.abs(np.abs(scaled)[:, :, None] - E2M1_TABLE[:8][None, None, :])
    idx = np.argmin(diffs, axis=2).astype(np.uint8)
    nib = idx | (signs << 3)
    packed = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
    result = np.zeros((n, 17), dtype=np.uint8)
    result[:, :16] = packed
    result[:, 16] = sb
    return result.tobytes()

def write_str(f, s):
    b = s.encode()
    f.write(struct.pack('<Q', len(b)))
    f.write(b)

def main():
    output = sys.argv[1] if len(sys.argv) > 1 else "test-mxfp4.gguf"

    tensors = [
        ('token_embd.weight', (V, D), FP16_TYPE, (np.random.randn(V, D)*0.02).astype(np.float16).tobytes()),
        ('output_norm.weight', (D,), FP16_TYPE, (np.random.randn(D)*0.1).astype(np.float16).tobytes()),
        ('output.weight', (V, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(V, D)*0.02).astype(np.float32))),
        ('blk.0.attn_q.weight', (D, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(D, D)*0.02).astype(np.float32))),
        ('blk.0.attn_k.weight', (D, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(D, D)*0.02).astype(np.float32))),
        ('blk.0.attn_v.weight', (D, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(D, D)*0.02).astype(np.float32))),
        ('blk.0.attn_output.weight', (D, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(D, D)*0.02).astype(np.float32))),
        ('blk.0.ffn_gate.weight', (FF, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(FF, D)*0.02).astype(np.float32))),
        ('blk.0.ffn_up.weight', (FF, D), MXFP4_TYPE, quantize_mxfp4((np.random.randn(FF, D)*0.02).astype(np.float32))),
        ('blk.0.ffn_down.weight', (D, FF), MXFP4_TYPE, quantize_mxfp4((np.random.randn(D, FF)*0.02).astype(np.float32))),
        ('blk.0.attn_norm.weight', (D,), FP16_TYPE, (np.random.randn(D)*0.1).astype(np.float16).tobytes()),
        ('blk.0.ffn_norm.weight', (D,), FP16_TYPE, (np.random.randn(D)*0.1).astype(np.float16).tobytes()),
    ]

    kv = [
        ('general.architecture', 's', 'llama'),
        ('general.name', 's', 'test-mxfp4'),
        ('llama.context_length', 'u', 256),
        ('llama.embedding_length', 'u', D),
        ('llama.feed_forward_length', 'u', FF),
        ('llama.block_count', 'u', HL),
        ('llama.attention.head_count', 'u', NH),
        ('llama.attention.head_count_kv', 'u', NH),
        ('llama.attention.layer_norm_rms_epsilon', 'f', 1e-6),
        ('llama.vocab_size', 'u', V),
        ('llama.rope.freq_base', 'f', 10000.0),
    ]

    with open(output, 'wb') as f:
        f.write(struct.pack('<I', MAGIC))
        f.write(struct.pack('<I', VERSION))
        f.write(struct.pack('<Q', len(tensors)))
        f.write(struct.pack('<Q', len(kv)))

        for k, typ, v in kv:
            write_str(f, k)
            if typ == 's':
                f.write(struct.pack('<I', 8))
                write_str(f, v)
            elif typ == 'f':
                f.write(struct.pack('<If', 6, v))
            elif typ == 'u':
                f.write(struct.pack('<II', 4, v))

        data_offset = 0
        entries = []
        for name, shape, dtype, data in tensors:
            padding = (32 - (data_offset % 32)) % 32
            data_offset += padding
            entries.append((name, shape, dtype, data_offset, data))
            data_offset += len(data)

        for name, shape, dtype, off, data in entries:
            write_str(f, name)
            f.write(struct.pack('<I', len(shape)))
            for d in reversed(shape):
                f.write(struct.pack('<Q', d))
            f.write(struct.pack('<I', dtype))
            f.write(struct.pack('<Q', off))

        pos = f.tell()
        f.write(b'\x00' * ((32 - pos % 32) % 32))
        data_start = f.tell()

        for _, _, _, off, data in entries:
            target = data_start + off
            cur = f.tell()
            if cur < target:
                f.write(b'\x00' * (target - cur))
            f.write(data)

    size = os.path.getsize(output)
    print(f"Wrote {output}: {size} bytes ({size/1024:.1f} KB)")

if __name__ == '__main__':
    main()
