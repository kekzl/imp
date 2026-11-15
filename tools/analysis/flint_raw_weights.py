"""Raw SafeTensors reader with Modelopt NVFP4 dequantization (W4, FP32 out): no torch fp8 dtype needed.

Modelopt layout per quantized Linear: `<name>` U8 [N, K/2] (two E2M1 nibbles per byte, low nibble = even
column), `<name>_scale` F8_E4M3 [N, K/16], `<name>_scale_2` F32 scalar; w = fp4 * scale * scale_2.
"""
import json
import mmap
import pathlib
import struct

import numpy as np
import torch

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def _e4m3_lut():
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if b & 0x80 else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        out[b] = s * (m / 8.0) * 2.0 ** -6 if e == 0 else s * (1 + m / 8.0) * 2.0 ** (e - 7)
    return out


E4M3 = _e4m3_lut()
NP = {"F32": np.float32, "F16": np.float16, "U8": np.uint8, "I8": np.int8, "F8_E4M3": np.uint8, "BF16": np.uint16,
      "I64": np.int64, "I32": np.int32}


class RawWeights:
    def __init__(self, model_dir):
        self.dir = pathlib.Path(model_dir)
        self.map = json.loads((self.dir / "model.safetensors.index.json").read_text())["weight_map"]
        self.files = {}

    def _file(self, f):
        if f not in self.files:
            fh = open(self.dir / f, "rb")
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            n = struct.unpack("<Q", mm[:8])[0]
            self.files[f] = (mm, json.loads(mm[8:8 + n]), 8 + n)
        return self.files[f]

    def has(self, name):
        return name in self.map

    def raw(self, name):
        mm, hdr, base = self._file(self.map[name])
        m = hdr[name]
        a = np.frombuffer(mm, dtype=NP[m["dtype"]], count=int(np.prod(m["shape"], dtype=np.int64)) if m["shape"] else 1,
                          offset=base + m["data_offsets"][0])
        return a.reshape(m["shape"]), m["dtype"]

    def get(self, name, dtype=torch.float32):
        a, dt = self.raw(name)
        if dt == "U8" and self.has(name + "_scale"):
            n, k2 = a.shape
            q = np.empty((n, k2 * 2), dtype=np.float32)
            q[:, 0::2] = E2M1[a & 0xF]
            q[:, 1::2] = E2M1[a >> 4]
            sc, _ = self.raw(name + "_scale")
            s2, _ = self.raw(name + "_scale_2")
            q *= np.repeat(E4M3[sc], 16, axis=1)
            q *= float(np.asarray(s2).reshape(-1)[0])
            return torch.from_numpy(q).to(dtype)
        if dt == "BF16":
            return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16).to(dtype)
        if dt == "F8_E4M3":
            return torch.from_numpy(E4M3[a]).to(dtype)
        return torch.from_numpy(a.copy()).to(dtype)
