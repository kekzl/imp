"""Writes tests/fixtures/jpeg: synthetic JPEGs (Pillow encode) plus Pillow's RGB decode of each.

Reference = Image.open(f).convert("RGB").tobytes(), the decode HF image processors see.
Odd size 45x37: partial MCUs and chroma edge rows on every subsampling.
"""

import hashlib
import random
import sys

import PIL
from PIL import Image, features

W, H = 45, 37


def synth() -> Image.Image:
    rng = random.Random(2381)
    im = Image.new("RGB", (W, H))
    px = im.load()
    for y in range(H):
        for x in range(W):
            r = (x * 255) // (W - 1)
            g = (y * 255) // (H - 1)
            b = 255 if (x // 6 + y // 5) % 2 else 30
            n = rng.randint(-40, 40)
            px[x, y] = tuple(max(0, min(255, v + n)) for v in (r, g, b))
    return im


def main(out: str) -> None:
    src = synth()
    cases = {
        "baseline_420.jpg": (src, dict(quality=90, subsampling=2)),
        "progressive_420.jpg": (src, dict(quality=90, subsampling=2, progressive=True)),
        "baseline_444.jpg": (src, dict(quality=90, subsampling=0)),
        "baseline_422.jpg": (src, dict(quality=75, subsampling=1)),
        "gray.jpg": (src.convert("L"), dict(quality=90)),
        "cmyk.jpg": (src.convert("CMYK"), dict(quality=90)),
    }
    lines = [
        f"# Pillow {PIL.__version__}, libjpeg-turbo {features.version('libjpeg_turbo')}; "
        "regenerate: tools/jpeg_fixture/run.sh",
        "# name width height sha256(rgb)",
    ]
    for name, (im, kw) in cases.items():
        im.save(f"{out}/{name}", **kw)
        rgb = Image.open(f"{out}/{name}").convert("RGB")
        data = rgb.tobytes()
        open(f"{out}/{name[:-4]}.rgb", "wb").write(data)
        lines.append(f"{name} {rgb.width} {rgb.height} {hashlib.sha256(data).hexdigest()}")
    open(f"{out}/pillow_ref.txt", "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
