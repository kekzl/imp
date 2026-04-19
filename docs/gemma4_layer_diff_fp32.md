# Gemma-4 Layer-Diff: imp vs llama.cpp

Per-tensor sum comparison. Snapshot B = `attn_out-N` / imp post-attn. Snapshot C = `l_out-N` / imp post-layer (incl. layer_out_scale).

Divergence threshold: |rel_diff| > 0.05

## First significant divergence

**Step 0, Layer 0, Snapshot B**: imp_sum=-0.0003, lc_sum=-2433.9531, rel_diff=+100.00%

## Step 0 (prefill)

| Layer | Snap | imp_sum | lc_sum | abs_diff | rel_diff |
|---|---|---|---|---|---|
|  0 | B | -0.0003 | -2433.9531 | +2433.9528 | +100.00% ⚠️ |
|  0 | C | +nan | -88.1075 | +nan | +nan% |
|  1 | B | +nan | +408.6655 | +nan | +nan% |
|  1 | C | +nan | -280.4632 | +nan | +nan% |
|  2 | B | +nan | +266.6638 | +nan | +nan% |
|  2 | C | +nan | -69.6994 | +nan | +nan% |
|  3 | B | +nan | +280.8509 | +nan | +nan% |
|  3 | C | +nan | +68.5844 | +nan | +nan% |
|  4 | B | +nan | +139.9945 | +nan | +nan% |
|  4 | C | +nan | -69.2975 | +nan | +nan% |
|  5 | B | +nan | -291.0345 | +nan | +nan% |
|  5 | C | +nan | -654.6313 | +nan | +nan% |
|  6 | B | +nan | -682.0203 | +nan | +nan% |
|  6 | C | +nan | -422.5416 | +nan | +nan% |
|  7 | B | +nan | -373.9698 | +nan | +nan% |
|  7 | C | +nan | -459.4284 | +nan | +nan% |
|  8 | B | +nan | -397.3661 | +nan | +nan% |
|  8 | C | +nan | -309.7111 | +nan | +nan% |
|  9 | B | +nan | -183.4028 | +nan | +nan% |
|  9 | C | +nan | -315.2777 | +nan | +nan% |
| 10 | B | +nan | -146.8542 | +nan | +nan% |
| 10 | C | +nan | +190.1010 | +nan | +nan% |
| 11 | B | +nan | +246.1817 | +nan | +nan% |
| 11 | C | +nan | +304.6377 | +nan | +nan% |
| 12 | B | +nan | +396.9237 | +nan | +nan% |
| 12 | C | +nan | +421.1830 | +nan | +nan% |
| 13 | B | +nan | +372.2950 | +nan | +nan% |
| 13 | C | +nan | +297.4785 | +nan | +nan% |
| 14 | B | +nan | +172.5291 | +nan | +nan% |
| 14 | C | +nan | -34.0960 | +nan | +nan% |
| 15 | B | +nan | -194.1064 | +nan | +nan% |
| 15 | C | +nan | -211.4562 | +nan | +nan% |
| 16 | B | +nan | -92.9029 | +nan | +nan% |
| 16 | C | +nan | -249.9294 | +nan | +nan% |
| 17 | B | +nan | -62.3388 | +nan | +nan% |
| 17 | C | +nan | +34.8163 | +nan | +nan% |
| 18 | B | +nan | +188.8068 | +nan | +nan% |
| 18 | C | +nan | -214.9173 | +nan | +nan% |
| 19 | B | +nan | -98.4247 | +nan | +nan% |
| 19 | C | +nan | -103.3466 | +nan | +nan% |
| 20 | B | +nan | -116.0539 | +nan | +nan% |
| 20 | C | +nan | -210.2422 | +nan | +nan% |
| 21 | B | +nan | -302.3726 | +nan | +nan% |
| 21 | C | +nan | -107.6459 | +nan | +nan% |
| 22 | B | +nan | -299.2846 | +nan | +nan% |
| 22 | C | +nan | -62.6904 | +nan | +nan% |
| 23 | B | +nan | -453.9839 | +nan | +nan% |
| 23 | C | +nan | -328.0114 | +nan | +nan% |
| 24 | B | +nan | -390.9024 | +nan | +nan% |
| 24 | C | +nan | -291.1242 | +nan | +nan% |
| 25 | B | +nan | -417.5493 | +nan | +nan% |
| 25 | C | +nan | -215.2384 | +nan | +nan% |
| 26 | B | +nan | -354.9112 | +nan | +nan% |
| 26 | C | +nan | +28.9825 | +nan | +nan% |
| 27 | B | +nan | -19.9284 | +nan | +nan% |
| 27 | C | +nan | -27.5155 | +nan | +nan% |
| 28 | B | +nan | -266.2364 | +nan | +nan% |
| 28 | C | +nan | +326.5697 | +nan | +nan% |
| 29 | B | +nan | -120.6251 | +nan | +nan% |
| 29 | C | +nan | -61.2732 | +nan | +nan% |

