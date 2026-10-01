#!/usr/bin/env bash
# Which local models take paged_attention_decode_kernel_generic on F16 KV decode (#2374).
# Rule (src/compute/attention_paged.cu paged_attention_decode, MHA branch): generic when
# n_heads/n_kv_heads is 1 or > 16 and head_dim has no template, or V is asymmetric (MLA).
# Before: templates 64 96 128 256 512, MLA vhd != hd always generic.
# After: + 192; MLA with nh == nkv decodes symmetric (zero-padded V) + compaction.
# Usage: tools/analysis/paged_decode_generic_models.sh [models_dir]   (bash + jq only)
set -u
DIR=${1:-$HOME/models}
printf '%-44s %4s %4s %4s %4s %-8s %-8s\n' model nh nkv hd vhd before after
for cfg in "$DIR"/*/config.json; do
    [ -f "$cfg" ] || continue
    name=$(basename "$(dirname "$cfg")")
    jq -r --arg name "$name" '
      (if .text_config != null then .text_config else . end) as $c
      | ($c.num_attention_heads // empty) as $nh
      | ($c.num_key_value_heads // $nh) as $nkv
      | (($c.qk_nope_head_dim // 0) + ($c.qk_rope_head_dim // 0)) as $mla_hd
      | (if $mla_hd > 0 then $mla_hd
         elif $c.head_dim != null then $c.head_dim
         else (($c.hidden_size // 0) / $nh | floor) end) as $hd
      | (if $mla_hd > 0 and $c.v_head_dim != null then $c.v_head_dim else $hd end) as $vhd
      | (if $nkv > 0 then ($nh / $nkv | floor) else 0 end) as $ratio
      | ($ratio == 1 or $ratio > 16) as $mha
      | def generic($set; $asym): $asym or ($mha and (($set | index($hd)) == null));
        generic([64, 96, 128, 256, 512]; $vhd != $hd) as $before
      | generic([64, 96, 128, 192, 256, 512]; $vhd != $hd and $nh != $nkv) as $after
      | [$name, $nh, $nkv, $hd, $vhd,
         (if $before then "GENERIC" else "ok" end), (if $after then "GENERIC" else "ok" end)]
      | @tsv' "$cfg" 2>/dev/null |
        while IFS=$'\t' read -r m nh nkv hd vhd b a; do
            printf '%-44s %4s %4s %4s %4s %-8s %-8s\n' "$m" "$nh" "$nkv" "$hd" "$vhd" "$b" "$a"
        done
done
