# shellcheck shell=bash
# Resolves the newest nvidia/cuda *devel* image tag; sourced by ptx_*_survey.sh so a survey
# always runs against the current CUDA toolkit.
# Honours $CUDA_IMG or --image; falls back to a pinned tag when curl/jq are missing or the
# registry query fails, so surveys still run offline.
latest_cuda_devel_img() {
    local fallback="nvidia/cuda:13.3.1-devel-ubuntu26.04"
    command -v curl >/dev/null 2>&1 && command -v jq >/dev/null 2>&1 || { echo "$fallback"; return; }
    local tag
    tag=$(curl -fsS --max-time 8 \
        "https://registry.hub.docker.com/v2/repositories/nvidia/cuda/tags?page_size=100&name=devel-ubuntu" 2>/dev/null \
        | jq -r '.results[].name
                 | select(test("^[0-9]+\\.[0-9]+\\.[0-9]+-devel-ubuntu[0-9]+\\.[0-9]+$"))' 2>/dev/null \
        | sort -Vr | head -1)
    [ -n "$tag" ] && echo "nvidia/cuda:$tag" || echo "$fallback"
}
