#!/usr/bin/env bash
# Dependency-pin gate.
#
# Three failures this catches. The first two happened on 2026-08-14 and were
# invisible until a Docker layer cache went cold:
#
#   1. DRIFT. The pins live in cmake/imp-deps.cmake and are injected into the
#      Docker build by scripts/dep_build_args.sh, but the Dockerfile also
#      carries ARG defaults for anyone running `docker build` directly. Those
#      defaults had fallen a release behind (CUTLASS v4.6.2 vs the pinned
#      4.7.0), so the same tree built two different dependency sets depending on
#      how it was invoked. AGENTS.md says "bump both dep-pin sites together";
#      nothing enforced it.
#
#   2. A TAG THAT DOES NOT EXIST UPSTREAM. The CUTLASS pin read `4.7.0` while
#      every tag in that repo carries a `v` prefix. `git clone --branch 4.7.0`
#      fails, so every build from a cold cache died -- while cached builds kept
#      working and CI stayed green.
#
#   4. AN UNVERIFIED DOWNLOAD (AUDIT_arch_2026 H-6). The pins guard the four
#      FetchContent deps and nothing else the build pulls: the CMake installer
#      was fetched and executed with no checksum, git-clang-format came off a
#      *branch* onto a runner holding GITHUB_TOKEN, base images floated on tags
#      and pip resolved its transitive set fresh on every run. Every download
#      now names the bytes it expects: image digests, sha256 for single files,
#      hash-pinned lock files for pip.
#
#   3. UPSTREAM MUTATION (AUDIT_arch_2026 H-8). Every pin used to be a mutable
#      ref: four tags, four `git clone --branch`, nine `uses:` action majors,
#      zero commit SHAs. A re-tag or a compromised action release changed what
#      the published image contains with nothing in this repo moving. Each dep
#      now carries a TAG *and* a SHA, the build fetches the SHA, and --online
#      asserts the tag still resolves to it -- that assertion failing is the
#      re-tag alarm. Actions are pinned to 40-hex SHAs, checked offline.
#
# All checks read the real sources (cmake/imp-deps.cmake, the Dockerfile's own
# ARG/fetch lines, the workflow files) rather than keeping a copy of the truth.
#
# usage: check_dep_pins.sh [--online]
#   default : drift + form + download checks, no network
#   --online: additionally resolve every tag against its upstream remote
#   --selftest: run every check against a fixture tree with known violations
set -uo pipefail

# IMP_PINS_ROOT lets --selftest point the checks at a fixture tree.
ROOT="${IMP_PINS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# --selftest: run the checks against a fixture tree whose violations are known.
# A gate that stops parsing (a renamed field, a reformatted line) goes quiet
# rather than red, and this repo has caught five such tools that way. Each case
# builds the fixture, plants one violation and asserts the gate names it.
if [ "${1:-}" = "--selftest" ]; then
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    st_pass=0; st_fail=0

    fixture() {  # writes a clean, passing tree into $tmp
        rm -rf "$tmp"; mkdir -p "$tmp/cmake" "$tmp/.github/workflows" "$tmp/reqs"
        cat > "$tmp/cmake/imp-deps.cmake" <<'EOF'
set(IMP_DEP_FOO_TAG v1.0.0)
set(IMP_DEP_FOO_SHA 0123456789abcdef0123456789abcdef01234567)
EOF
        cat > "$tmp/Dockerfile" <<'EOF'
FROM example.com/base:1.0@sha256:1111111111111111111111111111111111111111111111111111111111111111 AS toolchain
ARG IMP_DEP_FOO_TAG=v1.0.0
ARG IMP_DEP_FOO_SHA=0123456789abcdef0123456789abcdef01234567
RUN pin https://example.com/foo.git ${IMP_DEP_FOO_TAG} ${IMP_DEP_FOO_SHA} /deps/foo
RUN wget -qO /tmp/x.sh https://example.com/x.sh \
    && echo '2222222222222222222222222222222222222222222222222222222222222222  /tmp/x.sh' | sha256sum -c - \
    && sh /tmp/x.sh
COPY reqs.txt /tmp/reqs.txt
RUN pip install -r /tmp/reqs.txt
EOF
        cat > "$tmp/reqs.txt" <<'EOF'
foo==1.2.3 \
    --hash=sha256:3333333333333333333333333333333333333333333333333333333333333333
EOF
        cat > "$tmp/.github/workflows/w.yml" <<'EOF'
jobs:
  j:
    container:
      image: example.com/base:1.0@sha256:1111111111111111111111111111111111111111111111111111111111111111
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
      - run: |
          curl -fsSL https://raw.githubusercontent.com/o/r/3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff/f \
            -o /tmp/f
          echo '4444444444444444444444444444444444444444444444444444444444444444  /tmp/f' | sha256sum -c -
EOF
    }

    st_case() {  # name, expected-substring ("" = must pass)
        out="$(IMP_PINS_ROOT="$tmp" bash "${BASH_SOURCE[0]}" 2>&1)"; rc=$?
        if [ -z "$2" ]; then
            [ "$rc" = "0" ] && { st_pass=$((st_pass+1)); return; }
            printf 'SELFTEST FAIL  %s: expected a pass, got:\n%s\n' "$1" "$(printf '%s' "$out" | grep '^FAIL' | head -2)"
        elif [ "$rc" != "0" ] && printf '%s' "$out" | grep -qF "$2"; then
            st_pass=$((st_pass+1)); return
        else
            printf 'SELFTEST FAIL  %s: no FAIL naming %s (rc=%s)\n' "$1" "$2" "$rc"
        fi
        st_fail=$((st_fail+1))
    }

    fixture; st_case "clean fixture" ""
    fixture; sed -i 's|@sha256:1111[0-9a-f]*||' "$tmp/Dockerfile"
    st_case "image on a bare tag" "is tag-pinned"
    fixture; sed -i 's|@sha256:1111111111111111111111111111111111111111111111111111111111111111|@sha256:9999999999999999999999999999999999999999999999999999999999999999|' "$tmp/.github/workflows/w.yml"
    st_case "one tag, two digests" "one tag, one digest"
    fixture; sed -i '/sha256sum -c -/d; s|https://example.com/x.sh \\|https://example.com/x.sh|' "$tmp/Dockerfile"
    st_case "download with no checksum" "with no 'echo <sha256>"
    fixture; sed -i 's|/3b5b5c1ec4a3095ab096dd780e84d7ab81f3d7ff/|/main/|' "$tmp/.github/workflows/w.yml"
    st_case "raw fetch off a branch" "a branch or tag is a mutable ref"
    fixture; sed -i 's|pip install -r /tmp/reqs.txt|pip install foo==1.2.3|' "$tmp/Dockerfile"
    st_case "pip without a lock file" "without '-r <lock file>'"
    fixture; sed -i '/--hash=sha256:/d; s|^foo==1.2.3 \\|foo==1.2.3|' "$tmp/reqs.txt"
    st_case "requirement with no hash" "carries no --hash"
    fixture; sed -i 's|^foo==1.2.3|foo>=1.2.3|' "$tmp/reqs.txt"
    st_case "pin loosened to >=" "is not an exact pin"
    fixture; sed -i '/^ARG IMP_DEP_FOO_SHA=/d' "$tmp/Dockerfile"
    st_case "Dockerfile loses a SHA default" "no 'ARG IMP_DEP_FOO_SHA='"
    fixture; sed -i 's|actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1|actions/checkout@v7|' "$tmp/.github/workflows/w.yml"
    st_case "action back on a major tag" "pin it to a 40-hex commit SHA"

    printf 'selftest: %d/%d\n' "$st_pass" "$((st_pass + st_fail))"
    exit "$([ "$st_fail" = "0" ] && echo 0 || echo 1)"
fi

CMAKE_FILE="$ROOT/cmake/imp-deps.cmake"
DOCKERFILE="$ROOT/Dockerfile"
ONLINE=0
[ "${1:-}" = "--online" ] && ONLINE=1

fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL  %s\n' "$*"; fail=1; }

# name -> tag / name -> sha, from the single source of truth
declare -A PINNED
declare -A PINNED_SHA
while IFS='=' read -r name tag; do
    [ -n "$name" ] && PINNED["$name"]="$tag"
done < <(sed -n 's/^set(IMP_DEP_\([A-Z_]*\)_TAG[ \t]*\([^ \t)]*\).*/\1=\2/p' "$CMAKE_FILE")
while IFS='=' read -r name sha; do
    [ -n "$name" ] && PINNED_SHA["$name"]="$sha"
done < <(sed -n 's/^set(IMP_DEP_\([A-Z_]*\)_SHA[ \t]*\([^ \t)]*\).*/\1=\2/p' "$CMAKE_FILE")

if [ ${#PINNED[@]} -eq 0 ]; then
    bad "no IMP_DEP_*_TAG entries parsed from cmake/imp-deps.cmake"
    exit 1
fi

# name -> ARG default, from the Dockerfile (tags and SHAs in one table)
declare -A ARGDEF
declare -A ARGDEF_SHA
while IFS='=' read -r name tag; do
    [ -n "$name" ] && ARGDEF["$name"]="$tag"
done < <(sed -n 's/^ARG IMP_DEP_\([A-Z_]*\)_TAG=\(.*\)$/\1=\2/p' "$DOCKERFILE")
while IFS='=' read -r name sha; do
    [ -n "$name" ] && ARGDEF_SHA["$name"]="$sha"
done < <(sed -n 's/^ARG IMP_DEP_\([A-Z_]*\)_SHA=\(.*\)$/\1=\2/p' "$DOCKERFILE")

# name -> clone URL, from the Dockerfile's own fetch lines
declare -A URL
while read -r name url; do
    [ -n "$name" ] && URL["$name"]="$url"
done < <(grep -oE 'pin[[:space:]]+https://[^[:space:]]+[[:space:]]+\$\{IMP_DEP_[A-Z_]+_TAG\}' "$DOCKERFILE" \
         | sed -E 's|pin[[:space:]]+(https://[^[:space:]]+)[[:space:]]+\$\{IMP_DEP_([A-Z_]+)_TAG\}|\2 \1|')

note "dependency pins (source: cmake/imp-deps.cmake)"
for name in $(printf '%s\n' "${!PINNED[@]}" | sort); do
    tag="${PINNED[$name]}"
    sha="${PINNED_SHA[$name]:-}"
    printf '  %-16s %-9s %s\n' "$name" "$tag" "${sha:-<none>}"

    # 1. every tag pin needs a commit pin, and it must look like one
    if [ -z "$sha" ]; then
        bad "$name: tag '$tag' has no IMP_DEP_${name}_SHA in cmake/imp-deps.cmake (H-8: a tag is a mutable ref)"
    elif ! [[ "$sha" =~ ^[0-9a-f]{40}$ ]]; then
        bad "$name: IMP_DEP_${name}_SHA='$sha' is not a 40-hex commit"
    fi

    # 2. drift: the Dockerfile ARG defaults must match the pins exactly
    if [ -z "${ARGDEF[$name]:-}" ]; then
        bad "$name: pinned in cmake but no 'ARG IMP_DEP_${name}_TAG=' default in the Dockerfile"
    elif [ "${ARGDEF[$name]}" != "$tag" ]; then
        bad "$name: Dockerfile tag default '${ARGDEF[$name]}' != cmake pin '$tag' (bump both, AGENTS.md)"
    fi
    if [ -z "${ARGDEF_SHA[$name]:-}" ]; then
        bad "$name: pinned in cmake but no 'ARG IMP_DEP_${name}_SHA=' default in the Dockerfile"
    elif [ -n "$sha" ] && [ "${ARGDEF_SHA[$name]}" != "$sha" ]; then
        bad "$name: Dockerfile SHA default '${ARGDEF_SHA[$name]}' != cmake pin '$sha' (bump both, AGENTS.md)"
    fi

    # 3. the tag must still resolve, and to the pinned commit: a re-tag upstream
    #    is exactly the mutation the SHA pin defends against, and this is the
    #    only place it becomes visible.
    if [ "$ONLINE" = "1" ]; then
        u="${URL[$name]:-}"
        if [ -z "$u" ]; then
            bad "$name: no fetch URL found in the Dockerfile; cannot verify the tag"
            continue
        fi
        remote_sha="$(git ls-remote "$u" "refs/tags/$tag^{}" "refs/tags/$tag" "refs/heads/$tag" 2>/dev/null \
                      | grep -m1 -E "refs/tags/$tag\^\{\}$" | cut -f1)"
        [ -z "$remote_sha" ] && remote_sha="$(git ls-remote "$u" "refs/tags/$tag" "refs/heads/$tag" 2>/dev/null | head -1 | cut -f1)"
        if [ -z "$remote_sha" ]; then
            bad "$name: '$tag' resolves to no tag or branch at $u (a cold fetch would fail here)"
        elif [ -n "$sha" ] && [ "$remote_sha" != "$sha" ]; then
            bad "$name: RE-TAGGED upstream. '$tag' at $u is now $remote_sha, pinned $sha. The build still uses the pinned commit; decide, then bump both lines."
        fi
    fi
done

# Every Dockerfile ARG must be backed by a pin, or it is a fourth source of truth.
for name in "${!ARGDEF[@]}"; do
    [ -z "${PINNED[$name]:-}" ] && bad "$name: Dockerfile ARG tag default with no matching pin in cmake/imp-deps.cmake"
done
for name in "${!ARGDEF_SHA[@]}"; do
    [ -z "${PINNED_SHA[$name]:-}" ] && bad "$name: Dockerfile ARG SHA default with no matching pin in cmake/imp-deps.cmake"
done

# GitHub Actions: a major tag (actions/checkout@v7) is a mutable ref owned by
# someone else and runs with this repo's token. Offline, textual, no network.
note ""
note "action pins (source: .github/workflows/)"
n_actions=0
while IFS=: read -r file line ref; do
    n_actions=$((n_actions + 1))
    case "$ref" in
        ./*|docker://*) continue ;;  # local composite action / image, no ref to pin
    esac
    if ! [[ "$ref" =~ @[0-9a-f]{40}$ ]]; then
        bad "$(basename "$file"):$line uses '$ref' - pin it to a 40-hex commit SHA (Dependabot rewrites SHA pins with a version comment)"
    fi
done < <(grep -rnE '^[[:space:]]*(-[[:space:]]+)?uses:' "$ROOT/.github/workflows/" \
         | sed -E 's/^([^:]+):([0-9]+):[[:space:]]*(-[[:space:]]+)?uses:[[:space:]]*/\1:\2:/' \
         | sed -E 's/[[:space:]]*#.*$//')
if [ "$n_actions" = "0" ]; then
    bad "no 'uses:' lines parsed from .github/workflows/ (parser broken?)"
else
    printf '  %d uses: lines\n' "$n_actions"
fi

# ---------------------------------------------------------------------------
# What the build DOWNLOADS (AUDIT_arch_2026 H-6). The pins above cover the four
# FetchContent deps; these three classes are everything else the build pulls off
# the network: container base images, single files, Python packages. All checks
# are textual and offline.
#
# Scope: Dockerfiles and .github/workflows/. Makefile dev targets that run a
# throwaway `python:3.12-slim` (chat-goldens) are out: they generate goldens on
# a developer box and reach neither CI nor the published image.
# ---------------------------------------------------------------------------
note ""
note "downloads (source: Dockerfiles + .github/workflows/)"

# Every Dockerfile except the developer harnesses: tools/Dockerfile.{agents,
# agents-sdk,claude-code,ncu} and tools/analysis/* build on one operator box
# (agent_external_smoke.sh, ncu profiling) and reach neither CI nor the
# published image -- the same call .github/dependabot.yml makes. A new
# Dockerfile anywhere else is in scope by default.
DEV_HARNESS_RE='tools/(analysis/|Dockerfile\.(agents|agents-sdk|claude-code|ncu)$)'
mapfile -t DOCKERFILES < <(find "$ROOT" -name 'Dockerfile*' -not -path '*/build*/*' \
                                -not -path '*/.git/*' | grep -vE "$DEV_HARNESS_RE" | sort)
mapfile -t WORKFLOWS < <(find "$ROOT/.github/workflows" -name '*.yml' | sort)

# A shell or YAML command can span lines; a download and its checksum are one
# such command. Fold continuations into one logical line, keyed by its first.
logical_lines() {
    awk '{ if (buf == "") { start = NR; buf = $0 } else { buf = buf " " $0 }
           if (buf ~ /\\$/) { sub(/\\$/, "", buf); next }
           print start ":" buf; buf = "" }
         END { if (buf != "") print start ":" buf }' "$1"
}

# 4a. Container images: a tag is a mutable ref (same class as H-8, one layer
#     down). Every external image must carry an @sha256: digest, and the same
#     tag must carry the same digest everywhere -- the Dockerfile and the five
#     ci.yml `image:` lines are two copies of one decision.
declare -A IMG_DIGEST      # "repo:tag" -> digest, first sighting
declare -A IMG_WHERE
n_images=0
check_image_ref() {  # file line ref
    local file="$1" line="$2" ref="$3"
    case "$ref" in
        imp:*) return ;;          # built locally by make/roofline, never pulled
        scratch) return ;;
    esac
    n_images=$((n_images + 1))
    if [[ "$ref" != *"@sha256:"* ]]; then
        bad "$(basename "$file"):$line '$ref' is tag-pinned - add an @sha256: digest (docker buildx imagetools inspect '$ref')"
        return
    fi
    local tag="${ref%%@*}" digest="${ref##*@}"
    if ! [[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
        bad "$(basename "$file"):$line digest '$digest' is not a sha256:<64-hex>"
        return
    fi
    if [ -z "${IMG_DIGEST[$tag]:-}" ]; then
        IMG_DIGEST["$tag"]="$digest"
        IMG_WHERE["$tag"]="$(basename "$file"):$line"
    elif [ "${IMG_DIGEST[$tag]}" != "$digest" ]; then
        bad "$(basename "$file"):$line '$tag' is $digest here and ${IMG_DIGEST[$tag]} at ${IMG_WHERE[$tag]} - one tag, one digest"
    fi
}

for f in "${DOCKERFILES[@]}"; do
    # Stage names defined in this file are internal refs, not registry pulls.
    stages="$(sed -nE 's/^FROM[[:space:]]+.*[[:space:]]+[Aa][Ss][[:space:]]+([A-Za-z0-9_.-]+).*/\1/p' "$f" | tr '\n' ' ')"
    while IFS=: read -r line ref; do
        case " $stages " in *" $ref "*) continue ;; esac
        check_image_ref "$f" "$line" "$ref"
    done < <(grep -nE '^FROM[[:space:]]' "$f" | sed -E 's/^([0-9]+):FROM[[:space:]]+([^[:space:]]+).*/\1:\2/')
done
for f in "${WORKFLOWS[@]}"; do
    while IFS=: read -r line ref; do
        check_image_ref "$f" "$line" "$ref"
    done < <(grep -nE '^[[:space:]]*image:[[:space:]]' "$f" | sed -E 's/^([0-9]+):[[:space:]]*image:[[:space:]]*([^[:space:]#]+).*/\1:\2/')
done
printf '  %d image refs\n' "$n_images"

# 4b. Remote files. Two of these are fetched and then executed (the CMake
#     installer in the build image, git-clang-format on a runner that holds
#     GITHUB_TOKEN). A branch ref is a moving target; an unverified download is
#     whatever the network hands back that day.
n_fetch=0
for f in "${DOCKERFILES[@]}" "${WORKFLOWS[@]}"; do
    while IFS= read -r hit; do
        line="${hit%%:*}"
        text="${hit#*:}"
        # localhost health polls fetch nothing that gets stored or run
        case "$text" in *localhost*|*127.0.0.1*) continue ;; esac
        url="$(printf '%s' "$text" | grep -oE 'https?://[^[:space:]"'"'"']+' | head -1)"
        [ -z "$url" ] && continue
        n_fetch=$((n_fetch + 1))
        # the file this download lands in: -o/-O <path>, or -qO <path>
        dest="$(printf '%s' "$text" | grep -oE '\-[a-zA-Z]*[oO][[:space:]]+[^[:space:]]+' | head -1 | sed -E 's/^-[a-zA-Z]*[oO][[:space:]]+//')"
        case "$url" in
            *raw.githubusercontent.com*)
                # .../<owner>/<repo>/<ref>/<path>: the 5th field is the ref
                ref="$(printf '%s' "$url" | cut -d/ -f6)"
                [[ "$ref" =~ ^[0-9a-f]{40}$ ]] || \
                    bad "$(basename "$f"):$line fetches raw content at '$ref' - a branch or tag is a mutable ref, use a 40-hex commit"
                ;;
        esac
        if [ -z "$dest" ]; then
            bad "$(basename "$f"):$line downloads $url with no -o/-O target the checksum could name"
        elif ! logical_lines "$f" | grep -qE "[0-9a-f]{64}[[:space:]]+${dest//\//\\/}'[[:space:]]*\|[[:space:]]*sha256sum[[:space:]]+(-c|--check)"; then
            bad "$(basename "$f"):$line downloads $url into $dest with no 'echo <sha256>  $dest | sha256sum -c -' in the same file"
        fi
    done < <(logical_lines "$f" | grep -E '(curl|wget)[[:space:]]' | grep -E 'https?://' | grep -vE '^[0-9]+:[[:space:]]*#')
done
printf '  %d remote file fetches\n' "$n_fetch"

# 4c. Python. `pip install <pkg>==<ver>` still lets the index serve different
#     bytes for that version and says nothing about the transitive set, so every
#     install goes through a lock file in which every requirement carries a
#     hash (one --hash puts pip into --require-hashes mode for the whole run).
n_pip=0
declare -A SEEN_LOCK
for f in "${DOCKERFILES[@]}" "${WORKFLOWS[@]}"; do
    while IFS= read -r hit; do
        line="${hit%%:*}"
        text="${hit#*:}"
        n_pip=$((n_pip + 1))
        lock="$(printf '%s' "$text" | grep -oE '\-r[[:space:]]+[^[:space:]]+' | head -1 | sed -E 's/^-r[[:space:]]+//')"
        if [ -z "$lock" ]; then
            bad "$(basename "$f"):$line installs Python packages without '-r <lock file>' ($(printf '%s' "$text" | sed -E 's/^[[:space:]]*//'))"
            continue
        fi
        SEEN_LOCK["$lock|$(dirname "$f")"]=1
    done < <(logical_lines "$f" | grep -E '\bpip3?[[:space:]]+install\b' | grep -vE '^[0-9]+:[[:space:]]*#')
done
for entry in "${!SEEN_LOCK[@]}"; do
    lock="${entry%%|*}"
    dir="${entry##*|}"
    # either repo-relative, or an in-container path COPY'd from beside the Dockerfile
    path="$ROOT/${lock#/}"
    [ -f "$path" ] || path="$dir/$(basename "$lock")"
    if [ ! -f "$path" ]; then
        bad "referenced lock file '$lock' does not exist"
        continue
    fi
    # Per requirement, not per file: a lock file has more --hash lines than
    # requirements (sdist + wheels), so a total count hides a stripped pin.
    n_req=0
    while IFS='|' read -r req n_hash; do
        n_req=$((n_req + 1))
        printf '%s' "$req" | grep -qE '==' || bad "$lock: '$req' is not an exact pin"
        [ "$n_hash" = "0" ] && bad "$lock: '$req' carries no --hash=sha256: line"
    done < <(awk '/^[A-Za-z0-9]/ { if (name != "") print name "|" h
                                   name = $1; sub(/[[:space:]]*\\$/, "", name); h = 0; next }
                  /--hash=sha256:/ { h++ }
                  END { if (name != "") print name "|" h }' "$path")
    if [ "$n_req" = "0" ]; then
        bad "$lock: no requirements parsed (parser broken, or the file is empty?)"
    else
        printf '  %-34s %d requirements, all hashed\n' "$lock" "$n_req"
    fi
done
printf '  %d pip install lines\n' "$n_pip"

if [ "$fail" = "0" ]; then
    note "OK ($([ "$ONLINE" = "1" ] && echo 'drift + form + upstream tag->commit' || echo 'drift + form; pass --online to resolve tags'))"
else
    note ""
    note "Pins are the one thing a cached Docker layer will happily hide."
fi
exit "$fail"
