#!/usr/bin/env bash
# shellcheck disable=SC2016,SC2034 # check() evals its condition later; rc is read there
# make verify-ab resolves AB_BASE_REF once (#2226). Throwaway git repo, docker/git shims on PATH,
# no Docker daemon, no GPU. The git shim advances refs/remotes/origin/main after every
# `rev-parse ... origin/main`, as a concurrent `git fetch` in another worktree would.
# Wired into scripts/ci_static_gates.sh (group gpulock).
set -uo pipefail

cd "$(dirname "$(readlink -f "$0")")/.." || exit 1
SRC="$PWD"
REAL_GIT="$(command -v git)"

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
PASS=0 FAIL=0
ok()  { PASS=$((PASS + 1)); echo "  ok    $1"; }
bad() { FAIL=$((FAIL + 1)); echo "  FAIL  $1"; }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }

R="$T/repo"
git init -q "$R"
cd "$R" || exit 1
git config user.email t@t; git config user.name t
for i in 1 2 3 4 5 6; do git commit -q --allow-empty -m "c$i"; git tag "c$i"; done
mkdir scripts
cp "$SRC/Makefile" .
cp "$SRC/scripts/ab_base_image.sh" scripts/
printf '#!/bin/bash\n[ "${1:-}" = check ] && exit 0\necho imp:test\n' > scripts/image_tag.sh
printf '#!/bin/bash\necho "$IMG_A" > "$STUB/img_a"\n' > scripts/verify_ab.sh

# docker shim: every image "exists" (reuse path), `tag SRC DST` is logged.
mkdir "$T/bin"
printf '#!/bin/bash\ncase "$1" in image) exit 0;; tag) echo "$2 $3" >> "$STUB/tags";; esac\n' > "$T/bin/docker"
# git shim: after each `rev-parse ... origin/main`, count it and move origin/main one commit on.
cat > "$T/bin/git" <<EOF
#!/bin/bash
"$REAL_GIT" "\$@"; rc=\$?
case " \$* " in *" rev-parse "*" origin/main "*)
    n=\$(( \$(cat "\$STUB/n" 2>/dev/null || echo 0) + 1 )); echo \$n > "\$STUB/n"
    "$REAL_GIT" update-ref refs/remotes/origin/main "c\$((n + 1))";;
esac
exit \$rc
EOF
chmod +x "$T/bin/"*
export STUB="$T/s" PATH="$T/bin:$PATH"

reset() { rm -rf "$STUB"; mkdir "$STUB"; git update-ref refs/remotes/origin/main c1; }
built() { awk '$2 == "imp:ab-base" {print $1}' "$STUB/tags"; }
MK=(make -s -o build -o check-gpu GPU_LOCKED= verify-ab)

# Old recipe, verbatim from before the fix: base image built for one sha, IMG_A resolved again.
cat > old.mk <<'EOF'
SHELL := bash
AB_BASE_REF ?= origin/main
IMG_CHECK = true
DOCKER_IMG = imp:test
GPU_LOCKED =
ab-base-image:
	@bash scripts/ab_base_image.sh $(AB_BASE_REF)
verify-ab: ab-base-image
	@$(IMG_CHECK) && IMG_A=imp:ab-$$(git rev-parse --short=8 $(AB_BASE_REF)) IMG_B=$(DOCKER_IMG) \
	 $(GPU_LOCKED) bash scripts/verify_ab.sh
EOF
reset
make -s -f old.mk verify-ab >/dev/null 2>&1
check "old recipe: IMG_A ($(cat "$STUB/img_a" 2>/dev/null)) differs from built tag ($(built))" \
    '[ -n "$(built)" ] && [ "$(cat "$STUB/img_a")" != "$(built)" ]'
check "old recipe: origin/main resolved twice" '[ "$(cat "$STUB/n")" = 2 ]'

reset
"${MK[@]}" >"$T/out" 2>&1; rc=$?
check "new recipe: exit 0" '[ $rc -eq 0 ] || { cat "$T/out"; false; }'
check "new recipe: origin/main resolved once" '[ "$(cat "$STUB/n")" = 1 ]'
check "new recipe: IMG_A ($(cat "$STUB/img_a" 2>/dev/null)) is the built tag ($(built))" \
    '[ -n "$(built)" ] && [ "$(cat "$STUB/img_a")" = "$(built)" ]'
check "new recipe: tag is imp:ab-<sha of c1>" '[ "$(built)" = "imp:ab-$(git rev-parse --short=8 c1)" ]'

reset
printf '#!/bin/bash\nexit 1\n' > scripts/ab_base_image.sh
"${MK[@]}" >/dev/null 2>&1; rc=$?
check "new recipe: failing ab_base_image.sh fails verify-ab, verify_ab.sh not run" \
    '[ $rc -ne 0 ] && [ ! -e "$STUB/img_a" ]'

echo "test_verify_ab_sha_once: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
