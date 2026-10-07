#!/usr/bin/env bash
# Installs pinned LLVM tools from apt.llvm.org. Single source of the LLVM pin for CI,
# the Dockerfile `lint` stage and the Makefile. Usage (root): install_llvm.sh clang-format [clang-tidy clang libclang-rt]
# Pin is 23.1.*: apt.llvm.org keeps one build per branch in its pool, an exact =version dies at its next rebuild.
set -euo pipefail

LLVM_VERSION=23.1
LLVM_MAJOR="${LLVM_VERSION%%.*}"
LLVM_APT_VERSION="1:$LLVM_VERSION.*"
LLVM_KEY_URL=https://apt.llvm.org/llvm-snapshot.gpg.key
LLVM_KEY_SHA256=8b2a587ffd672c4687e7581dad4b2f6c1bb2ad6b480cd9771ba2ff48e0b8c75d
LLVM_KEY_FPR=6084F3CF814B57C1CF12EFD515CF4D18AF4F7421

[ "$#" -gt 0 ] || { echo "usage: $0 clang-format|clang-tidy|clang|libclang-rt ..." >&2; exit 2; }
[ "$(id -u)" = "0" ] || { echo "$0: needs root" >&2; exit 2; }

# A failed apt call switches the Ubuntu mirror and retries: azure, archive.ubuntu.com and
# kernel.org each failed alone on 2026-09-11 / 2026-10-07 (same list as .github/workflows/ci.yml).
apt_get() {
    local m
    apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 "$@" && return 0
    [ -f /etc/apt/sources.list.d/ubuntu.sources ] || return 1
    for m in azure.archive.ubuntu.com archive.ubuntu.com mirrors.edge.kernel.org; do
        echo "install_llvm: apt-get $1 failed, Ubuntu mirror -> $m" >&2
        sed -i -E "s#http://[a-z.]+/ubuntu/?#http://$m/ubuntu/#" /etc/apt/sources.list.d/ubuntu.sources
        apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 update -qq &&
            apt-get -o Acquire::Retries=3 -o Acquire::http::Timeout=30 "$@" && return 0
    done
    return 1
}

# shellcheck disable=SC1091
codename="$(. /etc/os-release && echo "${VERSION_CODENAME:?}")"
GNUPGHOME="$(mktemp -d)"
export DEBIAN_FRONTEND=noninteractive GNUPGHOME
apt_get update -qq
apt_get install -y -qq --no-install-recommends ca-certificates curl gpg >/dev/null

key=/tmp/llvm-snapshot.gpg.key
curl -fsSL --retry 5 "$LLVM_KEY_URL" -o "$key"
echo "$LLVM_KEY_SHA256  $key" | sha256sum -c -
fpr="$(gpg --show-keys --with-colons "$key" | awk -F: '/^fpr:/ {print $10; exit}')"
[ "$fpr" = "$LLVM_KEY_FPR" ] || { echo "LLVM key fingerprint $fpr != $LLVM_KEY_FPR" >&2; exit 1; }
gpg --dearmor < "$key" > /usr/share/keyrings/llvm-snapshot.gpg
echo "deb [signed-by=/usr/share/keyrings/llvm-snapshot.gpg] https://apt.llvm.org/$codename/ llvm-toolchain-$codename-$LLVM_MAJOR main" \
    > /etc/apt/sources.list.d/llvm-$LLVM_MAJOR.list
apt_get update -qq

# libclang-rt: libFuzzer + sanitizer runtimes (libclang-rt-N-dev), no binary of its own.
pkgs=()
for t in "$@"; do
    case "$t" in
        clang-format|clang-tidy|clang) pkgs+=("$t-$LLVM_MAJOR=$LLVM_APT_VERSION") ;;
        libclang-rt) pkgs+=("libclang-rt-$LLVM_MAJOR-dev=$LLVM_APT_VERSION") ;;
        *) echo "$0: unknown tool '$t'" >&2; exit 2 ;;
    esac
done
apt_get install -y -qq --no-install-recommends "${pkgs[@]}" >/dev/null

# Unversioned names on PATH: clang-format also ships git-clang-format, clang also clang++.
for t in "$@"; do
    [ "$t" = libclang-rt ] && continue
    ln -sf "/usr/bin/$t-$LLVM_MAJOR" "/usr/local/bin/$t"
    [ "$t" = clang-format ] && ln -sf "/usr/bin/git-clang-format-$LLVM_MAJOR" /usr/local/bin/git-clang-format
    [ "$t" = clang ] && ln -sf "/usr/bin/clang++-$LLVM_MAJOR" /usr/local/bin/clang++
    v="$("/usr/local/bin/$t" --version | grep -oE 'version [0-9]+\.[0-9]+\.[0-9]+' | head -1)"
    case "$v" in "version $LLVM_VERSION."*) ;; *) echo "$t: '$v' is not $LLVM_VERSION.x" >&2; exit 1 ;; esac
    echo "$t: $v"
done
