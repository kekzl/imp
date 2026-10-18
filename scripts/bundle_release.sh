#!/usr/bin/env bash
# Release tarball without Docker (roadmap row 55): libexec/ binaries, every shared library ldd
# resolves except the NVIDIA driver (glibc and its loader included), bin/ launchers that run the
# bundled loader. Target needs only the driver. Runs in the Dockerfile `bundle` stage.
# Usage: bundle_release.sh <out_dir> <binary>...
set -euo pipefail
out="$1"
shift
version=$(sed -n 's/^project(imp[^)]*VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)
[ -n "$version" ] || { echo "bundle: no project VERSION in CMakeLists.txt" >&2; exit 1; }
name="imp-${version}-linux-x86_64"
root="$out/$name"
mkdir -p "$root/bin" "$root/libexec" "$root/lib" "$root/share/doc/imp/copyright"

# The host provides the driver; the vDSO is the kernel's.
host_lib='^(linux-vdso\.so|libcuda\.so|libnvidia-)'

copy_lib() {  # <soname> <path>
    [ -e "$root/lib/$1" ] && return 0
    cp -L "$2" "$root/lib/$1"
    local pkg
    pkg=$(dpkg -S "$(readlink -f "$2")" 2>/dev/null | head -1 | cut -d: -f1 || true)
    if [ -n "$pkg" ] && [ -f "/usr/share/doc/$pkg/copyright" ]; then
        cp "/usr/share/doc/$pkg/copyright" "$root/share/doc/imp/copyright/$pkg"
    fi
}

shipped=()
for bin in "$@"; do
    [ -f "$bin" ] || continue
    b=$(basename "$bin")
    install -m 755 "$bin" "$root/libexec/"
    missing=$(ldd "$bin" | awk '/not found/ {print $1}' | grep -Ev "$host_lib" || true)
    [ -z "$missing" ] || { echo "bundle: $bin: unresolved $missing" >&2; exit 1; }
    while read -r soname path; do
        [[ "$soname" =~ $host_lib ]] || copy_lib "$soname" "$path"
    done < <(ldd "$bin" | awk '$2 == "=>" && $3 ~ /^\// {print $1, $3}')
    # --library-path covers the dependencies of every bundled library; argv[0] stays the launcher.
    cat > "$root/bin/$b" <<EOF
#!/bin/sh
d=\$(dirname "\$(readlink -f "\$0")")/..
exec "\$d/lib/ld-linux-x86-64.so.2" --library-path "\$d/lib" --argv0 "\$0" "\$d/libexec/$b" "\$@"
EOF
    chmod 755 "$root/bin/$b"
    shipped+=("$b")
done
[ ${#shipped[@]} -gt 0 ] || { echo "bundle: no binaries" >&2; exit 1; }
copy_lib ld-linux-x86-64.so.2 /lib64/ld-linux-x86-64.so.2

# The CUDA packages' copyright files carry the CUDA EULA (redistribution: Attachment A).
cp LICENSE THIRD_PARTY_LICENSES.md "$root/share/doc/imp/"
. /etc/os-release
cat > "$root/README.txt" <<EOF
imp $version for Linux x86_64, NVIDIA Blackwell sm_120.
Needs: NVIDIA driver with CUDA 13 support. glibc, CUDA runtime and cuBLAS are bundled.
Run: bin/imp-server --model <model> (bin/ launches libexec/ with the bundled loader).
Bundled libraries: $PRETTY_NAME packages; licenses in share/doc/imp/copyright/.
EOF
tar -C "$out" -czf "$out/$name.tar.gz" "$name"
(cd "$out" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "bundle: $out/$name.tar.gz $(du -h "$out/$name.tar.gz" | cut -f1), ${#shipped[@]} binaries, $(ls "$root/lib" | wc -l) libs"
