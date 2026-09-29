#!/bin/sh
# Broken relative links and repo path references in docs/**/*.md and every CLAUDE.md.
# Link `](target)`: resolved against the file's directory, anchor stripped, existence only.
# Path in backticks starting docs/ src/ tools/ scripts/: resolved against the repo root.
# Not checked: fenced code, globs, a backticked path that is the text of an http link (a
# permalink to a deleted file), gitignored paths, and entries in scripts/check_doc_links.allow.
# Usage: scripts/check_doc_links.sh [repo_root]. Exit 1 with one line per broken reference.
set -eu
ROOT=${1:-$(git rev-parse --show-toplevel)}
cd "$ROOT"
ALLOW=scripts/check_doc_links.allow
OUT=$(mktemp)
trap 'rm -f "$OUT"' EXIT

git ls-files -- 'docs/*.md' 'docs/**/*.md' 'CLAUDE.md' '**/CLAUDE.md' | sort -u |
while IFS= read -r f; do
    awk '
    function glob(t) {
        return index(t, "*") || index(t, "{") || index(t, "<") || index(t, "$") ||
               index(t, "[") || index(t, "|") || index(t, "...")
    }
    /^[ \t]*(```|~~~)/ { fence = !fence; next }
    fence { next }
    {
        s = $0
        gsub(/`[^`]*`/, "", s)
        while (match(s, /\]\([^)[:space:]]+(\)| )/)) {
            t = substr(s, RSTART + 2, RLENGTH - 3)
            s = substr(s, RSTART + RLENGTH)
            if (t ~ /^(https?:|mailto:|ftp:|#|\/)/) continue
            sub(/#.*/, "", t)
            if (t != "") print "L\t" FNR "\t" t
        }
        s = $0
        while (match(s, /`[^`]+`/)) {
            t = substr(s, RSTART + 1, RLENGTH - 2)
            s = substr(s, RSTART + RLENGTH)
            if (substr(s, 1, 6) == "](http") continue
            if (t !~ /^(docs|src|tools|scripts)\//) continue
            sub(/[ \t].*/, "", t)
            if (glob(t)) continue
            sub(/[:#].*/, "", t)
            sub(/\/\.[a-z].*$/, "", t)
            sub(/[.,;)]+$/, "", t)
            print "P\t" FNR "\t" t
        }
    }' "$f" |
    while IFS="$(printf '\t')" read -r kind ln t; do
        if [ "$kind" = L ]; then p="$(dirname "$f")/$t"; else p="$t"; fi
        # a name fragment ending in _ or - is a prefix: any match counts
        case "$p" in *_|*-) set -- "$p"*; p=$1 ;; esac
        [ -e "$p" ] && continue
        # an extensionless module stem (src/x/foo for foo.h + foo.cpp) counts if any foo.* exists
        set -- "$p".*; [ -e "$1" ] && continue
        git check-ignore -q "$p" && continue
        grep -qxF "$f $t" "$ALLOW" 2>/dev/null && continue
        printf '%s:%s: broken %s: %s\n' "$f" "$ln" "$([ "$kind" = L ] && echo link || echo path)" "$t"
    done
done > "$OUT"

n=$(wc -l < "$OUT")
cat "$OUT"
echo "check_doc_links: $n broken"
[ "$n" -eq 0 ]
