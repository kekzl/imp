"""Docs inventory for docs/audit/docs-rewrite/DOCS_INVENTORY.md. Run from repo root: python3 docs/audit/docs-rewrite/inventory.py > inv.json"""
import collections, json, re, subprocess, sys, pathlib

sys.path.insert(0, "scripts")
import docs_lint as dl

ROOT = pathlib.Path(".")
files = [f for f in subprocess.check_output(["git", "ls-files", "*.md"], text=True).split()
         if not f.startswith(("third_party/", ".github/"))]

FILLER = re.compile(r"\b(basically|simply|note that|it'?s worth|blazing|state[- ]of[- ]the[- ]art|"
                    r"originally|previously|coming soon|in this document|we decided|world[- ]class)\b", re.I)
TOKS = re.compile(r"\b\d+(?:\.\d+)?\s*tok/s\b")
SENT = re.compile(r"[.!?](?:\s|$)")
# Measured-looking figures (299.61, 1792 GB/s, 18.3 GiB): the same one in two files is a duplicated fact.
NUM = re.compile(r"\b\d+\.\d+\s*(?:tok/s|GiB|GB/s|GB|MiB|ms|us|%)|\b\d{3,}\.\d+\b")
VER = re.compile(r"\bv0\.(\d+)\.\d+\b")
nums = collections.defaultdict(set)

stale = collections.defaultdict(list)
sp = ROOT / "docs/audit/docs-rewrite/STALE.md"
for ln in sp.read_text().splitlines():
    m = re.match(r"- ([^:]+\.md)(:\d+)?: (.*)", ln)
    if m:
        e = re.search(r"edited (\d+)x", m.group(3))
        stale[m.group(1)].append(int(e.group(1)) if e else "prov")

def norm(s):
    s = re.sub(r"[`*_\[\]()>|#]", "", s).lower()
    return re.sub(r"\s+", " ", s).strip()

rows, shingles = {}, collections.defaultdict(set)
for f in files:
    text = (ROOT / f).read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    fm = dl.FRONTMATTER_RE.match(text)
    layer = None
    if fm:
        m = re.search(r"^layer:\s*(\S+)", fm.group(1) or fm.group(2), re.M)
        layer = m.group(1) if m else None
    layer = layer or dl._layer_for_path(f) or "-"
    fence = comment = False
    prose = nonblank = long_par = filler = oldv = 0
    par = []
    def flush():
        global long_par
        if par and len(SENT.findall(" ".join(par))) > 2:
            long_par += 1
        par.clear()
    for ln in lines:
        s = ln.strip()
        if s.startswith("```"):
            fence = not fence; flush(); continue
        if fence:
            continue
        if s.startswith("<!--"):
            comment = "-->" not in s; continue
        if comment:
            comment = "-->" not in s; continue
        if not s:
            flush(); continue
        nonblank += 1
        filler += len(FILLER.findall(s))
        if not re.match(r"(\||#|>|<|!\[)", s):
            for sent in re.split(r"(?<=[.!?;:])\s+", re.sub(r"^([-*+]|\d+\.) ", "", s)):
                n = norm(sent)
                if len(n) >= 50:
                    shingles[n].add(f)
            for num in NUM.findall(s):
                nums[num].add(f)
            oldv += len([v for v in VER.findall(s) if int(v) < 40])
        if re.match(r"(\||#|[-*+] |\d+\. |>|<|!\[|\[.*\]:)", s):
            flush(); continue
        prose += 1
        par.append(s)
    flush()
    rows[f] = dict(layer=layer, lines=len(lines), prose=prose, nonblank=nonblank, long_par=long_par,
                   filler=filler, oldv=oldv, toks=len(TOKS.findall(text)), scope=dl.in_scope(f),
                   stale_edits=sum(x for x in stale[f] if x != "prov"), stale_prov=stale[f].count("prov"))

dup = collections.defaultdict(collections.Counter)
for n, fs in shingles.items():
    if len(fs) > 1:
        for a in fs:
            for b in fs - {a}:
                dup[a][b] += 1
# Numbers only count between in-scope docs: audit logs and plans cite them by design.
numdup = collections.defaultdict(collections.Counter)
for n, fs in nums.items():
    fs = {f for f in fs if rows[f]["scope"]}
    for a in fs:
        for b in fs - {a}:
            numdup[a][b] += 1

json.dump({f: dict(r, dups=sum(dup[f].values()), dup_top=(dup[f].most_common(1) or [("", 0)])[0],
                  numdups=sum(numdup[f].values()), numdup_top=(numdup[f].most_common(1) or [("", 0)])[0])
           for f, r in rows.items()}, sys.stdout, indent=0)
