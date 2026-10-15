#!/usr/bin/env python3
"""Checks the web UI markdown renderer (index.html, esc .. tables) in headless Chromium, no GPU.

Loads index.html via file:// and calls the page's own markup() with page.evaluate.

    docker run --rm -v "$PWD":/src:ro mcr.microsoft.com/playwright/python:v1.55.0-noble \
        bash -c 'pip install -q playwright==1.55.0 && python /src/tools/imp-server/webui/dev/markup_test.py'

Prints PASS/FAIL per case and "FAILS: <n>"; exit code 1 if any case fails.
"""
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

INDEX = Path(__file__).resolve().parent.parent / "index.html"

CASES = [
    ("link", "see [docs](https://example.com/a?b=1&c=2) now",
     lambda o: '<a href="https://example.com/a?b=1&amp;c=2"' in o and ">docs</a>" in o),
    ("javascript: stays text", "[x](javascript:alert(1))", lambda o: "<a" not in o and "[x]" in o),
    ("data: stays text", "[x](data:text/html,hi)", lambda o: "<a" not in o),
    ("quote cannot break out", '[x](https://a.com/"onmouseover=alert(1))',
     lambda o: not re.search(r"<a [^>]*onmouseover", o)),
    ("html in text escaped", "<img src=x onerror=alert(1)>", lambda o: "<img" not in o and "&lt;img" in o),
    ("link inside inline code stays code", "`[a](https://x.com)`", lambda o: "<a" not in o and "<code>" in o),
    ("link inside fence stays code", "```\n[a](https://x.com)\n```", lambda o: "<a" not in o),
    ("table", "| A | B |\n|---|---|\n| 1 | **2** |",
     lambda o: "<table>" in o and "<th>A</th>" in o and "<td><strong>2</strong></td>" in o),
    ("prose over rule is not a table", "a | b\n---\nnext", lambda o: "<table>" not in o),
    ("table then text", "| A |\n|---|\n| 1 |\n\nafter", lambda o: "</table>" in o and "after" in o),
    ("heading", "## Title\nbody", lambda o: "<h4>Title</h4>" in o),
    ("inline math", "so $x^2 + \\alpha$ holds",
     lambda o: 'class="math"' in o and "<sup>2</sup>" in o and "α" in o),
    ("dollar amounts stay text", "costs $5 and $10 today",
     lambda o: 'class="math"' not in o and "$5 and $10" in o),
    ("display math frac", "$$\\frac{a}{b}$$", lambda o: "math display" in o and 'class="frac"' in o),
    ("paren math", "\\(\\sqrt{2} \\le 2\\)", lambda o: "√" in o and "≤" in o),
    ("math stays escaped", "$<img src=x onerror=alert(1)>$", lambda o: "<img" not in o and "&lt;img" in o),
    ("math star not italic", "$a*b*c$", lambda o: "<em>" not in o),
    ("math inside code stays code", "`$x^2$`", lambda o: 'class="math"' not in o),
    ("highlight keyword", "```js\nconst x = 1; // one\n```",
     lambda o: '<span class="tk-k">const</span>' in o
     and '<span class="tk-c">// one</span>' in o
     and '<span class="tk-n">1</span>' in o),
    ("highlight escapes", "```html\n<script>alert(1)</script>\n```",
     lambda o: "<script>" not in o and "&lt;" in o),
    ("highlight string with lt", '```python\ns = "a < b"\n```',
     lambda o: '<span class="tk-s">"a &lt; b"</span>' in o),
    ("plain fence not highlighted", "```\nif x\n```", lambda o: "tk-" not in o),
]


def main():
    fails = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(INDEX.as_uri())
        for name, src, ok in CASES:
            out = page.evaluate("(s) => markup(s)", src)
            passed = bool(ok(out))
            if not passed:
                fails += 1
            print(("PASS " if passed else "FAIL ") + name + ("" if passed else " :: " + out))
        browser.close()
    print("FAILS:", fails)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
