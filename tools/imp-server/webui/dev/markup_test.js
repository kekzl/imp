// Checks the web UI markdown renderer (index.html, esc .. tables) in node, no browser, no GPU:
//   docker run --rm -v "$PWD":/src:ro mcr.microsoft.com/playwright:v1.56.0-noble \
//       node /src/tools/imp-server/webui/dev/markup_test.js
const fs = require("fs");
const html = fs.readFileSync("/src/tools/imp-server/webui/index.html", "utf8");
const start = html.indexOf("const ESCAPES");
const end = html.indexOf("const rawText");
eval(html.slice(start, end) + "\nglobalThis.markup = markup;");

const cases = [
  ["link", "see [docs](https://example.com/a?b=1&c=2) now", (o) => o.includes('<a href="https://example.com/a?b=1&amp;c=2"') && o.includes(">docs</a>")],
  ["javascript: stays text", "[x](javascript:alert(1))", (o) => !o.includes("<a") && o.includes("[x]")],
  ["data: stays text", "[x](data:text/html,hi)", (o) => !o.includes("<a")],
  ["quote cannot break out", '[x](https://a.com/"onmouseover=alert(1))', (o) => !/<a [^>]*onmouseover/.test(o)],
  ["html in text escaped", "<img src=x onerror=alert(1)>", (o) => !o.includes("<img") && o.includes("&lt;img")],
  ["link inside inline code stays code", "`[a](https://x.com)`", (o) => !o.includes("<a") && o.includes("<code>")],
  ["link inside fence stays code", "```\n[a](https://x.com)\n```", (o) => !o.includes("<a")],
  ["table", "| A | B |\n|---|---|\n| 1 | **2** |", (o) => o.includes("<table>") && o.includes("<th>A</th>") && o.includes("<td><strong>2</strong></td>")],
  ["prose over rule is not a table", "a | b\n---\nnext", (o) => !o.includes("<table>")],
  ["table then text", "| A |\n|---|\n| 1 |\n\nafter", (o) => o.includes("</table>") && o.includes("after")],
  ["heading", "## Title\nbody", (o) => o.includes("<h4>Title</h4>")],
];
let fails = 0;
for (const [name, src, ok] of cases) {
  const out = markup(src);
  const pass = ok(out);
  if (!pass) fails++;
  console.log((pass ? "PASS " : "FAIL ") + name + (pass ? "" : " :: " + out));
}
console.log("FAILS:", fails);
process.exit(fails ? 1 : 0);
