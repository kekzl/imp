// Scenario runner for the imp web UI against dev/mock_server.py: drives a
// headless Chromium through one scenario, screenshots every step into /shots
// and prints the readouts as JSON. No host install: it runs in the public
// Playwright image, which carries the browsers.
//
//   D=tools/imp-server/webui/dev
//   docker run --rm --network host -v "$PWD/$D":/w -v "$PWD/build-dev/shots":/shots -w /w \
//       -e PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 mcr.microsoft.com/playwright:v1.56.0-noble \
//       bash -c 'npm i --silent playwright@1.56.0 && node drive.js chat http://localhost:9099/'
//
// scenarios: chat, length, errors, stop, live (mid-stream captures), narrow,
// swap, system, reload, actions, boot (needs a mock started with --boot-delay),
// features (every redesign feature, results in "checks"), shots (review captures).
// usage: node drive.js <scenario> [url]
const { chromium } = require("playwright");

const URL = process.argv[3] || "http://localhost:9099/";
const scenario = process.argv[2] || "baseline";
const SHOTS = "/shots";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function state(page) {
  return page.evaluate(() => ({
    state: document.getElementById("state")?.textContent,
    model: document.getElementById("model")?.textContent ?? document.querySelector("#model, select#model")?.value,
    readout: Object.fromEntries([...document.querySelectorAll(".readout dt")].map((dt) =>
      [dt.textContent.trim(), dt.nextElementSibling?.textContent.trim()])),
    turns: [...document.querySelectorAll(".turn")].map((t) => ({
      cls: t.className,
      text: t.querySelector(".body")?.textContent.slice(0, 80),
      note: t.querySelector(".note, .turn-note, .stats")?.textContent,
    })),
    faults: [...document.querySelectorAll(".fault")].map((f) => f.textContent),
    stripNote: document.getElementById("strip-note")?.textContent,
    rate: document.getElementById("m-rate")?.textContent,
  }));
}

async function send(page, text) {
  await page.fill("#prompt", text);
  await page.press("#prompt", "Enter");
}

// The right panel is tabbed (run, set, srv); inputs on a hidden tab cannot be filled.
async function openTab(page, name) {
  await page.evaluate(() => { if (innerWidth <= 900) document.getElementById("open-panel")?.click(); });
  await page.click("#tab-" + name);
}

// 1x1 PNG, for the attachment path.
const PNG = Buffer.from("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==", "base64");

async function waitIdle(page, timeout = 30000) {
  await page.waitForFunction(() => {
    const s = document.getElementById("state")?.textContent || "";
    return /idle|stopped|error|no server|no model/.test(s);
  }, null, { timeout });
}

(async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1280, height: 820 }, deviceScaleFactor: 1 });
  const page = await ctx.newPage();
  const errors = [];
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") errors.push(m.type() + ": " + m.text()); });
  page.on("pageerror", (e) => errors.push("pageerror: " + e.message));
  const bodies = [];
  page.on("request", (r) => { if (r.method() === "POST") { try { bodies.push(JSON.parse(r.postData())); } catch {} } });

  await page.goto(URL);
  await page.waitForFunction(() => !/starting|connecting/.test(document.getElementById("state")?.textContent || ""), null, { timeout: 15000 }).catch(() => {});
  const out = { scenario, steps: [] };

  const step = async (name, fn) => {
    await fn();
    const s = await state(page);
    await page.screenshot({ path: `${SHOTS}/${scenario}-${name}.png`, fullPage: false });
    out.steps.push({ name, ...s });
  };

  await step("00-loaded", async () => {});

  if (scenario === "baseline" || scenario === "chat") {
    await step("01-answer", async () => { await send(page, "Explain streaming, with code."); await waitIdle(page); await sleep(150); });
    await step("02-followup", async () => { await send(page, "And a follow-up, please."); await waitIdle(page); await sleep(150); });
  }
  if (scenario === "length") {
    await step("01-length", async () => { await openTab(page, "set"); await page.fill("#maxtok", "40"); await send(page, "length: cut me off"); await waitIdle(page); await sleep(150); });
    await step("02-thinkonly", async () => { await page.fill("#maxtok", "60"); await send(page, "thinkonly: only thinks"); await waitIdle(page); await sleep(150); });
  }
  if (scenario === "errors") {
    await step("01-fail400", async () => { await send(page, "fail400 please"); await waitIdle(page); await sleep(150); });
    await step("02-failmid", async () => { await send(page, "failmid please"); await waitIdle(page); await sleep(150); });
    await step("03-recover", async () => { await send(page, "fine now"); await waitIdle(page); await sleep(150); });
  }
  if (scenario === "stop") {
    await step("01-stop", async () => {
      await send(page, "long answer please");
      await page.waitForFunction(() => (document.querySelectorAll(".turn.is-model .body")[0]?.textContent.length || 0) > 40, null, { timeout: 15000 });
      await page.keyboard.press("Escape");
      await waitIdle(page); await sleep(150);
    });
    await step("02-after-stop", async () => { await send(page, "nothink short"); await waitIdle(page); await sleep(150); });
  }
  if (scenario === "live") {
    // mid-stream captures: thinking shimmer, then hot bars + caret
    await step("01-thinking", async () => { await send(page, "long answer please"); await page.waitForSelector("details.think.live", { timeout: 10000 }); await sleep(250); });
    await step("02-streaming", async () => { await page.waitForFunction(() => (document.querySelectorAll(".turn.is-model .body")[0]?.textContent.length || 0) > 200, null, { timeout: 20000 }); });
    await step("03-done", async () => { await waitIdle(page); await sleep(150); });
  }
  if (scenario === "narrow") {
    await page.setViewportSize({ width: 620, height: 900 });
    await step("01-narrow", async () => { await send(page, "nothink: narrow layout"); await waitIdle(page); await sleep(150); });
  }
  if (scenario === "swap") {
    await step("01-pick", async () => {
      const sel = await page.$("select#model, #model select, #model-select");
      out.hasSelect = !!sel;
      if (sel) await sel.selectOption({ index: 1 });
    });
    await step("02-swapped", async () => { await send(page, "nothink hello new model"); await waitIdle(page, 40000); await sleep(300); });
  }
  if (scenario === "system") {
    await step("01-system", async () => {
      await openTab(page, "set"); const sys = await page.$("#system");
      out.hasSystem = !!sys;
      if (sys) { await page.evaluate(() => { const d = document.querySelector("#system")?.closest("details"); if (d) d.open = true; }); await page.fill("#system", "You are terse."); }
      await send(page, "nothink hi"); await waitIdle(page); await sleep(150);
    });
    out.sentBody = bodies[bodies.length - 1] || null;
  }
  if (scenario === "reload") {
    await step("01-set", async () => { await openTab(page, "set"); await page.fill("#temp", "0.3"); await page.fill("#maxtok", "999"); await page.evaluate(() => document.getElementById("temp").dispatchEvent(new Event("change", { bubbles: true }))); await page.evaluate(() => document.getElementById("maxtok").dispatchEvent(new Event("change", { bubbles: true }))); });
    await step("02-reloaded", async () => { await page.reload(); await sleep(600); });
    out.values = await page.evaluate(() => ({ temp: document.getElementById("temp").value, maxtok: document.getElementById("maxtok").value }));
  }
  if (scenario === "boot") {
    // mock started with --boot-delay: the UI must recover without a reload
    await step("01-waiting", async () => { await sleep(1000); });
    await step("02-recovered", async () => {
      await page.waitForFunction(() => /idle/.test(document.getElementById("state")?.textContent || ""), null, { timeout: 30000 });
    });
  }
  if (scenario === "actions") {
    await step("01-answer", async () => { await send(page, "nothink with code"); await waitIdle(page); await sleep(150); });
    await step("02-copy", async () => {
      const btn = await page.$(".turn.is-model .act-copy, .turn.is-model button");
      out.hasCopy = !!btn;
      if (btn) await btn.click();
      await sleep(200);
    });
    await step("03-regen", async () => {
      const btn = await page.$(".turn.is-model .act-regen");
      out.hasRegen = !!btn;
      if (btn) { await btn.click(); await waitIdle(page); await sleep(150); }
    });
    await step("04-newchat", async () => {
      const btn = await page.$("#newchat, .act-new");
      out.hasNew = !!btn;
      if (btn) { await btn.click(); await sleep(200); }
    });
  }

  if (scenario === "features") {
    // One pass over every feature of the redesign; each check lands in out.checks.
    const checks = (out.checks = {});
    const q = (fn, ...a) => page.evaluate(fn, ...a);
    await step("01-answer", async () => { await send(page, "Explain streaming, with code."); await waitIdle(page); await sleep(1500); });
    checks.highlight = await q(() => !!document.querySelector(".turn.is-model pre .tk-k"));
    checks.thinkPane = await q(() => !!document.querySelector(".turn.is-model details.think"));
    checks.energyFoot = await q(() => /J, \d+ W avg/.test(document.querySelector(".turn.is-model .stats")?.textContent || ""));
    checks.energyRun = await q(() => document.getElementById("m-energy").textContent);
    checks.speedPoints = await q(() => document.querySelectorAll("#speed .hit").length);

    await step("02-tokens", async () => {
      await openTab(page, "set");
      await page.check("#logprobs");
      await openTab(page, "run");
      await send(page, "nothink token view"); await waitIdle(page); await sleep(200);
      await page.click(".turn.is-model:last-of-type .act-tokens");
      await page.hover(".turn.is-model:last-of-type .tk.p3, .turn.is-model:last-of-type .tk.p4");
      await sleep(150);
    });
    checks.logprobsSent = out.lastBodyTokens = bodies[bodies.length - 1]?.logprobs === true && bodies[bodies.length - 1]?.top_logprobs === 5;
    checks.tokenSpans = await q(() => document.querySelectorAll(".body.tokens .tk").length);
    checks.tokenTip = await q(() => !document.getElementById("tk-tip").hidden && document.querySelectorAll("#tk-tip tr").length);
    await openTab(page, "set"); await page.uncheck("#logprobs"); await openTab(page, "run");

    await step("03-continue", async () => {
      await openTab(page, "set"); await page.fill("#maxtok", "40"); await openTab(page, "run");
      await send(page, "nothink length: cut me off"); await waitIdle(page); await sleep(150);
      await page.click(".act-cont"); await waitIdle(page); await sleep(150);
    });
    checks.continued = await q(() => document.querySelectorAll(".turn.is-cont").length === 1 && !!document.querySelector(".cont-mark"));
    checks.continueBodyLastUser = (bodies[bodies.length - 1]?.messages || []).at(-1)?.content?.startsWith("Continue exactly");
    await openTab(page, "set"); await page.fill("#maxtok", "512"); await openTab(page, "run");

    await step("04-edit", async () => {
      const n0 = await q(() => document.querySelectorAll(".turn").length);
      checks.turnsBeforeEdit = n0;
      await page.click(".turn.is-user .act-edit");
      await page.fill(".edit-box textarea", "nothink edited first question");
      await page.click(".edit-box .btn.primary"); await waitIdle(page); await sleep(150);
    });
    checks.editTruncated = await q(() => document.querySelectorAll(".turn").length === 2 && document.querySelector(".turn.is-user .body").textContent === "nothink edited first question");

    await step("05-end-thinking", async () => {
      await send(page, "deepthink then answer");
      await page.waitForSelector(".end-think", { timeout: 10000 });
      await sleep(300);
      await page.click(".end-think");
      await waitIdle(page); await sleep(150);
    });
    checks.endThinkingAnswered = await q(() => (document.querySelector(".turn.is-model:last-of-type > .body")?.textContent || "").length > 50);
    checks.endThinkingShort = await q(() => (document.querySelector(".turn.is-model:last-of-type details.think .body")?.textContent || "").length < 2000);

    await step("06-schema-invalid", async () => {
      await openTab(page, "set");
      await page.selectOption("#fmt", "json_schema");
      await page.fill("#schema", "{not json");
      await send(page, "structured please"); await sleep(300);
    });
    checks.schemaBlocked = await q(() => /not usable/.test(document.querySelector(".fault")?.textContent || ""));
    checks.schemaState = await q(() => document.getElementById("schema-state").textContent);
    await step("07-schema-valid", async () => {
      await page.fill("#schema", '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}');
      await page.click("#send"); await waitIdle(page); await sleep(150);
      await page.selectOption("#fmt", "none");
      await openTab(page, "run");
    });
    checks.schemaSent = bodies[bodies.length - 1]?.response_format?.type === "json_schema" && !!bodies[bodies.length - 1]?.response_format?.json_schema?.schema;

    await step("08-image", async () => {
      await page.setInputFiles("#attach-file", { name: "imp.png", mimeType: "image/png", buffer: PNG });
      await sleep(200);
      await send(page, "nothink what is in the picture");
      await waitIdle(page); await sleep(150);
    });
    const ib = bodies[bodies.length - 1];
    checks.imageSent = Array.isArray(ib?.messages?.at(-1)?.content) && ib.messages.at(-1).content.some((p) => p.type === "image_url" && p.image_url.url.startsWith("data:image/"));
    checks.imageThumb = await q(() => !!document.querySelector(".turn.is-user .thumbs-row img"));

    await step("09-tokenize", async () => {
      await page.fill("#prompt", "count these few words for me please");
      await page.dispatchEvent("#prompt", "input");
      await sleep(1400);
    });
    checks.ctxNote = await q(() => document.getElementById("ctx-note").textContent);
    await page.fill("#prompt", "");

    await step("10-server", async () => { await openTab(page, "srv"); await sleep(3600); });
    checks.server = await q(() => Object.fromEntries(["s-flight", "s-kv", "s-vram", "s-power", "s-energy", "s-req", "i-ctx", "i-slots", "i-input"].map((id) => [id, document.getElementById(id).textContent])));
    checks.powerRowShown = await q(() => !document.getElementById("s-power-row").hidden);
    await openTab(page, "run");

    await step("11-fast", async () => { await send(page, "fast nothink speedy"); await waitIdle(page); await sleep(120); });
    checks.fastBubble = await q(() => document.getElementById("bubble").textContent);

    await step("12-convs", async () => {
      await page.keyboard.press("Escape");
      await page.click("#log");
      await page.keyboard.press("Control+k");
      await sleep(150);
      await send(page, "nothink second conversation about kittens"); await waitIdle(page); await sleep(150);
      await page.fill("#conv-search", "kittens"); await sleep(100);
    });
    checks.convCount = await q(() => document.querySelectorAll("#convs li[data-id]").length);
    await page.fill("#conv-search", "");
    checks.convTotal = await q(() => document.querySelectorAll("#convs li[data-id]").length);
    const dl = page.waitForEvent("download");
    await page.click("#export-json");
    const d = await dl;
    checks.exportName = d.suggestedFilename();
    const exported = JSON.parse(require("fs").readFileSync(await d.path(), "utf8"));
    exported.conversations[0].title = "Imported copy";
    await page.setInputFiles("#import-file", { name: "chat.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify(exported)) });
    await sleep(300);
    checks.imported = await q(() => [...document.querySelectorAll("#convs .t")].some((t) => t.textContent === "Imported copy"));
    const dm = page.waitForEvent("download");
    await page.click("#export-md");
    checks.exportMd = (await dm).suggestedFilename();

    await step("13-help", async () => { await page.click("#log"); await page.keyboard.press("?"); await sleep(150); });
    checks.helpOpen = await q(() => document.getElementById("help").open);
    await page.keyboard.press("Escape");

    await step("14-reload", async () => { await page.reload(); await sleep(800); });
    checks.afterReload = await q(() => ({ convs: document.querySelectorAll("#convs li[data-id]").length, turns: document.querySelectorAll(".turn").length }));
    await step("15-light", async () => { await page.click("#theme"); await sleep(200); });
    checks.theme = await q(() => document.documentElement.dataset.theme);
  }
  if (scenario === "shots") {
    // README / review captures: desktop dark and light, settings, server, mobile.
    await page.emulateMedia({ colorScheme: "dark" });
    await page.evaluate(() => { localStorage.clear(); });
    await page.reload(); await sleep(600);
    await send(page, "Explain streaming, with code."); await waitIdle(page); await sleep(1600);
    await send(page, "fast nothink And how fast was that?"); await waitIdle(page); await sleep(1000);  // fireball landed, remark still up
    await page.screenshot({ path: `${SHOTS}/desktop-dark.png` });
    await openTab(page, "set"); await sleep(150);
    await page.screenshot({ path: `${SHOTS}/settings.png` });
    await openTab(page, "srv"); await sleep(3500);
    await page.screenshot({ path: `${SHOTS}/server.png` });
    await openTab(page, "run");
    await page.emulateMedia({ colorScheme: "light" }); await sleep(250);
    await page.screenshot({ path: `${SHOTS}/desktop-light.png` });
    await page.emulateMedia({ colorScheme: "dark" });
    await page.setViewportSize({ width: 390, height: 844 }); await sleep(300);
    await page.screenshot({ path: `${SHOTS}/mobile.png` });
    await page.click("#open-side"); await sleep(350);
    await page.screenshot({ path: `${SHOTS}/mobile-chats.png` });
    await page.keyboard.press("Escape");
    await openTab(page, "srv"); await sleep(3500);
    await page.screenshot({ path: `${SHOTS}/mobile-server.png` });
    out.mobileOverflow = await page.evaluate(() => document.documentElement.scrollWidth > innerWidth || document.body.scrollWidth > innerWidth);
  }

  out.consoleErrors = errors;
  out.lastBody = bodies[bodies.length - 1] || null;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch((e) => { console.error("DRIVER FAIL", e); process.exit(1); });
