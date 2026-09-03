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
// scenarios: chat, length, errors, stop, narrow, swap, system, reload, actions,
// boot (needs a mock started with --boot-delay).
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
  }));
}

async function send(page, text) {
  await page.fill("#prompt", text);
  await page.press("#prompt", "Enter");
}

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
    await step("01-length", async () => { await page.fill("#maxtok", "40"); await send(page, "length: cut me off"); await waitIdle(page); await sleep(150); });
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
      const sys = await page.$("#system");
      out.hasSystem = !!sys;
      if (sys) { await page.evaluate(() => { const d = document.querySelector("#system")?.closest("details"); if (d) d.open = true; }); await page.fill("#system", "You are terse."); }
      await send(page, "nothink hi"); await waitIdle(page); await sleep(150);
    });
    out.sentBody = bodies[bodies.length - 1] || null;
  }
  if (scenario === "reload") {
    await step("01-set", async () => { await page.fill("#temp", "0.3"); await page.fill("#maxtok", "999"); await page.evaluate(() => document.getElementById("temp").dispatchEvent(new Event("change", { bubbles: true }))); await page.evaluate(() => document.getElementById("maxtok").dispatchEvent(new Event("change", { bubbles: true }))); });
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

  out.consoleErrors = errors;
  out.lastBody = bodies[bodies.length - 1] || null;
  console.log(JSON.stringify(out, null, 1));
  await browser.close();
})().catch((e) => { console.error("DRIVER FAIL", e); process.exit(1); });
