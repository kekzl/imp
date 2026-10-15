#!/usr/bin/env python3
"""Scenario runner for the imp web UI against dev/mock_server.py (or a real server).

Drives a headless Chromium through one scenario, screenshots every step into
/shots and prints the readouts as JSON. No host install: the Playwright Python
image carries the browsers; the package itself is pip-installed in the container.

    D=tools/imp-server/webui/dev
    docker run --rm --network host -v "$PWD/$D":/w -v "$PWD/build-dev/shots":/shots -w /w \
        mcr.microsoft.com/playwright/python:v1.55.0-noble \
        bash -c 'pip install -q playwright==1.55.0 && python drive.py chat http://localhost:9099/'

scenarios: chat, length, errors, stop, live (mid-stream captures), narrow,
swap, system, reload, actions, boot (needs a mock started with --boot-delay),
features (every redesign feature, results in "checks"), shots (review captures).
usage: python drive.py <scenario> [url]
Exit code 0 = scenario finished, 1 = driver failure (timeout, missing element).
"""
import base64
import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:9099/"
SCENARIO = sys.argv[1] if len(sys.argv) > 1 else "baseline"
SHOTS = "/shots"

# 1x1 PNG, for the attachment path.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)

STATE_JS = """() => ({
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
})"""

IDLE_JS = """() => {
  const s = document.getElementById("state")?.textContent || "";
  return /idle|stopped|error|no server|no model/.test(s);
}"""


def sleep(ms):
    time.sleep(ms / 1000)


def state(page):
    return page.evaluate(STATE_JS)


def send(page, text):
    page.fill("#prompt", text)
    page.press("#prompt", "Enter")


def open_tab(page, name):
    """The right panel is tabbed (run, set, srv); inputs on a hidden tab cannot be filled."""
    page.evaluate('() => { if (innerWidth <= 900) document.getElementById("open-panel")?.click(); }')
    page.click("#tab-" + name)


def wait_idle(page, timeout=30000):
    page.wait_for_function(IDLE_JS, timeout=timeout)


def dig(obj, *path):
    """Optional-chaining lookup: dict keys and list indexes, None on any miss."""
    for p in path:
        try:
            obj = obj[p]
        except (KeyError, IndexError, TypeError):
            return None
    return obj


def run(pw):
    browser = pw.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1280, "height": 820}, device_scale_factor=1)
    page = ctx.new_page()
    errors = []
    bodies = []

    def on_console(m):
        if m.type in ("error", "warning"):
            errors.append(m.type + ": " + m.text)

    def on_request(r):
        if r.method == "POST":
            try:
                bodies.append(json.loads(r.post_data))
            except (TypeError, ValueError):
                pass

    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append("pageerror: " + str(e)))
    page.on("request", on_request)

    page.goto(URL)
    try:
        page.wait_for_function(
            '() => !/starting|connecting/.test(document.getElementById("state")?.textContent || "")',
            timeout=15000,
        )
    except Exception:
        pass
    out = {"scenario": SCENARIO, "steps": []}
    last_body = lambda: bodies[-1] if bodies else None  # noqa: E731

    def step(name, fn):
        fn()
        s = state(page)
        page.screenshot(path=f"{SHOTS}/{SCENARIO}-{name}.png", full_page=False)
        out["steps"].append({"name": name, **s})

    def answer_and_settle(text, timeout=30000, settle=150):
        send(page, text)
        wait_idle(page, timeout)
        sleep(settle)

    step("00-loaded", lambda: None)

    if SCENARIO in ("baseline", "chat"):
        step("01-answer", lambda: answer_and_settle("Explain streaming, with code."))
        step("02-followup", lambda: answer_and_settle("And a follow-up, please."))

    if SCENARIO == "length":
        def s1():
            open_tab(page, "set")
            page.fill("#maxtok", "40")
            answer_and_settle("length: cut me off")

        def s2():
            page.fill("#maxtok", "60")
            answer_and_settle("thinkonly: only thinks")

        step("01-length", s1)
        step("02-thinkonly", s2)

    if SCENARIO == "errors":
        step("01-fail400", lambda: answer_and_settle("fail400 please"))
        step("02-failmid", lambda: answer_and_settle("failmid please"))
        step("03-recover", lambda: answer_and_settle("fine now"))

    if SCENARIO == "stop":
        def s1():
            send(page, "long answer please")
            page.wait_for_function(
                '() => (document.querySelectorAll(".turn.is-model .body")[0]?.textContent.length || 0) > 40',
                timeout=15000,
            )
            page.keyboard.press("Escape")
            wait_idle(page)
            sleep(150)

        step("01-stop", s1)
        step("02-after-stop", lambda: answer_and_settle("nothink short"))

    if SCENARIO == "live":
        # mid-stream captures: thinking shimmer, then hot bars + caret
        def s1():
            send(page, "long answer please")
            page.wait_for_selector("details.think.live", timeout=10000)
            sleep(250)

        def s2():
            page.wait_for_function(
                '() => (document.querySelectorAll(".turn.is-model .body")[0]?.textContent.length || 0) > 200',
                timeout=20000,
            )

        def s3():
            wait_idle(page)
            sleep(150)

        step("01-thinking", s1)
        step("02-streaming", s2)
        step("03-done", s3)

    if SCENARIO == "narrow":
        page.set_viewport_size({"width": 620, "height": 900})
        step("01-narrow", lambda: answer_and_settle("nothink: narrow layout"))

    if SCENARIO == "swap":
        def s1():
            sel = page.query_selector("select#model, #model select, #model-select")
            out["hasSelect"] = sel is not None
            if sel:
                sel.select_option(index=1)

        step("01-pick", s1)
        step("02-swapped", lambda: answer_and_settle("nothink hello new model", timeout=40000, settle=300))

    if SCENARIO == "system":
        def s1():
            open_tab(page, "set")
            sys_el = page.query_selector("#system")
            out["hasSystem"] = sys_el is not None
            if sys_el:
                page.evaluate(
                    '() => { const d = document.querySelector("#system")?.closest("details"); if (d) d.open = true; }'
                )
                page.fill("#system", "You are terse.")
            answer_and_settle("nothink hi")

        step("01-system", s1)
        out["sentBody"] = last_body()

    if SCENARIO == "reload":
        change = '(id) => document.getElementById(id).dispatchEvent(new Event("change", { bubbles: true }))'

        def s1():
            open_tab(page, "set")
            page.fill("#temp", "0.3")
            page.fill("#maxtok", "999")
            page.evaluate(change, "temp")
            page.evaluate(change, "maxtok")

        def s2():
            page.reload()
            sleep(600)

        step("01-set", s1)
        step("02-reloaded", s2)
        out["values"] = page.evaluate(
            '() => ({ temp: document.getElementById("temp").value, maxtok: document.getElementById("maxtok").value })'
        )

    if SCENARIO == "boot":
        # mock started with --boot-delay: the UI must recover without a reload
        step("01-waiting", lambda: sleep(1000))
        step(
            "02-recovered",
            lambda: page.wait_for_function(
                '() => /idle/.test(document.getElementById("state")?.textContent || "")', timeout=30000
            ),
        )

    if SCENARIO == "actions":
        def s2():
            btn = page.query_selector(".turn.is-model .act-copy, .turn.is-model button")
            out["hasCopy"] = btn is not None
            if btn:
                btn.click()
            sleep(200)

        def s3():
            btn = page.query_selector(".turn.is-model .act-regen")
            out["hasRegen"] = btn is not None
            if btn:
                btn.click()
                wait_idle(page)
                sleep(150)

        def s4():
            btn = page.query_selector("#newchat, .act-new")
            out["hasNew"] = btn is not None
            if btn:
                btn.click()
                sleep(200)

        step("01-answer", lambda: answer_and_settle("nothink with code"))
        step("02-copy", s2)
        step("03-regen", s3)
        step("04-newchat", s4)

    if SCENARIO == "features":
        features(page, out, bodies, step, answer_and_settle)

    if SCENARIO == "shots":
        shots(page, out)

    out["consoleErrors"] = errors
    out["lastBody"] = last_body()
    print(json.dumps(out, indent=1, ensure_ascii=False))
    browser.close()


def features(page, out, bodies, step, answer_and_settle):
    """One pass over every feature of the redesign; each check lands in out["checks"]."""
    checks = out["checks"] = {}
    q = page.evaluate
    lb = lambda: bodies[-1] if bodies else None  # noqa: E731

    step("01-answer", lambda: answer_and_settle("Explain streaming, with code.", settle=1500))
    checks["highlight"] = q('() => !!document.querySelector(".turn.is-model pre .tk-k")')
    checks["thinkPane"] = q('() => !!document.querySelector(".turn.is-model details.think")')
    checks["energyFoot"] = q(
        '() => /J, \\d+ W avg/.test(document.querySelector(".turn.is-model .stats")?.textContent || "")'
    )
    checks["energyRun"] = q('() => document.getElementById("m-energy").textContent')
    checks["speedPoints"] = q('() => document.querySelectorAll("#speed .hit").length')

    def s2():
        open_tab(page, "set")
        page.check("#logprobs")
        open_tab(page, "run")
        answer_and_settle("nothink token view", settle=200)
        page.click(".turn.is-model:last-of-type .act-tokens")
        page.hover(".turn.is-model:last-of-type .tk.p3, .turn.is-model:last-of-type .tk.p4")
        sleep(150)

    step("02-tokens", s2)
    checks["logprobsSent"] = out["lastBodyTokens"] = (
        dig(lb(), "logprobs") is True and dig(lb(), "top_logprobs") == 5
    )
    checks["tokenSpans"] = q('() => document.querySelectorAll(".body.tokens .tk").length')
    checks["tokenTip"] = q(
        '() => !document.getElementById("tk-tip").hidden && document.querySelectorAll("#tk-tip tr").length'
    )
    open_tab(page, "set")
    page.uncheck("#logprobs")
    open_tab(page, "run")

    def s3():
        open_tab(page, "set")
        page.fill("#maxtok", "40")
        open_tab(page, "run")
        answer_and_settle("nothink length: cut me off")
        page.click(".act-cont")
        wait_idle(page)
        sleep(150)

    step("03-continue", s3)
    checks["continued"] = q(
        '() => document.querySelectorAll(".turn.is-cont").length === 1 && !!document.querySelector(".cont-mark")'
    )
    last_user = dig(lb(), "messages", -1, "content")
    checks["continueBodyLastUser"] = last_user.startswith("Continue exactly") if isinstance(last_user, str) else None
    open_tab(page, "set")
    page.fill("#maxtok", "512")
    open_tab(page, "run")

    def s4():
        checks["turnsBeforeEdit"] = q('() => document.querySelectorAll(".turn").length')
        page.click(".turn.is-user .act-edit")
        page.fill(".edit-box textarea", "nothink edited first question")
        page.click(".edit-box .btn.primary")
        wait_idle(page)
        sleep(150)

    step("04-edit", s4)
    checks["editTruncated"] = q(
        '() => document.querySelectorAll(".turn").length === 2'
        ' && document.querySelector(".turn.is-user .body").textContent === "nothink edited first question"'
    )

    def s5():
        send(page, "deepthink then answer")
        page.wait_for_selector(".end-think", timeout=10000)
        sleep(300)
        page.click(".end-think")
        wait_idle(page)
        sleep(150)

    step("05-end-thinking", s5)
    checks["endThinkingAnswered"] = q(
        '() => (document.querySelector(".turn.is-model:last-of-type > .body")?.textContent || "").length > 50'
    )
    checks["endThinkingShort"] = q(
        '() => (document.querySelector(".turn.is-model:last-of-type details.think .body")?.textContent || "").length < 2000'
    )

    def s6():
        open_tab(page, "set")
        page.select_option("#fmt", "json_schema")
        page.fill("#schema", "{not json")
        send(page, "structured please")
        sleep(300)

    step("06-schema-invalid", s6)
    checks["schemaBlocked"] = q('() => /not usable/.test(document.querySelector(".fault")?.textContent || "")')
    checks["schemaState"] = q('() => document.getElementById("schema-state").textContent')

    def s7():
        page.fill("#schema", '{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}')
        page.click("#send")
        wait_idle(page)
        sleep(150)
        page.select_option("#fmt", "none")
        open_tab(page, "run")

    step("07-schema-valid", s7)
    checks["schemaSent"] = dig(lb(), "response_format", "type") == "json_schema" and bool(
        dig(lb(), "response_format", "json_schema", "schema")
    )

    def s8():
        page.set_input_files("#attach-file", files={"name": "imp.png", "mimeType": "image/png", "buffer": PNG})
        sleep(200)
        send(page, "nothink what is in the picture")
        wait_idle(page)
        sleep(150)

    step("08-image", s8)
    content = dig(lb(), "messages", -1, "content")
    checks["imageSent"] = isinstance(content, list) and any(
        isinstance(p, dict)
        and p.get("type") == "image_url"
        and str(dig(p, "image_url", "url") or "").startswith("data:image/")
        for p in content
    )
    checks["imageThumb"] = q('() => !!document.querySelector(".turn.is-user .thumbs-row img")')

    def s9():
        page.fill("#prompt", "count these few words for me please")
        page.dispatch_event("#prompt", "input")
        sleep(1400)

    step("09-tokenize", s9)
    checks["ctxNote"] = q('() => document.getElementById("ctx-note").textContent')
    page.fill("#prompt", "")

    def s10():
        open_tab(page, "srv")
        sleep(3600)

    step("10-server", s10)
    checks["server"] = q(
        '() => Object.fromEntries(["s-flight", "s-kv", "s-vram", "s-power", "s-energy", "s-req", "i-ctx", "i-slots", "i-input"]'
        ".map((id) => [id, document.getElementById(id).textContent]))"
    )
    checks["powerRowShown"] = q('() => !document.getElementById("s-power-row").hidden')
    open_tab(page, "run")

    step("11-fast", lambda: answer_and_settle("fast nothink speedy", settle=120))
    checks["fastBubble"] = q('() => document.getElementById("bubble").textContent')

    def s12():
        page.keyboard.press("Escape")
        page.click("#log")
        page.keyboard.press("Control+k")
        sleep(150)
        answer_and_settle("nothink second conversation about kittens")
        page.fill("#conv-search", "kittens")
        sleep(100)

    step("12-convs", s12)
    count_convs = '() => document.querySelectorAll("#convs li[data-id]").length'
    checks["convCount"] = q(count_convs)
    page.fill("#conv-search", "")
    checks["convTotal"] = q(count_convs)
    with page.expect_download() as dl:
        page.click("#export-json")
    d = dl.value
    checks["exportName"] = d.suggested_filename
    with open(d.path(), encoding="utf-8") as f:
        exported = json.load(f)
    exported["conversations"][0]["title"] = "Imported copy"
    page.set_input_files(
        "#import-file",
        files={"name": "chat.json", "mimeType": "application/json", "buffer": json.dumps(exported).encode()},
    )
    sleep(300)
    checks["imported"] = q('() => [...document.querySelectorAll("#convs .t")].some((t) => t.textContent === "Imported copy")')
    with page.expect_download() as dm:
        page.click("#export-md")
    checks["exportMd"] = dm.value.suggested_filename

    def s13():
        page.click("#log")
        page.keyboard.press("?")
        sleep(150)

    step("13-help", s13)
    checks["helpOpen"] = q('() => document.getElementById("help").open')
    page.keyboard.press("Escape")

    def s14():
        page.reload()
        sleep(800)

    step("14-reload", s14)
    checks["afterReload"] = q(
        '() => ({ convs: document.querySelectorAll("#convs li[data-id]").length,'
        ' turns: document.querySelectorAll(".turn").length })'
    )

    def s15():
        page.click("#theme")
        sleep(200)

    step("15-light", s15)
    checks["theme"] = q("() => document.documentElement.dataset.theme")


def shots(page, out):
    """README / review captures: desktop dark and light, settings, server, mobile."""
    page.emulate_media(color_scheme="dark")
    page.evaluate("() => { localStorage.clear(); }")
    page.reload()
    sleep(600)
    send(page, "Explain streaming, with code.")
    wait_idle(page)
    sleep(1600)
    send(page, "fast nothink And how fast was that?")  # fireball landed, remark still up
    wait_idle(page)
    sleep(1000)
    page.screenshot(path=f"{SHOTS}/desktop-dark.png")
    open_tab(page, "set")
    sleep(150)
    page.screenshot(path=f"{SHOTS}/settings.png")
    open_tab(page, "srv")
    sleep(3500)
    page.screenshot(path=f"{SHOTS}/server.png")
    open_tab(page, "run")
    page.emulate_media(color_scheme="light")
    sleep(250)
    page.screenshot(path=f"{SHOTS}/desktop-light.png")
    page.emulate_media(color_scheme="dark")
    page.set_viewport_size({"width": 390, "height": 844})
    sleep(300)
    page.screenshot(path=f"{SHOTS}/mobile.png")
    page.click("#open-side")
    sleep(350)
    page.screenshot(path=f"{SHOTS}/mobile-chats.png")
    page.keyboard.press("Escape")
    open_tab(page, "srv")
    sleep(3500)
    page.screenshot(path=f"{SHOTS}/mobile-server.png")
    out["mobileOverflow"] = page.evaluate(
        "() => document.documentElement.scrollWidth > innerWidth || document.body.scrollWidth > innerWidth"
    )


def main():
    try:
        with sync_playwright() as pw:
            run(pw)
    except Exception as e:  # noqa: BLE001
        print("DRIVER FAIL", e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
