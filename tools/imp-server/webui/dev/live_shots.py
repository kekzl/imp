#!/usr/bin/env python3
"""Docs screenshots (docs/webui*.png) against a live imp-server: one real chat, four captures.

The page itself comes from the mounted index.html (page.route), the API from the server, so the
captures show the working tree's UI with real numbers without restarting the server.

    docker run --rm --network host -v "$PWD/tools/imp-server/webui":/ui:ro -v "$PWD/docs":/shots \
        mcr.microsoft.com/playwright/python:v1.55.0-noble \
        bash -c 'pip install -q playwright==1.55.0 && python /ui/dev/live_shots.py http://localhost:8080/'
"""
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080/"
OUT = "/shots"
IDLE_JS = """() => /idle|stopped|error|no server|no model/.test(document.getElementById("state")?.textContent || "")"""
PROMPT = ("Compare quicksort, mergesort and heapsort in a short markdown table "
          "(average time, worst case, extra memory), then show quicksort in Python.")


def answer(page, text):
    page.fill("#prompt", text)
    page.press("#prompt", "Enter")
    time.sleep(1.5)
    page.wait_for_function(IDLE_JS, timeout=300000)
    time.sleep(1.2)


with sync_playwright() as p:
    b = p.chromium.launch()
    page = b.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
    html = open("/ui/index.html", encoding="utf-8").read()
    page.route(URL, lambda r: r.fulfill(status=200, content_type="text/html; charset=utf-8", body=html))
    page.emulate_media(color_scheme="dark")
    page.goto(URL)
    page.evaluate("() => localStorage.clear()")
    page.reload()
    page.wait_for_function(IDLE_JS, timeout=120000)
    answer(page, PROMPT)
    page.evaluate("() => { document.getElementById('log').scrollTop = 0; }")
    time.sleep(0.4)
    page.screenshot(path=f"{OUT}/webui.png")
    page.click("#tab-srv")
    time.sleep(3.5)
    page.screenshot(path=f"{OUT}/webui-server.png")
    page.click("#tab-run")
    page.emulate_media(color_scheme="light")
    time.sleep(0.4)
    page.screenshot(path=f"{OUT}/webui-light.png")
    page.emulate_media(color_scheme="dark")
    page.set_viewport_size({"width": 390, "height": 844})
    time.sleep(0.5)
    page.screenshot(path=f"{OUT}/webui-mobile.png")
    b.close()
print("ok")
