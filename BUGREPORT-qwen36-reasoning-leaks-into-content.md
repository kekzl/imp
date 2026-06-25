---
name: Bug report
about: Reasoning leaks into the `content` channel (streaming) for Qwen3.6-35B-A3B-NVFP4
labels: bug
---

## What happened

With `--reasoning-format deepseek` (the default) and **Qwen3.6-35B-A3B-NVFP4**, the
server **also leaks chain-of-thought into the `content` channel** instead of keeping
it confined to `reasoning_content`. `reasoning_content` *is* populated correctly
(~460–480 chars per turn), but the model's post-`</think>` rambling — "The user wants
to know…", "I should formulate a response…", "Response: …" — still ends up in
`message.content` (and in the streamed `delta.content`).

This breaks every OpenAI-compatible client that does the normal thing and reads
`content` as the answer: the operator sees (and, with a TTS frontend, hears) the
model's reasoning prose, sometimes followed by a second JSON tool-call. It is
**intermittent** — many turns are clean (`content` is answer-only) — and shows up far
more often at higher temperature (seen reliably at `temperature=0.8`, rarely at 0.2).

It reproduces on the **streaming** path. Non-streaming `/v1/chat/completions` for the
same prompts returns a clean `content` in my sampling, so the split appears less
reliable for the incremental `delta.content` emission.

## Reproducer

```bash
# Server: stock launch, default reasoning-format=deepseek, think-budget=0.5
docker run -d --name imp --gpus all -v /home/kekz/models:/models -p 8080:8080 \
  ghcr.io/kekzl/imp:latest --model /models/Qwen3.6-35B-A3B-NVFP4

# Drive 3 ReAct-style turns through the streaming endpoint and split the two
# channels. Flags any turn whose `content` contains reasoning markers.
python3 - <<'PY'
import json, urllib.request
SYS = ('You are a precise assistant. To use a tool, respond with ONE JSON object and '
       'nothing else: {"tool":"<name>","args":{...}}. Otherwise write plain text. '
       'Tools: web_search(query), news(topic), weather(city). For any real-world fact '
       'you MUST use a tool first. Reply in the operator language. Keep answers short.')
turns = [
 [{"role":"user","content":"wie wird das wetter morgen?"},
  {"role":"assistant","content":'{"tool": "weather", "args": {"city": "Karlsruhe"}}'},
  {"role":"user","content":'TOOL RESULT (weather): 18°C, bewölkt, 40% Regen'}],
 [{"role":"user","content":"was könnte ich kochen?"},
  {"role":"assistant","content":'{"tool": "web_search", "args": {"query": "schnelle Abendessen"}}'},
  {"role":"user","content":'TOOL RESULT (web_search): Chefkoch ... Lecker.de ... (noisy titles)'}],
 [{"role":"user","content":"gib mir nachrichten zu KI"},
  {"role":"assistant","content":'{"tool": "news", "args": {"topic": "KI"}}'},
  {"role":"user","content":'TOOL RESULT (news): 1) EU AI Act 2) OpenAI ... 3) ...'}],
]
MARK = ["The user","I will","I'll","I performed","I should","I need to","Let me","Tool:","Args:","Response:"]
def stream(msgs):
    body = json.dumps({"model":"Qwen3.6-35B-A3B-NVFP4","messages":msgs,
                       "max_tokens":220,"temperature":0.8,"stream":True}).encode()
    r = urllib.request.urlopen(urllib.request.Request(
        "http://localhost:8080/v1/chat/completions", data=body,
        headers={"Content-Type":"application/json"}), timeout=120)
    content = reason = ""
    for raw in r:
        line = raw.strip()
        if not line.startswith(b"data:"): continue
        d = line[5:].strip()
        if d == b"[DONE]": break
        try: delta = json.loads(d)["choices"][0].get("delta", {})
        except Exception: continue
        content += delta.get("content") or ""
        reason  += delta.get("reasoning_content") or ""
    return content, reason
for n, t in enumerate(turns):
    c, rr = stream([{"role":"system","content":SYS}] + t)
    leak = any(m in c for m in MARK)
    print(f"[{n}] content_leak={leak}  reasoning_content_len={len(rr)}")
    if leak: print("    content:", repr(c[:300]))
PY
```

Observed: **3/3 turns leaked** at temperature 0.8.

## Environment

- imp version: **0.12.4** (image revision `c2bc425b`, digest
  `sha256:ae3837ad4d48384bd19def1a3d9ad424ee3c3e98b2a2b1457497a699fd82a764`)
- GPU + driver: NVIDIA GeForce RTX 5090, driver **610.62**
- Build: prebuilt `ghcr.io/kekzl/imp:latest`
- Model: `/models/Qwen3.6-35B-A3B-NVFP4` (SafeTensors, NVFP4 prequant)
- Flags: stock — `--model` only ⇒ `--reasoning-format deepseek`, `--think-budget 0.5`
  (defaults per `docs/usage.md`)

## Output

Cleanest example (weather turn), captured from the streaming endpoint:

```
reasoning_content (len 460, correct):  "<the actual think block>"

content (LEAK — should be answer-only):
  "The user wants to know the weather forecast.
   The tool output provided is: 18°C, bewölkt, 40% Regen.
   I should formulate a response in German describing this weather.

   Response: Morgen wird es 18°C, es wird bewölkt sein und eine 40%ige
   Regenwahrscheinlichkeit geben. Or something more conversational bu…"
```

Other turns in the same run leaked similarly:
- cooking turn: `content` opened with a JSON fragment, then "…I already responded with
  the JSON. Now I need to process the tool output. Actually, the prompt structure
  implies I am generating a response based on the tool usage simulation…"
- news turn: `content` held two ```json fenced tool-calls, the first with doubled
  quotes (`{""tool"": ""recall""…}`) — malformed and also reasoning-channel material.

## Field data (106-prompt run)

A downstream agent (Nina) ran 106 diverse German prompts through the **streaming**
endpoint at its default **`temperature=0.2`** (0 errors, model otherwise solid —
19/20 tools exercised, avg 19 s/turn). Reasoning still bled into `content` on
**~4% of turns even at 0.2**, including a plain Q&A with **no tools involved**:

- Prompt: *"Erzähl mir einen kurzen Witz."* → `content`:
  `Ein Gummibärchen."\n\nLet's go with the bear one. It's short… Actually, let's try
  a dev-related one since I am a dev assistant… Maybe too obscure. Let's stick to the
  bear one… Okay, I'll just write…` — i.e. the model's deliberation lands in `content`
  with the actual answer buried inside. `reasoning_content` was populated in parallel.

So the leak is not specific to tool-call turns or high temperature; it's the
reasoning/answer demux on the `content` stream being incomplete for this model.

(Separately, the same run surfaced turns where the model emits **two** back-to-back
JSON objects in `content` — a client-side parsing issue we fixed on our end, not an
imp bug. Flagging only so the repro noise is attributed correctly.)

## Notes

- `reasoning_content` is populated on every turn, so the deepseek split partially
  works; the defect is that reasoning **also** reaches `content` rather than being
  fully confined. Likely a missed/duplicated `</think>` boundary on the incremental
  `delta.content` path for this model (Qwen3.6 NVFP4, fullwidth/added-token think
  markers — see `src/runtime/request.h:79-85`, `src/model/chat_template.cpp:187-192`).
- Temperature-sensitive (much worse at 0.8 than 0.2), so partly model rambling — but
  the serving contract is that `content` is answer-only when a `reasoning_content`
  channel exists.
- Worth checking whether the streaming and non-streaming paths share the same
  reasoning/content demux; non-streaming looked clean for the same prompts.
- Downstream workaround already applied in the Nina client (buffer the full turn,
  strip `<think>` and any embedded action JSON before showing/speaking), but clients
  reading `content` shouldn't have to defend against reasoning bleed.
