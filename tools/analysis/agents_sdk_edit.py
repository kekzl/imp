#!/usr/bin/env python3
"""Drive imp-server's /v1/responses with the real OpenAI Agents SDK.

Roadmap gap 10's remaining leg. The other two external legs cover the
chat-completions dialect (aider) and the Anthropic one (Claude Code); the
Responses API is what the Agents SDK and Codex speak, and nothing outside our
own probes had ever driven it.

The assertion is the same as the other legs': a real function call has to land
an edit in a throwaway repo. A model that merely TALKS about editing fails.
"""
import asyncio
import os
import pathlib
import sys

from agents import (Agent, ModelSettings, Runner, function_tool, set_default_openai_api,
                    set_default_openai_client, set_tracing_disabled)
from openai import AsyncOpenAI

WORKDIR = pathlib.Path(os.environ.get("IMP_AGENT_WORKDIR", "/work"))


@function_tool
def write_file(path: str, content: str) -> str:
    """Write `content` to `path`, replacing whatever is there. Returns a confirmation."""
    target = WORKDIR / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


@function_tool
def read_file(path: str) -> str:
    """Return the current contents of `path`, or an empty string if it does not exist."""
    target = WORKDIR / path
    return target.read_text() if target.exists() else ""


async def main() -> int:
    base = os.environ.get("OPENAI_BASE_URL", "http://localhost:8080/v1")
    model = os.environ.get("IMP_MODEL", "Qwen3-8B-Q8_0.gguf")

    # No telemetry: tracing POSTs to api.openai.com and this box is pointed at
    # imp. A trace upload failure would be noise, not signal.
    set_tracing_disabled(True)
    set_default_openai_client(AsyncOpenAI(base_url=base, api_key="dummy"))
    set_default_openai_api("responses")

    agent = Agent(
        name="editor",
        instructions=(
            "You edit files in the current repository. When asked for a change, call write_file "
            "with the COMPLETE new contents of the file. Do not explain, just make the edit."
        ),
        tools=[write_file, read_file],
        model=model,
        # temperature=0 because this gate asserts the LOOP works, not that
        # sampling got lucky.
        #
        # max_tokens is the setting that actually decides the outcome, and the
        # reason is the MODEL, not the server. Measured on Qwen3-8B-Q8_0 against
        # this exact request: at 400 the reply is `reasoning` + `function_call`
        # (232 output tokens); at 1400 it is a bare `message` (511) — given room,
        # the model reasons its way past the call and answers in prose instead.
        # imp emits both shapes correctly, which is why the leg pins the budget
        # rather than the dialect.
        model_settings=ModelSettings(temperature=0.0, max_tokens=400),
    )

    result = await Runner.run(
        agent,
        "Add a Python function add(a, b) that returns a + b to math_utils.py.",
        max_turns=6,
    )
    print("--- run items ---")
    for it in result.new_items:
        kind = type(it).__name__
        raw = getattr(it, "raw_item", None)
        rtype = getattr(raw, "type", None)
        print(f"  {kind} raw.type={rtype}")
    print("--- final output ---")
    print(result.final_output)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except Exception as exc:  # noqa: BLE001 — the harness wants the reason, not a traceback
        print(f"agents-sdk driver failed: {type(exc).__name__}: {exc}")
        sys.exit(1)
