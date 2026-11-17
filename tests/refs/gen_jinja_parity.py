#!/usr/bin/env python3
"""Generate tests/fixtures/jinja_parity/*.json: imp's Jinja engine vs the env transformers uses.

Each template file holds one upstream chat template (fetched by repo id, {% generation %} tags
stripped as imp's ChatTemplate does) and its renders for four contexts: basic, multi-turn,
an agent turn (tools with nested schemas, two tool_calls, two tool results, reasoning) and
the same agent turn with enable_thinking=false. probes.json pins one construct per entry.
Dict key order is part of the contract: Python dicts keep insertion order, so do the fixtures.

Regenerate:  make jinja-parity      (docker, network for the HF fetch)
"""
import json
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen_chat_goldens import make_env  # noqa: E402  the replicated transformers env

OUT = os.path.join(HERE, "..", "fixtures", "jinja_parity")

# Templates whose renders hit an engine feature beyond the chat goldens.
TEMPLATES = [
    ("qwen3_coder", "Qwen/Qwen3-Coder-30B-A3B-Instruct"),   # loop filter, nested schema keys
    ("nemotron3_nano", "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"),  # previtem, loop filter
    ("qwen38", "Qwen/Qwen3.8-27B"),                         # generation tags, previtem
    ("glm47_flash", "zai-org/GLM-4.7-Flash"),               # tojson(kw) if v is not string
    ("flint", "kekzle/Flint-NVFP4"),                        # same, Flint tool-call replay
    ("kimi_linear", "moonshotai/Kimi-Linear-48B-A3B-Instruct"),  # tojson(separators=...)
    ("gpt_oss", "openai/gpt-oss-20b"),                      # schema walk in key order
]

TOOLS = [
    {"type": "function", "function": {
        "name": "bash", "description": "Run a shell command.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The command."},
            "timeout": {"type": "integer", "description": "Seconds.", "default": 60}},
            "required": ["command"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Edit a file.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "mode": {"type": "string", "enum": ["replace", "append"]},
            "lines": {"type": "array", "items": {"type": "integer"}},
            "opts": {"type": "object", "properties": {"dry": {"type": "boolean"}}}},
            "required": ["path", "mode"]}}},
]
AGENT = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "List the files and fix a.txt."},
    {"role": "assistant", "content": "Looking first.", "reasoning_content": "Need the listing.",
     "tool_calls": [
         {"id": "call_1", "type": "function",
          "function": {"name": "bash", "arguments": {"command": "ls -la", "timeout": 30}}},
         {"id": "call_2", "type": "function",
          "function": {"name": "edit_file", "arguments": {"path": "a.txt", "mode": "append",
                                                           "lines": [1, 2], "opts": {"dry": True}}}}]},
    {"role": "tool", "tool_call_id": "call_1", "name": "bash", "content": "a.txt\nb.txt"},
    {"role": "tool", "tool_call_id": "call_2", "name": "edit_file", "content": "ok"},
    {"role": "assistant", "content": "Done: a.txt fixed.", "reasoning_content": "Both calls ok."},
    {"role": "user", "content": "Thanks. Now count lines."},
]
CONTEXTS = {
    "basic": {"messages": [{"role": "user", "content": "What is 2+2?"}]},
    "multi": {"messages": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Hi"},
                           {"role": "assistant", "content": "Hello."}, {"role": "user", "content": "Bye"}]},
    "agent": {"messages": AGENT, "tools": TOOLS},
    "agent_nothink": {"messages": AGENT, "tools": TOOLS, "enable_thinking": False},
}

PROBE_CTX = {
    "items": [1, 2, 3, 1],
    "vals": ["s", 3, True, None, {"k": "v"}, [1]],
    "obj": {"zeta": 1, "alpha": "x", "mid": [1, 2]},
    "msgs": [{"role": "user", "content": "u1", "n": 1}, {"role": "assistant", "content": "a1", "n": 2},
             {"role": "user", "content": "u2", "n": 3}],
}
PROBES = [
    ("loop_filter", "{% for x in items if x != 2 %}{{ x }}{{ loop.index }}/{{ loop.length }};{% endfor %}"),
    ("loop_vars", "{% for x in items %}{{ loop.revindex }}{{ loop.revindex0 }}{{ loop.depth }}"
                  "{{ loop.cycle('a', 'b') }}{% endfor %}"),
    ("prev_next", "{% for x in items %}{{ loop.previtem }}<{{ x }}>{{ loop.nextitem }};{% endfor %}"),
    ("filter_kwargs_ternary", "{% for v in vals %}[{{ v | tojson(ensure_ascii=False) if v is not string else v }}]"
                              "{% endfor %}"),
    ("nested_ternary", "{{ 'a' if false else 'b' if true else 'c' }}"),
    ("tojson_opts", "{{ obj | tojson(separators=(',', ':')) }}|{{ obj | tojson(indent=2) }}|"
                    "{{ obj | tojson(sort_keys=true) }}|{{ 'ä' | tojson(ensure_ascii=true) }}"),
    ("dict_order", "{% for k, v in obj.items() %}{{ k }}={{ v }};{% endfor %}{{ obj.keys() | list }}"
                   "{{ {'b': 1, 'a': 2} | tojson }}{{ msgs[0] | tojson }}"),
    ("floats", "{{ 3.0 }}|{{ 1e3 }}|{{ 0.1 + 0.2 }}|{{ [1.0, 2.5, 1e20] | tojson }}|{{ 3.7 | round }}"),
    ("arith", "{{ 7 // 2 }}|{{ -7 // 2 }}|{{ 2 ** 10 }}|{{ -7 % 3 }}|{{ '-' * 3 }}|{{ [1] * 2 }}|{{ 0 == false }}"),
    ("tests", "{{ 4 is even }}{{ 3 is odd }}{{ 9 is divisibleby 3 }}{{ 9 is divisibleby(4) }}{{ none is none }}"
              "{{ 'ab' is lower }}{{ 3 is number }}{{ obj is mapping }}{{ strftime_now is defined }}"
              "{{ raise_exception is defined }}"),
    ("map_select", "{{ msgs | map(attribute='role') | join(',') }}|{{ items | map('string') | join('-') }}|"
                   "{{ msgs | selectattr('role', 'equalto', 'user') | list | length }}|"
                   "{{ msgs | sum(attribute='n') }}|{{ items | max }}{{ items | min }}{{ items | sum }}"),
    ("collections", "{{ items | unique | list }}|{{ items | batch(3) | list }}|{{ [3, 1, 2] | sort }}|"
                    "{{ msgs | sort(attribute='role') | map(attribute='content') | list }}|"
                    "{% for g in msgs | groupby('role') %}{{ g.grouper }}:{{ g.list | length }};{% endfor %}|"
                    "{% for k, v in obj | dictsort %}{{ k }};{% endfor %}"),
    ("string_filters", "{{ 'xxaxx' | trim('x') }}|{{ 'aaa' | replace('a', 'b', 2) }}|{{ '' | default('d', true) }}|"
                       "{{ 'a\nb' | indent(2) }}|{{ 'a\nb' | indent(2, true) }}|{{ 'a b c' | wordcount }}|"
                       "[{{ 'ab' | center(6) }}]|{{ '%s-%d' | format('a', 3) }}|{{ '<a>' | e }}|"
                       "{{ '42' | int }}|{{ 'hello world' | title }}|{{ 'hELLO' | capitalize }}"),
    ("string_methods", "{{ ' Ab '.strip() }}|{{ 'a,b,c'.split(',', 1) }}|{{ 'a,b,c'.rsplit(',', 1) }}|"
                       "{{ 'ab'.startswith(('x', 'a')) }}|{{ 'a b'.title() }}|{{ 'abcb'.rfind('b') }}|"
                       "{{ ', '.join(['x', 'y']) }}|{{ '{}-{}'.format('a', 3) }}|{{ '{0}{name}'.format(1, name=2) }}|"
                       "{{ '12'.isdigit() }}{{ 'Ab'.isupper() }}|{{ 'hello world'.capitalize() }}"),
    ("list_methods", "{{ items.index(2) }}|{{ items.count(1) }}|{{ obj.get('nope', 'dflt') }}|"
                     "{{ obj.values() | list }}"),
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def load(repo):
    base = "https://huggingface.co/%s/resolve/main/" % repo
    cfg = json.loads(fetch(base + "tokenizer_config.json"))
    try:
        tpl = fetch(base + "chat_template.jinja")
    except Exception:  # noqa: BLE001
        tpl = cfg["chat_template"]

    def tok(v):
        return (v.get("content") if isinstance(v, dict) else v) or ""
    tpl = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", tpl)
    return tpl, tok(cfg.get("bos_token")), tok(cfg.get("eos_token"))


def main():
    env = make_env()
    os.makedirs(OUT, exist_ok=True)
    for slug, repo in TEMPLATES:
        tpl, bos, eos = load(repo)
        cases = []
        for name, c in CONTEXTS.items():
            ctx = dict(c, add_generation_prompt=True, bos_token=bos, eos_token=eos)
            cases.append({"name": name, "context": ctx, "expected": env.from_string(tpl).render(**ctx)})
        with open(os.path.join(OUT, slug + ".json"), "w") as fh:
            json.dump({"source": repo, "template": tpl, "cases": cases}, fh, ensure_ascii=False, indent=1)
            fh.write("\n")
        sys.stderr.write("%-16s %-48s tpl=%d\n" % (slug, repo, len(tpl)))
    probes = [{"name": n, "template": t, "expected": env.from_string(t).render(**PROBE_CTX)} for n, t in PROBES]
    with open(os.path.join(OUT, "probes.json"), "w") as fh:
        json.dump({"context": PROBE_CTX, "probes": probes}, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    sys.stderr.write("probes: %d\n" % len(probes))


if __name__ == "__main__":
    main()
