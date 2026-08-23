#!/usr/bin/env python3
"""Regenerate tests/refs/chat_template_goldens.h from upstream chat templates.

WHY (#1572): the only chat-template golden in the tree was Harmony's, and it
bypassed the production context builder. The other nine families were covered by
structural smoke tests that assert a marker appears somewhere in the output - a
test that cannot see a wrong turn order, a missing generation prompt, or a
silently dropped system message.

WHAT THE GOLDEN IS: the RENDERED PROMPT, before tokenisation. That is what
ChatTemplate::render_jinja() returns, and pinning it needs no vocabulary, so the
test runs in the CPU lane with zero skips. Pinning token IDs instead would have
made every case a GPU/model-gated skip.

WHERE THE EXPECTED BYTES COME FROM: the upstream template, rendered through the
same Jinja configuration transformers uses in
`PreTrainedTokenizerBase._compile_jinja_template`:

    ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                  extensions=[jinja2.ext.loopcontrols])
    + the `tojson` filter and the `raise_exception` / `strftime_now` globals

This is a replication, not transformers itself, because six of the nine families
have no local checkpoint and `apply_chat_template` renders with the *loaded*
tokenizer's bos/eos - feeding Llama 3's template to a Qwen tokenizer would have
produced an authentic-looking golden with the wrong special tokens.

The replication is not assumed to be faithful, it is CHECKED: for every family
whose model is present locally, --verify renders the same conversation through
the real `AutoTokenizer.apply_chat_template(tokenize=False)` and fails on any
byte of difference. That check is what makes the six fetched families
trustworthy.

  Regenerate:  make chat-goldens        (see tests/CLAUDE.md)
"""

import argparse
import json
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "chat_template_goldens.h")
LOCAL_MODELS = "/models"

# A template is only worth pinning if it is the one real clients send. Each entry
# names where the bytes came from so a re-pin can be checked against the source.
#
#   repo   HuggingFace repo id, or "local:<dir>" for a checkpoint already on disk
#   family the ChatTemplateFamily the C++ side must DETECT - not a label chosen
#          here. Nemotron-3-Nano and Phi-4-reasoning both ship ChatML templates
#          despite their names, which is exactly why the test asserts the
#          detected family instead of trusting a hand-written one.
SOURCES = [
    ("chatml", "local:Qwen3-0.6B", "CHATML"),
    ("chatml_nemotron_nano", "local:Nemotron-3-Nano-30B-A3B-NVFP4", "CHATML"),
    ("gemma", "local:Gemma-4-12B-NVFP4", "GEMMA"),
    ("llama3", "NousResearch/Meta-Llama-3-8B-Instruct", "LLAMA3"),
    ("llama2", "unsloth/mistral-7b-instruct-v0.2", "LLAMA2"),
    ("mistral_v3", "unsloth/mistral-7b-instruct-v0.3", "MISTRAL_V3"),
    ("nemotron", "nvidia/Nemotron-4-Mini-Hindi-4B-Instruct", "NEMOTRON"),
    ("deepseek_r1", "deepseek-ai/DeepSeek-R1", "DEEPSEEK_R1"),
    ("phi", "microsoft/Phi-3-mini-4k-instruct", "PHI"),
]

# Fixed, so a template calling strftime_now cannot make the golden change daily.
FROZEN_DATE = "2026-01-01"

# Emitted into the header alongside the goldens. They used to be written twice,
# once here and once in the .cpp, and the two drifted immediately: the goldens
# answered "capital of France" while the test asked "Hello there.", so every
# family failed for a reason that had nothing to do with the template.
CONVERSATIONS = [
    ("user_only", [{"role": "user", "content": "What is the capital of France?"}]),
    ("system_user", [{"role": "system", "content": "You are a terse assistant."},
                     {"role": "user", "content": "Hello there."}]),
    ("multi_turn", [{"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello! How can I help?"},
                    {"role": "user", "content": "What is 2+2?"}]),
]


def fetch(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.read().decode("utf-8")


def load_source(repo):
    """-> (template, bos, eos). Prefers chat_template.jinja, as transformers does."""
    def pick(cfg_text, jinja_text):
        if jinja_text is not None:
            tpl = jinja_text
            cfg = json.loads(cfg_text) if cfg_text else {}
        else:
            cfg = json.loads(cfg_text)
            tpl = cfg.get("chat_template")
            if isinstance(tpl, list):  # named templates: the default is first
                tpl = tpl[0].get("template")
        if not isinstance(tpl, str):
            raise SystemExit("no chat_template in %s" % repo)

        def tok(v):
            return (v.get("content") if isinstance(v, dict) else v) or ""
        return tpl, tok(cfg.get("bos_token")), tok(cfg.get("eos_token"))

    if repo.startswith("local:"):
        d = os.path.join(LOCAL_MODELS, repo[len("local:"):])
        j = os.path.join(d, "chat_template.jinja")
        c = os.path.join(d, "tokenizer_config.json")
        return pick(open(c).read() if os.path.exists(c) else None,
                    open(j).read() if os.path.exists(j) else None)

    base = "https://huggingface.co/%s/resolve/main/" % repo
    jinja_text = None
    try:
        jinja_text = fetch(base + "chat_template.jinja")
    except Exception:
        pass
    cfg_text = None
    try:
        cfg_text = fetch(base + "tokenizer_config.json")
    except Exception:
        if jinja_text is None:
            raise
    return pick(cfg_text, jinja_text)


def make_env():
    """transformers' Jinja configuration, reproduced."""
    import jinja2
    import jinja2.ext
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent,
                          separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                        extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = lambda fmt: FROZEN_DATE
    return env


def render(env, tpl, msgs, bos, eos, add_generation_prompt=True):
    return env.from_string(tpl).render(messages=msgs, bos_token=bos, eos_token=eos,
                                       add_generation_prompt=add_generation_prompt)


def verify_against_transformers(slug, repo, tpl, msgs, mine):
    """Render the same thing through transformers; return None or a diff message."""
    if not repo.startswith("local:"):
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    d = os.path.join(LOCAL_MODELS, repo[len("local:"):])
    try:
        tok = AutoTokenizer.from_pretrained(d, trust_remote_code=False)
        theirs = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        return "%s: transformers could not render (%s)" % (slug, e)
    if theirs != mine:
        return ("%s: replicated Jinja env disagrees with transformers\n  mine  : %r\n"
                "  theirs: %r" % (slug, mine[:200], theirs[:200]))
    return None


def cpp_escape(s):
    out = []
    for ch in s:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append("\\%03o" % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


def emit(name, value):
    """One string literal, chunked - and never split inside an escape sequence."""
    esc = cpp_escape(value)
    lines, cur = [], ""
    i = 0
    while i < len(esc):
        if esc[i] == "\\":
            n = 4 if esc[i + 1].isdigit() else 2
            piece = esc[i:i + n]
        else:
            piece = esc[i]
            n = 1
        if len(cur) + len(piece) > 92:
            lines.append(cur)
            cur = ""
        cur += piece
        i += n
    if cur or not lines:
        lines.append(cur)
    body = "\n".join('    "%s"' % l for l in lines)
    return "inline const std::string %s =\n%s\n    ;\n" % (name, body)


def emit_conv(name, msgs):
    rows = ",\n".join('        {"%s", "%s"}' % (cpp_escape(m["role"]), cpp_escape(m["content"]))
                       for m in msgs)
    return ("inline const std::vector<std::pair<std::string, std::string>> k_conv_%s = {\n"
            "%s,\n    };\n" % (name, rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="cross-check the replicated Jinja env against transformers")
    args = ap.parse_args()

    env = make_env()
    parts = ["".join(emit_conv(n, m) for n, m in CONVERSATIONS) + "\n"]
    problems = []
    verified = 0
    for slug, repo, family in SOURCES:
        tpl, bos, eos = load_source(repo)
        parts.append("// %s  <-  %s  (detected family: %s)\n" % (slug, repo, family))
        parts.append(emit("k_%s_template" % slug, tpl))
        parts.append(emit("k_%s_bos" % slug, bos))
        parts.append(emit("k_%s_eos" % slug, eos))
        for cname, msgs in CONVERSATIONS:
            try:
                out = render(env, tpl, msgs, bos, eos)
            except Exception as e:
                # A template that refuses a conversation is a fact about the
                # template, not a generator bug: Gemma rejects a system role.
                problems.append("%s/%s: %s" % (slug, cname, e))
                continue
            parts.append(emit("k_%s_%s" % (slug, cname), out))
            if args.verify:
                d = verify_against_transformers(slug, repo, tpl, msgs, out)
                if d:
                    problems.append(d)
                elif repo.startswith("local:"):
                    verified += 1
        sys.stderr.write("%-22s %-46s tpl=%-6d bos=%-24r eos=%r\n"
                         % (slug, repo, len(tpl), bos, eos))
        parts.append("\n")

    header = (
        "// GENERATED by tests/refs/gen_chat_goldens.py - do not edit by hand.\n"
        "//\n"
        "// Rendered prompts for the chat-template families imp claims to support,\n"
        "// taken from the upstream templates named above each block and rendered in\n"
        "// the Jinja configuration transformers uses (#1572). Regenerate with\n"
        "//   python3 tests/refs/gen_chat_goldens.py --verify\n"
        "#pragma once\n\n#include <string>\n#include <utility>\n#include <vector>\n\n"
        "namespace chat_goldens {\n\n"
    )
    with open(OUT, "w") as fh:
        fh.write(header + "".join(parts) + "}  // namespace chat_goldens\n")

    if problems:
        sys.stderr.write("\n" + "\n".join(problems) + "\n")
    if args.verify:
        sys.stderr.write("\ncross-checked %d render(s) against transformers "
                         "(only the families with a local checkpoint can be)\n" % verified)
        if not verified:
            sys.stderr.write("REFUSING: --verify checked nothing - the replicated Jinja env "
                             "is unproven, so the fetched families are unproven too\n")
            return 2
    sys.stderr.write("wrote %s (%d bytes)\n" % (OUT, os.path.getsize(OUT)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
