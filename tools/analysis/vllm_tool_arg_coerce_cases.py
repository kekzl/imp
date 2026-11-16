"""Regenerates tests/fixtures/vllm_tool_arg_coerce.json: vLLM 0.33 coerce_to_schema_type per (schema, text).

Run inside vllm/vllm-openai:v0.33.0: docker run --rm --entrypoint python3 -v $PWD/tools/analysis:/t:ro \
    vllm/vllm-openai:v0.33.0 /t/vllm_tool_arg_coerce_cases.py > tests/fixtures/vllm_tool_arg_coerce.json
"""
import json
import sys

from vllm.tool_parsers.utils import coerce_to_schema_type, extract_types_from_schema

schemas = {
    "string": {"type": "string"},
    "integer": {"type": "integer"},
    "number": {"type": "number"},
    "boolean": {"type": "boolean"},
    "array": {"type": "array"},
    "object": {"type": "object"},
    "str_null": {"type": ["string", "null"]},
    "int_str": {"type": ["integer", "string"]},
    "alias_int": {"type": "INT"},
    "alias_list": {"type": " list "},
    "enum_mixed": {"enum": [1, "a", None]},
    "anyof": {"anyOf": [{"type": "boolean"}, {"type": "array"}]},
    "untyped": {"description": "x"},
    "unknown": {"type": "widget"},
}
texts = ["42", " 42 ", "+7", "1_000", "1__0", "4.5", "2.0", "1e3", ".5", "5.", "inf", "nan", "1e999",
         "true", "TRUE", "1", "0", "false ", "null", "NULL", " null", "[1, 2]", "{\"a\": 1}", "[1e999]",
         "\"foo\"", "\"/home/x", "abc", "", "0x10", "-0", "99999999999999999999"]
rows = []
for sk, sch in schemas.items():
    types = extract_types_from_schema(sch)
    for t in texts:
        v = coerce_to_schema_type(t, types)
        rows.append([sk, t, v])
json.dump({"schemas": schemas, "rows": rows}, sys.stdout, ensure_ascii=False)
