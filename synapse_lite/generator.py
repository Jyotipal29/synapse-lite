"""Turns an analyzed project schema + natural-language query into a runnable MCP server."""

import json
import os
import re
from pathlib import Path

from openai import OpenAI

DEFAULT_MODEL = "gpt-4.1"

SYSTEM_PROMPT = """You are an expert at designing Model Context Protocol (MCP) servers from existing Python codebases.

You will be given:
1. A natural-language query describing what the user wants exposed via MCP.
2. A JSON array of candidate Python functions, each with: id, name, class_name, \
params (list of {name, type, required}), return_type, docstring, and a heuristic \
"kind" (resource/tool) and "destructive" flag already computed from the function name.

Select ONLY the functions relevant to the query. Never invent a function or an id \
that is not in the candidate list, and never change the "id" field of a candidate \
you select - it is used to match your selection back to the source code.

For each selected function, return an object with:
- "id": the exact id string copied from the candidate list
- "kind": "resource" (read-only, no side effects) or "tool" (mutates state or has \
side effects). You may override the heuristic kind if the docstring or params make \
the correct classification clear - trust the docstring over the name when they conflict.
- "destructive": boolean, true only if calling this function can irreversibly delete \
or destroy data
- "read_only": boolean, true only for "kind": "resource" functions with no side effects
- "description": a clean, single-sentence, user-facing description derived from the \
docstring (rewrite for clarity if the docstring is missing, vague, or unclear)
- "input_schema": a valid JSON Schema object ({"type": "object", "properties": {...}, \
"required": [...]}) built from the function's params list. Map Python types to JSON \
Schema types (int/float -> integer/number, str -> string, bool -> boolean, list -> \
array, dict -> object). Only include params that aren't *args/**kwargs.

Respond with a JSON object of the exact shape {"selected": [...]}. If no candidate \
functions are relevant to the query, respond with {"selected": []}."""


def _function_id(fn: dict) -> str:
    qualified = f"{fn['class_name']}.{fn['name']}" if fn.get("class_name") else fn["name"]
    return f"{fn['file']}::{qualified}"


def _candidate_payload(fn: dict) -> dict:
    return {
        "id": _function_id(fn),
        "name": fn["name"],
        "class_name": fn.get("class_name"),
        "params": fn.get("params", []),
        "return_type": fn.get("return_type"),
        "docstring": fn.get("docstring"),
        "kind": fn.get("kind"),
        "destructive": fn.get("destructive", False),
    }


_JSON_SCHEMA_TYPES = {"int": "integer", "float": "number", "str": "string", "bool": "boolean"}


def _param_json_schema_type(type_str: str | None) -> dict:
    if not type_str:
        return {}
    t = re.sub(r"\s*\|\s*None\s*$", "", type_str.strip())
    t = re.sub(r"^Optional\[(.*)\]$", r"\1", t)
    if t in _JSON_SCHEMA_TYPES:
        return {"type": _JSON_SCHEMA_TYPES[t]}
    if t.startswith("list[") or t.startswith("List["):
        return {"type": "array"}
    if t.startswith("dict[") or t.startswith("Dict["):
        return {"type": "object"}
    return {"type": "string"}


def build_input_schema(params: list[dict]) -> dict:
    """Deterministically build a JSON Schema from params - used as a fallback
    when the model's own input_schema is missing or malformed."""
    properties = {}
    required = []
    for p in params:
        if p["name"].startswith("*"):
            continue
        properties[p["name"]] = _param_json_schema_type(p.get("type"))
        if p.get("required"):
            required.append(p["name"])
    schema = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def select_functions(schema: dict, query: str, *, model: str = DEFAULT_MODEL, client: OpenAI | None = None) -> list[dict]:
    """Ask the OpenAI API to select and classify functions relevant to query.

    Returns a list of resolved selection dicts, each carrying enough source
    info (file, class_name, params) to render the MCP server.
    """
    if client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in the environment.")
        client = OpenAI(api_key=api_key)

    functions = schema.get("functions", [])
    by_id = {_function_id(fn): fn for fn in functions}
    user_payload = {"query": query, "functions": [_candidate_payload(fn) for fn in functions]}

    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ],
    )

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned invalid JSON: {exc}") from exc

    raw_selected = data.get("selected", [])
    if not isinstance(raw_selected, list):
        raise RuntimeError("OpenAI response 'selected' field is not a list.")

    resolved = []
    for item in raw_selected:
        if not isinstance(item, dict):
            continue
        source = by_id.get(item.get("id"))
        if source is None:
            continue  # model referenced an id we don't recognize; skip rather than guess

        kind = item.get("kind") if item.get("kind") in ("resource", "tool") else source.get("kind", "tool")
        destructive = bool(item.get("destructive", source.get("destructive", False)))
        read_only = bool(item.get("read_only", kind == "resource" and not destructive))
        description = item.get("description") or source.get("docstring") or source["name"]
        description = " ".join(str(description).split())

        input_schema = item.get("input_schema")
        if not isinstance(input_schema, dict) or "properties" not in input_schema:
            input_schema = build_input_schema(source.get("params", []))

        resolved.append(
            {
                "id": item["id"],
                "name": source["name"],
                "file": source["file"],
                "class_name": source.get("class_name"),
                "params": source.get("params", []),
                "kind": kind,
                "destructive": destructive,
                "read_only": read_only,
                "description": description,
                "input_schema": input_schema,
            }
        )

    return resolved


def _module_path(relative_file: str) -> str:
    path = Path(relative_file)
    parts = list(path.parts[:-1]) + [path.stem]
    return ".".join(parts)


# Only annotate params with types that need no import, so the generated file
# stays self-contained - anything else is left unannotated to avoid a NameError.
_SAFE_ANNOTATION_TYPES = {"int", "str", "bool", "float", "bytes"}


def _safe_annotation(type_str: str | None) -> str | None:
    if not type_str:
        return None
    t = type_str.strip()
    optional = False
    m = re.match(r"^(.*?)\s*\|\s*None$", t)
    if m:
        t, optional = m.group(1).strip(), True
    else:
        m2 = re.match(r"^Optional\[(.*)\]$", t)
        if m2:
            t, optional = m2.group(1).strip(), True
    if t in _SAFE_ANNOTATION_TYPES:
        return f"{t} | None" if optional else t
    return None


_SAFE_DEFAULT_LITERAL = re.compile(r"^(-?\d+(\.\d+)?|True|False|None|'[^'\\]*'|\"[^\"\\]*\")$")


def _safe_default(default_str: str | None) -> str | None:
    """Return the default's source text if it's a plain literal (e.g. `10`,
    `'foo'`, `None`) safe to inline with no import, else None."""
    if not default_str:
        return None
    default_str = default_str.strip()
    return default_str if _SAFE_DEFAULT_LITERAL.match(default_str) else None


def _filter_supported(selected: list[dict]) -> tuple[list[dict], list[str]]:
    """Split selected functions into (module-level functions we can render,
    warnings for anything we have to skip). Class methods aren't supported
    yet - they'd need an instance to call, which we have no way to construct."""
    warnings = []
    supported = []
    for fn in selected:
        if fn.get("class_name"):
            warnings.append(
                f"skipping {fn['class_name']}.{fn['name']} - method-based tools/resources aren't supported yet"
            )
            continue
        supported.append(fn)
    return supported, warnings


def _assign_aliases(supported: list[dict]) -> tuple[dict[str, str], dict[str, list[tuple[str, str]]]]:
    """Give each selected function a unique import alias, grouped by source module.

    Without an alias the wrapper (which reuses the original function's name)
    would shadow its own import and recurse into itself forever.
    """
    alias_by_id: dict[str, str] = {}
    imports_by_module: dict[str, list[tuple[str, str]]] = {}
    used_aliases: set[str] = set()

    for fn in supported:
        module_path = _module_path(fn["file"])
        alias = f"_impl_{fn['name']}"
        if alias in used_aliases:
            alias = f"_impl_{module_path.replace('.', '_')}_{fn['name']}"
        used_aliases.add(alias)
        alias_by_id[fn["id"]] = alias
        imports_by_module.setdefault(module_path, []).append((fn["name"], alias))

    return alias_by_id, imports_by_module


def _render_header(
    query: str, server_name: str, imports_by_module: dict[str, list[tuple[str, str]]]
) -> list[str]:
    lines = [
        f'"""Generated by synapse-lite for query: {query!r}. Do not edit by hand - run `synapse-lite build` again."""',
        "",
        "from mcp.server.fastmcp import FastMCP",
        "from mcp.types import ToolAnnotations",
        "",
    ]
    for module_path, entries in sorted(imports_by_module.items()):
        import_list = ", ".join(f"{name} as {alias}" for name, alias in entries)
        lines.append(f"from {module_path} import {import_list}")
    lines.append("")
    lines.append(f"mcp = FastMCP({server_name!r})")
    return lines


def _render_function_block(fn: dict, alias: str) -> list[str]:
    """Render the decorator + wrapper def for one selected function."""
    params = [p for p in fn["params"] if not p["name"].startswith("*")]

    # A default can only be kept if every param after it also has one -
    # required-after-optional isn't valid Python - so once that breaks,
    # drop defaults from that point backward instead of emitting bad syntax.
    defaults = [_safe_default(p.get("default")) for p in params]
    keep_default = True
    for i in range(len(defaults) - 1, -1, -1):
        if defaults[i] is None:
            keep_default = False
        elif not keep_default:
            defaults[i] = None

    sig_parts = []
    for p, default in zip(params, defaults):
        part = p["name"]
        if annotation := _safe_annotation(p.get("type")):
            part += f": {annotation}"
        if default is not None:
            part += f" = {default}"
        sig_parts.append(part)
    signature = ", ".join(sig_parts)
    call_args = ", ".join(f"{p['name']}={p['name']}" for p in params)
    description = fn["description"].replace('"""', "'''")

    lines = [""]
    if fn["kind"] == "resource":
        # URI scheme names can't contain underscores (RFC 3986), so hyphenate
        # for the scheme only - the function name itself stays snake_case.
        scheme = fn["name"].replace("_", "-")
        uri = (
            f"{scheme}://" + "/".join(f"{{{p['name']}}}" for p in params)
            if params
            else f"{scheme}://data"
        )
        lines.append(f"@mcp.resource({uri!r}, description={description!r})")
    else:
        annotations = f"ToolAnnotations(readOnlyHint={fn['read_only']}, destructiveHint={fn['destructive']})"
        lines.append(f"@mcp.tool(description={description!r}, annotations={annotations})")

    lines.append(f"def {fn['name']}({signature}):")
    lines.append(f'    """{description}"""')
    lines.append(f"    return {alias}({call_args})")
    return lines


def render_server(selected: list[dict], query: str, server_name: str = "synapse-lite") -> tuple[str, list[str]]:
    """Render mcp_server.py source text. Returns (source, warnings)."""
    supported, warnings = _filter_supported(selected)
    if not supported:
        raise RuntimeError("No supported (module-level) functions were selected - nothing to generate.")

    alias_by_id, imports_by_module = _assign_aliases(supported)

    lines = _render_header(query, server_name, imports_by_module)
    for fn in supported:
        lines += _render_function_block(fn, alias_by_id[fn["id"]])
    lines += ["", "", 'if __name__ == "__main__":', "    mcp.run()", ""]

    return "\n".join(lines), warnings