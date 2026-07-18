"""Static analysis of a Python codebase into an MCP resource/tool schema."""

import ast
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

SKIP_DIRS = {
    ".venv",
    "venv",
    "env",
    "site-packages",
    ".synapse-lite",
    "__pycache__",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".tox",
    "node_modules",
    "build",
    "dist",
}

# Classification checks the leading verb first; see classify_function for the fallback.
RESOURCE_VERBS = {"get", "list", "fetch", "find", "query", "search", "read", "check"}
TOOL_VERBS = {
    "create",
    "update",
    "delete",
    "trigger",
    "deploy",
    "write",
    "save",
    "insert",
    "set",
    "add",
    "remove",
    "rollback",
    "destroy",
    "revoke",
    "cancel",
}
DESTRUCTIVE_VERBS = {"delete", "remove", "destroy", "rollback", "revoke", "cancel"}

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokenize(name: str) -> list[str]:
    """Split a snake_case or camelCase identifier into lowercase tokens."""
    with_underscores = _CAMEL_BOUNDARY.sub("_", name)
    return [t.lower() for t in with_underscores.split("_") if t]


def classify_function(name: str) -> tuple[str, bool, str]:
    """Classify a function name as "resource" or "tool".

    Returns (kind, destructive, reason). The leading verb (first token in
    the name) is checked first; whole-name keyword matching is only used
    as a fallback when the leading token isn't a recognized verb, so e.g.
    `get_deploy_status` classifies as a resource on "get", not a tool on
    "deploy".
    """
    tokens = _tokenize(name)
    if not tokens:
        return "tool", False, "empty/unrecognized name; defaulting to tool"

    leading = tokens[0]

    if leading in RESOURCE_VERBS:
        return "resource", False, f"leading verb '{leading}_' is a known read-only verb"

    if leading in TOOL_VERBS:
        destructive = leading in DESTRUCTIVE_VERBS
        reason = f"leading verb '{leading}_' is a known mutating verb"
        if destructive:
            reason += " (destructive)"
        return "tool", destructive, reason

    # No leading verb match - fall back to whole-name keywords, tools first
    # (a stray read verb somewhere in the name doesn't prove it's safe).
    matched_tool = next((t for t in tokens if t in TOOL_VERBS), None)
    if matched_tool:
        destructive = matched_tool in DESTRUCTIVE_VERBS
        reason = f"no leading verb match; found mutating keyword '{matched_tool}' in name"
        if destructive:
            reason += " (destructive)"
        return "tool", destructive, reason

    matched_resource = next((t for t in tokens if t in RESOURCE_VERBS), None)
    if matched_resource:
        return (
            "resource",
            False,
            f"no leading verb match; found read-only keyword '{matched_resource}' in name",
        )

    return "tool", False, "no known verb found in name; defaulting to tool as a safe default"


@dataclass
class FunctionInfo:
    name: str
    file: str
    class_name: str | None
    params: list[dict] = field(default_factory=list)
    return_type: str | None = None
    docstring: str | None = None
    kind: str = "tool"
    destructive: bool = False
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "file": self.file,
            "class_name": self.class_name,
            "params": self.params,
            "return_type": self.return_type,
            "docstring": self.docstring,
            "kind": self.kind,
            "destructive": self.destructive,
            "reason": self.reason,
        }


def _unparse(node: ast.AST | None) -> str | None:
    return ast.unparse(node) if node is not None else None


def _extract_params(node: ast.FunctionDef | ast.AsyncFunctionDef, is_method: bool) -> list[dict]:
    args = node.args
    positional = list(args.posonlyargs) + list(args.args)
    defaults = list(args.defaults)
    num_no_default = len(positional) - len(defaults)

    params = []
    for i, arg in enumerate(positional):
        if is_method and i == 0 and arg.arg in ("self", "cls"):
            continue
        default_node = defaults[i - num_no_default] if i >= num_no_default else None
        params.append(
            {
                "name": arg.arg,
                "type": _unparse(arg.annotation),
                "required": i < num_no_default,
                "default": _unparse(default_node),
            }
        )

    if args.vararg:
        params.append(
            {
                "name": f"*{args.vararg.arg}",
                "type": _unparse(args.vararg.annotation),
                "required": False,
                "default": None,
            }
        )

    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            {
                "name": arg.arg,
                "type": _unparse(arg.annotation),
                "required": default is None,
                "default": _unparse(default),
            }
        )

    if args.kwarg:
        params.append(
            {
                "name": f"**{args.kwarg.arg}",
                "type": _unparse(args.kwarg.annotation),
                "required": False,
                "default": None,
            }
        )

    return params


def _build_function_info(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    relative_path: str,
    class_name: str | None,
) -> FunctionInfo:
    kind, destructive, reason = classify_function(node.name)
    return FunctionInfo(
        name=node.name,
        file=relative_path,
        class_name=class_name,
        params=_extract_params(node, is_method=class_name is not None),
        return_type=_unparse(node.returns),
        docstring=ast.get_docstring(node),
        kind=kind,
        destructive=destructive,
        reason=reason,
    )


def iter_python_files(project_root: Path) -> list[Path]:
    """Return all .py files under project_root, skipping SKIP_DIRS."""
    files = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")]
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(Path(dirpath) / filename)
    files.sort()
    return files


def parse_file(path: Path, project_root: Path) -> tuple[list[FunctionInfo], str | None]:
    """Parse a single Python file. Returns (functions, error_message)."""
    relative_path = str(path.relative_to(project_root))

    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return [], f"could not read file: {exc}"

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError as exc:
        return [], f"syntax error: {exc}"

    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_"):
                continue
            functions.append(_build_function_info(node, relative_path, class_name=None))
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if member.name.startswith("_"):
                        continue
                    functions.append(
                        _build_function_info(member, relative_path, class_name=node.name)
                    )

    return functions, None


def iter_analyzed_files(
    project_root: Path,
) -> Iterator[tuple[Path, list[FunctionInfo], str | None]]:
    """Yield (file_path, functions, error) for every Python file under project_root.

    Single place that walks + parses the project, so `analyze_project` and
    the CLI's `analyze` command don't each re-walk the tree.
    """
    for path in iter_python_files(project_root):
        functions, error = parse_file(path, project_root)
        yield path, functions, error


def summarize(functions: list[FunctionInfo]) -> dict:
    """Compute {total, resources, tools, destructive} counts for a list of
    functions - shared by per-file and whole-project summaries.
    """
    resources = sum(1 for fn in functions if fn.kind == "resource")
    destructive = sum(1 for fn in functions if fn.destructive)
    return {
        "total": len(functions),
        "resources": resources,
        "tools": len(functions) - resources,
        "destructive": destructive,
    }


def analyze_project(project_root: Path) -> dict:
    """Analyze every Python file under project_root and return a schema dict."""
    all_functions: list[FunctionInfo] = []
    errors: list[dict] = []
    files_scanned = 0

    for path, functions, error in iter_analyzed_files(project_root):
        files_scanned += 1
        if error:
            errors.append({"file": str(path.relative_to(project_root)), "error": error})
            continue
        all_functions.extend(functions)

    return {
        "files_scanned": files_scanned,
        "functions": [fn.to_dict() for fn in all_functions],
        "errors": errors,
        "summary": summarize(all_functions),
    }