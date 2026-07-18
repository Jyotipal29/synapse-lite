import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from synapse_lite import __version__, analyzer, generator

CONFIG_DIR_NAME = ".synapse-lite"
CONFIG_FILE_NAME = "config.json"
SCHEMA_FILE_NAME = "project_schema.json"
SERVER_FILE_NAME = "mcp_server.py"


def _config_path(project_root: Path) -> Path:
    return project_root / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _schema_path(project_root: Path) -> Path:
    return project_root / CONFIG_DIR_NAME / SCHEMA_FILE_NAME


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


@click.group()
@click.version_option(__version__, prog_name="synapse-lite")
def cli():
    """synapse-lite: turn a Python codebase into an MCP server."""


@cli.command()
@click.argument(
    "path",
    type=click.Path(file_okay=False, path_type=Path),
    default=".",
)
@click.option("--force", is_flag=True, help="Overwrite an existing config file.")
def init(path: Path, force: bool):
    """Initialize synapse-lite in PATH (defaults to the current directory)."""
    project_root = path.resolve()
    if not project_root.exists():
        raise click.ClickException(f"Path does not exist: {project_root}")

    config_path = _config_path(project_root)

    if config_path.exists() and not force:
        raise click.ClickException(
            f"{config_path} already exists. Use --force to overwrite."
        )

    config_path.parent.mkdir(exist_ok=True)
    config = {
        "version": __version__,
        "project_root": str(project_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "openai_api_key_set": bool(os.environ.get("OPENAI_API_KEY")),
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n")

    click.echo(f"Initialized synapse-lite in {config_path}")


def _describe_functions(functions: list) -> str:
    """One-line summary like '3 functions found (2 resources, 1 tool)' for a
    single file's functions - built on analyzer.summarize() so the counting
    logic isn't duplicated here."""
    s = analyzer.summarize(functions)
    text = (
        f"{_plural(s['total'], 'function')} found "
        f"({_plural(s['resources'], 'resource')}, {_plural(s['tools'], 'tool')}"
    )
    if s["destructive"]:
        text += f", {_plural(s['destructive'], 'destructive action')}"
    return text + ")"


def _print_verbose_functions(functions: list) -> None:
    for fn in functions:
        label = fn.kind + (" [destructive]" if fn.destructive else "")
        qualified_name = f"{fn.class_name}.{fn.name}" if fn.class_name else fn.name
        click.echo(f"  {qualified_name:<28} {label:<20} {fn.reason}")


@cli.command()
@click.argument(
    "path",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=".",
)
@click.option("--verbose", is_flag=True, help="Print each function's classification and reason.")
def analyze(path: Path, verbose: bool):
    """Analyze the Python codebase at PATH and classify its functions."""
    project_root = path.resolve()
    skip_list = ", ".join(sorted(analyzer.SKIP_DIRS))

    click.echo(f"Scanning {project_root} for Python files...")
    click.echo(f"  (skipping {skip_list})")

    all_functions = []
    errors = []
    files_scanned = 0

    for file_path, functions, error in analyzer.iter_analyzed_files(project_root):
        files_scanned += 1
        relative = file_path.relative_to(project_root)

        if error:
            errors.append({"file": str(relative), "error": error})
            click.echo(f"Parsing {relative}... skipped ({error})")
            continue

        all_functions.extend(functions)
        click.echo(f"Parsing {relative}... {_describe_functions(functions)}")
        if verbose:
            _print_verbose_functions(functions)

    if files_scanned == 0:
        click.echo("No Python files found.")
        return
    click.echo()

    summary = analyzer.summarize(all_functions)
    click.echo("Analysis complete.")
    click.echo(f"  Files scanned:   {files_scanned}")
    click.echo(f"  Functions found: {summary['total']}")
    click.echo(f"  Resources:       {summary['resources']}")
    click.echo(f"  Tools:           {summary['tools']}")
    click.echo(f"  Destructive:     {summary['destructive']}")
    if errors:
        click.echo(f"  Errors:          {len(errors)}")

    schema_path = _schema_path(project_root)
    schema_path.parent.mkdir(exist_ok=True)
    schema = {
        "version": __version__,
        "project_root": str(project_root),
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "files_scanned": files_scanned,
        "functions": [fn.to_dict() for fn in all_functions],
        "errors": errors,
        "summary": summary,
    }
    schema_path.write_text(json.dumps(schema, indent=2) + "\n")
    click.echo(f"\nWrote schema to {schema_path}")


@cli.command()
@click.argument(
    "path",
    type=click.Path(file_okay=False, exists=True, path_type=Path),
    default=".",
)
@click.option("--query", required=True, help="Natural-language description of what to expose via MCP.")
@click.option("--model", default=generator.DEFAULT_MODEL, show_default=True, help="OpenAI model to use for selection.")
@click.option(
    "--python",
    "python_executable",
    default=sys.executable,
    show_default=False,
    help="Python interpreter to validate/run the generated server with (defaults to the "
    "interpreter running synapse-lite itself). Point this at the target project's own "
    "venv if it has runtime dependencies synapse-lite's environment doesn't.",
)
def build(path: Path, query: str, model: str, python_executable: str):
    """Build mcp_server.py from the analyzed schema and a natural-language query."""
    project_root = path.resolve()

    if not os.environ.get("OPENAI_API_KEY"):
        raise click.ClickException("OPENAI_API_KEY is not set in the environment.")

    schema_path = _schema_path(project_root)
    if not schema_path.exists():
        raise click.ClickException(
            f"No schema found at {schema_path}. Run `synapse-lite analyze` first."
        )
    schema = json.loads(schema_path.read_text())

    total_candidates = schema.get("summary", {}).get("total", len(schema.get("functions", [])))
    click.echo(f"Discovering candidates... {_plural(total_candidates, 'function')} available in the schema.")
    click.echo(f"Calling OpenAI (model={model}) to select functions for query: {query!r}")

    try:
        selected = generator.select_functions(schema, query, model=model)
    except Exception as exc:
        raise click.ClickException(f"OpenAI selection failed: {exc}") from exc

    if not selected:
        click.echo("No relevant functions were selected for this query. Nothing to build.")
        return

    click.echo(f"\nSelected {_plural(len(selected), 'function')}:")
    for fn in selected:
        label = fn["kind"] + (" [destructive]" if fn["destructive"] else "")
        if fn["read_only"]:
            label += " [read-only]"
        qualified_name = f"{fn['class_name']}.{fn['name']}" if fn["class_name"] else fn["name"]
        click.echo(f"  {qualified_name:<28} {label:<28} {fn['description']}")

    try:
        source, warnings = generator.render_server(selected, query)
    except Exception as exc:
        raise click.ClickException(f"Failed to render mcp_server.py: {exc}") from exc

    for warning in warnings:
        click.echo(f"  Warning: {warning}")

    server_path = project_root / SERVER_FILE_NAME
    verb = "Overwriting" if server_path.exists() else "Writing"
    click.echo(f"\n{verb} {server_path}")
    server_path.write_text(source)

    click.echo(f"Validating generated server by importing it in a subprocess ({python_executable})...")
    result = subprocess.run(
        [python_executable, "-c", f"import {server_path.stem}"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise click.ClickException(
            f"Generated server failed to import:\n{result.stderr}"
        )
    click.echo("Import validation passed.")

    config_key = "".join(c if c.isalnum() or c in "-_" else "-" for c in project_root.name).lower() or "synapse-lite"
    config_snippet = {
        "mcpServers": {
            config_key: {
                "command": python_executable,
                "args": [str(server_path)],
            }
        }
    }
    click.echo("\nAdd this to your Claude Code / Claude Desktop MCP config (.mcp.json):\n")
    click.echo(json.dumps(config_snippet, indent=2))


@cli.command()
def status():
    """Show synapse-lite's current state for this project."""
    project_root = Path(".").resolve()
    config_path = _config_path(project_root)

    click.echo(f"Project:  {project_root}")

    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            click.echo(f"Init:     error - {config_path} is not valid JSON")
            config = None
        else:
            click.echo(f"Init:     yes ({config_path})")
    else:
        click.echo("Init:     no (run `synapse-lite init`)")
        config = None

    schema_path = _schema_path(project_root)
    if schema_path.exists():
        try:
            schema = json.loads(schema_path.read_text())
        except json.JSONDecodeError:
            click.echo(f"Analyzed: error - {schema_path} is not valid JSON")
        else:
            summary = schema.get("summary", {})
            click.echo(
                f"Analyzed: yes ({summary.get('total', '?')} functions: "
                f"{summary.get('resources', '?')} resources, {summary.get('tools', '?')} tools)"
            )
    else:
        click.echo("Analyzed: no (run `synapse-lite analyze`)")

    api_key_set = bool(os.environ.get("OPENAI_API_KEY"))
    click.echo(f"API key:  {'yes' if api_key_set else 'no'} (OPENAI_API_KEY env var)")
    if config is not None and config.get("openai_api_key_set") != api_key_set:
        click.echo(
            "          (differs from state recorded at init - env may have changed)"
        )


if __name__ == "__main__":
    cli()