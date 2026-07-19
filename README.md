# synapse-lite

Turn a Python codebase into a working [MCP](https://modelcontextprotocol.io) server — no manual tool wiring required.

`synapse-lite` scans your project's functions, classifies them as read-only **resources** or mutating **tools**, and then uses a natural-language query plus an LLM to pick the ones relevant to you and generate a ready-to-run `mcp_server.py`.

## How it works

1. **`init`** — sets up a `.synapse-lite/` config directory in your project.
2. **`analyze`** — walks every `.py` file in your project, parses top-level and class functions with `ast`, and classifies each one as a `resource` (read-only, e.g. `get_user`) or a `tool` (mutating, e.g. `delete_user`) based on its name. Destructive verbs (`delete`, `remove`, `destroy`, `rollback`, `revoke`, `cancel`) are flagged. Results are written to `.synapse-lite/project_schema.json`.
3. **`build --query "..."`** — sends the schema and your query to OpenAI, which selects the relevant functions, writes descriptions, and infers JSON Schema input shapes. `synapse-lite` then renders a self-contained `mcp_server.py` using [FastMCP](https://github.com/modelcontextprotocol/python-sdk) and validates it by importing it in a subprocess.
4. **`status`** — shows whether you're initialized, analyzed, and have an API key configured.

## Requirements

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) (only needed for `build`)

## Installation

```bash
git clone <this-repo>
cd synapse-lite
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This installs the `synapse-lite` command.

## Usage

Set your OpenAI API key (needed only for `build`):

```bash
export OPENAI_API_KEY=sk-...
```

Then, from the root of the Python project you want to expose via MCP:

```bash
# 1. Initialize
synapse-lite init /path/to/your/project

# 2. Analyze the codebase
synapse-lite analyze /path/to/your/project
# add --verbose to see how each function was classified and why

# 3. Build an MCP server for a specific use case
synapse-lite build /path/to/your/project --query "expose functions for managing users and reading their orders"

# Check current state at any time
synapse-lite status
```

`build` writes `mcp_server.py` into your project root and prints a config snippet like:

```json
{
  "mcpServers": {
    "your-project": {
      "command": "/path/to/python",
      "args": ["/path/to/your/project/mcp_server.py"]
    }
  }
}
```

Add that to your Claude Code / Claude Desktop MCP config (`.mcp.json`) to start using the generated server.

### Options

- `synapse-lite analyze PATH --verbose` — print each function's classification (`resource`/`tool`, `[destructive]`, and the reason).
- `synapse-lite build PATH --query "..." --model gpt-4.1` — choose a different OpenAI model.
- `synapse-lite build PATH --query "..." --python /path/to/venv/python` — validate/run the generated server with a different interpreter (e.g. the target project's own venv, if it has dependencies `synapse-lite` doesn't).

## How classification works

Function names are tokenized (snake_case or camelCase) and the **leading verb** is checked first:

- `get_`, `list_`, `fetch_`, `find_`, `query_`, `search_`, `read_`, `check_` → **resource**
- `create_`, `update_`, `delete_`, `trigger_`, `deploy_`, `write_`, `save_`, `insert_`, `set_`, `add_`, `remove_`, `rollback_`, `destroy_`, `revoke_`, `cancel_` → **tool**
- `delete_`, `remove_`, `destroy_`, `rollback_`, `revoke_`, `cancel_` → also flagged **destructive**

If the leading token isn't recognized, the whole name is checked for these keywords as a fallback (mutating keywords take priority over read-only ones). Functions with no recognizable verb default to `tool` as a safe default. Private functions (leading underscore) are skipped entirely.

This heuristic classification seeds the schema; `build` lets the LLM override it per-function when the docstring makes the real behavior clearer.

## Limitations

- Only **module-level** functions can currently be rendered into the generated server — class methods are detected and included in the schema, but `build` skips them with a warning (there's no way to construct an instance to call them on).
- `build` requires network access and an OpenAI API key.
- Classification is a heuristic based on function names; always review `--verbose` output or the generated server before trusting it in a sensitive environment (e.g. one with destructive tools).

## Project layout

```
synapse_lite/
  cli.py        # click CLI: init, analyze, build, status
  analyzer.py   # AST-based scanning + resource/tool classification
  generator.py  # OpenAI-based selection + mcp_server.py rendering
```

Generated/config files live in your target project, not this repo:

```
your-project/
  .synapse-lite/
    config.json          # created by `init`
    project_schema.json  # created by `analyze`
  mcp_server.py           # created by `build`
```
