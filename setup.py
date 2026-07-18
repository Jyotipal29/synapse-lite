from setuptools import find_packages, setup

setup(
    name="synapse-lite",
    version="0.1.0",
    description="Turn a Python codebase into a working MCP server.",
    packages=find_packages(include=["synapse_lite", "synapse_lite.*"]),
    python_requires=">=3.11",
    install_requires=[
        "click>=8.1",
        "openai>=1.0",
        "mcp>=1.0",
    ],
    entry_points={
        "console_scripts": [
            "synapse-lite=synapse_lite.cli:cli",
        ],
    },
)