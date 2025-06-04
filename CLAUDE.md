# GNTS - Single File Agents
A collection of powerful single-file agents built the Astral ecosystem (uv + ruff).

## Project Overview
This project leverages uv's ability to run Python scripts with inline dependencies, allowing each agent to be self-contained in a single file.

## Development Guidelines

### Core Principles
- Self-contained: Each script includes all dependencies via uv's inline syntax
- Well-documented: Clear docstrings, comments, and type hints throughout
- Astral ecosystem: Built on uv for dependency management and ruff for code quality
- Functional paradigm: Focus on "what", not "how" - prefer abstractions and leverage functools & itertools
- Minimal code philosophy: Less lines = less maintenance = fewer tokens = better
- NEVER use python command directly, uv is the python interface to use

### Creating New Agents
- Each agent must be a single Python file
- Include comprehensive docstrings for all functions and classes
- Apply functional programming patterns (as much as it makes sense):
  - Use `functools` & `itertools` 
  - Prefer immutable data structures and pure functions

### Code Quality
- Always use `uvx` interface for external tools
- Linter: `uvx ruff check <file.py>`
- Formatter: `uvx ruff format <file.py>`

### Code Clarity
- Avoid redundant or obvious comments; remove debug comments before committing
  - Comment complex algorithms or non-obvious design decisions
