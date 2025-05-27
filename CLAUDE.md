# GNTS - Single File Agents

A collection of powerful single-file agents built the Astral ecosystem (uv + ruff).

## Project Overview

This project leverages uv's ability to run Python scripts with inline dependencies, allowing each agent to be self-contained in a single file.

## Development Guidelines

### Core Principles
- **Self-contained**: Each script includes all dependencies via uv's inline syntax
- **Well-documented**: Clear docstrings, comments, and type hints throughout
- **Astral ecosystem**: Built on uv for dependency management and ruff for code quality
- **Functional paradigm**: Focus on "what", not "how" - prefer abstractions and leverage functools & itertools
- **Minimal code philosophy**: Less lines = less maintenance = fewer tokens = better

### Creating New Agents
- Each agent must be a single Python file
- Include comprehensive docstrings for all functions and classes
- No debug comments or obvious explanations
- Apply functional programming patterns:
  - Use `functools` & `itertools` 
  - Prefer immutable data structures and pure functions

### Code Quality
- Always use `uvx` interface for external tools
- **Linting**: `uvx ruff check <agent_file.py>`
- **Formatting**: `uvx ruff format <agent_file.py>`

### Code Clarity
- Avoid redundant or obvious comments; remove debug comments before committing
  - Comment complex algorithms or non-obvious design decisions
- Let the code speak for itself through clear naming

### Testing
- Ensure the correct environment variables are set
