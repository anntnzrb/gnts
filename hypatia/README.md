# Hypatia

An advanced analytical reasoning agent named after Hypatia of Alexandria, the renowned mathematician and philosopher. Powered by Google Gemini Pro, Hypatia excels at breaking down complex problems across mathematics, logic, science, philosophy, and practical domains.

## Features

- **Deep reasoning**: Systematic analytical approach to complex problems
- **Deterministic output**: Low temperature for consistent, logical results
- **Structured output**: Returns well-organized JSON with reasoning steps
- **Rich console interface**: Beautiful formatted output with color-coded sections
- **Alternative solutions**: Presents multiple valid approaches when applicable

## Usage

### Basic Usage

```bash
uv run hypatia.py "Your reasoning question here"
```

## Environment Setup

Set your Gemini API key:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Output Format

The agent returns structured JSON with the following fields:

- **problem_analysis**: Brief analysis of the query
- **assumptions**: List of assumptions made (if any)
- **reasoning_steps**: Step-by-step reasoning process
- **solution**: Final answer or conclusion
- **alternative_solutions**: Other valid solutions (optional)

## Examples

### Logic Puzzle
```bash
uv run hypatia.py "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much does the ball cost?"
```

### Scientific Reasoning
```bash
uv run hypatia.py "What would happen to Earth's climate if the moon suddenly disappeared?"
```

### Mathematical Problem
```bash
uv run hypatia.py "Prove that the square root of 2 is irrational"
```

### Philosophical Question
```bash
uv run hypatia.py "Can an AI system truly understand language, or is it merely pattern matching?"
```

## Model Information

Uses Gemini Pro Preview model for advanced reasoning capabilities.

## Dependencies

- `google-genai>=1.1.0` - Google's Gemini API client
- `rich>=13.7.0` - Rich console formatting

All dependencies are automatically handled via uv's inline script metadata.