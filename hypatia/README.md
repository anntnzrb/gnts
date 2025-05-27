# Hypatia

An advanced analytical reasoning agent named after Hypatia of Alexandria, the renowned mathematician and philosopher. Powered by Google Gemini Pro, Hypatia excels at breaking down complex problems across mathematics, logic, science, philosophy, and practical domains using iterative reasoning and parallel component analysis.

## Features

### Core Capabilities
- **Iterative reasoning**: Multi-iteration analysis for complex problems
- **Tool-based architecture**: Decomposes problems into manageable components
- **Parallel processing**: Analyzes multiple components simultaneously
- **Robust error handling**: Graceful fallbacks with retry logic
- **Rich console interface**: Beautiful formatted output with progress indicators

### Reasoning Pipeline
- **Problem decomposition**: Breaks complex queries into components
- **Component analysis**: Deep analysis of individual problem parts
- **Validation**: Checks logical consistency and assumptions
- **Synthesis**: Combines analyses into coherent solutions
- **Alternative perspectives**: Presents multiple valid approaches

## Usage

### Basic Usage (Default - Simple Mode)
```bash
uv run hypatia.py "Your reasoning question here"
```
Perfect for most questions. Uses single-iteration reasoning for fast, clear analysis.

### Advanced Mode (Multiple Iterations)
```bash
uv run hypatia.py "Complex reasoning question" -it 5
```
For complex problems requiring systematic decomposition and multi-step analysis.

### Arguments
- `-it, --iterations N`: Maximum reasoning iterations (default: 1, use 2+ for advanced mode)

## Environment Setup

Set your Gemini API key:

```bash
export GEMINI_API_KEY='your-api-key-here'
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

## Output Format

The agent returns structured output with the following sections:

- **Problem Analysis**: Analysis of what the query is asking
- **Assumptions**: List of assumptions made (if any)
- **Reasoning Process**: Step-by-step reasoning with numbered steps
- **Solution**: Final answer or conclusion
- **Alternative Solutions**: Other valid approaches when applicable

## Examples

### Simple Questions (Default Mode)
```bash
# Basic factual question
uv run hypatia.py "What is the capital of France?"

# Simple calculation
uv run hypatia.py "What is 2+2?"
```

### Moderate Complexity
```bash
# Conceptual comparison
uv run hypatia.py "Explain the difference between machine learning and traditional programming" -it 2

# Scientific reasoning
uv run hypatia.py "Why is the sky blue?" -it 2
```

### Complex Problems
```bash
# Advanced physics
uv run hypatia.py "Explain the relationship between quantum entanglement and the measurement problem" -it 5

# Mathematical proof
uv run hypatia.py "Prove that the square root of 2 is irrational using multiple approaches" -it 3

# Philosophical analysis
uv run hypatia.py "Analyze the mind-body problem from different philosophical perspectives" -it 4
```

## Architecture

### Tool-Based Reasoning
Hypatia uses four specialized reasoning tools:

1. **Decompose**: Breaks complex problems into components
2. **Analyze**: Performs deep analysis of individual components  
3. **Validate**: Checks logical consistency and assumptions
4. **Synthesize**: Combines analyses into final solutions

### Error Handling
- **Retry logic**: Up to 3 attempts for failed operations
- **Timeout protection**: 60-second timeout per component analysis
- **Graceful fallbacks**: Meaningful responses when technical issues occur
- **Parallel safety**: Robust error handling in concurrent operations

### Performance Features
- **Concurrent analysis**: Multiple components analyzed in parallel
- **Progressive refinement**: Each iteration builds on previous results
- **Smart caching**: Avoids re-analyzing completed components
- **Resource management**: Automatic cleanup and timeout handling

## Technical Details

### Model Configuration
- **Model**: `gemini-2.5-pro-preview-05-06`
- **Temperature**: 0.1 (deterministic reasoning)
- **Retry strategy**: Adaptive temperature increase on failures

## Troubleshooting

### Performance Tips
- Default mode (1 iteration) is perfect for most questions and provides fast responses
- Use `-it 2` or `-it 3` for moderate complexity before trying higher values
- Complex queries with 5+ iterations may take several minutes to complete

## Examples by Domain

### Mathematics
```bash
uv run hypatia.py "Explain the concept of infinity in set theory" -it 3
```

### Science
```bash
uv run hypatia.py "How does photosynthesis work at the molecular level?" -it 4
```

### Philosophy
```bash
uv run hypatia.py "What is consciousness and how might it emerge from physical processes?" -it 5
```

### Logic Puzzles
```bash
uv run hypatia.py "Five people with different nationalities live in five houses. Using the given clues, determine who owns the fish." -it 3
```

---

*Named after Hypatia of Alexandria (c. 350-415 CE), a brilliant mathematician, astronomer, and philosopher who exemplified systematic reasoning and intellectual rigor.*
