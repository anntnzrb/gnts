# /// script
# dependencies = [
#     "google-genai>=1.1.0",
#     "rich>=13.7.0",
#     "pydantic>=2.0.0",
# ]
# ///

"""Hypatia - Advanced analytical reasoning agent with Gemini thinking budget support."""

import os
import sys
import argparse
import json
import concurrent.futures
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

console = Console()
MODEL = "gemini-2.5-pro-preview-05-06"
TEMPERATURE = 0.1


class ProblemDecomposition(BaseModel):
    """Result of decomposing a complex problem."""

    components: List[str] = Field(description="Individual problem components")
    relationships: Dict[str, List[str]] = Field(
        default_factory=dict, description="How components relate"
    )
    complexity_score: float = Field(ge=0, le=1, description="Problem complexity (0-1)")
    approach_strategy: str = Field(description="Recommended approach")


class ComponentAnalysis(BaseModel):
    """Analysis of a single problem component."""

    component: str = Field(description="The component being analyzed")
    analysis: str = Field(description="Detailed analysis")
    assumptions: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Result of validating reasoning steps."""

    is_valid: bool
    logical_consistency: float = Field(ge=0, le=1)
    assumption_validity: float = Field(ge=0, le=1)
    issues: List[str] = Field(default_factory=list)
    suggestions: List[str] = Field(default_factory=list)


class ReasoningResult(BaseModel):
    """Complete reasoning result."""

    problem_analysis: str
    assumptions: List[str] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    solution: str
    alternative_solutions: List[str] = Field(default_factory=list)

    @field_validator("reasoning_steps")
    def validate_steps(cls, v):
        if len(v) < 2 and v:
            raise ValueError("Reasoning must have at least 2 steps")
        return v


@dataclass
class ReasoningState:
    """Tracks state across reasoning iterations."""

    query: str
    iteration: int = 0
    history: List[ReasoningResult] = field(default_factory=list)
    decomposition: Optional[ProblemDecomposition] = None
    component_analyses: List[ComponentAnalysis] = field(default_factory=list)


# tools
class ReasoningTool:
    """Base class for reasoning tools."""

    def __init__(self, name: str, description: str, prompt_template: str):
        self.name = name
        self.description = description
        self.prompt_template = prompt_template

    def create_prompt(self, **kwargs) -> str:
        return self.prompt_template.format(**kwargs)


DECOMPOSE_PROMPT = """<purpose>
Decompose a complex problem into manageable components for systematic analysis.
</purpose>

<query>
{query}
</query>

<context>
{context}
</context>

<instructions>
1. Identify the core problem and any sub-problems
2. List individual components that need analysis
3. Map relationships between components
4. Assess overall complexity
5. Recommend an approach strategy
</instructions>

<output_format>
Return a JSON object with this EXACT structure:
{{
    "components": ["list of problem components"],
    "relationships": {{"component1": ["related_to"]}},
    "complexity_score": 0.0 to 1.0,
    "approach_strategy": "recommended approach"
}}
</output_format>"""

ANALYZE_PROMPT = """<purpose>
Analyze a specific component of the problem in detail.
</purpose>

<component>
{component}
</component>

<context>
Original query: {query}
Previous analyses: {previous_analyses}
</context>

<instructions>
1. Provide detailed analysis of this component
2. List any assumptions you're making
3. Identify dependencies on other components
4. Provide clear analysis
</instructions>

<output_format>
Return a JSON object with this structure:
{{
    "component": "the component name",
    "analysis": "detailed analysis",
    "assumptions": ["list of assumptions"],
    "dependencies": ["list of dependencies"]
}}
</output_format>"""

VALIDATE_PROMPT = """<purpose>
Validate the logical consistency and soundness of reasoning.
</purpose>

<reasoning_chain>
{reasoning_chain}
</reasoning_chain>

<original_query>
{original_query}
</original_query>

<instructions>
1. Check logical flow between steps
2. Validate assumptions are reasonable
3. Identify any gaps or issues
4. Suggest improvements if needed
</instructions>

<output_format>
Return a JSON object:
{{
    "is_valid": true/false,
    "logical_consistency": 0.0 to 1.0,
    "assumption_validity": 0.0 to 1.0,
    "issues": ["list of issues found"],
    "suggestions": ["list of suggestions"]
}}
</output_format>"""

SYNTHESIZE_PROMPT = """<purpose>
Synthesize component analyses into a complete solution.
</purpose>

<phase>
{phase}
</phase>

<query>
{query}
</query>

<decomposition>
{decomposition}
</decomposition>

<analyses>
{analyses}
</analyses>

<validation>
{validation}
</validation>

<instructions>
1. Combine all analyses into a coherent solution
2. Show clear reasoning steps
3. State the final solution
4. Consider alternative approaches
5. Provide clear conclusions
</instructions>

<output_format>
Return a JSON object:
{{
    "problem_analysis": "summary of the problem",
    "assumptions": ["list of assumptions"],
    "reasoning_steps": ["step by step reasoning"],
    "solution": "final solution",
    "alternative_solutions": ["other valid approaches"]
}}
</output_format>"""

REASONING_TOOLS = {
    "decompose": ReasoningTool(
        "decompose_problem", "Break complex query into components", DECOMPOSE_PROMPT
    ),
    "analyze": ReasoningTool(
        "analyze_component", "Analyze a single component", ANALYZE_PROMPT
    ),
    "validate": ReasoningTool(
        "validate_reasoning", "Validate reasoning chain", VALIDATE_PROMPT
    ),
    "synthesize": ReasoningTool(
        "synthesize_solution", "Combine into final solution", SYNTHESIZE_PROMPT
    ),
}


def validate_environment() -> str:
    """Validates environment and returns API key."""
    if api_key := os.getenv("GEMINI_API_KEY"):
        return api_key

    console.print(
        Panel(
            "[red]Error: GEMINI_API_KEY environment variable is not set[/red]\n\n"
            "Please get your API key from Google AI Studio:\n"
            "https://aistudio.google.com/apikey\n\n"
            "Then set it with:\n"
            "[yellow]export GEMINI_API_KEY='your-api-key-here'[/yellow]",
            title="Configuration Error",
            border_style="red",
        )
    )
    sys.exit(1)


def execute_tool(
    client: genai.Client, tool: ReasoningTool, model: str, temperature: float, **kwargs
) -> Dict[str, Any]:
    """Execute a reasoning tool and parse response with retry logic."""
    prompt = tool.create_prompt(**kwargs)
    max_retries = 3

    for attempt in range(max_retries):
        try:
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature
                if attempt == 0
                else min(temperature + 0.1 * attempt, 0.9),
                response_mime_type="text/plain",
            )

            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=generate_content_config,
            )

            if not response.text:
                raise ValueError(f"Empty response from model on attempt {attempt + 1}")

            return extract_json(response.text)

        except (json.JSONDecodeError, ValueError) as e:
            if attempt == max_retries - 1:
                console.print(
                    f"[red]Tool execution failed after {max_retries} attempts: {e}[/red]"
                )
                return {
                    "error": f"Failed to execute {tool.name}",
                    "details": str(e),
                    "fallback": True,
                }
            else:
                console.print(
                    f"[yellow]Attempt {attempt + 1} failed, retrying...[/yellow]"
                )
                continue

    # no reach
    raise Exception("Unexpected error in execute_tool")


def extract_json(text: str) -> Dict[str, Any]:
    """Extracts JSON from model response with robust error handling."""
    if not text or not text.strip():
        raise json.JSONDecodeError("Empty response text", text or "", 0)

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        for delimiter in ["```json", "```"]:
            if delimiter in text:
                try:
                    parts = text.split(delimiter)
                    if len(parts) >= 2:
                        json_str = parts[1].split("```")[0].strip()
                        if json_str:
                            return json.loads(json_str)
                except (IndexError, json.JSONDecodeError):
                    continue

        brace_count = 0
        start_idx = text.find("{")
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    json_candidate = text[start_idx : i + 1]
                    try:
                        return json.loads(json_candidate)
                    except json.JSONDecodeError:
                        continue

        raise json.JSONDecodeError("No valid JSON found", text, 0)


def create_panel(content: str, title: str, color: str) -> Panel:
    """Creates a formatted panel."""
    return Panel(content, title=title, border_style=color)


def format_reasoning_steps(steps: list) -> str:
    """Formats reasoning steps with numbering."""
    return "\n\n".join(f"[bold]{i + 1}.[/bold] {step}" for i, step in enumerate(steps))


def format_alternatives(alternatives: list) -> str:
    """Formats alternative solutions."""
    return "\n\n".join(
        f"[bold]Option {i + 1}:[/bold] {alt}" for i, alt in enumerate(alternatives)
    )


def display_result(result: ReasoningResult) -> None:
    """Displays reasoning results with rich formatting."""
    data = result.model_dump(exclude_none=True)

    panels = [
        ("problem_analysis", "📊 Problem Analysis", "blue", lambda x: x),
        (
            "assumptions",
            "📌 Assumptions",
            "yellow",
            lambda x: "\n".join(f"• {a}" for a in x) if x else "None stated",
        ),
        ("reasoning_steps", "🧠 Reasoning Process", "cyan", format_reasoning_steps),
        ("solution", "💡 Solution", "green", lambda x: x),
        (
            "alternative_solutions",
            "🔄 Alternative Solutions",
            "magenta",
            format_alternatives,
        ),
    ]

    for key, title, color, formatter in panels:
        if key in data and data[key]:
            console.print(create_panel(formatter(data[key]), title, color))


def run_reasoning_pipeline(
    query: str,
    api_key: str,
    max_iterations: int = 1,
) -> None:
    """Execute iterative reasoning pipeline with tool-based approach."""
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    if max_iterations == 1:
        return run_simple_reasoning(query, api_key)

    state = ReasoningState(query=query)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for iteration in range(max_iterations):
            state.iteration = iteration

            task_id = progress.add_task(
                f"[cyan]Iteration {iteration + 1}/{max_iterations}...[/cyan]"
            )

            try:
                # Step 1: Decompose (only on first iteration)
                if iteration == 0:
                    decomp_result = execute_tool(
                        client,
                        REASONING_TOOLS["decompose"],
                        MODEL,
                        TEMPERATURE,
                        query=query,
                        context="Initial analysis",
                    )
                    state.decomposition = ProblemDecomposition(**decomp_result)
                    console.print(
                        Panel(
                            f"Components: {len(state.decomposition.components)}\n"
                            f"Complexity: {state.decomposition.complexity_score:.2f}\n"
                            f"Strategy: {state.decomposition.approach_strategy}",
                            title="Problem Decomposition",
                            border_style="dim",
                        )
                    )

                # Step 2: Analyze components in parallel (if needed)
                if iteration > 0 and state.decomposition:
                    # Get components that haven't been analyzed yet
                    unanalyzed_components = [
                        comp
                        for comp in state.decomposition.components[:3]
                        if not any(
                            ca.component == comp for ca in state.component_analyses
                        )
                    ]

                    if unanalyzed_components:

                        def analyze_component(
                            component: str,
                        ) -> Optional[ComponentAnalysis]:
                            try:
                                result = execute_tool(
                                    client,
                                    REASONING_TOOLS["analyze"],
                                    MODEL,
                                    TEMPERATURE,
                                    component=component,
                                    query=query,
                                    previous_analyses=json.dumps(
                                        [
                                            ca.model_dump()
                                            for ca in state.component_analyses
                                        ]
                                    ),
                                )

                                # fallback responses
                                if result.get("fallback"):
                                    console.print(
                                        f"[yellow]Skipping analysis of '{component}' due to errors[/yellow]"
                                    )
                                    return None

                                return ComponentAnalysis(**result)
                            except Exception as e:
                                console.print(
                                    f"[red]Failed to analyze component '{component}': {e}[/red]"
                                )
                                return None

                        # Execute analyses in parallel
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=3
                        ) as executor:
                            future_to_component = {
                                executor.submit(analyze_component, comp): comp
                                for comp in unanalyzed_components
                            }

                            for future in concurrent.futures.as_completed(
                                future_to_component
                            ):
                                try:
                                    analysis = future.result(
                                        timeout=60
                                    )
                                    if analysis is not None:
                                        state.component_analyses.append(analysis)
                                except concurrent.futures.TimeoutError:
                                    component = future_to_component[future]
                                    console.print(
                                        f"[yellow]Component analysis timeout for '{component}', skipping[/yellow]"
                                    )
                                except Exception as e:
                                    component = future_to_component[future]
                                    console.print(
                                        f"[red]Error analyzing component '{component}': {e}[/red]"
                                    )

                # Step 3: Validate if we have previous results
                validation = None
                if state.history and iteration > 0:
                    try:
                        last_result = state.history[-1]
                        val_result = execute_tool(
                            client,
                            REASONING_TOOLS["validate"],
                            MODEL,
                            TEMPERATURE,
                            reasoning_chain=json.dumps(last_result.model_dump()),
                            original_query=query,
                        )
                        if not val_result.get("fallback"):
                            validation = ValidationResult(**val_result)
                    except Exception as e:
                        console.print(f"[yellow]Validation step failed: {e}[/yellow]")

                # Step 4: Synthesize solution
                try:
                    synth_result = execute_tool(
                        client,
                        REASONING_TOOLS["synthesize"],
                        MODEL,
                        TEMPERATURE,
                        phase="iterative",
                        query=query,
                        decomposition=json.dumps(
                            state.decomposition.model_dump()
                            if state.decomposition
                            else {}
                        ),
                        analyses=json.dumps(
                            [ca.model_dump() for ca in state.component_analyses]
                        ),
                        validation=json.dumps(
                            validation.model_dump() if validation else {}
                        ),
                    )

                    if synth_result.get("fallback"):
                        console.print(
                            "[yellow]Synthesis failed, using simplified approach[/yellow]"
                        )
                        # simple fallback result
                        result = ReasoningResult(
                            problem_analysis=f"Analysis of: {query}",
                            reasoning_steps=[
                                "Complex analysis attempted but encountered technical issues"
                            ],
                            solution="Please try with fewer iterations or a simpler query",
                            assumptions=["Technical limitations encountered"],
                            alternative_solutions=["Retry with -it 1 for simple mode"],
                        )
                    else:
                        result = ReasoningResult(**synth_result)

                except Exception as e:
                    console.print(f"[red]Synthesis step failed: {e}[/red]")
                    # fallback result
                    result = ReasoningResult(
                        problem_analysis=f"Analysis of: {query}",
                        reasoning_steps=["Analysis encountered technical difficulties"],
                        solution="Unable to complete full analysis due to technical issues",
                        assumptions=["System limitations encountered"],
                        alternative_solutions=["Try with -it 1 for basic analysis"],
                    )

                state.history.append(result)
                console.print(f"\n[bold]Iteration {iteration + 1} Result:[/bold]")
                display_result(result)

            except Exception as e:
                console.print(
                    f"[red]Error in iteration {iteration + 1}: {str(e)}[/red]"
                )
                # simpler approach
                if iteration < max_iterations - 1:
                    continue
                else:
                    raise

            finally:
                progress.remove_task(task_id)

        # final summary
        if state.history:
            console.print("\n[bold]Final Result:[/bold]")
            display_result(state.history[-1])
        else:
            console.print("[red]No valid reasoning result generated.[/red]")


def main():
    parser = argparse.ArgumentParser(
        description="Hypatia - Advanced analytical reasoning agent",
        epilog='Example: uv run hypatia.py "Why is the sky blue?"',
    )
    parser.add_argument(
        "query", help="Question or problem requiring analytical reasoning"
    )
    parser.add_argument(
        "--iterations",
        "-it",
        type=int,
        default=1,
        help="Maximum reasoning iterations (default: 1, use 2+ for advanced mode)",
    )

    args = parser.parse_args()
    api_key = validate_environment()

    run_reasoning_pipeline(
        args.query,
        api_key,
        max_iterations=args.iterations,
    )


def run_simple_reasoning(query: str, api_key: str) -> None:
    """Simple one-shot reasoning."""
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})

    prompt = f"""<purpose>
You are an expert at analytical reasoning. Provide clear, systematic analysis.
</purpose>

<query>
{query}
</query>

<output_format>
Return JSON:
{{
    "problem_analysis": "analysis",
    "assumptions": ["assumptions"],
    "reasoning_steps": ["steps"],
    "solution": "solution",
    "alternative_solutions": ["alternatives"]
}}
</output_format>"""

    try:
        with console.status("[cyan]Reasoning...[/cyan]"):
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=types.GenerateContentConfig(temperature=TEMPERATURE),
            )

        if not response.text:
            console.print("[red]Empty response from model[/red]")
            sys.exit(1)

        result_dict = extract_json(response.text)
        result = ReasoningResult(**result_dict)
        display_result(result)

    except json.JSONDecodeError as e:
        console.print(f"[red]JSON Parse Error: {str(e)}[/red]")
        console.print(f"[yellow]Raw response:[/yellow]\n{response.text[:500]}...")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        import traceback

        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        sys.exit(1)


if __name__ == "__main__":
    main()
