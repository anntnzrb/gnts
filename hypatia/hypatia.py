# /// script
# dependencies = [
#     "google-genai>=1.1.0",
#     "rich>=13.7.0",
# ]
# ///

"""Hypatia - Advanced analytical reasoning agent with Gemini thinking budget support."""

import os
import sys
import argparse
import json
from typing import Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from google import genai
from google.genai import types

console = Console()
MODEL = "gemini-2.5-pro-preview-05-06"
TEMPERATURE = 0.1

AGENT_PROMPT = """<purpose>
You are a world-class expert at analytical reasoning, capable of breaking down complex
problems across mathematics, logic, science, philosophy, and practical domains.
Your goal is to provide clear, systematic, and insightful analysis.
</purpose>

<instructions>
<instruction>Analyze the query to identify the core problem or question</instruction>
<instruction>Break down complex problems into manageable components</instruction>
<instruction>Apply relevant reasoning frameworks and methodologies</instruction>
<instruction>Consider multiple perspectives and potential edge cases</instruction>
<instruction>Show your step-by-step thinking process explicitly</instruction>
<instruction>Identify and state any assumptions you're making</instruction>
<instruction>Provide clear, definitive conclusions with supporting logic</instruction>
<instruction>If the problem has multiple valid solutions, present them all</instruction>
</instructions>

<output_format>
You must return your response as valid JSON with the following structure:
{
    "problem_analysis": "Brief analysis of what the query is asking",
    "assumptions": ["List of assumptions made, if any"],
    "reasoning_steps": [
        "Step 1: First reasoning step",
        "Step 2: Second reasoning step",
        "..."
    ],
    "solution": "Final answer or conclusion",
    "alternative_solutions": ["Optional: other valid solutions or perspectives"]
}
</output_format>

<query>
{{USER_QUERY}}
</query>"""


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


def extract_json(text: str) -> Dict[str, Any]:
    """Extracts JSON from model response."""
    for delimiter in ["```json", "```"]:
        if delimiter in text:
            return json.loads(text.split(delimiter)[1].split("```")[0].strip())
    return json.loads(text)


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


def display_result(data: Dict[str, Any]) -> None:
    """Displays reasoning results with rich formatting."""
    panels = [
        ("problem_analysis", "📊 Problem Analysis", "blue", lambda x: x),
        (
            "assumptions",
            "📌 Assumptions",
            "yellow",
            lambda x: "\n".join(f"• {a}" for a in x),
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


def run_reasoning(query: str, api_key: str) -> None:
    """Executes the reasoning query."""
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    prompt = AGENT_PROMPT.replace("{{USER_QUERY}}", query)

    try:
        generate_content_config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
            response_mime_type="text/plain",
        )

        with console.status("[cyan]Reasoning...[/cyan]", spinner="dots"):
            response = client.models.generate_content(
                model=MODEL,
                contents=[
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=prompt)]
                    )
                ],
                config=generate_content_config,
            )

        try:
            display_result(extract_json(response.text))
        except json.JSONDecodeError as e:
            console.print(
                create_panel(
                    f"[red]Failed to parse JSON response:[/red]\n{str(e)}\n\n"
                    f"[yellow]Raw response:[/yellow]\n{response.text}",
                    "Parse Error",
                    "red",
                )
            )

    except Exception as e:
        console.print(
            create_panel(
                f"[red]Error during reasoning:[/red]\n{str(e)}",
                "Execution Error",
                "red",
            )
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Hypatia - Analytical reasoning agent",
        epilog='Example: uv run hypatia.py "Why is the sky blue?"',
    )
    parser.add_argument(
        "query", help="Question or problem requiring analytical reasoning"
    )

    args = parser.parse_args()
    run_reasoning(args.query, validate_environment())


if __name__ == "__main__":
    main()
