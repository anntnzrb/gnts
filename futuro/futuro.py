# /// script
# dependencies = [
#     "google-genai>=1.1.0",
#     "rich>=13.7.0",
#     "pydantic>=2.0.0",
# ]
# ///

"""Sports Table Extractor - Advanced image analysis agent for extracting football league table data."""

import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

from rich.console import Console
from rich.panel import Panel
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

console = Console()
MODEL = "gemini-2.5-pro-preview-05-06"
TEMPERATURE = 0.1


class TeamData(BaseModel):
    """Individual team statistics from the table."""

    team: str = Field(description="Team name")
    games: int = Field(description="Games played")
    goal_diff: int = Field(description="Goal difference")
    xG: float = Field(description="Expected Goals")
    xGA: float = Field(description="Expected Goals Against")
    net_xG: float = Field(description="Net Expected Goals")
    xG_pts: int = Field(description="Expected Goal Points")


class SportsTableData(BaseModel):
    """Complete sports table extraction result."""

    country: str = Field(description="Country/league (e.g., italy, spain, england)")
    match_type: str = Field(default="all", description="Match type: all, home, or away")
    time_period: str = Field(
        default="current", description="Time period: current, last5, last10, etc."
    )
    teams: list[TeamData] = Field(description="List of team data")

    @field_validator("teams")
    def validate_teams(cls, v):
        if len(v) < 10:
            console.print(
                "[yellow]Warning: Less than 10 teams detected. This might indicate incomplete extraction.[/yellow]"
            )
        return v


class ReflectionResult(BaseModel):
    """Result of reflection on extraction quality."""

    is_accurate: bool = Field(description="Whether extraction appears accurate")
    confidence_score: float = Field(
        ge=0, le=1, description="Confidence in extraction (0-1)"
    )
    issues_found: list[str] = Field(
        default_factory=list, description="Issues identified"
    )
    suggested_corrections: list[str] = Field(
        default_factory=list, description="Suggested fixes"
    )
    final_assessment: str = Field(description="Overall assessment")


class SportsTableExtractor:
    """Main extractor class for sports table data."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the extractor with Gemini API."""
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        elif not os.environ.get("GEMINI_API_KEY"):
            raise ValueError(
                "GEMINI_API_KEY must be provided or set as environment variable"
            )

        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY"),
            http_options={"api_version": "v1alpha"},
        )

    def _detect_mime_type(self, image_path: str) -> str:
        """Detect MIME type from file extension."""
        extension = Path(image_path).suffix.lower()
        if extension != ".png":
            raise ValueError(f"Only PNG files are supported. Got '{extension}'")
        return "image/png"

    def _create_extraction_prompt(self, filename_info: dict) -> str:
        """Create the comprehensive extraction prompt with filename context."""
        context_info = ""
        if filename_info:
            context_info = f"""
<filename_context>
Image filename suggests:
- Country/League: {filename_info.get("country", "unknown")}
- Match Type: {filename_info.get("match_type", "unknown")}
- Time Period: {filename_info.get("time_period", "unknown")}
Use this information to help validate your extraction.
</filename_context>
"""

        return f"""<purpose>
You are a data extraction specialist. Analyze this image of a sports league table and convert it to structured JSON data.
</purpose>
{context_info}
<instructions>
First, think through your analysis step by step, showing your reasoning process. Then provide the final JSON.

Step-by-step process:
1. **Initial Image Analysis** - Describe what you see in the image
2. **Context Validation** - Check if filename context matches what you see
3. **Column Identification** - List the column headers you can identify
4. **Team Recognition** - Note the teams and their order
5. **Data Extraction** - Go through each team systematically
6. **Quality Check** - Verify your extraction makes sense

After your analysis, provide the JSON in this format:
```json
{{
  "country": string,
  "match_type": string,
  "time_period": string, 
  "teams": [
    {{
      "team": "Team Name",
      "games": number,
      "goal_diff": number,
      "xG": number,
      "xGA": number,
      "net_xG": number,
      "xG_pts": number
    }}
  ]
}}
```
</instructions>

<data_fields>
Extract these key columns (may have different labels):
- Country/League: Identify from team names or context (italy, spain, england, etc.)
- Match Type: all, home, away (IMPORTANT: This comes from filename, not visible in image)
- Time Period: current, last5, last10, etc. (IMPORTANT: This comes from filename, not visible in image)
- Team name (usually first text column)
- Games played (GP, Games, Matches, etc.)
- Goal Difference (GD, +/-, Goal Diff, etc.)
- xG (Expected Goals) 
- xGA (Expected Goals Against)
- Net xG (xG difference, may be calculated as xG - xGA)
- xG Pts (Expected Goal Points, xG Points, etc.)
</data_fields>

<requirements>
- Show your thinking process clearly
- IMPORTANT: country, match_type, and time_period fields are derived from filename metadata, NOT from image content
- These filename-derived fields are non-verifiable from the image alone - this is expected and correct
- Maintain exact team order from image (top to bottom)
- Use exact team names as they appear
- Handle positive/negative numbers correctly
- Use null for missing data fields
- Ensure correct data types (integers vs decimals)
</requirements>"""

    def _analyze_image(self, image_path: str, prompt: str, step_name: str) -> str:
        """Send image and prompt to Gemini API."""
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode()

        mime_type = self._detect_mime_type(image_path)

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=base64.b64decode(image_data), mime_type=mime_type
                    ),
                ],
            ),
        ]

        config = types.GenerateContentConfig(
            temperature=TEMPERATURE,
        )

        response = ""
        for chunk in self.client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=config,
        ):
            if chunk.text:
                response += chunk.text

        return response.strip()

    def _create_reflection_prompt(self, extracted_data: dict) -> str:
        """Create reflection prompt to validate extraction quality."""
        return f"""<purpose>
You are a data validation specialist. Review the original image and the extracted JSON data to assess accuracy and identify any issues.
</purpose>

<extracted_data>
{json.dumps(extracted_data, indent=2)}
</extracted_data>

<instructions>
Think through your validation step by step, showing your reasoning process. Then provide the final assessment JSON.

Step-by-step validation:
1. **Visual Comparison** - Compare image with extracted data
2. **Team Names Check** - Verify spelling and completeness
3. **Numerical Validation** - Check if numbers match and make sense
4. **Order Verification** - Confirm team order matches image
5. **Completeness Review** - Ensure no teams missed or duplicated
6. **Filename Metadata Note** - country, match_type, and time_period are derived from filename, NOT image content (this is expected)
7. **Final Assessment** - Overall quality evaluation

After your analysis, provide the assessment in this format:
```json
{{
  "is_accurate": true/false,
  "confidence_score": 0.0,
  "issues_found": ["list of specific issues"],
  "suggested_corrections": ["list of suggested fixes"],
  "final_assessment": "overall assessment"
}}
```
</instructions>

<important_note>
The fields 'country', 'match_type', and 'time_period' are derived from filename metadata and are NOT visible in the image. 
Do NOT flag these as issues - they are intentionally non-verifiable from image content alone.
</important_note>

<validation_points>
- Team names: exact spelling and completeness
- Numerical accuracy: values match image
- Data consistency: reasonable and logical values
- Structural integrity: all fields present, correct types
- Completeness: no missing or extra teams
</validation_points>"""

    def _parse_filename(self, image_path: str) -> dict:
        """Parse filename to extract country, match_type, and time_period."""
        filename = Path(image_path).stem  # Remove extension
        parts = filename.split("_")

        info = {}
        if len(parts) >= 3:
            info["country"] = parts[0].lower()
            info["match_type"] = parts[1].lower()
            info["time_period"] = parts[2].lower()

        return info

    def extract_data(self, image_path: str) -> tuple[SportsTableData, ReflectionResult]:
        """Extract data from image with reflection validation."""
        filename_info = self._parse_filename(image_path)

        prompt = self._create_extraction_prompt(filename_info)
        response = self._analyze_image(image_path, prompt, "🔍 Step 1: Extracting Data")

        if not response:
            raise ValueError("Empty response from model")

        parsed_data = self._extract_json(response)
        table_data = SportsTableData(**parsed_data)

        reflection_prompt = self._create_reflection_prompt(parsed_data)
        reflection_response = self._analyze_image(
            image_path,
            reflection_prompt,
            "🤔 Step 2: Reflecting on Extraction",
        )

        if not reflection_response:
            raise ValueError("Empty reflection response from model")

        reflection_data = self._extract_json(reflection_response)
        reflection_result = ReflectionResult(**reflection_data)

        console.print("[green]✅ Extraction and reflection complete![/green]\n")

        return table_data, reflection_result

    def _extract_json(self, text: str) -> dict:
        """Extract JSON from model response with robust error handling."""
        if not text or not text.strip():
            raise json.JSONDecodeError("Empty response text", text or "", 0)

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for delimiter in ["```json", "```"]:
            if delimiter in text:
                try:
                    parts = text.split(delimiter)
                    if len(parts) >= 2 and (
                        json_str := parts[1].split("```")[0].strip()
                    ):
                        return json.loads(json_str)
                except (IndexError, json.JSONDecodeError):
                    continue

        start_idx = text.find("{")
        if start_idx == -1:
            raise json.JSONDecodeError("No JSON object found", text, 0)

        brace_count = 0
        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start_idx : i + 1])
                    except json.JSONDecodeError:
                        continue

        raise json.JSONDecodeError("No valid JSON found", text, 0)

    def save_to_file(self, data: SportsTableData, output_path: str) -> None:
        """Save extracted data to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data.model_dump(), f, indent=2, ensure_ascii=False)
        console.print(f"[green]Data saved to: {output_path}[/green]")


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


def display_results(data: SportsTableData, reflection: ReflectionResult) -> None:
    """Display extraction results with reflection analysis."""
    # Extraction summary
    summary = f"""[bold]Country:[/bold] {data.country}
[bold]Match Type:[/bold] {data.match_type}
[bold]Time Period:[/bold] {data.time_period}
[bold]Teams Extracted:[/bold] {len(data.teams)}"""

    console.print(Panel(summary, title="📊 Extraction Summary", border_style="blue"))

    # Reflection results
    accuracy_color = "green" if reflection.is_accurate else "red"
    confidence_color = (
        "green"
        if reflection.confidence_score > 0.8
        else "yellow"
        if reflection.confidence_score > 0.6
        else "red"
    )

    reflection_summary = f"""[bold]Accuracy:[/bold] [{accuracy_color}]{"✓ Accurate" if reflection.is_accurate else "✗ Issues Found"}[/{accuracy_color}]
[bold]Confidence:[/bold] [{confidence_color}]{reflection.confidence_score:.1%}[/{confidence_color}]
[bold]Assessment:[/bold] {reflection.final_assessment}"""

    console.print(
        Panel(reflection_summary, title="🔍 Reflection Analysis", border_style="cyan")
    )

    # Issues and suggestions if any
    if reflection.issues_found:
        issues_text = "\n".join(f"• {issue}" for issue in reflection.issues_found)
        console.print(Panel(issues_text, title="⚠️ Issues Found", border_style="yellow"))

    if reflection.suggested_corrections:
        suggestions_text = "\n".join(
            f"• {suggestion}" for suggestion in reflection.suggested_corrections
        )
        console.print(
            Panel(
                suggestions_text,
                title="💡 Suggested Corrections",
                border_style="magenta",
            )
        )


def main():
    parser = argparse.ArgumentParser(
        description="Sports Table Extractor - Extract league table data from images",
        epilog="Example: uv run futuro.py table_image.png",
    )
    parser.add_argument(
        "image_paths", nargs="+", help="Path(s) to the sports table image file(s)"
    )

    args = parser.parse_args()

    # Validate all image paths first
    image_paths = []
    for path_str in args.image_paths:
        image_path = Path(path_str)
        if not image_path.exists():
            console.print(f"[red]Error: Image file '{path_str}' not found[/red]")
            sys.exit(1)
        if image_path.suffix.lower() != ".png":
            console.print(
                f"[red]Error: Only PNG files are supported. Got '{image_path.suffix}' for '{path_str}'[/red]"
            )
            sys.exit(1)
        image_paths.append(image_path)

    api_key = validate_environment()

    try:
        extractor = SportsTableExtractor(api_key=api_key)

        if len(image_paths) == 1:
            console.print(f"[cyan]Processing: {image_paths[0].name}[/cyan]\n")
        else:
            console.print(
                f"[cyan]Processing {len(image_paths)} images in parallel...[/cyan]"
            )
            for i, path in enumerate(image_paths, 1):
                console.print(f"  {i}. {path.name}")
            console.print()

        if len(image_paths) == 1:
            image_path = image_paths[0]
            console.print(
                f"[bold blue]--- Processing: {image_path.name} ---[/bold blue]"
            )
            data, reflection = extractor.extract_data(str(image_path))
            output_path = f"{image_path.stem}_extracted.json"
            extractor.save_to_file(data, output_path)
            results = [(image_path, data, reflection)]
        else:

            def process_single_image(
                image_path: Path,
            ) -> Tuple[Path, SportsTableData, str]:
                console.print(
                    f"\n[bold blue]--- Processing: {image_path.name} ---[/bold blue]"
                )
                data, reflection = extractor.extract_data(str(image_path))
                output_path = f"{image_path.stem}_extracted.json"
                extractor.save_to_file(data, output_path)
                console.print(
                    f"[green]✅ {image_path.name} ({len(data.teams)} teams)[/green]"
                )
                return image_path, data, reflection

            results = []
            with ThreadPoolExecutor(max_workers=min(len(image_paths), 4)) as executor:
                future_to_path = {
                    executor.submit(process_single_image, path): path
                    for path in image_paths
                }

                for future in as_completed(future_to_path):
                    try:
                        image_path, data, reflection = future.result()
                        results.append((image_path, data, reflection))
                    except Exception as e:
                        failed_path = future_to_path[future]
                        console.print(
                            f"[red]❌ Failed to process {failed_path.name}: {e}[/red]"
                        )

        if len(results) == 1:
            image_path, data, reflection = results[0]
            if len(image_paths) > 1:
                display_results(data, reflection)
            console.print(
                f"[bold green]🎉 Successfully extracted {len(data.teams)} teams![/bold green]"
            )
        else:
            console.print("\n[bold cyan]📊 Results Summary:[/bold cyan]\n")

            total_teams = 0
            for image_path, data, reflection in results:
                total_teams += len(data.teams)
                console.print(
                    f"[bold blue]• {image_path.name}[/bold blue]: {len(data.teams)} teams extracted"
                )

            console.print(
                f"\n[bold green]🎉 Successfully processed {len(results)} images with {total_teams} total teams![/bold green]"
            )

            # Show detailed results for all images
            console.print("\n[dim]--- Detailed Results ---[/dim]")
            for image_path, data, reflection in results:
                console.print(f"\n[bold blue]{image_path.name}:[/bold blue]")
                display_results(data, reflection)
        return 0

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if "--debug" in sys.argv:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


if __name__ == "__main__":
    # Display usage if no arguments
    if len(sys.argv) == 1:
        console.print(
            Panel(
                "[bold]Sports Table Extractor[/bold]\n\n"
                "Extract football league table data from images to structured JSON\n\n"
                "[yellow]Usage:[/yellow]\n"
                "  uv run futuro.py image1.png\n"
                "  uv run futuro.py image1.png image2.png\n"
                "  uv run futuro.py *.png\n\n"
                "[yellow]Setup:[/yellow]\n"
                "  export GEMINI_API_KEY='your-api-key-here'\n"
                "  Get key from: https://aistudio.google.com/apikey",
                title="🏆 Sports Table Extractor",
                border_style="cyan",
            )
        )
        sys.exit(0)
    else:
        sys.exit(main())
