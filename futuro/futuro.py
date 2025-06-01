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
from datetime import datetime
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
    team: str = Field(description="Team name")
    games: int = Field(description="Games played")
    goal_diff: int = Field(description="Goal difference")
    xG: float = Field(description="Expected Goals")
    xGA: float = Field(description="Expected Goals Against")
    net_xG: float = Field(description="Net Expected Goals")
    xG_pts: int = Field(description="Expected Goal Points")


class SportsTableData(BaseModel):
    country: str = Field(description="Country/league (e.g., italy, spain, england)")
    match_type: str = Field(default="all", description="Match type: all, home, or away")
    time_period: str = Field(
        default="current", description="Time period: current, last5, last10, etc."
    )
    teams: list[TeamData] = Field(description="List of team data")

    @field_validator("teams")
    def validate_teams(cls, v):
        if len(v) < 10:
            console.print("[yellow]Warning: Less than 10 teams detected.[/yellow]")
        return v


class ReflectionResult(BaseModel):
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


class TeamStats(BaseModel):
    games: int = Field(description="Games played")
    goal_diff: int = Field(description="Goal difference")
    xG: float = Field(description="Expected Goals")
    xGA: float = Field(description="Expected Goals Against")
    net_xG: float = Field(description="Net Expected Goals")
    xG_pts: int = Field(description="Expected Goal Points")


class TeamPerformance(BaseModel):
    all: dict[str, TeamStats] = Field(
        description="All matches performance by time period"
    )
    home: dict[str, TeamStats] = Field(
        description="Home matches performance by time period"
    )
    away: dict[str, TeamStats] = Field(
        description="Away matches performance by time period"
    )


class ConsolidatedMetadata(BaseModel):
    timestamp: str = Field(description="Extraction timestamp")
    country: str = Field(description="Country/league identifier")
    definitions: dict = Field(description="Field definitions and explanations")


class ConsolidatedOutput(BaseModel):
    metadata: ConsolidatedMetadata = Field(description="Extraction metadata")
    data: dict[str, dict[str, TeamPerformance]] = Field(
        description="Team performance data"
    )


class SportsTableExtractor:
    def __init__(self, api_key: Optional[str] = None):
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
        if Path(image_path).suffix.lower() != ".png":
            raise ValueError(
                f"Only PNG files supported. Got '{Path(image_path).suffix}'"
            )
        return "image/png"

    def _create_extraction_prompt(self, filename_info: dict) -> str:
        context = ""
        if filename_info:
            context = f"""<filename_context>
Filename suggests: Country={filename_info.get("country", "unknown")}, Type={filename_info.get("match_type", "unknown")}, Period={filename_info.get("time_period", "unknown")}
</filename_context>
"""

        return f"""<purpose>Extract sports table data to JSON.</purpose>
{context}
<instructions>
Analyze step-by-step, then provide JSON:
1. Identify columns and teams
2. Extract data systematically
3. Verify extraction

JSON format:
```json
{{"country": string, "match_type": string, "time_period": string, "teams": [{{"team": string, "games": int, "goal_diff": int, "xG": float, "xGA": float, "net_xG": float, "xG_pts": int}}]}}
```
</instructions>

<requirements>
- country/match_type/time_period from filename, NOT image
- Exact team order and names
- Correct data types
- Use null for missing fields
</requirements>"""

    def _analyze_image(self, image_path: str, prompt: str) -> str:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=base64.b64decode(image_data),
                        mime_type=self._detect_mime_type(image_path),
                    ),
                ],
            )
        ]

        response = "".join(
            chunk.text
            for chunk in self.client.models.generate_content_stream(
                model=MODEL,
                contents=contents,
                config=types.GenerateContentConfig(temperature=TEMPERATURE),
            )
            if chunk.text
        )
        return response.strip()

    def _create_reflection_prompt(self, extracted_data: dict) -> str:
        return f"""<purpose>Validate extracted data against image.</purpose>

<extracted_data>{json.dumps(extracted_data, indent=2)}</extracted_data>

<instructions>
Validate step-by-step:
1. Compare image vs extracted data
2. Check team names, numbers, order
3. Verify completeness

Note: country/match_type/time_period from filename (not image) - don't flag as issues.

Provide assessment:
```json
{{"is_accurate": bool, "confidence_score": float, "issues_found": ["issues"], "suggested_corrections": ["fixes"], "final_assessment": "summary"}}
```
</instructions>"""

    def _parse_filename(self, image_path: str) -> dict:
        parts = Path(image_path).stem.split("_")
        return (
            {
                "country": parts[0].lower(),
                "match_type": parts[1].lower(),
                "time_period": parts[2].lower(),
            }
            if len(parts) >= 3
            else {}
        )

    def extract_data(self, image_path: str) -> tuple[SportsTableData, ReflectionResult]:
        filename_info = self._parse_filename(image_path)

        extraction_response = self._analyze_image(
            image_path, self._create_extraction_prompt(filename_info)
        )
        if not extraction_response:
            raise ValueError("Empty extraction response")

        parsed_data = self._extract_json(extraction_response)
        table_data = SportsTableData(**parsed_data)

        reflection_response = self._analyze_image(
            image_path, self._create_reflection_prompt(parsed_data)
        )
        if not reflection_response:
            raise ValueError("Empty reflection response")

        reflection_result = ReflectionResult(**self._extract_json(reflection_response))
        console.print("[green]✅ Complete![/green]\n")

        return table_data, reflection_result

    def _extract_json(self, text: str) -> dict:
        if not text.strip():
            raise json.JSONDecodeError("Empty response", text, 0)

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        for delimiter in ["```json", "```"]:
            if delimiter in text:
                try:
                    json_str = text.split(delimiter)[1].split("```")[0].strip()
                    if json_str:
                        return json.loads(json_str)
                except (IndexError, json.JSONDecodeError):
                    continue

        start = text.find("{")
        if start == -1:
            raise json.JSONDecodeError("No JSON found", text, 0)

        brace_count = 0
        for i, char in enumerate(text[start:], start):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        continue

        raise json.JSONDecodeError("No valid JSON found", text, 0)

    def save_to_file(self, data: SportsTableData, output_path: str) -> None:
        Path(output_path).write_text(
            json.dumps(data.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[green]Saved: {output_path}[/green]")

    def save_consolidated_to_file(
        self, data: ConsolidatedOutput, output_path: str
    ) -> None:
        Path(output_path).write_text(
            json.dumps(data.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        console.print(f"[green]Saved consolidated: {output_path}[/green]")

    def _create_metadata_definitions(self) -> dict:
        """Create the definitions section for metadata."""
        return {
            "time_periods": {
                "current": "Full season statistics",
                "last5": "Statistics from the last 5 games played",
                "last10": "Statistics from the last 10 games played",
            },
            "match_types": {
                "all": "All matches (home and away combined)",
                "home": "Home matches only",
                "away": "Away matches only",
            },
            "statistics": {
                "games": "Number of games played",
                "goal_diff": "Goal difference (goals scored minus goals conceded)",
                "xG": "Expected Goals - likelihood of scoring based on shot quality",
                "xGA": "Expected Goals Against - likelihood of conceding based on shots faced",
                "net_xG": "Net Expected Goals (xG minus xGA)",
                "xG_pts": "Expected Goal Points - points a team would have based on xG performance",
            },
        }

    def consolidate_extractions(
        self, extractions: list[SportsTableData]
    ) -> ConsolidatedOutput:
        if not extractions:
            raise ValueError("No extractions provided")

        country = extractions[0].country
        for extraction in extractions:
            if extraction.country != country:
                raise ValueError(
                    f"Mixed countries found: {country} vs {extraction.country}"
                )

        teams_data = {}

        for extraction in extractions:
            for team_data in extraction.teams:
                team_name = team_data.team

                if team_name not in teams_data:
                    teams_data[team_name] = {"all": {}, "home": {}, "away": {}}

                team_stats = TeamStats(
                    games=team_data.games,
                    goal_diff=team_data.goal_diff,
                    xG=team_data.xG,
                    xGA=team_data.xGA,
                    net_xG=team_data.net_xG,
                    xG_pts=team_data.xG_pts,
                )

                teams_data[team_name][extraction.match_type][extraction.time_period] = (
                    team_stats
                )

        team_performances = {
            team_name: TeamPerformance(
                all=team_data["all"], home=team_data["home"], away=team_data["away"]
            )
            for team_name, team_data in teams_data.items()
        }

        metadata = ConsolidatedMetadata(
            timestamp=datetime.now().isoformat(),
            country=country,
            definitions=self._create_metadata_definitions(),
        )

        return ConsolidatedOutput(metadata=metadata, data={"teams": team_performances})


def validate_environment() -> str:
    if api_key := os.getenv("GEMINI_API_KEY"):
        return api_key

    console.print(
        Panel(
            "[red]GEMINI_API_KEY not set[/red]\n\n"
            "Get key: https://aistudio.google.com/apikey\n"
            "Set: [yellow]export GEMINI_API_KEY='key'[/yellow]",
            title="Config Error",
            border_style="red",
        )
    )
    sys.exit(1)


def display_results(data: SportsTableData, reflection: ReflectionResult) -> None:
    summary = f"Country: {data.country} | Type: {data.match_type} | Period: {data.time_period} | Teams: {len(data.teams)}"
    console.print(Panel(summary, title="📊 Summary", border_style="blue"))

    accuracy_color = "green" if reflection.is_accurate else "red"
    confidence_color = (
        "green"
        if reflection.confidence_score > 0.8
        else "yellow"
        if reflection.confidence_score > 0.6
        else "red"
    )

    reflection_summary = f"Accuracy: [{accuracy_color}]{'✓' if reflection.is_accurate else '✗'}[/{accuracy_color}] | Confidence: [{confidence_color}]{reflection.confidence_score:.1%}[/{confidence_color}]\n{reflection.final_assessment}"
    console.print(Panel(reflection_summary, title="🔍 Analysis", border_style="cyan"))

    if reflection.issues_found:
        console.print(
            Panel(
                "\n".join(f"• {i}" for i in reflection.issues_found),
                title="⚠️ Issues",
                border_style="yellow",
            )
        )

    if reflection.suggested_corrections:
        console.print(
            Panel(
                "\n".join(f"• {s}" for s in reflection.suggested_corrections),
                title="💡 Fixes",
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

    image_paths = []
    for path_str in args.image_paths:
        path = Path(path_str)
        if not path.exists():
            console.print(f"[red]File not found: {path_str}[/red]")
            sys.exit(1)
        if path.suffix.lower() != ".png":
            console.print(f"[red]Only PNG supported: {path.suffix}[/red]")
            sys.exit(1)
        image_paths.append(path)

    try:
        extractor = SportsTableExtractor(api_key=validate_environment())
        console.print(
            f"[cyan]Processing {len(image_paths)} image{'s' if len(image_paths) > 1 else ''}...[/cyan]"
        )

        if len(image_paths) > 1:
            for i, path in enumerate(image_paths, 1):
                console.print(f"  {i}. {path.name}")

        def process_image(path: Path) -> Tuple[Path, SportsTableData, ReflectionResult]:
            console.print(f"[blue]Processing: {path.name}[/blue]")
            data, reflection = extractor.extract_data(str(path))
            console.print(f"[green]✅ {path.name} ({len(data.teams)} teams)[/green]")
            return path, data, reflection

        if len(image_paths) == 1:
            results = [process_image(image_paths[0])]
        else:
            results = []
            with ThreadPoolExecutor(max_workers=min(len(image_paths), 4)) as executor:
                future_to_path = {
                    executor.submit(process_image, path): path for path in image_paths
                }
                for future in as_completed(future_to_path):
                    try:
                        results.append(future.result())
                    except Exception as e:
                        console.print(
                            f"[red]❌ {future_to_path[future].name}: {e}[/red]"
                        )

        total_teams = sum(len(data.teams) for _, data, _ in results)

        if len(results) == 1:
            path, data, reflection = results[0]
            extractor.save_to_file(data, f"{path.stem}_extracted.json")
            console.print(f"[green]🎉 Extracted {total_teams} teams![/green]")
            display_results(data, reflection)
        else:
            console.print("\n[cyan]📊 Creating consolidated output...[/cyan]")
            extractions = [data for _, data, _ in results]
            consolidated = extractor.consolidate_extractions(extractions)
            consolidated_path = f"{extractions[0].country}.json"
            extractor.save_consolidated_to_file(consolidated, consolidated_path)

            console.print("\n[cyan]📊 Summary:[/cyan]")
            for path, data, _ in results:
                console.print(f"• {path.name}: {len(data.teams)} teams")
            console.print(
                f"\n[green]🎉 Processed {len(results)} images, {total_teams} total teams![/green]"
            )
            console.print(
                f"[green]Consolidated data saved to: {consolidated_path}[/green]"
            )
        return 0

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        if "--debug" in sys.argv:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return 1


if __name__ == "__main__":
    if len(sys.argv) == 1:
        console.print(
            Panel(
                "[bold]Sports Table Extractor[/bold]\n\n"
                "Extract league table data from images\n\n"
                "[yellow]Usage:[/yellow] uv run futuro.py image.png\n"
                "[yellow]Setup:[/yellow] export GEMINI_API_KEY='key'\n"
                "Get key: https://aistudio.google.com/apikey",
                title="🏆 Futuro",
                border_style="cyan",
            )
        )
        sys.exit(0)
    sys.exit(main())
