# /// script
# dependencies = [
#     "google-genai>=1.1.0",
#     "rich>=13.7.0",
#     "pydantic>=2.0.0",
# ]
# ///

"""
Sports Table Extractor - Advanced image analysis agent for extracting football league table data.

This single-file agent uses Google's Gemini AI model to extract structured data from images of
football league tables, focusing specifically on the last 10 games performance statistics.
The tool processes images with the naming format '{country}-{match_type}.png' and extracts
performance metrics like expected goals (xG) and other advanced statistics.

Features:
- Two-phase extraction: data extraction followed by self-validation
- Multi-image processing with consolidated output
- Support for different match types (all/home/away)
- Advanced error handling and reflection capabilities
- Rich console output with confidence scores

Usage: 
    uv run futuro.py image1.png image2.png image3.png
"""

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
    """Team performance data for the last 10 matches."""
    team: str = Field(description="Team name")
    games: int = Field(description="Games played")
    goal_diff: int = Field(description="Goal difference")
    xG: float = Field(description="Expected Goals")
    xGA: float = Field(description="Expected Goals Against")
    net_xG: float = Field(description="Net Expected Goals")
    xG_pts: int = Field(description="Expected Goal Points")


class SportsTableData(BaseModel):
    """Sports table data extracted from an image, containing team performance data for the last 10 matches."""
    country: str = Field(description="Country/league (e.g., italy, spain, england)")
    match_type: str = Field(default="all", description="Match type: all, home, or away")
    teams: list[TeamData] = Field(description="List of team data")

    @field_validator("teams")
    def validate_teams(cls, v):
        if len(v) < 10:
            console.print("[yellow]Warning: Less than 10 teams detected.[/yellow]")
        return v


class ReflectionResult(BaseModel):
    """Self-reflection assessment on the quality of the data extraction."""
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
    """Team performance statistics for a specific match type."""
    games: int = Field(description="Games played")
    goal_diff: int = Field(description="Goal difference")
    xG: float = Field(description="Expected Goals")
    xGA: float = Field(description="Expected Goals Against")
    net_xG: float = Field(description="Net Expected Goals")
    xG_pts: int = Field(description="Expected Goal Points")


class TeamPerformance(BaseModel):
    """Team performance across different match types (all, home, away)."""
    all: TeamStats = Field(description="All matches performance")
    home: TeamStats = Field(description="Home matches performance")
    away: TeamStats = Field(description="Away matches performance")


class ConsolidatedMetadata(BaseModel):
    """Metadata for consolidated output, including extraction timestamp and field definitions."""
    timestamp: str = Field(description="Extraction timestamp")
    country: str = Field(description="Country/league identifier")
    definitions: dict = Field(description="Field definitions and explanations")


class ConsolidatedOutput(BaseModel):
    """Consolidated output combining data from multiple extractions."""
    metadata: ConsolidatedMetadata = Field(description="Extraction metadata")
    data: dict[str, TeamPerformance] = Field(description="Team performance data indexed by team name")


class SportsTableExtractor:
    """Core extraction engine for processing sports table images using Gemini AI.
    
    This class handles the image analysis, data extraction, and validation workflow,
    leveraging Google's Gemini 2.5 Pro model for image understanding.
    """
    
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
        """Detect the MIME type of an image file.
        
        Currently only supports PNG files.
        """
        if Path(image_path).suffix.lower() != ".png":
            raise ValueError(
                f"Only PNG files supported. Got '{Path(image_path).suffix}'"
            )
        return "image/png"

    def _create_extraction_prompt(self, filename_info: dict) -> str:
        context = ""
        if filename_info:
            context = f"""<filename_context>
Filename suggests: Country={filename_info.get("country", "unknown")}, Type={filename_info.get("match_type", "unknown")}
</filename_context>
"""

        return f"""<purpose>Extract sports table data (last 10 games) to JSON.</purpose>
{context}
<instructions>
Analyze step-by-step, then provide JSON:
1. Identify columns and teams
2. Extract data systematically
3. Verify extraction

JSON format:
```json
{{"country": string, "match_type": string, "teams": [{{"team": string, "games": int, "goal_diff": int, "xG": float, "xGA": float, "net_xG": float, "xG_pts": int}}]}}
```
</instructions>

<requirements>
- country/match_type from filename, NOT image
- Data should represent LAST 10 GAMES only
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

Note: country/match_type from filename (not image) - don't flag as issues. All data should be for last 10 games only.

Provide assessment:
```json
{{"is_accurate": bool, "confidence_score": float, "issues_found": ["issues"], "suggested_corrections": ["fixes"], "final_assessment": "summary"}}
```
</instructions>"""

    def _parse_filename(self, image_path: str) -> dict:
        parts = Path(image_path).stem.split("-")
        return (
            {
                "country": parts[0].lower(),
                "match_type": parts[1].lower() if len(parts) >= 2 else "all",
            }
            if parts
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
            "match_types": {
                "all": "All matches (home and away combined)",
                "home": "Home matches only",
                "away": "Away matches only",
            },
            "statistics": {
                "games": "Number of games played in last 10 matches",
                "goal_diff": "Goal difference in last 10 matches (goals scored minus goals conceded)",
                "xG": "Expected Goals in last 10 matches - likelihood of scoring based on shot quality",
                "xGA": "Expected Goals Against in last 10 matches - likelihood of conceding based on shots faced",
                "net_xG": "Net Expected Goals in last 10 matches (xG minus xGA)",
                "xG_pts": "Expected Goal Points in last 10 matches - points a team would have based on xG performance",
            },
        }

    def consolidate_extractions(
        self, extractions: list[SportsTableData]
    ) -> ConsolidatedOutput:
        if not extractions:
            raise ValueError("No extractions provided")

        country = extractions[0].country
        if any(e.country != country for e in extractions):
            mixed = next(e.country for e in extractions if e.country != country)
            raise ValueError(f"Mixed countries found: {country} vs {mixed}")

        extractions_by_type = {e.match_type: e for e in extractions}

        all_teams = {team.team for extraction in extractions for team in extraction.teams}
        team_performances = {}

        default_stats = TeamStats(games=0, goal_diff=0, xG=0.0, xGA=0.0, net_xG=0.0, xG_pts=0)
        
        for team_name in all_teams:
            stats = {"all": default_stats, "home": default_stats, "away": default_stats}
            
            for match_type, extraction in extractions_by_type.items():
                if extraction and (team_data := next((t for t in extraction.teams if t.team == team_name), None)):
                    stats[match_type] = TeamStats(
                        games=team_data.games,
                        goal_diff=team_data.goal_diff,
                        xG=team_data.xG,
                        xGA=team_data.xGA,
                        net_xG=team_data.net_xG,
                        xG_pts=team_data.xG_pts,
                    )
            
            team_performances[team_name] = TeamPerformance(**stats)

        metadata = ConsolidatedMetadata(
            timestamp=datetime.now().isoformat(),
            country=country,
            definitions=self._create_metadata_definitions(),
        )

        return ConsolidatedOutput(metadata=metadata, data=team_performances)


def validate_environment() -> str:
    """Validate environment variables and return API key if available."""
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




def main() -> int:
    """Process sports table images and extract structured data using Gemini AI.
    
    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="Sports Table Extractor - Extract league table data from images",
        epilog="Example: uv run futuro.py italy-all.png italy-home.png italy-away.png",
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
        console.print(f"[cyan]Processing {len(image_paths)} image{'s' if len(image_paths) > 1 else ''}...[/cyan]")
        
        for i, path in enumerate(image_paths, 1):
            console.print(f"  {i}. {path.name}")

        def process_image(path: Path) -> Tuple[Path, SportsTableData, ReflectionResult]:
            console.print(f"[blue]Processing: {path.name}[/blue]")
            data, reflection = extractor.extract_data(str(path))
            console.print(f"[green]✅ {path.name} ({len(data.teams)} teams)[/green]")
            return path, data, reflection

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

        console.print("\n[cyan]📊 Creating consolidated output...[/cyan]")
        extractions = [data for _, data, _ in results]
        consolidated = extractor.consolidate_extractions(extractions)
        consolidated_path = f"{extractions[0].country}.json"
        extractor.save_consolidated_to_file(consolidated, consolidated_path)

        total_teams = sum(len(data.teams) for data in extractions)
        console.print("\n[cyan]📊 Summary:[/cyan]")
        for path, data, _ in results:
            console.print(f"• {path.name}: {len(data.teams)} teams")
        console.print(f"\n[green]🎉 Processed {len(results)} images, {total_teams} total teams![/green]")
        console.print(f"[green]Consolidated data saved to: {consolidated_path}[/green]")
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
                "[yellow]Usage:[/yellow] uv run futuro.py images/*.png\n"
                "[yellow]Setup:[/yellow] export GEMINI_API_KEY='key'\n"
                "Get key: https://aistudio.google.com/apikey",
                title="🏆 Futuro",
                border_style="cyan",
            )
        )
        sys.exit(0)
    sys.exit(main())
