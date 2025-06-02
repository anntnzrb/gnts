# /// script
# dependencies = [
#     "rich>=13.7.0",
#     "pydantic>=2.0.0",
# ]
# ///

"""
Team Comparison Tool - Generate focused matchup comparisons from consolidated sports data.

This tool reads consolidated JSON output from futuro.py and creates focused 2-team comparisons
with interactive selection and home/away context. Generates self-contained comparison files
suitable for analysis and sharing.

Features:
- Interactive team selection with Rich UI
- Home/Away assignment with preview
- Robust file naming with collision prevention
- Data validation and completeness checking
- Self-contained output files

Usage: 
    uv run futuro_compare.py consolidated_data.json
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from pydantic import BaseModel, Field

# Define models inline to avoid import issues in uv script mode
# These mirror the models from futuro.py

class TeamStats(BaseModel):
    """Team performance statistics for a specific match type."""
    games: int = Field(description="Games played in the analyzed period")
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

console = Console()


class MatchupInfo(BaseModel):
    """Matchup-specific information for team comparisons."""
    home_team: str = Field(description="Team playing at home")
    away_team: str = Field(description="Team playing away")
    home_away_context: dict[str, str] = Field(description="Maps each team to their role: 'home' or 'away'")


class ComparisonMetadata(BaseModel):
    """Metadata for comparison output, extends ConsolidatedMetadata."""
    timestamp: str = Field(description="Comparison creation timestamp")
    country: str = Field(description="Country/league identifier")
    matchup: MatchupInfo = Field(description="Matchup information")
    definitions: dict = Field(description="Field definitions and explanations")


class ComparisonOutput(BaseModel):
    """Comparison output containing metadata and filtered team data."""
    metadata: ComparisonMetadata = Field(description="Comparison metadata")
    data: dict[str, TeamPerformance] = Field(description="Team performance data for selected teams only")


def load_consolidated_data(file_path: str) -> ConsolidatedOutput:
    """Load and validate consolidated JSON file."""
    try:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ConsolidatedOutput(**data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load data: {e}")


def validate_team_count(data: ConsolidatedOutput) -> None:
    """Ensure at least 2 teams available for comparison."""
    if len(data.data) < 2:
        raise ValueError(f"Need at least 2 teams for comparison, found {len(data.data)}")


def validate_team_data_completeness(data: ConsolidatedOutput, teams: list[str]) -> None:
    """Ensure selected teams have complete home/away data."""
    warnings = [f"{team}: Missing {stat_type} statistics" 
                for team in teams for stat_type in ["home", "away"] 
                if getattr(data.data[team], stat_type).games == 0]
    
    if warnings:
        console.print("[yellow]⚠️  Data Completeness Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")
        
        if Prompt.ask("\nProceed with incomplete data?", choices=["y", "n"], default="n") == "n":
            console.print("[yellow]Comparison cancelled.[/yellow]")
            sys.exit(0)


def truncate_team_name(name: str, max_length: int = 30) -> str:
    """Truncate team name for display if too long."""
    if len(name) <= max_length:
        return name
    return name[:max_length-3] + "..."


def create_team_table(teams: list[str]) -> Table:
    """Create a formatted table for team selection."""
    table = Table(show_header=False, box=None)
    table.add_column("Index", style="dim")
    table.add_column("Team", style="bold")
    
    for i, team in enumerate(teams, 1):
        table.add_row(f"{i:2d}.", truncate_team_name(team))
    
    return table


def get_team_choice(teams: list[str], prompt_text: str) -> str:
    """Get valid team choice from user with error handling."""
    while True:
        try:
            choice = Prompt.ask(f"\n→ Enter choice (1-{len(teams)})")
            idx = int(choice) - 1
            if 0 <= idx < len(teams):
                return teams[idx]
            console.print(f"[red]Please enter a number between 1 and {len(teams)}[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def select_teams(teams: list[str]) -> tuple[str, str]:
    """Interactive selection of two teams using rich menus."""
    console.print(f"\n📂 Loaded teams: {len(teams)} available")
    
    console.print("\n[cyan]Select first team:[/cyan]")
    console.print(create_team_table(teams))
    team1 = get_team_choice(teams, "first team")
    
    remaining_teams = [t for t in teams if t != team1]
    console.print("\n[cyan]Select second team:[/cyan]")
    console.print(create_team_table(remaining_teams))
    team2 = get_team_choice(remaining_teams, "second team")
    
    return team1, team2


def select_home_team(team1: str, team2: str) -> tuple[str, str]:
    """Let user choose which team plays home."""
    console.print("\n[cyan]Select home team for matchup:[/cyan]")
    
    options = [
        f"{truncate_team_name(team1)} (home) vs {truncate_team_name(team2)} (away)",
        f"{truncate_team_name(team2)} (home) vs {truncate_team_name(team1)} (away)"
    ]
    
    for i, option in enumerate(options, 1):
        console.print(f" {i}. {option}")
    
    while True:
        try:
            choice = Prompt.ask("\n→ Enter choice (1-2)")
            idx = int(choice) - 1
            if idx == 0:
                return team1, team2  # team1 home, team2 away
            elif idx == 1:
                return team2, team1  # team2 home, team1 away
            else:
                console.print("[red]Please enter 1 or 2[/red]")
        except ValueError:
            console.print("[red]Please enter a valid number[/red]")


def sanitize_team_name(name: str, max_length: int = 20) -> str:
    """Sanitize team name for filename with robust rules."""
    sanitized = re.sub(r'[^\w\-]', '-', name.lower())
    sanitized = re.sub(r'-+', '-', sanitized).strip('-')
    
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length].rstrip('-')
    
    return sanitized or "team"


def generate_filename(country: str, home_team: str, away_team: str) -> str:
    """Generate clean filename for comparison output."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [sanitize_team_name(country), sanitize_team_name(home_team), 
             "vs", sanitize_team_name(away_team), timestamp]
    
    base_filename = f"{'_'.join(parts)}.json"
    
    # Handle collisions
    path = Path(base_filename)
    counter = 1
    while path.exists():
        name_without_ext = base_filename.rsplit('.', 1)[0]
        base_filename = f"{name_without_ext}_{counter}.json"
        path = Path(base_filename)
        counter += 1
    
    return base_filename


def show_preview(data: ConsolidatedOutput, home_team: str, away_team: str) -> None:
    """Show preview of team stats before creating comparison."""
    console.print("\n[cyan]📊 Preview:[/cyan]")
    
    for team, role in [(home_team, "home"), (away_team, "away")]:
        team_data = data.data[team]
        stats = getattr(team_data, role) if getattr(team_data, role).games > 0 else team_data.all
        console.print(f"  {truncate_team_name(team)} ({role}): {stats.games} games, {stats.goal_diff:+d} goal diff, {stats.net_xG:+.1f} net xG")


def create_comparison_data(
    original: ConsolidatedOutput, 
    home_team: str, 
    away_team: str
) -> ComparisonOutput:
    """Create filtered comparison output."""
    matchup_info = MatchupInfo(
        home_team=home_team,
        away_team=away_team,
        home_away_context={home_team: "home", away_team: "away"}
    )
    
    comparison_metadata = ComparisonMetadata(
        timestamp=datetime.now().isoformat(),
        country=original.metadata.country,
        matchup=matchup_info,
        definitions=original.metadata.definitions
    )
    
    return ComparisonOutput(
        metadata=comparison_metadata,
        data={home_team: original.data[home_team], away_team: original.data[away_team]}
    )


def save_comparison(data: ComparisonOutput, filename: str) -> None:
    """Save comparison data to JSON file."""
    try:
        path = Path(filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data.model_dump(), f, indent=2, ensure_ascii=False)
        console.print(f"[green]💾 Saved: {filename}[/green]")
    except Exception as e:
        console.print(f"[red]Error saving file: {e}[/red]")
        sys.exit(1)


def main() -> int:
    """Team comparison tool main function."""
    parser = argparse.ArgumentParser(
        description="Team Comparison Tool - Generate focused matchup comparisons",
        epilog="Example: uv run futuro_compare.py ecuador.json",
    )
    parser.add_argument(
        "input_file", 
        help="Path to consolidated JSON file (from futuro.py output)"
    )
    args = parser.parse_args()
    
    try:
        # Display header
        console.print(
            Panel(
                "[bold]Team Comparison Tool[/bold]\n\n"
                "Generate focused matchup comparisons from consolidated data",
                title="📊 Futuro Compare",
                border_style="cyan",
            )
        )
        
        # Load and validate data
        console.print(f"[cyan]Loading data from: {args.input_file}[/cyan]")
        data = load_consolidated_data(args.input_file)
        validate_team_count(data)
        
        # Team selection
        teams = sorted(data.data.keys())
        team1, team2 = select_teams(teams)
        
        # Validate data completeness
        validate_team_data_completeness(data, [team1, team2])
        
        # Home/away selection
        home_team, away_team = select_home_team(team1, team2)
        
        # Show preview
        show_preview(data, home_team, away_team)
        
        # Create comparison
        console.print(f"\n✅ Creating matchup: {truncate_team_name(home_team)} (home) vs {truncate_team_name(away_team)} (away)")
        comparison = create_comparison_data(data, home_team, away_team)
        
        # Generate filename and save
        filename = generate_filename(data.metadata.country, home_team, away_team)
        save_comparison(comparison, filename)
        
        return 0
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        return 1
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
                "[bold]Team Comparison Tool[/bold]\n\n"
                "Generate focused matchup comparisons from consolidated data\n\n"
                "[yellow]Usage:[/yellow] uv run futuro_compare.py input.json\n"
                "[yellow]Example:[/yellow] uv run futuro_compare.py ecuador.json",
                title="📊 Futuro Compare",
                border_style="cyan",
            )
        )
        sys.exit(0)
    sys.exit(main())