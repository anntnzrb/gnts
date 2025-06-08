# /// script
# dependencies = [
#     "typer>=0.9.0",
#     "rich>=13.7.0",
# ]
# ///

"""
Betting Expert Agent - xG-based betting recommendations across 9 core markets
Interactive team selection with mathematical probability calculations
"""

import json
import math
from dataclasses import dataclass
from itertools import chain
from typing import List, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(help="Betting Expert Agent - xG Based Analysis")


class BettingAgentError(Exception):
    pass


class DataValidationError(BettingAgentError):
    pass


class BettingConfig:
    """Configuration constants for betting thresholds and parameters."""

    STRONG_CONFIDENCE_THRESHOLD = 0.75
    MEDIUM_CONFIDENCE_THRESHOLD = 0.60
    WEAK_CONFIDENCE_THRESHOLD = 0.52
    BTTS_SCORING_THRESHOLD = 0.7
    CLEAN_SHEET_THRESHOLD = 0.8
    WIN_TO_NIL_THRESHOLD = 0.65
    TOTAL_GOALS_LINES = [1.5, 2.5, 3.5]
    HANDICAP_EDGE_THRESHOLD = 0.3


@dataclass
class TeamStats:
    """Statistical data for a team across multiple games."""

    games: int
    goal_diff: int
    xG: float
    xGA: float
    net_xG: float
    xG_pts: int

    @property
    def xG_per_game(self) -> float:
        return self.xG / self.games if self.games > 0 else 0.0

    @property
    def xGA_per_game(self) -> float:
        return self.xGA / self.games if self.games > 0 else 0.0

    @property
    def net_xG_per_game(self) -> float:
        return self.net_xG / self.games if self.games > 0 else 0.0


@dataclass
class TeamData:
    """Complete team data including overall, home, and away statistics."""

    name: str
    all_stats: TeamStats
    home_stats: TeamStats
    away_stats: TeamStats


@dataclass
class MarketRecommendation:
    """Betting recommendation for a specific market with confidence metrics."""

    market: str
    recommendation: str
    confidence: float
    reasoning: str
    calculated_probability: float

    @property
    def confidence_level(self) -> str:
        thresholds = [
            (BettingConfig.STRONG_CONFIDENCE_THRESHOLD, "Strong"),
            (BettingConfig.MEDIUM_CONFIDENCE_THRESHOLD, "Medium"),
            (0, "Weak"),
        ]
        return next(
            level for threshold, level in thresholds if self.confidence >= threshold
        )


class DataProcessor:
    """Handles loading and processing of team data from JSON files."""

    def load_json(self, file_path: str) -> dict:
        """Load and validate JSON data file."""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            if "data" not in data:
                raise DataValidationError("JSON file missing 'data' section")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            raise DataValidationError(f"Invalid JSON format: {e}")

    def extract_team_data(self, team_name: str, data: dict) -> TeamData:
        """Extract and structure team data from raw JSON."""
        team_raw = data["data"][team_name]
        return TeamData(
            name=team_name,
            all_stats=TeamStats(**team_raw["all"]),
            home_stats=TeamStats(**team_raw["home"]),
            away_stats=TeamStats(**team_raw["away"]),
        )

    def calculate_per_game_rates(self, stats: TeamStats) -> dict:
        """Calculate per-game statistical rates for a team."""
        if stats.games == 0:
            return {
                "xG_per_game": 0.0,
                "xGA_per_game": 0.0,
                "net_xG_per_game": 0.0,
                "goal_diff_per_game": 0.0,
            }

        return {
            "xG_per_game": stats.xG / stats.games,
            "xGA_per_game": stats.xGA / stats.games,
            "net_xG_per_game": stats.net_xG / stats.games,
            "goal_diff_per_game": stats.goal_diff / stats.games,
        }


class MatchAnalyzer:
    """Analyzes match data to calculate expected goals and team strengths."""

    def __init__(self, home_team_data: TeamData, away_team_data: TeamData):
        self.home_team = home_team_data
        self.away_team = away_team_data

    def get_contextual_stats(self, team_data: TeamData, is_home: bool) -> TeamStats:
        """Get team statistics based on home/away context."""
        return team_data.home_stats if is_home else team_data.away_stats

    def calculate_expected_goals(self) -> Tuple[float, float]:
        """Calculate expected goals for both teams in their respective contexts."""
        return (
            self.home_team.home_stats.xG_per_game,
            self.away_team.away_stats.xG_per_game,
        )

    def calculate_total_expected_goals(self) -> float:
        """Calculate total expected goals for the match."""
        return sum(self.calculate_expected_goals())

    def calculate_team_strengths(self) -> dict:
        """Calculate relative team strengths based on net xG."""
        home_net_xG = self.home_team.home_stats.net_xG_per_game
        away_net_xG = self.away_team.away_stats.net_xG_per_game
        return {
            "home_strength": home_net_xG,
            "away_strength": away_net_xG,
            "differential": home_net_xG - away_net_xG,
        }


def xG_to_scoring_probability(xG_rate: float) -> float:
    """Convert expected goals rate to scoring probability using Poisson distribution."""
    return 1 - math.exp(-xG_rate)


def calculate_win_probabilities(home_xG: float, away_xG: float) -> dict:
    """Calculate match outcome probabilities based on expected goals."""
    if home_xG + away_xG == 0:
        return {"home": 0.33, "draw": 0.33, "away": 0.33}

    xG_diff, total_xG = home_xG - away_xG, home_xG + away_xG
    multiplier = 0.27 if abs(xG_diff) > 1.5 else 0.24 if abs(xG_diff) > 0.8 else 0.20

    home_prob = max(0.15, min(0.8, 0.5 + xG_diff * multiplier))
    away_prob = max(0.15, min(0.8, 0.5 - xG_diff * multiplier))
    base_draw_prob = max(0.12, min(0.25, 0.30 - total_xG * 0.06))

    total = home_prob + away_prob + base_draw_prob
    return {
        "home": home_prob / total,
        "draw": base_draw_prob / total,
        "away": away_prob / total,
    }


class MatchWinnerCalculator:
    def generate_recommendation(
        self, probabilities: dict, home_team: str, away_team: str
    ) -> MarketRecommendation:
        home_prob, away_prob = probabilities["home"], probabilities["away"]

        if home_prob > max(away_prob, probabilities["draw"]) + 0.1:
            return MarketRecommendation(
                market="Match Winner",
                recommendation=f"Back {home_team}",
                confidence=home_prob,
                reasoning=f"Home team favored with {home_prob:.1%} probability",
                calculated_probability=home_prob,
            )
        elif away_prob > max(home_prob, probabilities["draw"]) + 0.1:
            return MarketRecommendation(
                market="Match Winner",
                recommendation=f"Back {away_team}",
                confidence=away_prob,
                reasoning=f"Away team favored with {away_prob:.1%} probability",
                calculated_probability=away_prob,
            )
        else:
            return MarketRecommendation(
                market="Match Winner",
                recommendation="Skip",
                confidence=0.45,
                reasoning="No clear favorite - too close to call",
                calculated_probability=max(home_prob, away_prob),
            )


class OverUnderGoalsCalculator:
    def _calculate_probabilities(
        self, expected_total: float, line: float = 2.5
    ) -> dict:
        def poisson_prob(k, lambda_val):
            if k > 20:
                return 0
            return (lambda_val**k) * math.exp(-lambda_val) / math.factorial(k)

        def cumulative_poisson_prob(n, lambda_val):
            return sum(poisson_prob(k, lambda_val) for k in range(n + 1))

        if line == int(line):
            under_prob = cumulative_poisson_prob(int(line), expected_total)
        else:
            under_prob = cumulative_poisson_prob(int(line), expected_total)

        over_prob = 1 - under_prob

        over_prob = max(0.05, min(0.95, over_prob))
        under_prob = 1 - over_prob

        return {"over": over_prob, "under": under_prob}

    def recommend_over_under(
        self, expected_total: float, line: float = 2.5
    ) -> Tuple[MarketRecommendation, MarketRecommendation]:
        probabilities = self._calculate_probabilities(expected_total, line)

        def create_recommendation(
            market: str, condition: bool, rec_type: str
        ) -> MarketRecommendation:
            return MarketRecommendation(
                market=market,
                recommendation=f"{rec_type} {line}" if condition else "Skip",
                confidence=probabilities[rec_type.lower()] if condition else 0.45,
                reasoning=f"Expected total {expected_total:.2f} {'favors' if condition else "doesn't favor"} {rec_type.lower()} {line}",
                calculated_probability=probabilities[rec_type.lower()],
            )

        over_rec = create_recommendation(
            "Over Goals", expected_total > line + 0.2, "Over"
        )
        under_rec = create_recommendation(
            "Under Goals", expected_total < line - 0.2, "Under"
        )

        return over_rec, under_rec

    def generate_goals_analysis(self, expected_total: float) -> list:
        """Generate comprehensive over/under analysis for multiple goal lines using functional patterns."""

        def analyze_line(line: float) -> dict:
            probs = self._calculate_probabilities(expected_total, line)

            def confidence_level(p: float) -> str:
                return "Strong" if p > 0.65 else "Medium" if p > 0.55 else "Weak"

            best_bet, best_prob = max(probs.items(), key=lambda x: x[1])

            return {
                "line": line,
                "over_prob": probs["over"],
                "under_prob": probs["under"],
                "over_confidence": confidence_level(probs["over"]),
                "under_confidence": confidence_level(probs["under"]),
                "best_bet": best_bet.title(),
                "best_prob": best_prob,
                "best_confidence": confidence_level(best_prob),
                "expected_total": expected_total,
            }

        return list(map(analyze_line, [0.5 + i*0.25 for i in range(18)]))


class BothTeamsScoreCalculator:
    def calculate_btts_probability(self, home_xG: float, away_xG: float) -> dict:
        home_scores_prob = xG_to_scoring_probability(home_xG)
        away_scores_prob = xG_to_scoring_probability(away_xG)
        btts_yes_prob = home_scores_prob * away_scores_prob

        return {"yes": btts_yes_prob, "no": 1 - btts_yes_prob}

    def generate_recommendation(
        self, home_xG: float, away_xG: float, home_xGA: float, away_xGA: float
    ) -> MarketRecommendation:
        probabilities = self.calculate_btts_probability(home_xG, away_xG)

        if probabilities["yes"] > BettingConfig.MEDIUM_CONFIDENCE_THRESHOLD:
            return MarketRecommendation(
                market="Both Teams Score",
                recommendation="Yes",
                confidence=probabilities["yes"],
                reasoning="Both teams have good scoring rates",
                calculated_probability=probabilities["yes"],
            )
        elif probabilities["no"] > BettingConfig.MEDIUM_CONFIDENCE_THRESHOLD:
            return MarketRecommendation(
                market="Both Teams Score",
                recommendation="No",
                confidence=probabilities["no"],
                reasoning="At least one team unlikely to score",
                calculated_probability=probabilities["no"],
            )
        else:
            return MarketRecommendation(
                market="Both Teams Score",
                recommendation="Skip",
                confidence=0.45,
                reasoning="BTTS probability too uncertain",
                calculated_probability=max(probabilities["yes"], probabilities["no"]),
            )


class AsianHandicapCalculator:
    def calculate_handicap_edge(self, net_xG_diff: float) -> dict:
        if abs(net_xG_diff) < BettingConfig.HANDICAP_EDGE_THRESHOLD:
            return {"edge": False, "team": None, "confidence": 0.45}

        team = "home" if net_xG_diff > BettingConfig.HANDICAP_EDGE_THRESHOLD else "away"
        confidence = min(0.8, 0.6 + abs(net_xG_diff) * 0.1)

        return {"edge": True, "team": team, "confidence": confidence}

    def calculate_handicap_probability(self, effective_advantage: float) -> float:
        """Calculate probability based on effective advantage after handicap"""
        base_prob = 0.5 + (effective_advantage * 0.12)
        return max(0.15, min(0.85, base_prob))

    def generate_handicap_analysis(
        self, net_xG_diff: float, home_team: str, away_team: str
    ) -> list:
        """Generate comprehensive handicap analysis from -2.0 to +2.0 using functional patterns."""

        def analyze_handicap(handicap: float) -> dict:
            team = home_team if handicap <= 0 else away_team
            display_handicap = f"{handicap:+.2f}" if handicap != 0 else "0.00"
            effective_advantage = (
                net_xG_diff if handicap <= 0 else -net_xG_diff
            ) + handicap
            prob = self.calculate_handicap_probability(effective_advantage)
            confidence = (
                "Strong" if prob > 0.65 else "Medium" if prob > 0.55 else "Weak"
            )

            return {
                "handicap": display_handicap,
                "team": team,
                "probability": prob,
                "confidence": confidence,
            }

        handicaps = [-5.0 + i*0.25 for i in range(41)]
        return list(map(analyze_handicap, handicaps))

    def recommend_handicap_bet(
        self, net_xG_diff: float, home_team: str, away_team: str
    ) -> MarketRecommendation:
        edge_data = self.calculate_handicap_edge(net_xG_diff)

        if not edge_data["edge"]:
            return MarketRecommendation(
                market="Asian Handicap",
                recommendation="Skip",
                confidence=0.45,
                reasoning="Teams too evenly matched",
                calculated_probability=0.5,
            )

        if edge_data["team"] == "home":
            handicap = -1 if net_xG_diff > 0.6 else -0.5
            effective_advantage = net_xG_diff + handicap
            prob = self.calculate_handicap_probability(effective_advantage)

            if prob < 0.50:
                return MarketRecommendation(
                    market="Asian Handicap",
                    recommendation="Skip",
                    confidence=0.45,
                    reasoning="Insufficient edge for reliable recommendation",
                    calculated_probability=prob,
                )

            return MarketRecommendation(
                market="Asian Handicap",
                recommendation=f"{home_team} {handicap}",
                confidence=prob,
                reasoning=f"Home team xG advantage: {net_xG_diff:+.2f}",
                calculated_probability=prob,
            )
        else:
            handicap = +1 if net_xG_diff < -0.6 else +0.5
            effective_advantage = -net_xG_diff + handicap
            prob = self.calculate_handicap_probability(effective_advantage)

            if prob < 0.50:
                return MarketRecommendation(
                    market="Asian Handicap",
                    recommendation="Skip",
                    confidence=0.45,
                    reasoning="Insufficient edge for reliable recommendation",
                    calculated_probability=prob,
                )

            return MarketRecommendation(
                market="Asian Handicap",
                recommendation=f"{away_team} {handicap:+}",
                confidence=prob,
                reasoning=f"Away team xG advantage: {-net_xG_diff:+.2f}",
                calculated_probability=prob,
            )


class DrawNoBetCalculator:
    def calculate_probability(self, match_analyzer: MatchAnalyzer) -> dict:
        home_xG, away_xG = match_analyzer.calculate_expected_goals()
        win_probs = calculate_win_probabilities(home_xG, away_xG)

        total_win_prob = win_probs["home"] + win_probs["away"]
        if total_win_prob > 0:
            return {
                "home": win_probs["home"] / total_win_prob,
                "away": win_probs["away"] / total_win_prob,
            }
        return {"home": 0.5, "away": 0.5}

    def generate_recommendation(
        self, probabilities: dict, home_team: str, away_team: str
    ) -> MarketRecommendation:
        home_prob, away_prob = probabilities["home"], probabilities["away"]

        if home_prob > away_prob + 0.15:
            return MarketRecommendation(
                market="Draw No Bet",
                recommendation=f"Back {home_team}",
                confidence=home_prob,
                reasoning="Home team favored, draw risk removed",
                calculated_probability=home_prob,
            )
        elif away_prob > home_prob + 0.15:
            return MarketRecommendation(
                market="Draw No Bet",
                recommendation=f"Back {away_team}",
                confidence=away_prob,
                reasoning="Away team favored, draw risk removed",
                calculated_probability=away_prob,
            )
        else:
            return MarketRecommendation(
                market="Draw No Bet",
                recommendation="Skip",
                confidence=0.45,
                reasoning="Teams too evenly matched for DNB value",
                calculated_probability=max(home_prob, away_prob),
            )


class DoubleChanceCalculator:
    def calculate_double_chance_probabilities(self, win_probs: dict) -> dict:
        return {
            "home_or_draw": win_probs["home"] + win_probs["draw"],
            "away_or_draw": win_probs["away"] + win_probs["draw"],
            "home_or_away": win_probs["home"] + win_probs["away"],
        }

    def recommend_double_chance(
        self, win_probs: dict, home_team: str, away_team: str
    ) -> MarketRecommendation:
        dc_probs = self.calculate_double_chance_probabilities(win_probs)
        max_prob = max(dc_probs.values())

        if max_prob < 0.7:
            return MarketRecommendation(
                market="Double Chance",
                recommendation="Skip",
                confidence=0.45,
                reasoning="No value in double chance markets",
                calculated_probability=max_prob,
            )

        recommendations = {
            "home_or_draw": f"{home_team} or Draw",
            "away_or_draw": f"{away_team} or Draw",
            "home_or_away": "Either Team to Win",
        }

        best_option = max(dc_probs, key=dc_probs.get)

        return MarketRecommendation(
            market="Double Chance",
            recommendation=recommendations[best_option],
            confidence=max_prob,
            reasoning=f"High probability ({max_prob:.1%}) covering scenarios",
            calculated_probability=max_prob,
        )


class CleanSheetCalculator:
    def calculate_clean_sheet_probability(self, team_xGA_rate: float) -> float:
        return math.exp(-team_xGA_rate)

    def recommend_clean_sheet_bets(
        self, home_xGA: float, away_xGA: float, home_team: str, away_team: str
    ) -> MarketRecommendation:
        home_cs_prob = self.calculate_clean_sheet_probability(away_xGA)
        away_cs_prob = self.calculate_clean_sheet_probability(home_xGA)

        if home_cs_prob > BettingConfig.MEDIUM_CONFIDENCE_THRESHOLD:
            return MarketRecommendation(
                market="Clean Sheet",
                recommendation=f"{home_team} Yes",
                confidence=home_cs_prob,
                reasoning=f"Away team weak attack ({away_xGA:.2f} xG/game)",
                calculated_probability=home_cs_prob,
            )
        elif away_cs_prob > BettingConfig.MEDIUM_CONFIDENCE_THRESHOLD:
            return MarketRecommendation(
                market="Clean Sheet",
                recommendation=f"{away_team} Yes",
                confidence=away_cs_prob,
                reasoning=f"Home team weak attack ({home_xGA:.2f} xG/game)",
                calculated_probability=away_cs_prob,
            )
        else:
            return MarketRecommendation(
                market="Clean Sheet",
                recommendation="Skip",
                confidence=0.45,
                reasoning="Both teams likely to score",
                calculated_probability=max(home_cs_prob, away_cs_prob),
            )


class WinToNilCalculator:
    def calculate_win_to_nil_probability(
        self, team_xG: float, opponent_xGA: float, win_prob: float
    ) -> float:
        cs_prob = math.exp(-opponent_xGA)
        return win_prob * cs_prob

    def generate_recommendation(
        self,
        home_xG: float,
        away_xG: float,
        home_xGA: float,
        away_xGA: float,
        win_probs: dict,
        home_team: str,
        away_team: str,
    ) -> MarketRecommendation:
        home_wtn_prob = self.calculate_win_to_nil_probability(
            home_xG, away_xGA, win_probs["home"]
        )
        away_wtn_prob = self.calculate_win_to_nil_probability(
            away_xG, home_xGA, win_probs["away"]
        )

        if home_wtn_prob > BettingConfig.WIN_TO_NIL_THRESHOLD:
            return MarketRecommendation(
                market="Win to Nil",
                recommendation=f"{home_team} Win to Nil",
                confidence=home_wtn_prob,
                reasoning="Strong home attack vs weak away attack",
                calculated_probability=home_wtn_prob,
            )
        elif away_wtn_prob > BettingConfig.WIN_TO_NIL_THRESHOLD:
            return MarketRecommendation(
                market="Win to Nil",
                recommendation=f"{away_team} Win to Nil",
                confidence=away_wtn_prob,
                reasoning="Strong away attack vs weak home defense",
                calculated_probability=away_wtn_prob,
            )
        else:
            return MarketRecommendation(
                market="Win to Nil",
                recommendation="Skip",
                confidence=0.45,
                reasoning="Win to nil probability too low",
                calculated_probability=max(home_wtn_prob, away_wtn_prob),
            )


class RecommendationFormatter:
    def __init__(self, console: Console):
        self.console = console

    def display_match_header(self, home_team: str, away_team: str, data_country: str):
        header_text = f"Match: {home_team} vs {away_team} ({data_country.title()})\nAnalysis Date: 2025-06-03"
        panel = Panel(header_text, title="BETTING EXPERT AGENT", border_style="blue")
        self.console.print(panel)

    def display_team_comparison_table(self, home_data: TeamData, away_data: TeamData):
        table = Table(title="TEAM COMPARISON", border_style="blue")

        columns = [
            "Team",
            "Home xG/Game",
            "Home xGA/Game",
            "Away xG/Game",
            "Away xGA/Game",
            "Net xG (Home)",
            "Net xG (Away)",
        ]
        for col in columns:
            table.add_column(col, justify="center" if "xG" in col else "left")

        for team_data in [home_data, away_data]:
            suffix = " (Home)" if team_data == home_data else " (Away)"
            table.add_row(
                f"{team_data.name}{suffix}",
                f"{team_data.home_stats.xG_per_game:.2f}",
                f"{team_data.home_stats.xGA_per_game:.2f}",
                f"{team_data.away_stats.xG_per_game:.2f}",
                f"{team_data.away_stats.xGA_per_game:.2f}",
                f"{team_data.home_stats.net_xG_per_game:+.1f}",
                f"{team_data.away_stats.net_xG_per_game:+.1f}",
            )

        self.console.print(table)

    def display_key_calculations(self, analyzer: MatchAnalyzer):
        home_xG, away_xG = analyzer.calculate_expected_goals()
        total_xG = analyzer.calculate_total_expected_goals()

        calc_text = f"Expected Goals: {analyzer.home_team.name} {home_xG:.2f} - {away_xG:.2f} {analyzer.away_team.name} (Total: {total_xG:.2f})"
        self.console.print(f"\n{calc_text}\n")

    def display_recommendations_table(
        self, recommendations: List[MarketRecommendation]
    ):
        table = Table(title="BETTING RECOMMENDATIONS", border_style="green")

        columns = [
            "Market",
            "Recommendation",
            "Confidence",
            "Calculated Probability",
            "Reasoning",
        ]
        for col in columns:
            table.add_column(
                col,
                justify="center"
                if "Confidence" in col or "Probability" in col
                else "left",
            )

        for rec in recommendations:
            confidence_color = (
                "green"
                if rec.confidence > 0.7
                else "yellow"
                if rec.confidence > 0.6
                else "red"
            )

            table.add_row(
                rec.market,
                rec.recommendation,
                f"[{confidence_color}]{rec.confidence:.0%}[/{confidence_color}]",
                f"{rec.calculated_probability:.1%}",
                rec.reasoning,
            )

        self.console.print(table)

    def display_asian_handicap_table(self, handicap_analysis: list):
        table = Table(title="ASIAN HANDICAP ANALYSIS", border_style="yellow")

        table.add_column("Handicap", justify="center")
        table.add_column("Team", justify="left")
        table.add_column("Probability", justify="center")
        table.add_column("Confidence", justify="center")

        for analysis in handicap_analysis:
            prob = analysis["probability"]
            confidence = analysis["confidence"]

            confidence_color = (
                "green"
                if confidence == "Strong"
                else "yellow"
                if confidence == "Medium"
                else "red"
            )

            table.add_row(
                analysis["handicap"],
                analysis["team"],
                f"{prob:.1%}",
                f"[{confidence_color}]{confidence}[/{confidence_color}]",
            )

        self.console.print(table)

    def display_goals_analysis_table(self, goals_analysis: list):
        table = Table(title="OVER/UNDER GOALS ANALYSIS", border_style="cyan")

        table.add_column("Line", justify="center")
        table.add_column("Over Prob", justify="center")
        table.add_column("Under Prob", justify="center")
        table.add_column("Best Bet", justify="center")
        table.add_column("Best Prob", justify="center")
        table.add_column("Confidence", justify="center")

        for analysis in goals_analysis:
            best_prob = analysis["best_prob"]
            confidence = analysis["best_confidence"]

            confidence_color = (
                "green"
                if confidence == "Strong"
                else "yellow"
                if confidence == "Medium"
                else "red"
            )

            table.add_row(
                f"{analysis['line']}",
                f"{analysis['over_prob']:.1%}",
                f"{analysis['under_prob']:.1%}",
                f"[bold]{analysis['best_bet']} {analysis['line']}[/bold]",
                f"{best_prob:.1%}",
                f"[{confidence_color}]{confidence}[/{confidence_color}]",
            )

        self.console.print(table)

    def display_summary_panel(self, recommendations: List[MarketRecommendation]):
        strong_recs = [
            r
            for r in recommendations
            if r.confidence > BettingConfig.MEDIUM_CONFIDENCE_THRESHOLD
            and r.recommendation != "Skip"
        ]

        summary_text = f"SUMMARY: {len(strong_recs)} strong recommendations found."
        if strong_recs:
            markets = [r.market for r in strong_recs]
            summary_text += f" Focus on: {', '.join(markets[:3])}"

        panel = Panel(summary_text, border_style="green")
        self.console.print(panel)


def interactive_team_selection(available_teams: list) -> Tuple[str, str]:
    """Interactive selection of home and away teams from available options."""
    console.print("\n[bold blue]Available Teams:[/bold blue]")
    for i, team in enumerate(available_teams, 1):
        console.print(f"[cyan]{i:2d}.[/cyan] {team}")
    console.print()

    def get_team_choice(prompt: str, exclude_team: str = None) -> str:
        while True:
            try:
                choice = typer.prompt(prompt)
                idx = int(choice) - 1
                if 0 <= idx < len(available_teams):
                    team = available_teams[idx]
                    if exclude_team and team == exclude_team:
                        console.print(
                            "[red]Away team must be different from home team[/red]"
                        )
                        continue
                    return team
                else:
                    console.print(
                        f"[red]Please enter a number between 1 and {len(available_teams)}[/red]"
                    )
            except ValueError:
                console.print("[red]Please enter a valid number[/red]")

    home_team = get_team_choice("Select HOME team (enter number)")
    away_team = get_team_choice(
        "Select AWAY team (enter number)", exclude_team=home_team
    )
    return home_team, away_team


@app.command()
def analyze(
    json_file: str = typer.Argument(..., help="Path to JSON data file"),
    confidence_threshold: float = typer.Option(
        0.6, help="Minimum confidence for recommendations"
    ),
    show_calculations: bool = typer.Option(False, help="Show detailed calculations"),
):
    """Analyze match and provide betting recommendations with interactive team selection"""

    try:
        processor = DataProcessor()
        data = processor.load_json(json_file)

        available_teams = sorted(list(data["data"].keys()))
        console.print(
            f"\n[bold green]Loaded data for {len(available_teams)} teams from {data.get('metadata', {}).get('country', 'Unknown').title()}[/bold green]"
        )

        home_team, away_team = interactive_team_selection(available_teams)

        console.print(
            f"\n[bold]Selected Match:[/bold] [blue]{home_team}[/blue] vs [red]{away_team}[/red]"
        )
        console.print()

        home_data = processor.extract_team_data(home_team, data)
        away_data = processor.extract_team_data(away_team, data)

        analyzer = MatchAnalyzer(home_data, away_data)

        home_xG, away_xG = analyzer.calculate_expected_goals()
        home_xGA = home_data.home_stats.xGA_per_game
        away_xGA = away_data.away_stats.xGA_per_game
        strengths = analyzer.calculate_team_strengths()
        win_probs = calculate_win_probabilities(home_xG, away_xG)

        calculator_configs = [
            (
                MatchWinnerCalculator(),
                "generate_recommendation",
                (win_probs, home_team, away_team),
            ),
            (
                BothTeamsScoreCalculator(),
                "generate_recommendation",
                (home_xG, away_xG, home_xGA, away_xGA),
            ),
            (
                AsianHandicapCalculator(),
                "recommend_handicap_bet",
                (strengths["differential"], home_team, away_team),
            ),
            (
                CleanSheetCalculator(),
                "recommend_clean_sheet_bets",
                (home_xGA, away_xGA, home_team, away_team),
            ),
            (
                WinToNilCalculator(),
                "generate_recommendation",
                (home_xG, away_xG, home_xGA, away_xGA, win_probs, home_team, away_team),
            ),
            (
                DrawNoBetCalculator(),
                "generate_recommendation",
                (
                    DrawNoBetCalculator().calculate_probability(analyzer),
                    home_team,
                    away_team,
                ),
            ),
            (
                DoubleChanceCalculator(),
                "recommend_double_chance",
                (win_probs, home_team, away_team),
            ),
        ]

        def execute_calculator(config: tuple) -> List[MarketRecommendation]:
            calc, method_name, args = config
            result = getattr(calc, method_name)(*args)
            return result if isinstance(result, tuple) else [result]

        recommendations = list(
            chain.from_iterable(map(execute_calculator, calculator_configs))
        )

        formatter = RecommendationFormatter(console)
        ah_calc = AsianHandicapCalculator()
        ou_calc = OverUnderGoalsCalculator()
        handicap_analysis = ah_calc.generate_handicap_analysis(
            strengths["differential"], home_team, away_team
        )
        goals_analysis = ou_calc.generate_goals_analysis(
            analyzer.calculate_total_expected_goals()
        )

        country = data.get("metadata", {}).get("country", "Unknown")
        formatter.display_match_header(home_team, away_team, country)
        formatter.display_team_comparison_table(home_data, away_data)
        formatter.display_key_calculations(analyzer)
        formatter.display_recommendations_table(recommendations)
        formatter.display_goals_analysis_table(goals_analysis)
        formatter.display_asian_handicap_table(handicap_analysis)
        formatter.display_summary_panel(recommendations)

        if show_calculations:
            console.print("\n[bold]Detailed Calculations:[/bold]")
            console.print(f"Win Probabilities: {win_probs}")
            console.print(f"Team Strengths: {strengths}")

    except FileNotFoundError:
        console.print("[red]Error: JSON file not found[/red]")
        raise typer.Exit(1)
    except DataValidationError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Analysis cancelled by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
