# Futuro Agent - Complete Specification 🎯⚽

*Advanced Soccer Betting Analysis Agent using Expected Goals (xG) Data*

## Table of Contents
- [Executive Summary](#executive-summary)
- [User Workflow & Use Cases](#user-workflow--use-cases)
- [Technical Architecture](#technical-architecture)
- [Data Requirements & Structure](#data-requirements--structure)
- [Betting Markets Framework](#betting-markets-framework)
- [Multi-Step Analysis Pipeline](#multi-step-analysis-pipeline)
- [Risk Assessment Framework](#risk-assessment-framework)
- [Output Specifications](#output-specifications)
- [Implementation Details](#implementation-details)
- [Usage Documentation](#usage-documentation)
- [Expert Betting Persona](#expert-betting-persona)
- [Future Considerations](#future-considerations)

---

## Executive Summary

**Futuro** is a single-file Python agent that analyzes soccer betting opportunities using Expected Goals (xG) data and Gemini's multimodal AI capabilities. The agent processes league-wide xG statistics and upcoming fixtures to identify high-value betting opportunities across 8 specialized markets.

### Core Value Proposition
- **Exhaustive Analysis**: Deep, multi-step reasoning without shortcuts
- **Multimodal Intelligence**: Direct image analysis of xG tables using Gemini 2.5 Pro
- **Focused Output**: Only shows favorable betting opportunities
- **Risk-Aware**: Intelligent confidence and risk assessment
- **League Agnostic**: Works with any soccer league (Premier League, La Liga, etc.)

---

## User Workflow & Use Cases

### Primary Use Case: Weekend Matchday Analysis

```
📅 Friday/Saturday Morning
        ↓
🏟️ Premier League Matchday Coming Up
        ↓
📊 User Gathers 7 Image Files:
   • 6 xG Tables (all 20 teams, different views)
   • 1 Fixtures List (who's playing this weekend)
        ↓
📁 Drop Files into data-xg/ Folder
        ↓
💻 Run: uv run futuro.py
        ↓
📈 Get Betting Recommendations
   • Only for teams playing this weekend
   • Only favorable opportunities shown
   • 8 betting markets analyzed
   • Risk/confidence levels provided
```

### Key Workflow Insights
1. **League-Wide Data → Matchday Focus**: System ingests data for all teams but analyzes only those playing
2. **Weekend Timing**: Designed for pre-matchday preparation
3. **Multi-League Support**: Same workflow works for any league
4. **Simple Execution**: Single command, comprehensive analysis

---

## Technical Architecture

### Single-File Design
- **File**: `futuro.py`
- **Dependencies**: Inline with uv script syntax
- **Model**: Gemini 2.5 Pro (multimodal capabilities)
- **Framework**: Built on Hypatia's (./hypatia/hypatia.py) multi-step reasoning approach

### Required Dependencies
```python
# /// script
# dependencies = [
#     "google-genai>=1.1.0",
#     "rich>=13.7.0", 
#     "pydantic>=2.0.0",
#     "pillow>=10.0.0",  # Image processing support
# ]
# ///
```

### Core Components
1. **Image Analysis Engine** - Gemini 2.5 Pro processing
2. **Data Extraction Layer** - xG metrics parsing
3. **Team Matching System** - Fixture to xG data correlation
4. **Market Analysis Engine** - 8 betting markets evaluation
5. **Risk Assessment Module** - Confidence/risk scoring
6. **Output Formatter** - Rich console tables

---

## Data Requirements & Structure

### Required File Structure
```
data-xg/
├── all-full.png        # Complete season (home + away) - ALL TEAMS
├── home-full.png       # Full season home games only - ALL TEAMS
├── away-full.png       # Full season away games only - ALL TEAMS
├── all-10.png          # Last 10 games (home + away) - ALL TEAMS
├── home-10.png         # Last 10 home games only - ALL TEAMS
├── away-10.png         # Last 10 away games only - ALL TEAMS
└── matches.png         # 🚨 REQUIRED: Upcoming fixtures for matchday
```

### Data Flow Architecture
```
League-Wide xG Data (20 teams) → Fixture Filter → Matchday Analysis
```

### Critical File: matches.png
- **Purpose**: Defines which teams are playing this matchday
- **Content**: Fixture list (Team A vs Team B, dates, times)
- **Role**: Primary filter for analysis scope
- **Format**: Screenshot of upcoming matches

### xG Table Structure Expected
Each xG table should contain standardized metrics:
- **Team Names** (consistent across all tables)
- **Games Played**
- **Expected Goals (xG)**
- **Expected Goals Against (xGA)**
- **Goal Difference vs xG**
- **Net xG** (xG - xGA)
- **Additional metrics** (form indicators, trends)

---

## Betting Markets Framework

### 8 Supported Markets

TODO: Investigate further

| Market | Description | xG Analysis Focus |
|--------|-------------|-------------------|
| **Draw No Bet** | Remove draw risk | Team strength differential |
| **Asian Handicap** | Level playing field | xG gap analysis |
| **Match Winner** | Straight win probability | Overall xG superiority |
| **Over/Under Goals** | Total goals prediction | Combined xG expectation |
| **Both Teams to Score** | Both teams score | Individual xG rates |
| **Double Chance** | Reduced risk combinations | Defensive/attacking balance |
| **Clean Sheet** | Team keeps clean sheet | Defensive xG analysis |
| **Win to Nil** | Win without conceding | Combined offensive/defensive |

### Market Selection Logic
- **Show**: Only markets with clear xG-supported edges
- **Hide**: Markets without statistical advantage
- **Priority**: Strongest xG edges highlighted first

### Market Analysis Methodology
1. **Historical Performance**: Season-long xG patterns
2. **Recent Form**: Last 10 games weighting
3. **Location Factors**: Home vs away performance
4. **Matchup Specific**: Head-to-head xG modeling
5. **Variance Consideration**: Consistency vs volatility
6. **Regression Detection**: Over/under-performing teams

---

## Multi-Step Analysis Pipeline

### Phase 1: Image Processing & Data Extraction 📸
**Objective**: Convert 7 images into structured data

**Process**:
1. **Multimodal Analysis**: Send all 7 images to Gemini 2.5 Pro simultaneously
2. **Data Extraction**: Parse xG metrics for all teams from 6 tables
3. **Fixture Parsing**: Extract matchday fixtures from matches.png
4. **Data Validation**: Ensure consistency across tables
5. **Team Normalization**: Handle name variations/abbreviations

**Output**: Structured database of team performance + fixture list

### Phase 2: Team Performance Profiling 📊
**Objective**: Build comprehensive performance profiles

**Process**:
1. **Season Analysis**: Full season xG performance (home/away/combined)
2. **Form Analysis**: Last 10 games performance trends
3. **Location Splits**: Home vs away performance gaps
4. **Regression Detection**: Teams over/under-performing xG
5. **Trend Identification**: Improving vs declining teams
6. **Consistency Metrics**: Variance in performance

**Output**: Rich team profiles with multiple performance dimensions

### Phase 3: Fixture-Specific Matching 🔗
**Objective**: Connect fixtures with relevant team data

**Process**:
1. **Team Mapping**: Match fixture teams with xG database
2. **Context Selection**: Choose relevant xG data (home team → home stats, etc.)
3. **Missing Data Handling**: Flag incomplete team information
4. **Quality Assurance**: Verify data completeness for analysis

**Output**: Validated fixture-team data mappings

### Phase 4: Statistical Matchup Modeling 📈
**Objective**: Calculate statistical edges for each fixture

**Process**:
1. **xG Gap Analysis**: Quantify attacking vs defensive matchups
2. **Form vs Season**: Weight recent performance vs long-term trends
3. **Location Advantages**: Factor in home field advantages
4. **Variance Assessment**: Consider consistency in performance
5. **Context Weighting**: Adjust for opponent strength
6. **Edge Quantification**: Numerical advantage calculation

**Output**: Statistical edges and advantages for each matchup

### Phase 5: Market Opportunity Detection 🎯
**Objective**: Evaluate all 8 betting markets per fixture

**Process**:
1. **Market-Specific Modeling**: Apply xG data to each betting market
2. **Threshold Analysis**: Determine favorable vs unfavorable opportunities
3. **Edge Quantification**: Calculate strength of statistical advantage
4. **Cross-Market Validation**: Ensure consistency across related markets
5. **Opportunity Ranking**: Prioritize strongest edges

**Output**: Filtered list of favorable betting opportunities

### Phase 6: Risk Assessment & Confidence Scoring 🛡️
**Objective**: Assign risk and confidence levels

**Process**:
1. **Data Strength Assessment**: Quality and quantity of supporting evidence
2. **Sample Size Evaluation**: Games played reliability
3. **Variance Consideration**: Account for potential outliers
4. **Market Efficiency**: Consider how sharp bookmaker pricing might be
5. **Historical Validation**: Back-test similar scenarios
6. **Tier Assignment**: Classify into Tier 1 (Elite) or Tier 2 (Solid Value)

**Output**: Risk/confidence classifications for each opportunity

### Phase 7: Output Generation & Formatting 🎨
**Objective**: Create rich, ADHD-friendly console output

**Process**:
1. **Table Generation**: Create structured tables for each fixture
2. **Visual Hierarchy**: Prioritize strongest opportunities
3. **Insight Generation**: Key statistical insights
4. **Formatting**: Apply colors, emojis, and clear structure
5. **Quality Control**: Ensure output clarity and completeness

**Output**: Formatted betting recommendations ready for user consumption

---

## Risk Assessment Framework

### Risk Levels

| Risk Level | Criteria | Characteristics |
|------------|----------|-----------------|
| **LOW** | Strong xG support + Consistent patterns + Large sample size | "Sleep well at night" bets |
| **MEDIUM** | Good xG backing + Some conflicting indicators + Adequate sample | Solid value with minor concerns |
| **HIGH** | Limited xG support OR conflicting signals OR small sample | Exceptional value required |

### Confidence Levels

| Confidence | Criteria | Decision Making |
|------------|----------|-----------------|
| **HIGH** | Multiple data sources agree + Strong statistical edge + Clear patterns | High conviction plays |
| **MEDIUM** | Good statistical support + Minor contradictions + Reasonable edge | Solid opportunities |
| **LOW** | Weak statistical support OR conflicting data OR marginal edge | Generally avoided |

### Tier Assignment

**Tier 1 (Elite Plays) 💎**
- Risk: LOW + Confidence: HIGH
- Strong xG edge across multiple data sources
- Consistent patterns in both season and recent form
- Clear statistical advantage
- Bankroll allocation: 60-70%

**Tier 2 (Solid Value) ⚡**
- Risk: MEDIUM + Confidence: MEDIUM/HIGH
- Good xG support with minor contradictions
- Decent statistical edge
- Some conflicting indicators but overall positive
- Bankroll allocation: 30-40%

**Hidden (Not Shown)**
- Risk: HIGH OR Confidence: LOW
- Insufficient statistical support
- Conflicting data sources
- Marginal or negative expected value

---

## Output Specifications

### ADHD-Friendly Design Principles
- **Scannable**: Quick visual hierarchy
- **Colorful**: Rich colors and emojis for categorization
- **Structured**: Consistent table formats
- **Focused**: Only show favorable opportunities
- **Concise**: Key insights prominently displayed

### Main Output Format

#### Match Analysis Header
```
╭─────────────────────────────────────────────────────────────────╮
│  🏟️  MATCH ANALYSIS: Manchester City vs Burnley                 │
│  📅  Premier League - Gameweek 38                               │
╰─────────────────────────────────────────────────────────────────╯
```

#### Statistical Overview Table
```
┌─────────────────────┬─────────────┬─────────────┬──────────────┐
│ 📊 Metric           │ Man City    │ Burnley     │ 🔥 Edge      │
├─────────────────────┼─────────────┼─────────────┼──────────────┤
│ Season xG (Home)    │ 2.8         │ 0.9 (Away)  │ +1.9 💪      │
│ Last 10 xG          │ 3.1 📈      │ 0.7 📉      │ +2.4 🔥      │
│ Defensive xGA       │ 0.8         │ 2.1         │ +1.3 🛡️      │
│ Form Trend          │ Improving   │ Declining   │ ✅ Bullish   │
│ Home/Away Factor    │ Strong home │ Weak away   │ ++ Advantage │
└─────────────────────┴─────────────┴─────────────┴──────────────┘
```

#### Betting Opportunities Table
```
╭─────────────────────────────────────────────────────────────────╮
│  💰 BETTING OPPORTUNITIES                                       │
╰─────────────────────────────────────────────────────────────────╯

┌─────┬────────────────────────┬──────────────┬────────┬──────────┐
│ 🏆  │ Market                 │ Recommend.   │ Risk   │ Confid.  │
├─────┼────────────────────────┼──────────────┼────────┼──────────┤
│ 💎  │ Asian Handicap         │ City -2      │ LOW    │ HIGH     │
│ 💎  │ Match Winner           │ City Win     │ LOW    │ HIGH     │
│ 💎  │ Win to Nil             │ City         │ LOW    │ MEDIUM   │
├─────┼────────────────────────┼──────────────┼────────┼──────────┤
│ ⚡  │ Over/Under Goals       │ Over 3.5     │ MEDIUM │ HIGH     │
│ ⚡  │ Draw No Bet            │ City         │ MEDIUM │ MEDIUM   │
└─────┴────────────────────────┴──────────────┴────────┴──────────┘
```

#### Key Insights Box
```
╭─────────────────────────────────────────────────────────────────╮
│  🎯 KEY INSIGHTS                                                │
├─────────────────────────────────────────────────────────────────┤
│  💡 City's attacking xG jumped 0.3 in last 10 games             │
│  🔥 Burnley conceding 0.4 more goals than xG suggests           │
│  📈 2.4 xG gap is the largest on this matchday                  │
│  🛡️ Burnley's clean sheet rate: 20% season → 0% last 10         │
│  🏠 City's home xG advantage: +0.6 vs their away form           │
╰─────────────────────────────────────────────────────────────────╯
```

### Multi-Match Output
When multiple fixtures are analyzed, each gets its own section with clear separation:

```
═══════════════════════════════════════════════════════════════════

🏟️  Arsenal vs Tottenham
[Analysis tables for this match]

═══════════════════════════════════════════════════════════════════

🏟️  Liverpool vs Chelsea  
[Analysis tables for this match]

═══════════════════════════════════════════════════════════════════
```

### Summary Section
```
╭─────────────────────────────────────────────────────────────────╮
│  📊 MATCHDAY SUMMARY                                            │
├─────────────────────────────────────────────────────────────────┤
│  🏆 Total Tier 1 Opportunities: 5                               │
│  ⚡ Total Tier 2 Opportunities: 8                               │
│  🎯 Strongest Edge: Man City vs Burnley (+2.4 xG)               │
│  🔥 Most Confident Play: Arsenal Asian Handicap                 │
╰─────────────────────────────────────────────────────────────────╯
```

### Important Output Rules
- **No Odds Displayed**: System doesn't have access to betting odds
- **No Unfavorable Markets**: Only show markets with clear edges
- **Consistent Formatting**: Same table structure for all matches
- **Visual Hierarchy**: Tier 1 (💎) always above Tier 2 (⚡)
- **Clear Recommendations**: Specific market picks, not just analysis

---

## Implementation Details

### Gemini Integration

#### Model Configuration
```python
MODEL = "gemini-2.5-pro-preview-05-06"
TEMPERATURE = 0.1  # Low temperature for consistent analysis
```

#### Multimodal Prompt Structure
```python
VISION_ANALYSIS_PROMPT = """
You are an elite soccer betting analyst with 15+ years of Expected Goals (xG) expertise.

IMAGES PROVIDED:
1. all-full.png - Complete season xG data (home + away)
2. home-full.png - Full season home games only
3. away-full.png - Full season away games only  
4. all-10.png - Last 10 games combined
5. home-10.png - Last 10 home games only
6. away-10.png - Last 10 away games only
7. matches.png - Upcoming fixtures for this matchday

TASK: Extract and analyze all xG data for betting opportunities.

[Detailed analysis instructions...]
"""
```

#### Error Handling & Retries
- **Retry Logic**: 3 attempts with increasing temperature
- **Fallback Responses**: Graceful degradation if analysis fails
- **Validation**: JSON schema validation for all outputs
- **Timeout Handling**: 60-second timeouts for API calls

### Data Models (Pydantic)

```python
class TeamPerformance(BaseModel):
    team_name: str
    games_played: int
    xg_for: float
    xg_against: float
    net_xg: float
    form_trend: str  # "improving", "declining", "stable"
    
class Fixture(BaseModel):
    home_team: str
    away_team: str
    date: Optional[str]
    competition: Optional[str]

class BettingOpportunity(BaseModel):
    market: str
    recommendation: str
    confidence: Literal["HIGH", "MEDIUM", "LOW"]
    risk: Literal["HIGH", "MEDIUM", "LOW"] 
    tier: Literal[1, 2]
    xg_edge: float
    reasoning: str

class MatchAnalysis(BaseModel):
    fixture: Fixture
    opportunities: List[BettingOpportunity]
    key_insights: List[str]
    statistical_edge: float
```

### File Processing Logic

#### Image Loading
```python
def load_images(data_dir: str) -> Dict[str, bytes]:
    required_files = [
        "all-full.png", "home-full.png", "away-full.png",
        "all-10.png", "home-10.png", "away-10.png", "matches.png"
    ]
    # Load and validate all required images
```

#### Team Name Matching
```python
def normalize_team_name(name: str) -> str:
    # Handle common abbreviations and variations
    # "Man City" → "Manchester City"
    # "Spurs" → "Tottenham"
    # etc.
```

### Environment Requirements
```bash
export GEMINI_API_KEY='your-api-key-here'
```

---

## Usage Documentation

### Installation
```bash
# No installation needed - uses uv script syntax
# Ensure GEMINI_API_KEY is set in environment
```

### Basic Usage
```bash
# Standard analysis (looks for data-xg/ folder)
uv run futuro.py

# Custom data directory
uv run futuro.py --data-dir ./my-xg-data/
```

### Command Line Arguments
```bash
uv run futuro.py [options]

Options:
  --data-dir DIR    Path to directory containing xG data files
                    (default: ./data-xg/)
  --help           Show help message
```

### Data Preparation Checklist
- [ ] Create data-xg/ directory
- [ ] Add all-full.png (season xG - all teams)
- [ ] Add home-full.png (season home xG - all teams)  
- [ ] Add away-full.png (season away xG - all teams)
- [ ] Add all-10.png (last 10 games xG - all teams)
- [ ] Add home-10.png (last 10 home xG - all teams)
- [ ] Add away-10.png (last 10 away xG - all teams)
- [ ] Add matches.png (upcoming fixtures)
- [ ] Verify team names consistent across all files
- [ ] Run analysis: `uv run futuro.py`

### Expected Output Timeline
- **Data Processing**: 30-60 seconds (7 images)
- **Analysis Phase**: 60-120 seconds (exhaustive reasoning)
- **Output Generation**: 10-20 seconds
- **Total Runtime**: 2-4 minutes (depending on matchday size)

### Troubleshooting

#### Common Issues
1. **Missing Files**: Ensure all 7 required files are present
2. **Team Name Mismatches**: Check spelling consistency across images
3. **API Errors**: Verify GEMINI_API_KEY is set correctly
4. **Empty Output**: Check if any fixtures have favorable opportunities

---

## Expert Betting Persona

### "The xG Whisperer" Profile
**Background**: 15+ years of professional sports betting with deep specialization in Expected Goals methodology and market inefficiency detection.

**Expertise Areas**:
- Advanced xG interpretation and regression analysis
- Market psychology and bookmaker pricing patterns  
- Risk management and bankroll optimization
- Multi-league soccer betting (Premier League, La Liga, Serie A, etc.)
- Live betting and in-game adjustments

**Analytical Approach**:
- **Data-First**: Never rely on gut feelings or narratives
- **Multi-Perspective**: Season form vs recent form vs location splits
- **Conservative Risk**: Prioritize preservation of capital
- **Value-Focused**: Only bet when statistical edge is clear
- **Systematic**: Consistent methodology regardless of emotional attachment

**Communication Style**:
- **Precise**: Specific numbers and statistical backing
- **Confident**: Clear conviction when data supports
- **Honest**: Acknowledges uncertainty when data is mixed
- **Educational**: Explains reasoning without jargon
- **Visual**: Uses tables, emojis, and clear formatting

### Persona Integration in Prompts

#### Core Personality Traits
```
- 15+ years of professional xG analysis experience
- Track record of identifying market inefficiencies  
- Specializes in low-to-medium risk, high-value opportunities
- Known for statistical rigor combined with practical betting wisdom
- Communicates with precision and clarity
- Conservative approach to risk management
- Focus on "why" behind recommendations, not just "what"
```

#### Analysis Philosophy
```
- Expected Goals data tells the story that league tables often hide
- Recent form (last 10) can reveal trend changes before markets adjust
- Home/away splits expose location-based advantages
- Regression to the mean is the bettor's best friend
- Sample size matters - more games = more reliable predictions
- Multiple data perspectives reduce blind spots
- Risk management is more important than any single bet
```

#### Decision-Making Framework
```
1. What does the xG data actually say?
2. How strong is the statistical edge?
3. What's the confidence level in this data?
4. What's the appropriate risk level?
5. Does this fit our value criteria?
6. Can we back this with multiple data points?
```

---

## Future Considerations

### Potential Enhancements

#### Phase 2 Features
- **Historical Validation**: Back-testing analysis against historical results
- **League Customization**: League-specific adjustments (different playing styles)
- **Weather Integration**: Weather data for over/under goals analysis
- **Injury Reports**: Key player absence impact on xG expectations
- **Head-to-Head History**: Historical matchup analysis

#### Phase 3 Features  
- **Live Updates**: In-game xG monitoring and live betting suggestions
- **Multi-League**: Simultaneous analysis across multiple leagues
- **Portfolio Management**: Bankroll tracking and bet sizing recommendations
- **API Integration**: Direct bookmaker odds comparison
- **Mobile Interface**: Web-based interface for mobile usage

#### Technical Improvements
- **Performance Optimization**: Parallel processing for large datasets
- **Enhanced Vision**: Better OCR and table parsing accuracy
- **Database Storage**: Historical data storage and trend analysis
- **Machine Learning**: Pattern recognition for market inefficiencies
- **Real-Time Data**: Live xG feeds and automatic updates

### Scalability Considerations

#### Multi-League Support
- **Data Structure**: Standardized across different leagues
- **Team Mapping**: Handle different naming conventions
- **League Rules**: Adjust for playoff systems, different season lengths
- **Market Variations**: Different betting markets per region

#### Volume Handling
- **Large Matchdays**: Handle 20+ fixtures efficiently  
- **Multiple Leagues**: Simultaneous analysis across leagues
- **Historical Data**: Process season-long datasets
- **Performance**: Maintain <5 minute analysis times

### Research Areas

#### Advanced xG Metrics
- **Shot Quality**: Beyond just shot quantity
- **Defensive Actions**: Blocks, interceptions, pressure metrics
- **Set Pieces**: Corner kicks, free kicks xG modeling
- **Game State**: xG in different scoreline situations
- **Player-Level**: Individual player xG contributions

#### Market Efficiency Studies
- **Closing Line Value**: How often xG predictions beat closing odds
- **Market Movement**: How xG insights predict line movement
- **Seasonal Patterns**: Market efficiency changes throughout season
- **League Differences**: Which leagues have softer markets

### Integration Opportunities

#### Data Sources
- **Official xG Providers**: FBRef, Understat, Opta
- **Betting Exchanges**: Betfair, Smarkets for true market prices
- **News APIs**: Injury reports, team news
- **Weather Services**: Conditions that affect gameplay

#### Platform Integration
- **Telegram Bot**: Automated matchday analysis delivery
- **Discord Integration**: Community sharing of analysis
- **Spreadsheet Export**: CSV output for manual tracking
- **API Development**: Allow third-party integrations

---

## Conclusion

Futuro represents a sophisticated approach to soccer betting analysis, combining the power of Expected Goals data with advanced AI reasoning capabilities. The system is designed to be:

- **Comprehensive**: Exhaustive analysis without shortcuts
- **Practical**: Fits real-world weekend matchday workflow  
- **Reliable**: Conservative risk management and statistical rigor
- **User-Friendly**: Simple usage with rich, clear output
- **Scalable**: Foundation for future enhancements

The agent serves as a powerful tool for bettors who value data-driven decision making and systematic approaches to sports betting, while maintaining the flexibility to work across different leagues and betting markets.

**Key Success Metrics**:
- Identification of profitable betting opportunities
- Clear risk and confidence assessment
- User-friendly output that aids decision making
- Consistent methodology across all analysis
- Foundation for long-term betting success

---

*This specification serves as the complete blueprint for implementing the Futuro soccer betting analysis agent.*
