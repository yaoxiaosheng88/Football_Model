# Football Prediction Model Analysis

## Overview
This is a comprehensive football prediction system that uses statistical models (Poisson, Dixon-Coles, Monte Carlo) combined with team strength calculations to predict match outcomes, probabilities, and expected stats.

---

## Data Sources

### Primary Data Sources:
1. **Historical Match Data** (`data/*_final.csv`):
   - 7 CSV files: `epl_final.csv`, `laliga_final.csv`, `bundesliga_final.csv`, `seriea_final.csv`, `ligue1_final.csv`, `mls_final.csv`, `ucl_final.csv`
   - Contains team-level match data for past 3+ seasons + current season
   - Columns include: `team`, `opponent`, `date`, `league`, `GF` (goals for), `GA` (goals against), `xG`, `xGA`, `Poss` (possession), `Standard_Sh` (shots), `Standard_SoT` (shots on target), `Performance_PSxG`, `Crosses_Stp%`, `Venue` (Home/Away), `Season`, `Round`

2. **Schedule Data** (`data/*_Schedule.xlsx` or `.csv`):
   - 7 schedule files for upcoming/future matches
   - Contains: `Team`, `Opponent`, `Date`, `Venue`, `Round`
   - Used to display upcoming fixtures in the Streamlit app

3. **API Data** (`data/api_results_basic.csv`) - NEW:
   - Real-time match data from Football-Data.org API
   - Contains: `league`, `utcDate`, `status`, `matchday`, `home_team_id`, `home_team_name`, `away_team_id`, `away_team_name`, `score_fullTime_home`, `score_fullTime_away`, `score_winner`
   - Team names are normalized using `team_name_map.json` (ID-based mapping)

---

## Model Architecture

### 1. **Data Loading & Normalization** (`utils/data_loader.py`)

**Function**: `load_all_league_data()`
- Loads all 7 `*_final.csv` files
- Normalizes team names using `team_name_normalizer.py`
- Merges into master dataframe
- Standardizes dates and extracts season years
- Ensures numeric columns are properly typed

**Output**: Master dataframe with all historical match data

---

### 2. **Feature Engineering** (`utils/feature_engineering.py`)

#### `compute_team_season_averages()`:
- **Purpose**: Calculate weighted seasonal averages for a team
- **Input**: Team name, venue (Home/Away), season filter
- **Process**:
  1. Filters matches by team and venue
  2. Applies **dynamic season weights** (recent seasons weighted higher)
  3. Computes weighted averages for:
     - `xg_avg`: Expected goals
     - `xga_avg`: Expected goals against
     - `gf_avg`: Goals for (actual)
     - `ga_avg`: Goals against (actual)
     - `shots_avg`: Shots
     - `sot_avg`: Shots on target
     - `poss_avg`: Possession %
     - `psxg_avg`: Post-shot expected goals
     - `crosses_stp_pct_avg`: Crosses stopped percentage

#### `compute_league_averages()`:
- **Purpose**: Calculate league-wide averages
- **Process**: Same metrics as team averages, but aggregated across all teams in a league

#### `get_head_to_head()`:
- **Purpose**: Extract historical match data between two teams
- **Usage**: Used for H2H adjustments in predictions (20% weight)

---

### 3. **Team Strength Calculation** (`utils/metrics_calculator.py`)

#### `compute_team_strengths()`:
- **Purpose**: Calculate attack/defense strengths for all teams in a league
- **Formula**:
  ```
  home_attack_strength = (team_home_goals_scored / league_avg_home_goals)
  away_attack_strength = (team_away_goals_scored / league_avg_away_goals)
  home_defense_strength = (league_avg_away_goals / team_home_goals_conceded)
  away_defense_strength = (league_avg_home_goals / team_away_goals_conceded)
  ```

**Process**:
1. **Computes league base rates**: Average goals scored/conceded at home/away across the league
2. **Detects new teams**: Teams with < 10 matches use league priors (default strengths)
3. **Calculates team averages**: Uses weighted averages (recent seasons weighted more)
4. **Blends goals and xG**: Adaptively blends actual goals with xG based on performance ratio
5. **Handles partial data**: Teams with 4-10 matches get blended with league baseline
6. **Calculates overall strength**: 
   - Primary (70%): Average of attack and defense strengths
   - Secondary (30%): Metric-based (xG, shots, possession, PSxG)

**Output**: Dictionary saved to `data/team_strengths/{league}_strengths.json`
- Keys: `home_attack`, `away_attack`, `home_defense`, `away_defense`, `strength`, `is_new_team`

---

### 4. **Expected Goals Calculation** (`model/poisson_model.py`)

#### `expected_goals()`:
- **Purpose**: Calculate λ (lambda) - expected goals for home and away teams
- **Formula**:
  ```
  λ_home = base_home_goals × (0.8 × situational_home + 0.2 × overall_ratio_home)
  λ_away = base_away_goals × (0.8 × situational_away + 0.2 × overall_ratio_away)
  
  Where:
  situational_home = home_attack_strength × (1 / away_defense_strength)
  situational_away = away_attack_strength × (1 / home_defense_strength)
  overall_ratio = team_strength / opponent_strength
  ```

**Cross-League Normalization**:
- For UCL or cross-league matches, applies league strength factors:
  - EPL/La Liga: 1.45
  - Bundesliga/Serie A/Ligue 1: 1.40
  - MLS: 1.20
  - UCL: 1.55

**Lambda Clipping**: Values clipped between 0.1 and 3.5 to avoid extreme outliers

---

### 5. **Dixon-Coles Adjustment** (`model/dixon_coles.py`)

#### `calculate_dixon_coles_probabilities()`:
- **Purpose**: Correct Poisson model for low-score games (underestimates 0-0, 1-0, 0-1, 1-1)
- **Correction Factor**:
  ```
  τ(i, j, λ_home, λ_away) = 1 - λ_home × λ_away × ρ
  Where ρ = -0.13 (correlation parameter)
  Applied only to scores: (0,0), (0,1), (1,0), (1,1)
  ```

**Process**:
1. Builds probability matrix (0-6 goals for each team)
2. Applies correction factor to low-score scenarios
3. Normalizes matrix so probabilities sum to 1
4. Calculates probabilities for:
   - Home win / Draw / Away win
   - BTTS (Both Teams To Score)
   - Over 1.5 / 2.5 / 3.5 goals

---

### 6. **Monte Carlo Simulation** (`model/monte_carlo.py`)

#### `simulate_match()`:
- **Purpose**: Simulate match 10,000 times using Poisson distributions
- **Process**:
  1. Generates 10,000 random goal outcomes using Poisson(λ_home) and Poisson(λ_away)
  2. Counts outcomes:
     - Home wins: `home_goals > away_goals`
     - Away wins: `away_goals > home_goals`
     - Draws: `home_goals == away_goals`
     - BTTS: `home_goals > 0 AND away_goals > 0`
     - Over 1.5/2.5/3.5: Total goals > threshold
  3. Calculates correct score probabilities (top 5 most common)
  4. Returns probabilities as fractions of 10,000

---

### 7. **Probability Blending** (`model/monte_carlo.py`)

#### `blend_probabilities()`:
- **Purpose**: Combine Dixon-Coles and Monte Carlo probabilities
- **Formula**:
  ```
  final_prob = 0.7 × dixon_coles_prob + 0.3 × monte_carlo_prob
  ```
- **Rationale**: Dixon-Coles is more theoretically sound, Monte Carlo adds robustness

---

### 8. **Main Prediction Orchestrator** (`model/predictor.py`)

#### `predict_match()`:
- **Purpose**: Complete prediction pipeline
- **Process**:

1. **Normalize team names** using `team_name_normalizer.py`

2. **Load/Compute team strengths**:
   - Loads from `data/team_strengths/{league}_strengths.json`
   - If missing, computes using `compute_team_strengths()`
   - For UCL matches, adjusts strengths by blending with domestic league data (60% UCL, 40% domestic)

3. **Calculate expected goals** (`λ_home`, `λ_away`) using `expected_goals()`

4. **Apply H2H adjustment** (if available):
   - Gets last 3+ matches between teams
   - Calculates average xG from H2H
   - Blends: `λ = 0.8 × strength_λ + 0.2 × h2h_xg`

5. **Calculate probabilities**:
   - Dixon-Coles probabilities
   - Monte Carlo probabilities
   - Blended final probabilities (70% DC, 30% MC)

6. **Calculate expected stats**:
   - Expected shots, possession, xG using `calculate_expected_stats()`

7. **Get recent form**:
   - Last 5 matches results for each team

8. **Determine best bets**:
   - Best bet: Outcome with >55% probability
   - Best prop: BTTS >65% or Over 2.5 >65%

**Output**: Dictionary containing:
- Team names, league
- Lambda values (expected goals)
- Team strengths
- Probabilities (home win, draw, away win, BTTS, over 1.5/2.5/3.5)
- Correct score probabilities
- Expected stats
- Recent form
- Best bet recommendations

---

## Key Features

### Season Weighting (`utils/weighting.py`):
- **Dynamic weights**: Recent seasons weighted higher than older seasons
- **Team-specific weighting**: Adjusts based on team's historical performance consistency
- **Formula**: `weight = base_weight × recency_factor × consistency_factor`

### UCL Cross-League Handling:
- Teams in UCL get strength adjustments by blending:
  - 60% UCL strength (from UCL matches)
  - 40% Domestic league strength (from EPL/La Liga/etc.)
- Cross-league normalization factors applied

### New Team Handling:
- Teams with < 10 matches use **league priors** (default strengths ~1.0)
- Teams with 4-10 matches get **blended** with league baseline
- Prevents extreme predictions for promoted/new teams

### Data Normalization:
- All team names normalized using `team_name_normalizer.py`
- API data uses ID-based mapping (`team_name_map.json`) for consistency
- Ensures "Arsenal FC" = "Arsenal" = "Arsenal F.C."

---

## Data Flow Summary

```
Historical CSVs (*_final.csv)
    ↓
load_all_league_data()
    ↓
Master DataFrame (all leagues, normalized)
    ↓
compute_team_strengths()
    ↓
Team Strengths JSON (saved to data/team_strengths/)
    ↓
predict_match()
    ↓
expected_goals() → λ_home, λ_away
    ↓
calculate_dixon_coles_probabilities() → DC probs
    ↓
simulate_match() → MC probs
    ↓
blend_probabilities() → Final probs
    ↓
Return Prediction Dictionary
```

---

## Files Used by Models

### Core Model Files:
- `model/predictor.py` - Main orchestrator
- `model/poisson_model.py` - Expected goals calculation
- `model/dixon_coles.py` - Low-score correction
- `model/monte_carlo.py` - Simulation and blending

### Supporting Utils:
- `utils/data_loader.py` - Load CSVs and schedules
- `utils/feature_engineering.py` - Calculate team/league averages
- `utils/metrics_calculator.py` - Compute team strengths
- `utils/strengths.py` - Save/load strengths from JSON
- `utils/weighting.py` - Season weighting logic
- `utils/team_name_normalizer.py` - Normalize team names

### Data Files:
- `data/*_final.csv` - Historical match data (7 files)
- `data/*_Schedule.xlsx` - Upcoming fixtures (7 files)
- `data/api_results_basic.csv` - API match data
- `data/team_strengths/*_strengths.json` - Computed team strengths
- `data/team_name_map.json` - API team ID to normalized name mapping

---

## Strengths of This System

1. **Multi-Model Approach**: Combines Poisson, Dixon-Coles, and Monte Carlo for robustness
2. **Dynamic Weighting**: Recent form weighted more heavily
3. **Cross-League Support**: Handles UCL matches with proper normalization
4. **New Team Handling**: Prevents extreme predictions for teams with limited data
5. **Comprehensive Metrics**: Uses goals, xG, shots, possession, PSxG for strength calculation
6. **H2H Integration**: Incorporates head-to-head history when available
7. **Normalized Data**: Consistent team names across all sources

---

## Limitations & Considerations

1. **Static Weights**: Blending ratios (70/30, 80/20) are fixed - could be tuned
2. **No Contextual Factors**: Doesn't account for injuries, weather, manager changes
3. **Historical Data Dependency**: Requires sufficient historical data (3+ seasons ideal)
4. **League Factors**: Cross-league factors are manually set, could be data-driven
5. **H2H Small Sample**: Only uses H2H if ≥3 matches exist

---

## Future Integration Points

- **API Data Integration**: The `api_results_basic.csv` can be merged into master dataframe for real-time predictions
- **Live Updates**: Team strengths can be recalculated as new matches are added
- **Model Tuning**: Weights and factors can be optimized using backtesting

