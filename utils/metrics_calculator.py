"""
Calculate team strengths and expected match metrics.
Computes attack/defense strengths using weighted averages relative to league baselines.
"""

import numpy as np
import pandas as pd
from .feature_engineering import compute_team_season_averages, compute_league_averages
from .weighting import get_current_season, apply_season_weights
from .strengths import save_team_strengths


# Global dictionary to store league base rates
LEAGUE_BASE_RATES = {}

# Minimum matches required for reliable team strength calculation
MIN_MATCHES = 10

# Global dictionary to store league baselines (xG, xGA, attack/defense averages)
LEAGUE_BASELINES = {}

# Global dictionary to store team metadata (is_new, n_matches)
TEAM_METADATA = {}


def compute_league_base_rates(dataframe, league, current_season=None):
    """
    Compute league base rates (home and away goal averages).
    
    Args:
        dataframe: Master dataframe
        league: League name
        current_season: Most recent season
        
    Returns:
        tuple: (base_home_goals, base_away_goals)
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    league_df = dataframe[dataframe['league'] == league.upper()].copy()
    
    if len(league_df) == 0:
        # Fallback to default values
        return 1.5, 1.2
    
    # Apply season weights
    league_df = apply_season_weights(league_df, current_season)
    
    # Calculate base rates: total goals / total matches
    home_matches = league_df[league_df['Venue'] == 'Home']
    away_matches = league_df[league_df['Venue'] == 'Away']
    
    if len(home_matches) == 0 or len(away_matches) == 0:
        return 1.5, 1.2
    
    # Use weighted average
    weights_home = home_matches['season_weight'].values if 'season_weight' in home_matches.columns else np.ones(len(home_matches))
    weights_away = away_matches['season_weight'].values if 'season_weight' in away_matches.columns else np.ones(len(away_matches))
    
    total_home_goals = np.sum(home_matches['GF'].fillna(0) * weights_home)
    total_home_matches = np.sum(weights_home)
    base_home_goals = total_home_goals / total_home_matches if total_home_matches > 0 else 1.5
    
    total_away_goals = np.sum(away_matches['GF'].fillna(0) * weights_away)
    total_away_matches = np.sum(weights_away)
    base_away_goals = total_away_goals / total_away_matches if total_away_matches > 0 else 1.2
    
    # Store in global cache
    LEAGUE_BASE_RATES[league.upper()] = (base_home_goals, base_away_goals)
    
    return base_home_goals, base_away_goals


def get_league_base_rates(league):
    """
    Get cached league base rates.
    
    Args:
        league: League name
        
    Returns:
        tuple: (base_home_goals, base_away_goals) or (1.5, 1.2) if not found
    """
    return LEAGUE_BASE_RATES.get(league.upper(), (1.5, 1.2))


def compute_league_baselines(dataframe, league, current_season=None):
    """
    Compute league-wide baselines for xG, xGA, attack, and defense strengths.
    These serve as priors for new teams.
    
    Args:
        dataframe: Master dataframe
        league: League name
        current_season: Most recent season
        
    Returns:
        dict: League baselines with keys:
            - home_attack_avg: Average home attack strength (normalized, should be ~1.0)
            - away_attack_avg: Average away attack strength
            - home_defense_avg: Average home defense strength
            - away_defense_avg: Average away defense strength
            - xg_home_avg: Average home xG
            - xga_home_avg: Average home xGA
            - xg_away_avg: Average away xG
            - xga_away_avg: Average away xGA
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    league_key = league.upper()
    
    # Check cache first
    if league_key in LEAGUE_BASELINES:
        return LEAGUE_BASELINES[league_key]
    
    league_df = dataframe[dataframe['league'] == league_key].copy()
    
    if len(league_df) == 0:
        # Return default baselines
        return {
            "home_attack_avg": 1.0,
            "away_attack_avg": 1.0,
            "home_defense_avg": 1.0,
            "away_defense_avg": 1.0,
            "xg_home_avg": 1.5,
            "xga_home_avg": 1.2,
            "xg_away_avg": 1.2,
            "xga_away_avg": 1.5
        }
    
    # Apply season weights
    league_df = apply_season_weights(league_df, current_season)
    
    # Get home and away matches
    home_matches = league_df[league_df['Venue'] == 'Home']
    away_matches = league_df[league_df['Venue'] == 'Away']
    
    if len(home_matches) == 0 or len(away_matches) == 0:
        return {
            "home_attack_avg": 1.0,
            "away_attack_avg": 1.0,
            "home_defense_avg": 1.0,
            "away_defense_avg": 1.0,
            "xg_home_avg": 1.5,
            "xga_home_avg": 1.2,
            "xg_away_avg": 1.2,
            "xga_away_avg": 1.5
        }
    
    # Calculate weighted averages
    weights_home = home_matches['season_weight'].values if 'season_weight' in home_matches.columns else np.ones(len(home_matches))
    weights_away = away_matches['season_weight'].values if 'season_weight' in away_matches.columns else np.ones(len(away_matches))
    
    # xG and xGA averages
    xg_home_avg = np.average(home_matches['xG'].fillna(0), weights=weights_home)
    xga_home_avg = np.average(home_matches['xGA'].fillna(0), weights=weights_home)
    xg_away_avg = np.average(away_matches['xG'].fillna(0), weights=weights_away)
    xga_away_avg = np.average(away_matches['xGA'].fillna(0), weights=weights_away)
    
    # Get base rates for normalization
    base_home_goals, base_away_goals = compute_league_base_rates(dataframe, league, current_season)
    
    # Calculate average attack strengths (goals scored relative to league average)
    home_gf_avg = np.average(home_matches['GF'].fillna(0), weights=weights_home)
    away_gf_avg = np.average(away_matches['GF'].fillna(0), weights=weights_away)
    
    home_attack_avg = home_gf_avg / base_home_goals if base_home_goals > 0 else 1.0
    away_attack_avg = away_gf_avg / base_away_goals if base_away_goals > 0 else 1.0
    
    # Calculate average defense strengths (goals conceded relative to league average)
    home_ga_avg = np.average(home_matches['GA'].fillna(0), weights=weights_home)
    away_ga_avg = np.average(away_matches['GA'].fillna(0), weights=weights_away)
    
    # Defense strength formula: league_avg_goals_against / team_goals_conceded
    # For league average, this simplifies to 1.0 (since avg_ga = base_away for home, base_home for away)
    home_defense_avg = base_away_goals / home_ga_avg if home_ga_avg > 0 else 1.0
    away_defense_avg = base_home_goals / away_ga_avg if away_ga_avg > 0 else 1.0
    
    baselines = {
        "home_attack_avg": round(home_attack_avg, 3),
        "away_attack_avg": round(away_attack_avg, 3),
        "home_defense_avg": round(home_defense_avg, 3),
        "away_defense_avg": round(away_defense_avg, 3),
        "xg_home_avg": round(xg_home_avg, 3),
        "xga_home_avg": round(xga_home_avg, 3),
        "xg_away_avg": round(xg_away_avg, 3),
        "xga_away_avg": round(xga_away_avg, 3)
    }
    
    # Cache and return
    LEAGUE_BASELINES[league_key] = baselines
    return baselines


def get_league_baselines(league):
    """
    Get cached league baselines.
    
    Args:
        league: League name
        
    Returns:
        dict: League baselines or None if not computed
    """
    return LEAGUE_BASELINES.get(league.upper())


def detect_new_teams(dataframe, league, current_season=None):
    """
    Detect new or low-sample teams in a league.
    
    Args:
        dataframe: Master dataframe
        league: League name
        current_season: Most recent season
        
    Returns:
        dict: Dictionary mapping team names to metadata:
            {"is_new": bool, "n_matches": int}
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    league_key = f"{league.upper()}"
    league_df = dataframe[dataframe['league'] == league_key].copy()
    
    if len(league_df) == 0:
        return {}
    
    teams = league_df['team'].unique()
    metadata = {}
    
    for team in teams:
        team_matches = league_df[league_df['team'] == team]
        total_matches = len(team_matches)
        
        is_new = total_matches < MIN_MATCHES
        
        metadata[team] = {
            "is_new": is_new,
            "n_matches": total_matches
        }
    
    # Store in global metadata
    if league_key not in TEAM_METADATA:
        TEAM_METADATA[league_key] = {}
    TEAM_METADATA[league_key].update(metadata)
    
    return metadata


def assign_prior_strengths(league_baselines):
    """
    Assign prior strengths for new teams based on league baselines.
    
    New/promoted teams typically underperform initially:
    - Attack: 0.9x league average (weaker attack)
    - Defense: 1.1x league average (weaker defense, concedes more)
    
    Args:
        league_baselines: Dict from compute_league_baselines
        
    Returns:
        dict: Prior strengths for a new team
    """
    return {
        "home_attack": 0.9 * league_baselines.get("home_attack_avg", 1.0),
        "away_attack": 0.9 * league_baselines.get("away_attack_avg", 1.0),
        "home_defense": 1.1 * league_baselines.get("home_defense_avg", 1.0),
        "away_defense": 1.1 * league_baselines.get("away_defense_avg", 1.0),
        "strength": 1.0  # Overall strength starts at league average
    }


def blend_partial_data(team_strength, league_baseline, n_matches):
    """
    Blend team strength with league baseline for teams with partial data (4-10 matches).
    
    Args:
        team_strength: Computed team strength value
        league_baseline: League baseline value
        n_matches: Number of matches played
        
    Returns:
        float: Blended strength value
    """
    # Weight increases with number of matches, up to 1.0 at MIN_MATCHES
    weight_current = min(1.0, n_matches / MIN_MATCHES)
    
    blended = weight_current * team_strength + (1 - weight_current) * league_baseline
    
    return blended


def compute_team_strengths(dataframe, league, current_season=None):
    """
    Compute comprehensive team strengths for all teams in a league.
    
    Now includes:
    - Detection of new/low-sample teams (< MIN_MATCHES)
    - Priors for new teams based on league baselines
    - Blending for teams with partial data (4-10 matches)
    
    Uses formulas:
    - home_attack_strength = (team_home_goals / league_avg_home_goals)
    - away_attack_strength = (team_away_goals / league_avg_away_goals)
    - home_defence_strength = (league_avg_away_goals / team_home_goals_conceded)
    - away_defence_strength = (league_avg_home_goals / team_away_goals_conceded)
    
    Args:
        dataframe: Master dataframe with all match data
        league: League name (e.g., 'EPL')
        current_season: Most recent season (auto-detected if None)
        
    Returns:
        dict: Dictionary mapping team names to their strengths
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    # Compute and cache league base rates and baselines
    base_home_goals, base_away_goals = compute_league_base_rates(dataframe, league, current_season)
    league_baselines = compute_league_baselines(dataframe, league, current_season)
    
    # Detect new teams
    team_metadata = detect_new_teams(dataframe, league, current_season)
    
    # Filter to league
    league_df = dataframe[dataframe['league'] == league.upper()].copy()
    
    if len(league_df) == 0:
        return {}
    
    # Get league-wide averages for all metrics (for overall strength calculation)
    league_home_avg = compute_league_averages(dataframe, league, venue='Home', current_season=current_season)
    league_away_avg = compute_league_averages(dataframe, league, venue='Away', current_season=current_season)
    
    if not league_home_avg or not league_away_avg:
        return {}
    
    # Get all unique teams in league
    teams = sorted(league_df['team'].unique())
    
    strengths_dict = {}
    
    for team in teams:
        # Get team metadata
        team_info = team_metadata.get(team, {"is_new": False, "n_matches": 0})
        is_new_team = team_info["is_new"]
        total_matches = team_info["n_matches"]
        
        # For completely new teams (< MIN_MATCHES total), use priors
        if is_new_team and total_matches < MIN_MATCHES:
            prior_strengths = assign_prior_strengths(league_baselines)
            strengths_dict[team] = {
                "home_attack": round(prior_strengths["home_attack"], 3),
                "away_attack": round(prior_strengths["away_attack"], 3),
                "home_defense": round(prior_strengths["home_defense"], 3),
                "away_defense": round(prior_strengths["away_defense"], 3),
                "strength": round(prior_strengths["strength"], 3),
                "is_new_team": True  # Flag for logging
            }
            continue
        
        # Get team averages (home and away) - these are already weighted
        team_home_stats = compute_team_season_averages(
            dataframe, team, venue='Home', current_season=current_season
        )
        team_away_stats = compute_team_season_averages(
            dataframe, team, venue='Away', current_season=current_season
        )
        
        if not team_home_stats or not team_away_stats:
            continue
        
        home_matches = team_home_stats.get('matches', 0)
        away_matches = team_away_stats.get('matches', 0)
        
        # Calculate attack strengths (as per requirements)
        # Use a blend of actual goals and xG for more reliable estimates, especially for small samples
        home_goals_scored = team_home_stats['gf_avg']
        home_xg = team_home_stats.get('xg_avg', home_goals_scored)
        
        # Adaptive blending: If actual goals significantly exceed xG (e.g., >25%), trust actual goals more
        # This accounts for teams that consistently outperform their xG (good finishing)
        goal_xg_ratio = home_goals_scored / home_xg if home_xg > 0.1 else 1.0
        
        if goal_xg_ratio > 1.25:
            # Team is outperforming xG significantly - trust actual goals more (85% vs 15%)
            home_goals_scored = 0.85 * home_goals_scored + 0.15 * home_xg
        elif goal_xg_ratio < 0.8:
            # Team is underperforming xG significantly - use more xG (60% actual, 40% xG)
            home_goals_scored = 0.6 * home_goals_scored + 0.4 * home_xg
        else:
            # Normal case: blend actual goals (70%) with xG (30%)
            home_goals_scored = 0.7 * home_goals_scored + 0.3 * home_xg
        
        # Apply smoothing for small sample sizes to prevent extreme values
        # Use a Bayesian-style adjustment: blend team average with league average based on sample size
        min_matches_for_reliability = 10
        if home_matches < min_matches_for_reliability and home_matches > 0:
            # Smooth towards league average for small samples
            # More matches = more weight on team average
            sample_weight = home_matches / min_matches_for_reliability
            smoothed_gf = sample_weight * home_goals_scored + (1 - sample_weight) * base_home_goals
            home_goals_scored = max(0.3, smoothed_gf)  # Minimum 0.3 to prevent extreme values
        
        home_attack_strength = home_goals_scored / base_home_goals if base_home_goals > 0 else 1.0
        
        # Blend with league baseline if partial data (4-10 matches)
        if total_matches < MIN_MATCHES:
            home_attack_baseline = league_baselines.get("home_attack_avg", 1.0)
            home_attack_strength = blend_partial_data(home_attack_strength, home_attack_baseline, total_matches)
        
        # Cap attack strength at reasonable maximum (2.0) to prevent outliers
        home_attack_strength = min(2.0, home_attack_strength)
        
        away_goals_scored = team_away_stats['gf_avg']
        away_xg = team_away_stats.get('xg_avg', away_goals_scored)
        away_matches = team_away_stats.get('matches', 0)
        
        # Adaptive blending for away goals
        goal_xg_ratio = away_goals_scored / away_xg if away_xg > 0.1 else 1.0
        
        if goal_xg_ratio > 1.25:
            away_goals_scored = 0.85 * away_goals_scored + 0.15 * away_xg
        elif goal_xg_ratio < 0.8:
            away_goals_scored = 0.6 * away_goals_scored + 0.4 * away_xg
        else:
            away_goals_scored = 0.7 * away_goals_scored + 0.3 * away_xg
        
        # Apply smoothing for small sample sizes
        if away_matches < min_matches_for_reliability and away_matches > 0:
            sample_weight = away_matches / min_matches_for_reliability
            smoothed_gf = sample_weight * away_goals_scored + (1 - sample_weight) * base_away_goals
            away_goals_scored = max(0.3, smoothed_gf)
        
        away_attack_strength = away_goals_scored / base_away_goals if base_away_goals > 0 else 1.0
        
        # Blend with league baseline if partial data
        if total_matches < MIN_MATCHES:
            away_attack_baseline = league_baselines.get("away_attack_avg", 1.0)
            away_attack_strength = blend_partial_data(away_attack_strength, away_attack_baseline, total_matches)
        
        # Cap attack strength at reasonable maximum (2.0)
        away_attack_strength = min(2.0, away_attack_strength)
        
        # Calculate defense strengths (as per requirements)
        # Home defense: league_avg_away_goals / team_home_goals_conceded
        home_goals_conceded = team_home_stats['ga_avg']
        home_matches = team_home_stats.get('matches', 0)
        
        # Apply smoothing for small sample sizes to prevent extreme values
        # Use a Bayesian-style adjustment: blend team average with league average based on sample size
        min_matches_for_reliability = 10
        if home_matches < min_matches_for_reliability and home_matches > 0:
            # Smooth towards league average for small samples
            # More matches = more weight on team average
            sample_weight = home_matches / min_matches_for_reliability
            smoothed_ga = sample_weight * home_goals_conceded + (1 - sample_weight) * base_away_goals
            home_goals_conceded = max(0.3, smoothed_ga)  # Minimum 0.3 to prevent extreme values
        
        # Cap goals conceded at minimum to prevent extreme defense strengths
        home_goals_conceded = max(0.3, home_goals_conceded)
        home_defence_strength = base_away_goals / home_goals_conceded if home_goals_conceded > 0 else 1.0
        
        # Blend with league baseline if partial data
        if total_matches < MIN_MATCHES:
            home_defense_baseline = league_baselines.get("home_defense_avg", 1.0)
            home_defence_strength = blend_partial_data(home_defence_strength, home_defense_baseline, total_matches)
        
        # Cap defense strength at reasonable maximum (2.5) to prevent outliers
        home_defence_strength = min(2.5, home_defence_strength)
        
        # Away defense: league_avg_home_goals / team_away_goals_conceded
        away_goals_conceded = team_away_stats['ga_avg']
        away_matches = team_away_stats.get('matches', 0)
        
        # Apply smoothing for small sample sizes
        if away_matches < min_matches_for_reliability and away_matches > 0:
            sample_weight = away_matches / min_matches_for_reliability
            smoothed_ga = sample_weight * away_goals_conceded + (1 - sample_weight) * base_home_goals
            away_goals_conceded = max(0.3, smoothed_ga)
        
        # Cap goals conceded at minimum to prevent extreme values
        away_goals_conceded = max(0.3, away_goals_conceded)
        away_defence_strength = base_home_goals / away_goals_conceded if away_goals_conceded > 0 else 1.0
        
        # Blend with league baseline if partial data
        if total_matches < MIN_MATCHES:
            away_defense_baseline = league_baselines.get("away_defense_avg", 1.0)
            away_defence_strength = blend_partial_data(away_defence_strength, away_defense_baseline, total_matches)
        
        # Cap defense strength at reasonable maximum (2.5)
        away_defence_strength = min(2.5, away_defence_strength)
        
        # Compute normalized metric strengths for overall strength
        xg_strength_home = team_home_stats['xg_avg'] / league_home_avg['xg_avg'] if league_home_avg['xg_avg'] > 0 else 1.0
        xg_strength_away = team_away_stats['xg_avg'] / league_away_avg['xg_avg'] if league_away_avg['xg_avg'] > 0 else 1.0
        
        shots_strength_home = team_home_stats['shots_avg'] / league_home_avg['shots_avg'] if league_home_avg['shots_avg'] > 0 else 1.0
        shots_strength_away = team_away_stats['shots_avg'] / league_away_avg['shots_avg'] if league_away_avg['shots_avg'] > 0 else 1.0
        
        poss_strength_home = team_home_stats['poss_avg'] / league_home_avg['poss_avg'] if league_home_avg['poss_avg'] > 0 else 1.0
        poss_strength_away = team_away_stats['poss_avg'] / league_away_avg['poss_avg'] if league_away_avg['poss_avg'] > 0 else 1.0
        
        # Handle PSxG if available
        league_psxg_home = league_home_avg.get('psxg_avg', 0)
        league_psxg_away = league_away_avg.get('psxg_avg', 0)
        
        if league_psxg_home > 0:
            psxg_strength_home = team_home_stats['psxg_avg'] / league_psxg_home
        else:
            psxg_strength_home = 1.0
        
        if league_psxg_away > 0:
            psxg_strength_away = team_away_stats['psxg_avg'] / league_psxg_away
        else:
            psxg_strength_away = 1.0
        
        # Overall strength calculation
        # Primary: Average of attack and defense strengths (most important indicators)
        avg_attack_strength = (home_attack_strength + away_attack_strength) / 2
        avg_defense_strength = (home_defence_strength + away_defence_strength) / 2
        primary_strength = (avg_attack_strength + avg_defense_strength) / 2
        
        # Secondary: Metric-based strengths (xG, shots, possession, PSxG)
        metric_strength = (
            0.4 * ((xg_strength_home + xg_strength_away) / 2) +
            0.2 * ((shots_strength_home + shots_strength_away) / 2) +
            0.2 * ((poss_strength_home + poss_strength_away) / 2) +
            0.2 * ((psxg_strength_home + psxg_strength_away) / 2)
        )
        
        # Final overall strength: 70% primary (attack/defense), 30% metric-based
        # Attack and defense directly reflect goals scored/conceded, so weighted higher
        overall_strength = 0.7 * primary_strength + 0.3 * metric_strength
        
        strengths_dict[team] = {
            "home_attack": round(home_attack_strength, 3),
            "away_attack": round(away_attack_strength, 3),
            "home_defense": round(home_defence_strength, 3),
            "away_defense": round(away_defence_strength, 3),
            "strength": round(overall_strength, 3),
            "is_new_team": False  # Only True for teams that used pure priors
        }
    
    # Save strengths for future use
    save_team_strengths(strengths_dict, league)
    
    return strengths_dict


def get_league_goal_averages(dataframe, league, current_season=None):
    """
    Get league-wide goal averages for home and away.
    (Kept for backward compatibility)
    
    Args:
        dataframe: Master dataframe
        league: League name
        current_season: Most recent season
        
    Returns:
        tuple: (league_home_avg_goals, league_away_avg_goals)
    """
    # Check cache first
    if league.upper() in LEAGUE_BASE_RATES:
        return LEAGUE_BASE_RATES[league.upper()]
    
    # Compute if not cached
    return compute_league_base_rates(dataframe, league, current_season)


def calculate_expected_stats(dataframe, home_team, away_team, league, 
                             use_h2h=True, current_season=None):
    """
    Calculate expected match statistics (shots, possession, PSxG, crosses stopped %).
    """
    from .feature_engineering import get_head_to_head
    from .weighting import calculate_match_decay_weights
    
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    # Get team averages
    home_stats = compute_team_season_averages(dataframe, home_team, venue='Home', current_season=current_season)
    away_stats = compute_team_season_averages(dataframe, away_team, venue='Away', current_season=current_season)
    
    if not home_stats or not away_stats:
        return None
    
    # Try to get H2H data
    h2h_weight = 0.0
    h2h_home_stats = None
    h2h_away_stats = None
    
    if use_h2h:
        h2h_df = get_head_to_head(dataframe, home_team, away_team, current_season=current_season)
        
        if len(h2h_df) >= 2:
            recent_h2h = h2h_df.tail(5)
            decay_weights = calculate_match_decay_weights(recent_h2h)
            
            home_h2h = recent_h2h[recent_h2h['team'] == home_team]
            away_h2h = recent_h2h[recent_h2h['team'] == away_team]
            
            if len(home_h2h) > 0:
                weights_h = decay_weights[recent_h2h['team'] == home_team]
                total_weight_h = weights_h.sum()
                if total_weight_h > 0:
                    h2h_home_stats = {
                        'shots_avg': np.average(home_h2h['Standard_Sh'].fillna(0), weights=weights_h),
                        'poss_avg': np.average(home_h2h['Poss'].fillna(0), weights=weights_h),
                        'psxg_avg': np.average(home_h2h['Performance_PSxG'].fillna(0), weights=weights_h),
                        'crosses_stp_pct_avg': np.average(home_h2h['Crosses_Stp%'].fillna(0), weights=weights_h),
                    }
            
            if len(away_h2h) > 0:
                weights_a = decay_weights[recent_h2h['team'] == away_team]
                total_weight_a = weights_a.sum()
                if total_weight_a > 0:
                    h2h_away_stats = {
                        'shots_avg': np.average(away_h2h['Standard_Sh'].fillna(0), weights=weights_a),
                        'poss_avg': np.average(away_h2h['Poss'].fillna(0), weights=weights_a),
                        'psxg_avg': np.average(away_h2h['Performance_PSxG'].fillna(0), weights=weights_a),
                        'crosses_stp_pct_avg': np.average(away_h2h['Crosses_Stp%'].fillna(0), weights=weights_a),
                    }
            
            if h2h_home_stats and h2h_away_stats:
                h2h_weight = 0.3
    
    # Blend seasonal and H2H stats
    season_weight = 1.0 - h2h_weight
    
    home_shots = (season_weight * home_stats.get('shots_avg', 0) + 
                  h2h_weight * (h2h_home_stats.get('shots_avg', 0) if h2h_home_stats else home_stats.get('shots_avg', 0)))
    
    away_shots = (season_weight * away_stats.get('shots_avg', 0) + 
                  h2h_weight * (h2h_away_stats.get('shots_avg', 0) if h2h_away_stats else away_stats.get('shots_avg', 0)))
    
    home_poss = (season_weight * home_stats.get('poss_avg', 0) + 
                 h2h_weight * (h2h_home_stats.get('poss_avg', 0) if h2h_home_stats else home_stats.get('poss_avg', 0)))
    
    away_poss = (season_weight * away_stats.get('poss_avg', 0) + 
                 h2h_weight * (h2h_away_stats.get('poss_avg', 0) if h2h_away_stats else away_stats.get('poss_avg', 0)))
    
    # Possession should be normalized (sum to 100%)
    total_poss = home_poss + away_poss
    if total_poss > 0:
        home_poss = (home_poss / total_poss) * 100
        away_poss = (away_poss / total_poss) * 100
    else:
        home_poss = 50.0
        away_poss = 50.0
    
    home_psxg = (season_weight * home_stats.get('psxg_avg', 0) + 
                 h2h_weight * (h2h_home_stats.get('psxg_avg', 0) if h2h_home_stats else home_stats.get('psxg_avg', 0)))
    
    away_psxg = (season_weight * away_stats.get('psxg_avg', 0) + 
                 h2h_weight * (h2h_away_stats.get('psxg_avg', 0) if h2h_away_stats else away_stats.get('psxg_avg', 0)))
    
    home_crosses_stp = (season_weight * home_stats.get('crosses_stp_pct_avg', 0) + 
                       h2h_weight * (h2h_home_stats.get('crosses_stp_pct_avg', 0) if h2h_home_stats else home_stats.get('crosses_stp_pct_avg', 0)))
    
    away_crosses_stp = (season_weight * away_stats.get('crosses_stp_pct_avg', 0) + 
                       h2h_weight * (h2h_away_stats.get('crosses_stp_pct_avg', 0) if h2h_away_stats else away_stats.get('crosses_stp_pct_avg', 0)))
    
    return {
        'home_shots': round(home_shots, 1),
        'away_shots': round(away_shots, 1),
        'home_poss': round(home_poss, 1),
        'away_poss': round(away_poss, 1),
        'home_psxg': round(home_psxg, 2),
        'away_psxg': round(away_psxg, 2),
        'home_crosses_stp': round(home_crosses_stp, 1),
        'away_crosses_stp': round(away_crosses_stp, 1)
    }


def get_recent_form(dataframe, team, n_matches=5, use_api_data=True):
    """
    Get recent form for a team (last N matches).
    
    Now uses API data only for recent form (cross-league for teams in multiple competitions).
    For teams in UCL + domestic league: shows last 5 matches across all leagues
    For teams in single league: shows last 5 matches from their league only
    
    Args:
        dataframe: Master dataframe (kept for compatibility, but not used if use_api_data=True)
        team: Team name (normalized)
        n_matches: Number of recent matches to return
        use_api_data: If True, use API data instead of historical CSVs
        
    Returns:
        list: List of result symbols (✅, ❌, ➖)
    """
    import os
    
    if use_api_data:
        # Load API data
        api_path = 'data/api_results_basic.csv'
        if not os.path.exists(api_path):
            # Fallback to historical data if API not available
            return _get_recent_form_from_historical(dataframe, team, n_matches)
        
        try:
            api_df = pd.read_csv(api_path, dtype=str)
            
            # Clean dates
            if 'utcDate' in api_df.columns:
                api_df['date'] = pd.to_datetime(api_df['utcDate'], errors='coerce').dt.normalize()
            else:
                api_df['date'] = pd.NaT
            
            # Normalize team names
            from utils.team_name_normalizer import normalize_team_name
            api_df['home_team_normalized'] = api_df['home_team_name'].apply(normalize_team_name)
            api_df['away_team_normalized'] = api_df['away_team_name'].apply(normalize_team_name)
            
            # Filter: Only finished matches with valid scores
            api_df = api_df[
                (api_df['status'] == 'FINISHED') &
                (api_df['home_team_name'].notna()) &
                (api_df['home_team_name'] != 'nan') &
                (api_df['away_team_name'].notna()) &
                (api_df['away_team_name'] != 'nan') &
                (api_df['score_fullTime_home'].notna()) &
                (api_df['score_fullTime_home'] != 'nan') &
                (api_df['score_fullTime_away'].notna()) &
                (api_df['score_fullTime_away'] != 'nan')
            ].copy()
            
            if len(api_df) == 0:
                return _get_recent_form_from_historical(dataframe, team, n_matches)
            
            # Find all matches where team appears (home or away)
            team_matches = api_df[
                (api_df['home_team_normalized'] == team) | 
                (api_df['away_team_normalized'] == team)
            ].copy()
            
            if len(team_matches) == 0:
                return _get_recent_form_from_historical(dataframe, team, n_matches)
            
            # Check if team is in multiple leagues (UCL + domestic)
            team_leagues = set(team_matches['league'].dropna().unique())
            has_ucl = 'UEFA Champions League' in team_leagues
            has_domestic = len([lg for lg in team_leagues if lg != 'UEFA Champions League']) > 0
            
            # If team is in UCL + domestic league, show matches from all leagues
            # Otherwise, show matches from their primary league only
            if has_ucl and has_domestic:
                # Cross-league: Use all matches
                filtered_matches = team_matches.copy()
            else:
                # Single league: Use matches from their league only
                # Get the most common league (primary league)
                primary_league = team_matches['league'].mode()[0] if len(team_matches['league'].mode()) > 0 else None
                if primary_league:
                    filtered_matches = team_matches[team_matches['league'] == primary_league].copy()
                else:
                    filtered_matches = team_matches.copy()
            
            if len(filtered_matches) == 0:
                return _get_recent_form_from_historical(dataframe, team, n_matches)
            
            # Sort by date (most recent last) and get last N matches
            filtered_matches = filtered_matches.sort_values('date').tail(n_matches)
            
            # Calculate form from scores
            form = []
            for _, row in filtered_matches.iterrows():
                home_team = row.get('home_team_normalized', '')
                away_team = row.get('away_team_normalized', '')
                
                try:
                    home_score = float(row.get('score_fullTime_home', 0) or 0)
                    away_score = float(row.get('score_fullTime_away', 0) or 0)
                    
                    if team == home_team or str(team).lower() == str(home_team).lower():
                        # Team is home
                        if home_score > away_score:
                            form.append('✅')
                        elif home_score < away_score:
                            form.append('❌')
                        else:
                            form.append('➖')
                    elif team == away_team or str(team).lower() == str(away_team).lower():
                        # Team is away
                        if away_score > home_score:
                            form.append('✅')
                        elif away_score < home_score:
                            form.append('❌')
                        else:
                            form.append('➖')
                    else:
                        form.append('?')
                except:
                    form.append('?')
            
            return form
            
        except Exception as e:
            print(f"Error loading API data for recent form: {e}")
            return _get_recent_form_from_historical(dataframe, team, n_matches)
    
    else:
        # Fallback to historical data
        return _get_recent_form_from_historical(dataframe, team, n_matches)


def _get_recent_form_from_historical(dataframe, team, n_matches=5):
    """
    Get recent form from historical CSVs (fallback method).
    """
    team_df = dataframe[dataframe['team'] == team].copy()
    
    if len(team_df) == 0:
        return []
    
    # Sort by date (most recent last)
    team_df = team_df.sort_values('date').tail(n_matches)
    
    form = []
    for _, row in team_df.iterrows():
        # Check if Result column exists (historical CSVs)
        if 'Result' in row and pd.notna(row.get('Result')):
            result = str(row.get('Result', '')).upper()
            if result == 'W':
                form.append('✅')
            elif result == 'L':
                form.append('❌')
            elif result == 'D':
                form.append('➖')
            else:
                form.append('?')
        # If no Result column, calculate from scores
        elif 'GF' in row and 'GA' in row and pd.notna(row.get('GF')) and pd.notna(row.get('GA')):
            try:
                gf = float(row.get('GF', 0))
                ga = float(row.get('GA', 0))
                if gf > ga:
                    form.append('✅')
                elif gf < ga:
                    form.append('❌')
                else:
                    form.append('➖')
            except:
                form.append('?')
        else:
            form.append('?')
    
    return form
