"""
Poisson model for goal prediction using team strengths and league base rates.
"""

import numpy as np
from scipy import stats
from utils.metrics_calculator import get_league_base_rates, compute_league_base_rates
from utils.strengths import get_all_league_strengths

# Cross-league strength factors
LEAGUE_FACTORS = {
    'EPL': 1.45,
    'LALIGA': 1.45,
    'BUNDESLIGA': 1.40,
    'SERIEA': 1.40,
    'LIGUE1': 1.40,
    'MLS': 1.20,
    'UCL': 1.55
}
GLOBAL_LEAGUE_MEAN = sum(LEAGUE_FACTORS.values()) / len(LEAGUE_FACTORS)


def expected_goals(home_team, away_team, home_league, away_league, dataframe=None, strengths_dict=None):
    """
    Calculate expected goals using team strengths and league base rates.
    
    Formula:
    lambda_home = base_home_goals * home_attack_strength * (1 / away_defence_strength)
    lambda_away = base_away_goals * away_attack_strength * (1 / home_defence_strength)
    
    Note: Defense strength is inverted because low defense_strength (< 1.0) means weak defense,
    so we invert to get higher multiplier for expected goals.
    
    For cross-league matches, normalize by league base rates.
    
    Args:
        home_team: Home team name
        away_team: Away team name
        home_league: League of home team (e.g., 'EPL')
        away_league: League of away team (e.g., 'UCL')
        dataframe: Master dataframe (for computing base rates if needed)
        strengths_dict: Dictionary of all league strengths (if None, loads from files)
        
    Returns:
        tuple: (lambda_home, lambda_away) - clipped between 0.1 and 3.5
    """
    # Load strengths if not provided
    if strengths_dict is None:
        strengths_dict = get_all_league_strengths()
    
    # Get team strengths
    home_strengths = strengths_dict.get(home_league.upper(), {}).get(home_team)
    away_strengths = strengths_dict.get(away_league.upper(), {}).get(away_team)
    
    if not home_strengths or not away_strengths:
        # Fallback to default values
        return 1.5, 1.2
    
    # Get league base rates
    # For same league, use the same base rates for both teams
    if home_league.upper() == away_league.upper():
        base_home_goals, base_away_goals = get_league_base_rates(home_league)
        
        # If not in cache, compute them
        if base_home_goals == 1.5 and dataframe is not None:
            base_home_goals, base_away_goals = compute_league_base_rates(dataframe, home_league)
        
        # Situational factors (attack vs defense) with defense inverted
        situational_home = home_strengths['home_attack'] * (1.0 / max(away_strengths['away_defense'], 1e-6))
        situational_away = away_strengths['away_attack'] * (1.0 / max(home_strengths['home_defense'], 1e-6))

        # Overall power ratios
        overall_home = max(home_strengths.get('strength', 1.0), 1e-6)
        overall_away = max(away_strengths.get('strength', 1.0), 1e-6)
        overall_ratio_home = overall_home / overall_away
        overall_ratio_away = overall_away / overall_home

        # Blend situational and overall
        λ_home = base_home_goals * (0.8 * situational_home + 0.2 * overall_ratio_home)
        λ_away = base_away_goals * (0.8 * situational_away + 0.2 * overall_ratio_away)
    else:
        # Cross-league match - get base rates for both leagues
        home_base_home, home_base_away = get_league_base_rates(home_league)
        away_base_home, away_base_away = get_league_base_rates(away_league)
        
        # If not in cache, compute them
        if home_base_home == 1.5 and dataframe is not None:
            home_base_home, home_base_away = compute_league_base_rates(dataframe, home_league)
        if away_base_home == 1.5 and dataframe is not None:
            away_base_home, away_base_away = compute_league_base_rates(dataframe, away_league)
        
        # Situational and overall ratios
        situational_home = home_strengths['home_attack'] * (1.0 / max(away_strengths['away_defense'], 1e-6))
        situational_away = away_strengths['away_attack'] * (1.0 / max(home_strengths['home_defense'], 1e-6))
        overall_home = max(home_strengths.get('strength', 1.0), 1e-6)
        overall_away = max(away_strengths.get('strength', 1.0), 1e-6)
        overall_ratio_home = overall_home / overall_away
        overall_ratio_away = overall_away / overall_home

        # Base from each league
        λ_home = home_base_home * (0.8 * situational_home + 0.2 * overall_ratio_home)
        λ_away = away_base_away * (0.8 * situational_away + 0.2 * overall_ratio_away)

        # Apply league strength normalization factors
        lf_home = LEAGUE_FACTORS.get(home_league.upper(), GLOBAL_LEAGUE_MEAN)
        lf_away = LEAGUE_FACTORS.get(away_league.upper(), GLOBAL_LEAGUE_MEAN)
        if GLOBAL_LEAGUE_MEAN > 0:
            λ_home *= (lf_home / GLOBAL_LEAGUE_MEAN)
            λ_away *= (lf_away / GLOBAL_LEAGUE_MEAN)
    
    # Clip lambda values between 0.1 and 3.5 to avoid extreme outliers
    λ_home = np.clip(λ_home, 0.1, 3.5)
    λ_away = np.clip(λ_away, 0.1, 3.5)
    
    return λ_home, λ_away


def poisson_probability(goals, lambda_param):
    """
    Calculate Poisson probability for a number of goals.
    
    Args:
        goals: Number of goals
        lambda_param: Expected goals (lambda)
        
    Returns:
        float: Probability
    """
    if lambda_param <= 0:
        return 0.0
    
    return stats.poisson.pmf(goals, lambda_param)


def build_poisson_matrix(lambda_home, lambda_away, max_goals=6):
    """
    Build probability matrix for all goal combinations.
    
    Args:
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        max_goals: Maximum goals to consider per team
        
    Returns:
        np.array: 2D array of probabilities (home_goals, away_goals)
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = poisson_probability(i, lambda_home) * poisson_probability(j, lambda_away)
    
    return matrix


def calculate_match_probabilities(lambda_home, lambda_away, max_goals=6):
    """
    Calculate match outcome probabilities using Poisson model.
    
    Args:
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        max_goals: Maximum goals to consider
        
    Returns:
        dict: Probabilities for home win, draw, away win, BTTS, over 1.5, over 2.5
    """
    matrix = build_poisson_matrix(lambda_home, lambda_away, max_goals)
    
    # Home win: home goals > away goals (i > j, which is lower triangle in numpy)
    home_win = np.sum(np.tril(matrix, k=-1))
    
    # Away win: away goals > home goals (i < j, which is upper triangle in numpy)
    away_win = np.sum(np.triu(matrix, k=1))
    
    # Draw: home goals == away goals
    draw = np.trace(matrix)
    
    # Normalize
    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw /= total
        away_win /= total
    
    # BTTS (Both Teams To Score): both teams score at least 1
    btts = 0.0
    for i in range(1, max_goals + 1):
        for j in range(1, max_goals + 1):
            btts += matrix[i, j]
    
    # Over 1.5 goals
    over_15 = 1.0 - (matrix[0, 0] + matrix[0, 1] + matrix[1, 0])
    
    # Over 2.5 goals
    over_25 = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j > 2.5:
                over_25 += matrix[i, j]
    
    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        'btts': btts,
        'over_15': over_15,
        'over_25': over_25
    }
