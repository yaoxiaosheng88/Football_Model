"""
Dixon-Coles adjustment for low-score correction in Poisson model.
"""

import numpy as np
from scipy import stats


def dixon_coles_correction(goals_home, goals_away, lambda_home, lambda_away, rho=-0.13):
    """
    Apply Dixon-Coles correction factor for low-score games (0-0, 1-0, 0-1, 1-1).
    
    The correction factor is:
    τ(i, j, λ_home, λ_away) = 1 - λ_home * λ_away * ρ if (i, j) in {(0,0), (0,1), (1,0), (1,1)}
    Otherwise: 1
    
    Args:
        goals_home: Home team goals
        goals_away: Away team goals
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        rho: Correlation parameter (typically -0.13)
        
    Returns:
        float: Correction factor
    """
    # Low-score scenarios that need correction
    low_scores = [(0, 0), (0, 1), (1, 0), (1, 1)]
    
    if (goals_home, goals_away) in low_scores:
        correction = 1 - lambda_home * lambda_away * rho
        return max(0, correction)  # Ensure non-negative
    else:
        return 1.0


def dixon_coles_probability(goals_home, goals_away, lambda_home, lambda_away, rho=-0.13):
    """
    Calculate Dixon-Coles adjusted probability.
    
    Args:
        goals_home: Home team goals
        goals_away: Away team goals
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        rho: Correlation parameter
        
    Returns:
        float: Adjusted probability
    """
    # Base Poisson probability
    poisson_prob = (stats.poisson.pmf(goals_home, lambda_home) * 
                   stats.poisson.pmf(goals_away, lambda_away))
    
    # Apply correction
    correction = dixon_coles_correction(goals_home, goals_away, lambda_home, lambda_away, rho)
    
    return poisson_prob * correction


def build_dixon_coles_matrix(lambda_home, lambda_away, max_goals=6, rho=-0.13):
    """
    Build Dixon-Coles adjusted probability matrix.
    
    Args:
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        max_goals: Maximum goals to consider
        rho: Correlation parameter
        
    Returns:
        np.array: 2D array of adjusted probabilities
    """
    matrix = np.zeros((max_goals + 1, max_goals + 1))
    
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            matrix[i, j] = dixon_coles_probability(i, j, lambda_home, lambda_away, rho)
    
    # Normalize matrix so probabilities sum to 1
    total = matrix.sum()
    if total > 0:
        matrix = matrix / total
    
    return matrix


def calculate_dixon_coles_probabilities(lambda_home, lambda_away, max_goals=6, rho=-0.13):
    """
    Calculate match outcome probabilities using Dixon-Coles model.
    
    Args:
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        max_goals: Maximum goals to consider
        rho: Correlation parameter
        
    Returns:
        dict: Probabilities for all outcomes
    """
    matrix = build_dixon_coles_matrix(lambda_home, lambda_away, max_goals, rho)
    
    # Home win: home goals > away goals (i > j, which is lower triangle in numpy)
    home_win = np.sum(np.tril(matrix, k=-1))
    
    # Away win: away goals > home goals (i < j, which is upper triangle in numpy)
    away_win = np.sum(np.triu(matrix, k=1))
    
    # Draw: home goals == away goals
    draw = np.trace(matrix)
    
    # BTTS (Both Teams To Score)
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
    
    # Over 3.5 goals
    over_35 = 0.0
    for i in range(max_goals + 1):
        for j in range(max_goals + 1):
            if i + j > 3.5:
                over_35 += matrix[i, j]
    
    return {
        'home_win': home_win,
        'draw': draw,
        'away_win': away_win,
        'btts': btts,
        'over_15': over_15,
        'over_25': over_25,
        'over_35': over_35
    }

