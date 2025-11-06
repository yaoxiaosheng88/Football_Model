"""
Monte Carlo simulation for match outcome prediction.
Runs 10,000 simulations and computes comprehensive probabilities.
"""

import numpy as np
from scipy import stats


def simulate_match(lambda_home, lambda_away, n_simulations=10000):
    """
    Simulate a match N times using Poisson distributions.
    
    Args:
        lambda_home: Expected goals for home team
        lambda_away: Expected goals for away team
        n_simulations: Number of simulations to run (default: 10000)
        
    Returns:
        dict: Simulation results with probabilities for all outcomes
    """
    # Set random seed for reproducibility (optional)
    np.random.seed(None)
    
    # Simulate goals for each team
    home_goals = np.random.poisson(lambda_home, n_simulations)
    away_goals = np.random.poisson(lambda_away, n_simulations)
    
    # Count outcomes
    # Note: home_goals and away_goals are correctly labeled
    home_wins = np.sum(home_goals > away_goals)
    away_wins = np.sum(away_goals > home_goals)
    draws = np.sum(home_goals == away_goals)
    
    # BTTS (Both Teams To Score)
    btts = np.sum((home_goals > 0) & (away_goals > 0))
    
    # Total goals
    total_goals = home_goals + away_goals
    over_15 = np.sum(total_goals > 1.5)
    over_25 = np.sum(total_goals > 2.5)
    over_35 = np.sum(total_goals > 3.5)
    
    # Calculate correct score probabilities (most common scores)
    score_counts = {}
    for h, a in zip(home_goals, away_goals):
        # Limit to reasonable scores (0-5 goals each)
        if h <= 5 and a <= 5:
            score_key = f"{h}-{a}"
            score_counts[score_key] = score_counts.get(score_key, 0) + 1
    
    # Convert to probabilities and get top 5
    total_valid = sum(count for key, count in score_counts.items())
    correct_scores = {}
    if total_valid > 0:
        for score, count in sorted(score_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            correct_scores[score] = count / total_valid
    
    # Calculate probabilities
    return {
        'home_win': home_wins / n_simulations,
        'draw': draws / n_simulations,
        'away_win': away_wins / n_simulations,
        'btts': btts / n_simulations,
        'over_15': over_15 / n_simulations,
        'over_25': over_25 / n_simulations,
        'over_35': over_35 / n_simulations,
        'home_goals_mean': np.mean(home_goals),
        'away_goals_mean': np.mean(away_goals),
        'total_goals_mean': np.mean(total_goals),
        'correct_scores': correct_scores
    }


def blend_probabilities(poisson_probs, monte_carlo_probs, poisson_weight=0.7):
    """
    Blend Poisson/Dixon-Coles and Monte Carlo probabilities.
    
    Args:
        poisson_probs: Dictionary of probabilities from Poisson/Dixon-Coles
        monte_carlo_probs: Dictionary of probabilities from Monte Carlo
        poisson_weight: Weight for Poisson model (0-1)
        
    Returns:
        dict: Blended probabilities
    """
    mc_weight = 1.0 - poisson_weight
    
    blended = {}
    for key in poisson_probs.keys():
        if key in monte_carlo_probs:
            blended[key] = (poisson_weight * poisson_probs[key] + 
                           mc_weight * monte_carlo_probs[key])
        else:
            blended[key] = poisson_probs[key]
    
    # Add over_3.5 and correct_scores from Monte Carlo if available
    if 'over_35' in monte_carlo_probs:
        blended['over_35'] = monte_carlo_probs['over_35']
    
    if 'correct_scores' in monte_carlo_probs:
        blended['correct_scores'] = monte_carlo_probs['correct_scores']
    
    return blended
