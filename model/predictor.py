"""
Main prediction orchestrator using team strengths system.
"""

import numpy as np
import pandas as pd
from utils.metrics_calculator import (
    compute_team_strengths,
    get_league_goal_averages,
    calculate_expected_stats,
    get_recent_form
)
from utils.weighting import get_current_season
from utils.strengths import (
    get_all_league_strengths,
    load_team_strengths,
    get_ucl_adjusted_team_strength,
    compute_global_strength_baseline,
)
from model.poisson_model import expected_goals, calculate_match_probabilities
from model.dixon_coles import calculate_dixon_coles_probabilities
from model.monte_carlo import simulate_match, blend_probabilities
from utils.team_name_normalizer import normalize_team_name
from utils.feature_engineering import get_head_to_head


def predict_match(dataframe, home_team, away_team, league, current_season=None):
    """
    Predict match outcome using strengths-based Poisson + Dixon-Coles + Monte Carlo.
    
    Args:
        dataframe: Master dataframe with all match data
        home_team: Home team name
        away_team: Away team name
        league: League name (or league of home team for cross-league matches)
        current_season: Most recent season (auto-detected if None)
        
    Returns:
        dict: Complete prediction results
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    # Normalize team names
    home_team = normalize_team_name(home_team)
    away_team = normalize_team_name(away_team)
    
    # Determine actual leagues for both teams
    # IMPORTANT: Prioritize the specified league, then check other leagues
    home_league = league.upper()
    away_league = league.upper()
    
    # Check if teams exist in the SPECIFIED league first (most important)
    home_team_in_specified = len(dataframe[(dataframe['team'] == home_team) & (dataframe['league'] == league.upper())]) > 0
    away_team_in_specified = len(dataframe[(dataframe['team'] == away_team) & (dataframe['league'] == league.upper())]) > 0
    
    # If teams exist in specified league, use that league
    if home_team_in_specified:
        home_league = league.upper()
    else:
        # Check if team exists in other leagues (for cross-league matches)
        home_team_leagues = dataframe[dataframe['team'] == home_team]['league'].unique()
        if len(home_team_leagues) > 0:
            # Prefer domestic leagues over UCL for non-UCL predictions
            if league.upper() != 'UCL':
                domestic_leagues = [lg for lg in home_team_leagues if lg != 'UCL']
                if len(domestic_leagues) > 0:
                    home_league = domestic_leagues[0]
                else:
                    home_league = home_team_leagues[0]
            else:
                home_league = home_team_leagues[0]
    
    if away_team_in_specified:
        away_league = league.upper()
    else:
        # Check if team exists in other leagues (for cross-league matches)
        away_team_leagues = dataframe[dataframe['team'] == away_team]['league'].unique()
        if len(away_team_leagues) > 0:
            # Prefer domestic leagues over UCL for non-UCL predictions
            if league.upper() != 'UCL':
                domestic_leagues = [lg for lg in away_team_leagues if lg != 'UCL']
                if len(domestic_leagues) > 0:
                    away_league = domestic_leagues[0]
                else:
                    away_league = away_team_leagues[0]
            else:
                away_league = away_team_leagues[0]
    
    # Load or compute team strengths for both leagues
    all_strengths = get_all_league_strengths()
    
    # Compute strengths if not already computed
    if home_league not in all_strengths or home_team not in all_strengths.get(home_league, {}):
        print(f"Computing strengths for {home_league}...")
        compute_team_strengths(dataframe, home_league, current_season)
        all_strengths = get_all_league_strengths()
    
    if away_league not in all_strengths or away_team not in all_strengths.get(away_league, {}):
        print(f"Computing strengths for {away_league}...")
        compute_team_strengths(dataframe, away_league, current_season)
        all_strengths = get_all_league_strengths()
    
    # Get team strengths (with UCL adjustments when predicting UCL matches)
    if home_league.upper() == 'UCL' or away_league.upper() == 'UCL' or league.upper() == 'UCL':
        # Clone strengths to avoid mutating cached dict
        adjusted_strengths = {lg: dict(teams) for lg, teams in all_strengths.items()}
        
        # Adjust home team if in UCL context
        if home_league.upper() == 'UCL':
            adjusted_home = get_ucl_adjusted_team_strength(home_team, adjusted_strengths)
            adjusted_strengths.setdefault('UCL', {})[home_team] = adjusted_home
        
        # Adjust away team if in UCL context
        if away_league.upper() == 'UCL':
            adjusted_away = get_ucl_adjusted_team_strength(away_team, adjusted_strengths)
            adjusted_strengths.setdefault('UCL', {})[away_team] = adjusted_away
        
        # Use adjusted strengths for lookups and downstream λ computation
        all_strengths = adjusted_strengths
    
    home_strengths = all_strengths.get(home_league, {}).get(home_team)
    away_strengths = all_strengths.get(away_league, {}).get(away_team)
    
    # CRITICAL FIX: If teams don't have strengths, use league priors
    # This handles teams that exist in fixtures but not in master data
    if not home_strengths:
        from utils.metrics_calculator import compute_league_baselines, assign_prior_strengths
        # Check if team exists in dataframe at all before warning
        team_exists = len(dataframe[dataframe['team'] == home_team]) > 0
        if team_exists:
            print(f"[WARNING] {home_team} exists in data but no strengths found in {home_league} - recomputing...")
            # Try recomputing strengths once more
            compute_team_strengths(dataframe, home_league, current_season)
            all_strengths = get_all_league_strengths()
            home_strengths = all_strengths.get(home_league, {}).get(home_team)
            
            # If still not found, use priors
            if not home_strengths:
                print(f"[WARNING] {home_team} not found in {home_league} after recompute - using league priors")
                league_baselines = compute_league_baselines(dataframe, home_league, current_season)
                if league_baselines:
                    prior_strengths = assign_prior_strengths(league_baselines)
                    home_strengths = {
                        "home_attack": round(prior_strengths["home_attack"], 3),
                        "away_attack": round(prior_strengths["away_attack"], 3),
                        "home_defense": round(prior_strengths["home_defense"], 3),
                        "away_defense": round(prior_strengths["away_defense"], 3),
                        "strength": round(prior_strengths["strength"], 3),
                        "is_new_team": True
                    }
                    all_strengths.setdefault(home_league, {})[home_team] = home_strengths
                else:
                    home_strengths = {
                        "home_attack": 1.0,
                        "away_attack": 1.0,
                        "home_defense": 1.0,
                        "away_defense": 1.0,
                        "strength": 1.0,
                        "is_new_team": True
                    }
                    all_strengths.setdefault(home_league, {})[home_team] = home_strengths
        else:
            print(f"[WARNING] {home_team} not found in data - using league priors")
            league_baselines = compute_league_baselines(dataframe, home_league, current_season)
            if league_baselines:
                prior_strengths = assign_prior_strengths(league_baselines)
                home_strengths = {
                    "home_attack": round(prior_strengths["home_attack"], 3),
                    "away_attack": round(prior_strengths["away_attack"], 3),
                    "home_defense": round(prior_strengths["home_defense"], 3),
                    "away_defense": round(prior_strengths["away_defense"], 3),
                    "strength": round(prior_strengths["strength"], 3),
                    "is_new_team": True
                }
                all_strengths.setdefault(home_league, {})[home_team] = home_strengths
            else:
                home_strengths = {
                    "home_attack": 1.0,
                    "away_attack": 1.0,
                    "home_defense": 1.0,
                    "away_defense": 1.0,
                    "strength": 1.0,
                    "is_new_team": True
                }
                all_strengths.setdefault(home_league, {})[home_team] = home_strengths
    
    if not away_strengths:
        from utils.metrics_calculator import compute_league_baselines, assign_prior_strengths
        # Check if team exists in dataframe at all before warning
        team_exists = len(dataframe[dataframe['team'] == away_team]) > 0
        if team_exists:
            print(f"[WARNING] {away_team} exists in data but no strengths found in {away_league} - recomputing...")
            # Try recomputing strengths once more
            compute_team_strengths(dataframe, away_league, current_season)
            all_strengths = get_all_league_strengths()
            away_strengths = all_strengths.get(away_league, {}).get(away_team)
            
            # If still not found, use priors
            if not away_strengths:
                print(f"[WARNING] {away_team} not found in {away_league} after recompute - using league priors")
                league_baselines = compute_league_baselines(dataframe, away_league, current_season)
                if league_baselines:
                    prior_strengths = assign_prior_strengths(league_baselines)
                    away_strengths = {
                        "home_attack": round(prior_strengths["home_attack"], 3),
                        "away_attack": round(prior_strengths["away_attack"], 3),
                        "home_defense": round(prior_strengths["home_defense"], 3),
                        "away_defense": round(prior_strengths["away_defense"], 3),
                        "strength": round(prior_strengths["strength"], 3),
                        "is_new_team": True
                    }
                    all_strengths.setdefault(away_league, {})[away_team] = away_strengths
                else:
                    away_strengths = {
                        "home_attack": 1.0,
                        "away_attack": 1.0,
                        "home_defense": 1.0,
                        "away_defense": 1.0,
                        "strength": 1.0,
                        "is_new_team": True
                    }
                    all_strengths.setdefault(away_league, {})[away_team] = away_strengths
        else:
            print(f"[WARNING] {away_team} not found in data - using league priors")
            league_baselines = compute_league_baselines(dataframe, away_league, current_season)
            if league_baselines:
                prior_strengths = assign_prior_strengths(league_baselines)
                away_strengths = {
                    "home_attack": round(prior_strengths["home_attack"], 3),
                    "away_attack": round(prior_strengths["away_attack"], 3),
                    "home_defense": round(prior_strengths["home_defense"], 3),
                    "away_defense": round(prior_strengths["away_defense"], 3),
                    "strength": round(prior_strengths["strength"], 3),
                    "is_new_team": True
                }
                all_strengths.setdefault(away_league, {})[away_team] = away_strengths
            else:
                away_strengths = {
                    "home_attack": 1.0,
                    "away_attack": 1.0,
                    "home_defense": 1.0,
                    "away_defense": 1.0,
                    "strength": 1.0,
                    "is_new_team": True
                }
                all_strengths.setdefault(away_league, {})[away_team] = away_strengths
    
    # Log if teams are new/promoted
    from utils.metrics_calculator import TEAM_METADATA
    
    if home_strengths and home_strengths.get('is_new_team', False):
        print(f"[INFO] {home_team} flagged as a new or promoted team — using league priors.")
    
    if away_strengths and away_strengths.get('is_new_team', False):
        print(f"[INFO] {away_team} flagged as a new or promoted team — using league priors.")
    
    # Also check metadata for partial data teams
    home_league_meta = TEAM_METADATA.get(home_league.upper(), {})
    away_league_meta = TEAM_METADATA.get(away_league.upper(), {})
    
    home_team_meta = home_league_meta.get(home_team, {})
    away_team_meta = away_league_meta.get(away_team, {})
    
    if home_team_meta.get('n_matches', 0) < 10 and home_team_meta.get('n_matches', 0) > 0:
        print(f"[INFO] {home_team} has limited data ({home_team_meta.get('n_matches', 0)} matches) — using blended strengths.")
    
    if away_team_meta.get('n_matches', 0) < 10 and away_team_meta.get('n_matches', 0) > 0:
        print(f"[INFO] {away_team} has limited data ({away_team_meta.get('n_matches', 0)} matches) — using blended strengths.")
    
    # Calculate expected goals using strengths
    λ_home, λ_away = expected_goals(
        home_team, away_team, home_league, away_league,
        dataframe=dataframe, strengths_dict=all_strengths
    )
    
    # Apply H2H adjustment if available
    h2h_df = get_head_to_head(dataframe, home_team, away_team, current_season=current_season)
    
    if len(h2h_df) >= 3:
        home_h2h = h2h_df[h2h_df['team'] == home_team]
        away_h2h = h2h_df[h2h_df['team'] == away_team]
        
        if len(home_h2h) > 0 and len(away_h2h) > 0:
            h2h_home_xg = home_h2h['xG'].fillna(0).mean()
            h2h_away_xg = away_h2h['xG'].fillna(0).mean()
            
            # Blend with strengths-based lambdas (20% H2H weight)
            λ_home = 0.8 * λ_home + 0.2 * h2h_home_xg
            λ_away = 0.8 * λ_away + 0.2 * h2h_away_xg
    
    # Lambda clipping is done in expected_goals function
    # No need to clip again here
    
    # Dixon-Coles probabilities
    dixon_coles_probs = calculate_dixon_coles_probabilities(λ_home, λ_away)
    
    # Monte Carlo simulation
    monte_carlo_probs = simulate_match(λ_home, λ_away, n_simulations=10000)
    
    # Blend probabilities (70% Dixon-Coles, 30% Monte Carlo)
    final_probs = blend_probabilities(dixon_coles_probs, monte_carlo_probs, poisson_weight=0.7)
    
    # Calculate expected stats
    expected_stats = calculate_expected_stats(dataframe, home_team, away_team, league, current_season=current_season)
    
    # Get recent form
    home_form = get_recent_form(dataframe, home_team)
    away_form = get_recent_form(dataframe, away_team)
    
    # Determine best bets
    best_bet = determine_best_bet(final_probs['home_win'], final_probs['draw'], final_probs['away_win'])
    best_prop = determine_best_prop(final_probs['btts'], final_probs['over_15'], final_probs['over_25'])
    
    return {
        'home_team': home_team,
        'away_team': away_team,
        'league': league,
        'lambda_home': round(λ_home, 2),
        'lambda_away': round(λ_away, 2),
        'home_strengths': home_strengths,
        'away_strengths': away_strengths,
        'probabilities': {
            'home_win': round(final_probs['home_win'] * 100, 1),
            'draw': round(final_probs['draw'] * 100, 1),
            'away_win': round(final_probs['away_win'] * 100, 1),
            'btts': round(final_probs['btts'] * 100, 1),
            'over_15': round(final_probs['over_15'] * 100, 1),
            'over_25': round(final_probs['over_25'] * 100, 1),
            'over_35': round(final_probs.get('over_35', 0) * 100, 1)
        },
        'correct_scores': final_probs.get('correct_scores', {}),
        'expected_stats': expected_stats,
        'home_form': home_form,
        'away_form': away_form,
        'best_bet': best_bet,
        'best_prop': best_prop
    }


def determine_best_bet(home_win_prob, draw_prob, away_win_prob):
    """
    Determine best bet based on probabilities.
    
    Args:
        home_win_prob: Probability of home win (0-1)
        draw_prob: Probability of draw (0-1)
        away_win_prob: Probability of away win (0-1)
        
    Returns:
        str: Best bet suggestion
    """
    max_prob = max(home_win_prob, draw_prob, away_win_prob)
    
    if max_prob >= 0.55:  # 55% threshold (as per requirements)
        if max_prob == home_win_prob:
            return "Home Win"
        elif max_prob == away_win_prob:
            return "Away Win"
        else:
            return "Draw"
    else:
        return "No strong bet"


def determine_best_prop(btts_prob, over_15_prob, over_25_prob):
    """
    Determine best prop bet(s).
    
    Args:
        btts_prob: Probability of BTTS (0-1)
        over_15_prob: Probability of Over 1.5 goals (0-1)
        over_25_prob: Probability of Over 2.5 goals (0-1)
        
    Returns:
        str: Best prop bet suggestion(s) - shows all props above 65% threshold
    """
    props = []
    threshold = 0.65
    
    # Check each prop and add if above threshold
    if btts_prob >= threshold:
        props.append("BTTS Yes")
    
    if over_25_prob >= threshold:
        props.append("Over 2.5")
    
    if over_15_prob >= threshold:
        props.append("Over 1.5")
    
    # Return all props that meet threshold, or "No Value Bet" if none
    if len(props) > 0:
        return " | ".join(props)
    else:
        return "No Value Bet"
