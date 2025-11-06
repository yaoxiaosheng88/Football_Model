"""
Feature engineering module.
Computes team averages, league averages, and relative strengths.
"""

import pandas as pd
import numpy as np
from .weighting import apply_season_weights, get_current_season, apply_dynamic_season_weights


def compute_team_season_averages(dataframe, team, season_year=None, venue=None, 
                                  current_season=None):
    """
    Compute weighted seasonal averages for a team.
    
    Args:
        dataframe: Master dataframe with all match data
        team: Team name
        season_year: If provided, filter to this season
        venue: 'Home' or 'Away' (optional filter)
        current_season: Most recent season for weighting
        
    Returns:
        dict: Dictionary of average metrics
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    # Filter by team
    team_df = dataframe[dataframe['team'] == team].copy()
    
    if len(team_df) == 0:
        return None
    
    # Apply dynamic, team-specific season weights
    team_df = apply_dynamic_season_weights(team_df, team, current_season)
    
    # Filter by venue if specified
    if venue:
        team_df = team_df[team_df['Venue'] == venue]
    
    # Filter by season if specified
    if season_year:
        team_df = team_df[team_df['season_year'] == season_year]
    
    if len(team_df) == 0:
        return None
    
    # Compute weighted averages
    weights = team_df['season_weight'].values
    total_weight = weights.sum()
    
    if total_weight == 0:
        return None
    
    metrics = {
        'xg_avg': np.average(team_df['xG'].fillna(0), weights=weights),
        'xga_avg': np.average(team_df['xGA'].fillna(0), weights=weights),
        'gf_avg': np.average(team_df['GF'].fillna(0), weights=weights),
        'ga_avg': np.average(team_df['GA'].fillna(0), weights=weights),
        'shots_avg': np.average(team_df['Standard_Sh'].fillna(0), weights=weights),
        'sot_avg': np.average(team_df['Standard_SoT'].fillna(0), weights=weights),
        'poss_avg': np.average(team_df['Poss'].fillna(0), weights=weights),
        'psxg_avg': np.average(team_df['Performance_PSxG'].fillna(0), weights=weights),
        'crosses_stp_pct_avg': np.average(team_df['Crosses_Stp%'].fillna(0), weights=weights),
        'matches': len(team_df)
    }
    
    return metrics


def compute_league_averages(dataframe, league, venue=None, current_season=None):
    """
    Compute league-wide averages.
    
    Args:
        dataframe: Master dataframe
        league: League name (e.g., 'EPL')
        venue: 'Home' or 'Away' (optional)
        current_season: Most recent season for weighting
        
    Returns:
        dict: Dictionary of league averages
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    # Filter by league
    league_df = dataframe[dataframe['league'] == league].copy()
    
    if len(league_df) == 0:
        return None
    
    # Apply season weights
    league_df = apply_season_weights(league_df, current_season)
    
    # Filter by venue if specified
    if venue:
        league_df = league_df[league_df['Venue'] == venue]
    
    # Compute weighted averages
    weights = league_df['season_weight'].values
    total_weight = weights.sum()
    
    if total_weight == 0:
        return None
    
    metrics = {
        'xg_avg': np.average(league_df['xG'].fillna(0), weights=weights),
        'xga_avg': np.average(league_df['xGA'].fillna(0), weights=weights),
        'gf_avg': np.average(league_df['GF'].fillna(0), weights=weights),
        'ga_avg': np.average(league_df['GA'].fillna(0), weights=weights),
        'shots_avg': np.average(league_df['Standard_Sh'].fillna(0), weights=weights),
        'poss_avg': np.average(league_df['Poss'].fillna(0), weights=weights),
    }
    
    # Add PSxG if available
    if 'Performance_PSxG' in league_df.columns:
        metrics['psxg_avg'] = np.average(league_df['Performance_PSxG'].fillna(0), weights=weights)
    
    return metrics


def compute_relative_strengths(team_home_stats, team_away_stats, league_home_stats, league_away_stats):
    """
    Compute relative attack and defense strengths.
    
    Args:
        team_home_stats: Team's home averages (from compute_team_season_averages)
        team_away_stats: Team's away averages
        league_home_stats: League home averages
        league_away_stats: League away averages
        
    Returns:
        dict: Relative strengths
    """
    if not all([team_home_stats, team_away_stats, league_home_stats, league_away_stats]):
        return None
    
    strengths = {
        'attack_strength_home': (team_home_stats['xg_avg'] / league_home_stats['xg_avg'] 
                                 if league_home_stats['xg_avg'] > 0 else 1.0),
        'defense_strength_home': (team_home_stats['xga_avg'] / league_away_stats['xg_avg'] 
                                  if league_away_stats['xg_avg'] > 0 else 1.0),
        'attack_strength_away': (team_away_stats['xg_avg'] / league_away_stats['xg_avg'] 
                                if league_away_stats['xg_avg'] > 0 else 1.0),
        'defense_strength_away': (team_away_stats['xga_avg'] / league_home_stats['xg_avg'] 
                                 if league_home_stats['xg_avg'] > 0 else 1.0),
    }
    
    return strengths


def get_head_to_head(dataframe, team_a, team_b, current_season=None, max_matches=20):
    """
    Get head-to-head match history between two teams.
    
    Args:
        dataframe: Master dataframe
        team_a: First team name
        team_b: Second team name
        current_season: Most recent season for weighting
        max_matches: Maximum number of matches to return
        
    Returns:
        pd.DataFrame: H2H matches, sorted by date (most recent last)
    """
    if current_season is None:
        current_season = get_current_season(dataframe)
    
    # Find matches where A played B or B played A
    h2h = dataframe[
        ((dataframe['team'] == team_a) & (dataframe['opponent'] == team_b)) |
        ((dataframe['team'] == team_b) & (dataframe['opponent'] == team_a))
    ].copy()
    
    if len(h2h) == 0:
        return pd.DataFrame()
    
    # Apply season weights
    h2h = apply_season_weights(h2h, current_season)
    
    # Sort by date (most recent last)
    h2h = h2h.sort_values('date').tail(max_matches)
    
    return h2h


def compute_h2h_averages(h2h_df):
    """
    Compute weighted averages from H2H data.
    
    Args:
        h2h_df: Head-to-head matches dataframe
        
    Returns:
        dict: H2H averages (split by team perspective)
    """
    if len(h2h_df) == 0:
        return None
    
    weights = h2h_df['season_weight'].values
    total_weight = weights.sum()
    
    if total_weight == 0:
        return None
    
    # For team A's perspective when they played B
    # We'll compute overall averages and let caller handle perspective
    h2h_avg = {
        'xg_avg': np.average(h2h_df['xG'].fillna(0), weights=weights),
        'xga_avg': np.average(h2h_df['xGA'].fillna(0), weights=weights),
        'gf_avg': np.average(h2h_df['GF'].fillna(0), weights=weights),
        'ga_avg': np.average(h2h_df['GA'].fillna(0), weights=weights),
        'shots_avg': np.average(h2h_df['Standard_Sh'].fillna(0), weights=weights),
        'poss_avg': np.average(h2h_df['Poss'].fillna(0), weights=weights),
        'matches': len(h2h_df)
    }
    
    return h2h_avg

