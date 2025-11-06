"""
Weighting module for season and match recency weighting.
Uses specific seasonal weights as per requirements.
"""

import numpy as np
import pandas as pd


# Season weights mapping
SEASON_WEIGHTS = {
    2025: 1.0,
    2024: 0.7,
    2023: 0.5,
    2022: 0.3
}


def calculate_season_weight(season_year, current_season):
    """
    Calculate season weight based on recency.
    Uses specific weights: 2025: 1.0, 2024: 0.7, 2023: 0.5, 2022: 0.3.
    
    Args:
        season_year: Year of the season
        current_season: Most recent season year
        
    Returns:
        float: Season weight
    """
    if pd.isna(season_year) or pd.isna(current_season):
        return 1.0
    
    # Convert to int for lookup
    season_int = int(season_year) if not pd.isna(season_year) else None
    
    if season_int is None:
        return 1.0
    
    # Use specific weight if available
    if season_int in SEASON_WEIGHTS:
        return SEASON_WEIGHTS[season_int]
    
    # For seasons outside the mapping, use exponential decay
    diff = abs(current_season - season_int)
    if diff >= len(SEASON_WEIGHTS):
        # For older seasons, use the minimum weight
        return min(SEASON_WEIGHTS.values())
    
    # Linear interpolation for years between mapped values
    weights_list = sorted(SEASON_WEIGHTS.items(), key=lambda x: x[0], reverse=True)
    for i, (year, weight) in enumerate(weights_list):
        if season_int >= year:
            return weight
    
    # Fallback
    return 0.3


def get_current_season(dataframe):
    """
    Get the most recent season year from dataframe.
    
    Args:
        dataframe: DataFrame with 'season_year' column
        
    Returns:
        int: Most recent season year
    """
    if 'season_year' not in dataframe.columns:
        return None
    
    valid_seasons = dataframe['season_year'].dropna()
    if len(valid_seasons) == 0:
        return None
    
    return int(valid_seasons.max())


def calculate_match_decay_weights(matches, decay_factor=0.2):
    """
    Calculate decay weights for recent matches.
    More recent matches get higher weights.
    
    Args:
        matches: DataFrame with matches (should be sorted by date, most recent last)
        decay_factor: How quickly weight decays (higher = slower decay)
        
    Returns:
        np.array: Array of weights for each match
    """
    n = len(matches)
    if n == 0:
        return np.array([])
    
    # Create exponential decay weights
    # Most recent match gets weight 1.0
    weights = np.exp(decay_factor * np.arange(n))
    weights = weights / weights.sum() * n  # Normalize
    
    return weights


def apply_season_weights(dataframe, current_season=None):
    """
    Add season weights column to dataframe.
    
    Args:
        dataframe: DataFrame with 'season_year' column
        current_season: Most recent season (if None, calculated from data)
        
    Returns:
        pd.DataFrame: DataFrame with added 'season_weight' column
    """
    df = dataframe.copy()
    
    if current_season is None:
        current_season = get_current_season(df)
    
    if current_season is None:
        df['season_weight'] = 1.0
        return df
    
    df['season_weight'] = df['season_year'].apply(
        lambda x: calculate_season_weight(x, current_season)
    )
    
    return df


def _compute_team_season_performance(team_df):
    """
    Compute per-season performance metrics for a team.
    Returns DataFrame indexed by season_year with columns:
      matches, gf, ga, xg, xga, ppg, gdiff_per_match, xgdiff_per_match, score
    where score is a composite performance score in [~0, >1].
    """
    df = team_df.copy()
    # Derive points per match from Result if present
    def result_to_points(r):
        r = str(r).upper()
        if r == 'W':
            return 3
        if r == 'D':
            return 1
        if r == 'L':
            return 0
        return np.nan
    if 'Points' in df.columns:
        df['points'] = pd.to_numeric(df['Points'], errors='coerce')
    else:
        df['points'] = df['Result'].apply(result_to_points) if 'Result' in df.columns else np.nan
    grp = df.groupby('season_year', dropna=True)
    agg = grp.agg(
        matches=('GF', 'count'),
        gf=('GF', 'mean'),
        ga=('GA', 'mean'),
        xg=('xG', 'mean'),
        xga=('xGA', 'mean'),
        ppg=('points', 'mean')
    ).reset_index()
    if len(agg) == 0:
        return agg.set_index('season_year')
    agg['gdiff_per_match'] = agg['gf'] - agg['ga']
    agg['xgdiff_per_match'] = (agg['xg'].fillna(0) - agg['xga'].fillna(0))
    # Normalize ppg to [0,1] scale by dividing by 3
    agg['ppg_norm'] = (agg['ppg'].fillna(0) / 3.0).clip(lower=0, upper=1)
    # Composite score: favor actual goal difference, include xG diff and PPG
    agg['score'] = (
        0.4 * agg['gdiff_per_match'].fillna(0) +
        0.3 * agg['xgdiff_per_match'].fillna(0) +
        0.3 * agg['ppg_norm']
    )
    return agg.set_index('season_year')


def compute_team_dynamic_season_weights(team_df, current_season, k: float = 3.0):
    """
    Compute dynamic, team-specific season weights starting from SEASON_WEIGHTS,
    then adjust the current season weight up/down based on deviation from
    multi-season average performance.
    Returns dict mapping season_year -> adjusted_weight (normalized to sum=1 for the team's seasons).
    """
    if current_season is None or 'season_year' not in team_df.columns:
        # Fallback to static weights normalized over this team's seasons
        seasons = team_df['season_year'].dropna().astype(int).unique().tolist()
        if not seasons:
            return {}
        base = {y: calculate_season_weight(y, max(seasons)) for y in seasons}
        s = sum(base.values()) or 1.0
        return {y: w / s for y, w in base.items()}
    perf = _compute_team_season_performance(team_df)
    seasons = perf.index.astype(int).tolist()
    if not seasons:
        return {}
    # Base weights for this team
    base_weights = {y: calculate_season_weight(y, current_season) for y in seasons}
    # Compute logistic adjustments per season based on performance vs prior seasons
    adjusted = {}
    for y in seasons:
        score_y = perf.loc[y, 'score'] if y in perf.index else np.nan
        # prior seasons for comparison
        prior = perf[perf.index < y]
        if len(prior) == 0 or pd.isna(score_y):
            adjusted[y] = base_weights.get(y, 0.0)
            continue
        prior_mean = prior['score'].mean()
        if pd.isna(prior_mean) or abs(prior_mean) < 1e-6:
            adjusted[y] = base_weights.get(y, 0.0)
            continue
        performance_ratio = score_y / prior_mean
        # Logistic scaling in (0,1): higher when ratio>1
        logistic = 1.0 / (1.0 + np.exp(-k * (performance_ratio - 1.0)))
        # Rescale around base weight by mixing: emphasize seasons with higher logistic
        # Combine base with logistic emphasis; map logistic (0..1) to multiplier ~ (0.8..1.2)
        mult = 0.8 + 0.4 * logistic
        adjusted[y] = base_weights.get(y, 0.0) * mult
    # Normalize to sum=1 over this team's seasons
    total = sum(adjusted.values())
    if total <= 0:
        n = len(adjusted)
        return {y: 1.0 / n for y in adjusted}
    return {y: w / total for y, w in adjusted.items()}


def apply_dynamic_season_weights(dataframe, team, current_season=None):
    """
    Add team-specific dynamic season weights as 'season_weight' using
    compute_team_dynamic_season_weights.
    """
    df = dataframe.copy()
    if current_season is None:
        current_season = get_current_season(df)
    if current_season is None:
        df['season_weight'] = 1.0
        return df
    team_df = df[df['team'] == team].copy()
    if len(team_df) == 0:
        df['season_weight'] = 1.0
        return df
    weight_map = compute_team_dynamic_season_weights(team_df, current_season)
    # If we failed to compute, fall back to static weights
    if not weight_map:
        return apply_season_weights(df, current_season)
    df['season_weight'] = df['season_year'].apply(lambda y: weight_map.get(int(y) if not pd.isna(y) else y, 0.0))
    # For rows not in the team (when caller passes full df), keep original weight 0; caller should subset before using
    return df
