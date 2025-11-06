"""
Team strengths persistence module.
Saves and loads computed team strengths for reuse across leagues.
"""

import json
from pathlib import Path
import pandas as pd


def save_team_strengths(strength_dict, league):
    """
    Save team strengths to JSON file.
    
    Args:
        strength_dict: Dictionary mapping team names to their strengths
        league: League name (e.g., 'EPL', 'UCL')
    """
    path = Path("data/team_strengths")
    path.mkdir(parents=True, exist_ok=True)
    
    file_path = path / f"{league.lower()}_strengths.json"
    
    with open(file_path, "w") as f:
        json.dump(strength_dict, f, indent=4)
    
    print(f"Saved {len(strength_dict)} team strengths for {league} to {file_path}")


def load_team_strengths(league):
    """
    Load team strengths from JSON file.
    
    Args:
        league: League name (e.g., 'EPL', 'UCL')
        
    Returns:
        dict: Dictionary of team strengths, or None if file doesn't exist
    """
    path = Path("data/team_strengths") / f"{league.lower()}_strengths.json"
    
    if not path.exists():
        return None
    
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading strengths for {league}: {e}")
        return None


def get_all_league_strengths():
    """
    Load strengths for all leagues.
    
    Returns:
        dict: Dictionary mapping league names to their strengths
    """
    path = Path("data/team_strengths")
    if not path.exists():
        return {}
    
    all_strengths = {}
    leagues = ['epl', 'laliga', 'bundesliga', 'seriea', 'ligue1', 'mls', 'ucl']
    
    for league in leagues:
        strengths = load_team_strengths(league)
        if strengths:
            all_strengths[league.upper()] = strengths
    
    return all_strengths


def save_all_strengths_to_csv(all_strengths_dict):
    """
    Save all league strengths to a master CSV file.
    
    Args:
        all_strengths_dict: Dictionary mapping league names to team strengths
    """
    path = Path("data/team_strengths_master.csv")
    
    rows = []
    for league, strengths in all_strengths_dict.items():
        for team, team_data in strengths.items():
            row = {
                'league': league,
                'team': team,
                'home_attack': team_data.get('home_attack', 1.0),
                'away_attack': team_data.get('away_attack', 1.0),
                'home_defense': team_data.get('home_defense', 1.0),
                'away_defense': team_data.get('away_defense', 1.0),
                'strength': team_data.get('strength', 1.0),
                'is_new_team': team_data.get('is_new_team', False)
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"Saved {len(rows)} team strengths to {path}")


def _average_strengths(rows):
    """
    Compute average of strengths across provided rows.
    Rows: iterable of dicts with keys: home_attack, away_attack, home_defense, away_defense, strength.
    Returns dict with same keys. Ignores missing keys gracefully.
    """
    keys = [
        'home_attack', 'away_attack', 'home_defense', 'away_defense', 'strength'
    ]
    sums = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    for row in rows:
        if not isinstance(row, dict):
            continue
        for k in keys:
            v = row.get(k)
            if isinstance(v, (int, float)):
                sums[k] += float(v)
                counts[k] += 1
    averages = {}
    for k in keys:
        averages[k] = (sums[k] / counts[k]) if counts[k] > 0 else 1.0
    return averages


def compute_global_strength_baseline(all_strengths_dict):
    """
    Compute dynamic global baseline across all leagues (including UCL) using existing strengths.
    Used as fallback for missing teams. No hardcoded 1.0.
    """
    rows = []
    for league_strengths in all_strengths_dict.values():
        for team_data in league_strengths.values():
            rows.append(team_data)
    if not rows:
        return {
            'home_attack': 1.0, 'away_attack': 1.0,
            'home_defense': 1.0, 'away_defense': 1.0,
            'strength': 1.0
        }
    return _average_strengths(rows)


def _find_domestic_strength_for_team(team_name, all_strengths_dict):
    """
    Find the first domestic league strengths for a team (exclude UCL).
    Returns tuple (league_code, strength_dict) or (None, None) if not found.
    """
    for league_code, league_strengths in all_strengths_dict.items():
        if league_code.upper() == 'UCL':
            continue
        team_strength = league_strengths.get(team_name)
        if team_strength:
            return league_code, team_strength
    return None, None


def get_ucl_adjusted_team_strength(team_name, all_strengths_dict, ucl_weight=0.6, domestic_weight=0.4):
    """
    Return adjusted strength for a UCL team using combined UCL + domestic data when possible.

    Rules:
    - If team has both UCL and domestic strengths: combine (ucl_weight, domestic_weight) per metric
    - If team only has UCL: use UCL strength only
    - If team only has domestic: use domestic strength only
    - If missing both: use dynamic global average baseline
    """
    ucl_strengths = all_strengths_dict.get('UCL', {}).get(team_name)
    _, domestic_strengths = _find_domestic_strength_for_team(team_name, all_strengths_dict)

    if ucl_strengths and domestic_strengths:
        combined = {}
        for k in ['home_attack', 'away_attack', 'home_defense', 'away_defense', 'strength']:
            u = ucl_strengths.get(k)
            d = domestic_strengths.get(k)
            if isinstance(u, (int, float)) and isinstance(d, (int, float)):
                combined[k] = ucl_weight * float(u) + domestic_weight * float(d)
            elif isinstance(u, (int, float)):
                combined[k] = float(u)
            elif isinstance(d, (int, float)):
                combined[k] = float(d)
        return combined

    if ucl_strengths and not domestic_strengths:
        return ucl_strengths

    if domestic_strengths and not ucl_strengths:
        return domestic_strengths

    # Missing both → global baseline
    global_baseline = compute_global_strength_baseline(all_strengths_dict)
    return global_baseline

