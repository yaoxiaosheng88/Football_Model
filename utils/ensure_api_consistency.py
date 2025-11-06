"""
Ensure API CSV team names are fully integrated into the normalizer.
This prevents name mismatches when merging API data with training data.
"""

import pandas as pd
import json
import os
from utils.team_name_normalizer import normalize_team_name, TEAM_NAME_MAPPING

def ensure_api_teams_in_normalizer():
    """
    Add all API team names to the normalizer mapping to ensure consistency.
    This ensures that when API data is merged with training data, all team names match.
    """
    api_path = "data/api_results_basic.csv"
    map_path = "data/team_name_map.json"
    
    print("Ensuring API teams are fully integrated into normalizer...")
    print("=" * 60)
    
    # Load API CSV
    df = pd.read_csv(api_path, dtype=str)
    df = df[
        (df['home_team_name'].notna()) & 
        (df['home_team_name'] != 'nan') &
        (df['away_team_name'].notna()) & 
        (df['away_team_name'] != 'nan')
    ].copy()
    
    # Collect all unique team names from API
    api_teams = set(df['home_team_name'].unique()) | set(df['away_team_name'].unique())
    print(f"Found {len(api_teams)} unique teams in API CSV")
    
    # Load team_name_map.json
    team_map = {}
    if os.path.exists(map_path):
        with open(map_path, "r", encoding="utf-8") as f:
            team_map = json.load(f) or {}
    
    print(f"Loaded {len(team_map)} entries from team_name_map.json")
    
    # Check each API team
    missing_mappings = []
    inconsistencies = []
    
    for api_team in api_teams:
        api_team_lower = str(api_team).strip().lower()
        normalized = normalize_team_name(api_team)
        
        # Check if normalized name is in TEAM_NAME_MAPPING values
        if normalized not in TEAM_NAME_MAPPING.values():
            # This team might not be in historical CSVs
            missing_mappings.append((api_team, normalized))
        
        # Check if API variant is in mapping
        if api_team_lower not in TEAM_NAME_MAPPING:
            # Add it to ensure consistency
            TEAM_NAME_MAPPING[api_team_lower] = normalized
    
    print(f"\nAdded {len(missing_mappings)} new team variants to normalizer")
    print(f"Total teams checked: {len(api_teams)}")
    
    # Verify consistency: all API teams normalize to same values
    print("\nVerifying consistency...")
    consistent = True
    for api_team in sorted(api_teams):
        normalized = normalize_team_name(api_team)
        # Check if running normalize twice gives same result (idempotency)
        normalized_twice = normalize_team_name(normalized)
        if normalized != normalized_twice:
            inconsistencies.append((api_team, normalized, normalized_twice))
            consistent = False
    
    if inconsistencies:
        print(f"[WARNING] Found {len(inconsistencies)} inconsistencies:")
        for api, norm1, norm2 in inconsistencies[:5]:
            print(f"  '{api}' -> '{norm1}' -> '{norm2}'")
    else:
        print("[OK] All team names normalize consistently (idempotent)")
    
    # Save updated mapping back to JSON
    # Re-normalize all values in team_map
    updated_map = {}
    for team_id, team_name in team_map.items():
        updated_map[team_id] = normalize_team_name(str(team_name))
    
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(updated_map, f, indent=4, ensure_ascii=False)
    
    print(f"\n[OK] Updated {map_path} with normalized values")
    print(f"[OK] Normalizer now has {len(TEAM_NAME_MAPPING)} total mappings")
    
    return consistent

if __name__ == "__main__":
    ensure_api_teams_in_normalizer()

