"""
Check and normalize UCL teams in API CSV.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name

def check_ucl_teams():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    ucl = df[df['league'] == 'UEFA Champions League']
    teams = sorted(set(ucl['home_team_name'].dropna().unique()) | 
                  set(ucl['away_team_name'].dropna().unique()))
    
    print('='*70)
    print('UCL TEAM NORMALIZATION CHECK')
    print('='*70)
    print(f'\nTotal UCL teams: {len(teams)}\n')
    
    mismatches = []
    for team in teams:
        normalized = normalize_team_name(team)
        if team != normalized:
            mismatches.append((team, normalized))
            print(f'  "{team}" -> "{normalized}"')
    
    print(f'\nTotal mismatches: {len(mismatches)}')
    
    if len(mismatches) == 0:
        print('\n[OK] All UCL teams normalized correctly!')
    else:
        print('\n[WARNING] Some teams need normalization fixes')
    
    return mismatches

if __name__ == "__main__":
    check_ucl_teams()

