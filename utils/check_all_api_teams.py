"""
Comprehensive check of ALL teams in API CSV to verify normalization.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name

def check_all_api_teams():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    
    # Get all unique teams across all leagues
    all_teams = sorted(set(df['home_team_name'].dropna().unique()) | 
                      set(df['away_team_name'].dropna().unique()))
    
    print('='*70)
    print('COMPREHENSIVE API TEAM NORMALIZATION CHECK')
    print('='*70)
    print(f'\nTotal unique teams in API CSV: {len(all_teams)}\n')
    
    mismatches = []
    normalized_teams = []
    
    for team in all_teams:
        normalized = normalize_team_name(team)
        if team != normalized:
            mismatches.append((team, normalized))
        else:
            normalized_teams.append(team)
    
    print(f'Teams that need normalization: {len(mismatches)}')
    print(f'Teams already normalized: {len(normalized_teams)}')
    print('\n' + '-'*70)
    
    if mismatches:
        print('\nTeams that need normalization fixes:')
        print('-'*70)
        for orig, norm in mismatches:
            print(f'  "{orig}" -> "{norm}"')
    else:
        print('\n[OK] All teams are properly normalized!')
    
    print('\n' + '='*70)
    print(f'SUMMARY: {len(mismatches)} teams need normalization out of {len(all_teams)} total')
    print('='*70)
    
    if len(mismatches) == 0:
        print('\n[OK] ALL API TEAMS ARE FULLY NORMALIZED!')
    
    return mismatches

if __name__ == "__main__":
    check_all_api_teams()

