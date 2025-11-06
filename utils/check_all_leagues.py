"""
Comprehensive check of all league team normalization in API CSV.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name

def check_all_leagues():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    leagues = ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Ligue 1']
    
    print('='*70)
    print('COMPREHENSIVE LEAGUE NORMALIZATION CHECK')
    print('='*70)
    
    total_issues = 0
    all_mismatches = {}
    
    for league in leagues:
        ll_df = df[df['league'] == league]
        teams = sorted(set(ll_df['home_team_name'].dropna().unique()) | 
                      set(ll_df['away_team_name'].dropna().unique()))
        mismatches = [(t, normalize_team_name(t)) for t in teams if t != normalize_team_name(t)]
        
        print(f'\n{league}:')
        print(f'  Total teams: {len(teams)}')
        print(f'  Mismatches: {len(mismatches)}')
        
        if mismatches:
            total_issues += len(mismatches)
            all_mismatches[league] = mismatches
            for a, b in mismatches:
                print(f'    "{a}" -> "{b}"')
        else:
            print('  [OK] All teams normalized correctly!')
    
    print('\n' + '='*70)
    print(f'OVERALL: {total_issues} total mismatches across all leagues')
    print('='*70)
    
    if total_issues == 0:
        print('\n[OK] ALL LEAGUES FULLY NORMALIZED!')
    else:
        print(f'\n[WARNING] {len(all_mismatches)} leagues have normalization issues')
    
    return all_mismatches

if __name__ == "__main__":
    check_all_leagues()

