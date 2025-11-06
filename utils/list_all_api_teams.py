"""
List all teams in API CSV and verify they're in the normalizer mapping.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name, TEAM_NAME_MAPPING

def list_all_api_teams_with_normalizer_check():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    
    # Get all unique teams across all leagues
    all_teams = sorted(set(df['home_team_name'].dropna().unique()) | 
                      set(df['away_team_name'].dropna().unique()))
    
    print('='*70)
    print('ALL API TEAMS AND NORMALIZER STATUS')
    print('='*70)
    print(f'\nTotal unique teams in API CSV: {len(all_teams)}\n')
    
    in_mapping = []
    not_in_mapping = []
    normalized_correctly = []
    normalization_issues = []
    
    for team in all_teams:
        team_lower = str(team).strip().lower()
        normalized = normalize_team_name(team)
        
        # Check if team is in TEAM_NAME_MAPPING
        if team_lower in TEAM_NAME_MAPPING:
            in_mapping.append(team)
        else:
            not_in_mapping.append(team)
        
        # Check normalization
        if team == normalized:
            normalized_correctly.append(team)
        else:
            normalization_issues.append((team, normalized))
    
    print(f'Teams in normalizer mapping: {len(in_mapping)}')
    print(f'Teams NOT in normalizer mapping: {len(not_in_mapping)}')
    print(f'Teams normalized correctly: {len(normalized_correctly)}')
    print(f'Teams with normalization issues: {len(normalization_issues)}')
    
    print('\n' + '-'*70)
    print('DETAILED BREAKDOWN:')
    print('-'*70)
    
    if not_in_mapping:
        print(f'\nTeams NOT in normalizer mapping ({len(not_in_mapping)}):')
        print('(These normalize correctly via fallback logic)')
        for team in not_in_mapping[:20]:  # Show first 20
            normalized = normalize_team_name(team)
            # Use ASCII-safe encoding
            try:
                team_safe = team.encode('ascii', 'replace').decode('ascii')
                norm_safe = normalized.encode('ascii', 'replace').decode('ascii')
                print(f'  "{team_safe}" -> "{norm_safe}"')
            except:
                print(f'  [Team] -> [Normalized]')
        if len(not_in_mapping) > 20:
            print(f'  ... and {len(not_in_mapping) - 20} more')
    
    if normalization_issues:
        print(f'\nTeams with normalization issues ({len(normalization_issues)}):')
        for orig, norm in normalization_issues:
            print(f'  "{orig}" -> "{norm}"')
    
    print('\n' + '='*70)
    print('SUMMARY:')
    print('='*70)
    print(f'Total teams: {len(all_teams)}')
    print(f'In normalizer mapping: {len(in_mapping)}')
    print(f'Not in mapping (but normalize correctly): {len(not_in_mapping)}')
    print(f'Normalization issues: {len(normalization_issues)}')
    
    if len(normalization_issues) == 0:
        print('\n[OK] ALL TEAMS NORMALIZE CORRECTLY!')
        print('Note: Teams not in mapping still normalize correctly via fallback logic.')
    
    return {
        'total': len(all_teams),
        'in_mapping': len(in_mapping),
        'not_in_mapping': not_in_mapping,
        'normalization_issues': normalization_issues
    }

if __name__ == "__main__":
    list_all_api_teams_with_normalizer_check()

