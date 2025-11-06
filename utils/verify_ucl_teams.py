"""
Verify UCL teams normalization by showing each team and its normalized form.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name

def verify_ucl_teams():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    ucl = df[df['league'] == 'UEFA Champions League']
    teams = sorted(set(ucl['home_team_name'].dropna().unique()) | 
                  set(ucl['away_team_name'].dropna().unique()))
    
    print('='*70)
    print('UCL TEAM NORMALIZATION VERIFICATION')
    print('='*70)
    print(f'\nTotal UCL teams: {len(teams)}\n')
    
    print('Team name -> Normalized form:')
    print('-'*70)
    
    issues = []
    for team in teams:
        normalized = normalize_team_name(team)
        status = '[OK]' if team == normalized else '[FIX]'
        print(f'{status} {team:35} -> {normalized}')
        if team != normalized:
            issues.append((team, normalized))
    
    print('\n' + '='*70)
    print(f'Issues found: {len(issues)}')
    
    if issues:
        print('\nTeams that need normalization:')
        for orig, norm in issues:
            print(f'  "{orig}" -> "{norm}"')
    
    return issues

if __name__ == "__main__":
    verify_ucl_teams()

