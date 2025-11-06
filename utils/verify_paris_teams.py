"""
Verify Paris FC and Paris Saint-Germain are correctly separated.
"""

import pandas as pd
from utils.team_name_normalizer import normalize_team_name

def verify_paris_teams():
    df = pd.read_csv('data/api_results_basic.csv', dtype=str)
    
    print('='*70)
    print('PARIS TEAMS VERIFICATION')
    print('='*70)
    
    id1045_rows = df[(df['home_team_id']=='1045.0') | (df['away_team_id']=='1045.0')]
    id524_rows = df[(df['home_team_id']=='524.0') | (df['away_team_id']=='524.0')]
    
    print(f'\nTeam ID 1045 instances: {len(id1045_rows)}')
    print(f'Team ID 524 instances: {len(id524_rows)}')
    
    # Check team names
    if len(id1045_rows) > 0:
        id1045_name = id1045_rows.iloc[0]['home_team_name'] if id1045_rows.iloc[0]['home_team_id']=='1045.0' else id1045_rows.iloc[0]['away_team_name']
        print(f'\nTeam ID 1045 name: "{id1045_name}"')
        print(f'  Normalizes to: "{normalize_team_name(id1045_name)}"')
    
    if len(id524_rows) > 0:
        id524_name = id524_rows.iloc[0]['home_team_name'] if id524_rows.iloc[0]['home_team_id']=='524.0' else id524_rows.iloc[0]['away_team_name']
        print(f'\nTeam ID 524 name: "{id524_name}"')
        print(f'  Normalizes to: "{normalize_team_name(id524_name)}"')
    
    # Check all unique names for these IDs (only their own names, not opponents)
    id1045_home_names = set(id1045_rows[id1045_rows['home_team_id']=='1045.0']['home_team_name'].dropna().unique())
    id1045_away_names = set(id1045_rows[id1045_rows['away_team_id']=='1045.0']['away_team_name'].dropna().unique())
    id1045_names = id1045_home_names | id1045_away_names
    
    id524_home_names = set(id524_rows[id524_rows['home_team_id']=='524.0']['home_team_name'].dropna().unique())
    id524_away_names = set(id524_rows[id524_rows['away_team_id']=='524.0']['away_team_name'].dropna().unique())
    id524_names = id524_home_names | id524_away_names
    
    print(f'\nUnique names for ID 1045 (own team only): {id1045_names}')
    print(f'Unique names for ID 524 (own team only): {id524_names}')
    
    if id1045_names == {'Paris FC'} and id524_names == {'Paris Saint-Germain'}:
        print('\n[OK] Paris teams are correctly separated!')
        return True
    else:
        print('\n[WARNING] Paris teams may not be correctly separated')
        return False

if __name__ == "__main__":
    verify_paris_teams()

